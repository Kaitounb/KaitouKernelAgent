"""Trusted development deployment store. Workers never see or write this directory."""
import contextlib,fcntl,hashlib,json,os,shutil,stat,uuid
from pathlib import Path

DATA=Path(__file__).with_name('data')


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def canonical(value):return (json.dumps(value,sort_keys=True,indent=2)+'\n').encode()


def read_regular(path,limit):
    fd=os.open(path,os.O_RDONLY|os.O_NOFOLLOW|os.O_NONBLOCK)
    with os.fdopen(fd,'rb') as file:
        info=os.fstat(file.fileno())
        if not stat.S_ISREG(info.st_mode) or info.st_size>limit:raise ValueError('registry file type/size invalid')
        data=file.read(limit+1)
        if len(data)>limit:raise ValueError('registry file too large')
        return data


def index(channel='development'):
    if channel not in ['development','qualified']:raise ValueError('unknown release channel')
    if channel=='qualified':
        try:
            value=json.loads(read_regular(DATA/'qualified_releases.json',65536));raw=read_regular(DATA/'gemm_qualification.json',65536);qualification=json.loads(raw)
        except FileNotFoundError:raise ValueError('qualified GEMM release is not packaged; use an accepted release build') from None
        if not isinstance(value,dict) or not isinstance(qualification,dict) or value.get('qualification_sha256')!=hashlib.sha256(raw).hexdigest() or qualification.get('status')!='QUALIFIED' or qualification.get('backend')!='cublas' or qualification.get('production_release') is not True or qualification.get('scope')!='SM86_FIXED_IMAGE_GEMM_CUBLAS' or any(qualification.get(k) is not True for k in ['independent_correctness_pass','recovery_pass','final_pairing_pass','raw_audit_pass']):raise ValueError('qualified release certificate invalid')
        environment=qualification.get('environment');binaries=qualification.get('binary_sha256')
        if not isinstance(environment,dict) or set(environment)!={'image_id','torch','cuda','device'} or any(not isinstance(v,str) for v in environment.values()) or not isinstance(binaries,list) or len(binaries)!=2 or any(not isinstance(v,str) or len(v)!=64 or any(c not in '0123456789abcdef' for c in v) for v in binaries) or len(set(binaries))!=2:raise ValueError('qualified environment/artifact schema invalid')
        if value.get('schema_version')!=1 or not isinstance(value.get('versions'),dict) or len(value['versions'])!=2 or any(not isinstance(value.get(k),str) or value[k] not in value['versions'] for k in ['default','fallback']) or value['default']==value['fallback'] or any(not isinstance(meta,dict) or not {'version','contract_sha256','backend','production_release','binary_sha256'}<=set(meta) for meta in value['versions'].values()):raise ValueError('qualified release catalog invalid')
    else:value=json.loads(read_regular(DATA/'releases.json',65536))
    contract=sha(DATA/'gemm_contract_v1.json')
    for version,meta in value['versions'].items():
        if meta['version']!=version or meta['contract_sha256']!=contract or meta['backend']!='cublas' or meta['production_release'] is not (channel=='qualified'):raise ValueError('bundled release contract mismatch')
        if channel=='qualified' and (meta['binary_sha256'] not in qualification['binary_sha256'] or any(meta.get(k)!=qualification['environment'][k] for k in ['image_id','torch','cuda','device'])):raise ValueError('qualified artifact/environment differs')
    return value


def atomic_json(path,value):
    path=Path(path);temp=path.with_name('.'+path.name+'-'+uuid.uuid4().hex)
    try:
        with temp.open('xb') as file:file.write(canonical(value));file.flush();os.fsync(file.fileno())
        os.replace(temp,path)
        fd=os.open(path.parent,os.O_RDONLY)
        try:os.fsync(fd)
        finally:os.close(fd)
    finally:temp.unlink(missing_ok=True)


class Registry:
    def __init__(self,root,*,channel='development',data=None,catalog=None):
        self.root=Path(root).resolve();self.channel=channel
        self.data=DATA if data is None else Path(data)
        self.catalog=index(channel) if catalog is None else catalog

    @contextlib.contextmanager
    def locked(self):
        fd=os.open(self.root/'lock',os.O_CREAT|os.O_RDWR|os.O_NOFOLLOW,0o600)
        try:
            fcntl.flock(fd,fcntl.LOCK_EX);yield
        finally:fcntl.flock(fd,fcntl.LOCK_UN);os.close(fd)

    def initialize(self):
        self.root.mkdir(mode=0o700,parents=False,exist_ok=False)
        (self.root/'versions').mkdir();(self.root/'runs').mkdir()
        for version,meta in self.catalog['versions'].items():
            folder=self.root/'versions'/version;folder.mkdir()
            source=self.data/'artifacts'/(meta['binary_sha256']+'.so')
            if hashlib.sha256(read_regular(source,8*1024*1024)).hexdigest()!=meta['binary_sha256']:raise ValueError('bundled binary corrupt')
            shutil.copyfile(source,folder/'nka_gemm_v1.so');(folder/'nka_gemm_v1.so').chmod(0o444)
            (folder/'manifest.json').write_bytes(canonical(meta));(folder/'manifest.json').chmod(0o444)
        fallback=self.catalog['versions'][self.catalog['fallback']]
        shutil.copyfile(self.data/'artifacts'/(fallback['binary_sha256']+'.so'),self.root/'fallback.so');(self.root/'fallback.so').chmod(0o444)
        with self.locked():atomic_json(self.root/'current.json',{'active':self.catalog['default'],'previous':self.catalog['fallback']})
        return self.status()

    def pointer(self):
        value=json.loads(read_regular(self.root/'current.json',65536))
        if set(value)!= {'active','previous'} or any(type(value[k]) is not str or value[k] not in self.catalog['versions'] for k in value):raise ValueError('invalid version pointer')
        return value

    def release(self,version):
        if version not in self.catalog['versions']:raise ValueError('unknown approved development version')
        expected=self.catalog['versions'][version];folder=self.root/'versions'/version
        if folder.is_symlink():raise ValueError('version directory symlink rejected')
        if read_regular(folder/'manifest.json',65536)!=canonical(expected):raise ValueError('release metadata mismatch')
        binary=folder/'nka_gemm_v1.so'
        if hashlib.sha256(read_regular(binary,8*1024*1024)).hexdigest()!=expected['binary_sha256']:raise ValueError('release binary mismatch')
        return binary,expected.copy()

    def fallback(self):
        meta=self.catalog['versions'][self.catalog['fallback']].copy();binary=self.root/'fallback.so'
        if hashlib.sha256(read_regular(binary,8*1024*1024)).hexdigest()!=meta['binary_sha256']:raise ValueError('fallback binary corrupt')
        return binary,meta

    def select(self):
        try:
            pointer=self.pointer();binary,meta=self.release(pointer['active'])
            return binary,meta,None
        except (ValueError,OSError,RecursionError,TypeError) as error:
            binary,meta=self.fallback();return binary,meta,'active_unavailable: '+str(error)[:256]

    def activate(self,version):
        with self.locked():
            self.release(version);old=self.pointer()
            if version!=old['active']:atomic_json(self.root/'current.json',{'active':version,'previous':old['active']})
        return self.status()

    def rollback(self):
        with self.locked():
            old=self.pointer();self.release(old['previous'])
            atomic_json(self.root/'current.json',{'active':old['previous'],'previous':old['active']})
        return self.status()

    def status(self):
        binary,meta,reason=self.select()
        try:pointer=self.pointer()
        except (OSError,ValueError,TypeError,RecursionError):pointer=None
        return {'status':'qualified-runtime' if self.channel=='qualified' else 'development-only','production_release':meta['production_release'],'channel':self.channel,'pointer':pointer,'selected_version':meta['version'],'binary_sha256':meta['binary_sha256'],'fallback_reason':reason}
