"""Low-privilege fixed SDK runner using setpriv/libseccomp and the existing bounded I/O."""
import hashlib,json,os,platform,shutil,signal,subprocess,tempfile,time
from pathlib import Path
import numpy as np
from kernel_agent.bounded_io import capture
from kernel_agent.protocol import HEADER,META_LIMIT,decode_frame
from kernel_agent.output import load_output
from kernel_agent.registry import read_regular
from kernel_agent.runner import validate_inputs,digest


def validate_boundary(meta):
    expected={'uid':65534,'gid':65534,'capabilities_zero':True,'no_new_privs':True,'ipv4_socket_denied':True,'ipv6_socket_denied':True,'unix_ipc_available':True,'private_read_denied':True,'artifact_write_denied':True,'input_write_denied':True,'controller_signal_denied':True,'control_canary_in_env':False,'credentials_in_env':False,'torch':'2.7.1+cu126','cuda':'12.6','device':'sm89','execution':'reviewed-native','input_immutable':True,'producer_consumer_stream':True,'fresh_output':True}
    if any(type(meta.get(k)) is not type(v) or meta[k]!=v for k,v in expected.items()):raise ValueError('SM89 execution boundary invalid')


class NativeRunner:
    def __init__(self,binary,sha256,*,runtime):
        if os.geteuid()!=0 or platform.system()!='Linux' or platform.machine()!='x86_64':raise ValueError('SM89 trusted controller requires Linux x86_64 root to drop worker privileges')
        self.binary=Path(binary).resolve();self.sha256=sha256
        if digest(self.binary)!=sha256:raise ValueError('SM89 artifact hash mismatch')
        cfg=json.loads(read_regular(Path(runtime),65536));self.base=Path(cfg['worker_root']);self.python=Path(cfg['python']);self.cfg=cfg
        if cfg.get('status')!='PUBLIC_RUNTIME_READY' or not self.base.is_absolute() or self.base.is_symlink() or not self.python.resolve().is_relative_to(self.base):raise ValueError('SM89 worker runtime invalid')
        info=self.base.stat()
        if info.st_uid!=0 or info.st_gid!=65534 or info.st_mode&0o027:raise ValueError('SM89 worker software permissions invalid')
        if hashlib.sha256(read_regular(self.python.resolve(),64*1024*1024)).hexdigest()!=cfg['python_sha256'] or hashlib.sha256(read_regular(self.base/'software-manifest.json',16*1024*1024)).hexdigest()!=cfg['manifest_sha256']:raise ValueError('SM89 public runtime fingerprint invalid')
        if 'gpu_identity' in cfg:
            r=subprocess.run(['nvidia-smi','--query-gpu=uuid,name,driver_version','--format=csv,noheader'],capture_output=True,text=True,timeout=15)
            if r.returncode or r.stdout.strip()!=cfg['gpu_identity']:raise ValueError('SM89 validated GPU/driver differs')

    def run(self,a,b,work,*,case='cublas'):
        if case not in ['cublas','boundaries','crash','timeout','stdout_flood','stderr_flood']:raise ValueError('unsupported fixed SM89 fixture')
        validate_inputs(a,b);work=Path(work).resolve();work.mkdir(mode=0o700,parents=True,exist_ok=False)
        exposed=Path(tempfile.mkdtemp(prefix='job-',dir=self.base));os.chown(exposed,0,65534);exposed.chmod(0o750)
        artifact=exposed/'artifact';inputs=exposed/'inputs';scratch=exposed/'scratch'
        for p in [artifact,inputs]:p.mkdir();os.chown(p,0,65534);p.chmod(0o750)
        scratch.mkdir();os.chown(scratch,65534,65534);scratch.chmod(0o700)
        for name in ['pipe_worker.py','restrictions.py']:shutil.copyfile(Path(__file__).with_name(name),artifact/name)
        shutil.copyfile(Path(__file__).resolve().parents[1]/'protocol.py',artifact/'protocol.py');shutil.copyfile(self.binary,artifact/'nka_gemm_v1.so');(artifact/'readonly-guard').write_text('unchanged')
        np.save(inputs/'a.npy',a,allow_pickle=False);np.save(inputs/'b.npy',b,allow_pickle=False)
        private=work/'control-only-canary';private.write_text('trusted-controller-only')
        req={'case':case,'binary_sha256':self.sha256,'host_private_path':str(private),'host_pid':os.getpid()};(inputs/'request.json').write_text(json.dumps(req))
        for folder in [artifact,inputs]:
            for p in folder.iterdir():os.chown(p,0,65534);p.chmod(0o640)
            shutil.copytree(folder,work/folder.name)
        protected={str(p.relative_to(exposed)):digest(p) for folder in [artifact,inputs] for p in folder.iterdir()}
        cmd=['/usr/bin/env','-i','PATH=/usr/bin:/bin','HOME='+str(scratch),'TMPDIR='+str(scratch),'OMP_NUM_THREADS=1','OPENBLAS_NUM_THREADS=1','MKL_NUM_THREADS=1','CUDA_CACHE_DISABLE=1','/usr/bin/setpriv','--reuid=65534','--regid=65534','--clear-groups','--no-new-privs','--bounding-set=-all','--inh-caps=-all','--ambient-caps=-all',str(self.python),'-I','-B',str(artifact/'pipe_worker.py'),'--request',str(inputs/'request.json')]
        def owned():
            values=[]
            for p in Path('/proc').iterdir():
                if not p.name.isdigit():continue
                try:
                    args=(p/'cmdline').read_bytes().split(b'\0')
                    if str(inputs/'request.json').encode() in args and b'--request' in args:values.append(int(p.name))
                except OSError:pass
            return values
        def cleanup(reason):
            before=owned()
            for sig in [signal.SIGTERM,signal.SIGKILL]:
                for pid in owned():
                    try:
                        if os.getpgid(pid)==pid:os.killpg(pid,sig)
                        else:os.kill(pid,sig)
                    except ProcessLookupError:pass
                if not owned():break
                time.sleep(.2)
            remaining=owned();unchanged=all(digest(exposed/n)==h for n,h in protected.items())
            if not remaining:shutil.rmtree(exposed)
            return {'removed':not remaining and not exposed.exists(),'observed_owned_pids':before,'remaining_owned_pids':remaining,'exposed_files_unchanged':unchanged}
        limit=a.shape[0]*b.shape[1]*4+16384
        record=capture(cmd,work/'capture',timeout=8 if case=='timeout' else 45,stdout_limit=HEADER.size+META_LIMIT+limit,stderr_limit=65536,cleanup=cleanup)
        record.update(command=cmd,case=case,binary_sha256=self.sha256,protected_hashes=protected,protected_files_unchanged=all(digest(work/n)==h for n,h in protected.items()) and record['cleanup'].get('exposed_files_unchanged') is True,execution_boundary='uid65534-nnp-zero-caps-seccomp-fixed-cublas',container_isolation_verified=False,status='REJECT')
        actual=None
        if record['exit']==0 and record['reason'] is None and record['cleanup']['removed'] and record['protected_files_unchanged']:
            try:
                meta,payload=decode_frame(work/'capture/stdout.bin',limit);validate_boundary(meta)
                if meta['binary_sha256']!=self.sha256 or meta['case']!=case:raise ValueError('SM89 frame identity invalid')
                output=work/'result.npy';output.write_bytes(payload);actual=load_output(output,(a.shape[0],b.shape[1]))
                if not np.isfinite(actual).all():raise ValueError('nonfinite SM89 output')
                record.update(metadata=meta,status='OUTPUT_VALID')
            except (OSError,ValueError,EOFError) as error:actual=None;record['rejection']=str(error)[:512]
        else:record['rejection']=record['reason'] or 'worker_or_boundary_failure'
        (work/'record.json').write_text(json.dumps(record,indent=2)+'\n');return actual,record
