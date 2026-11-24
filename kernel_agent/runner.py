"""Bounded Docker adapter for reviewed GEMM artifacts. No model or promotion logic."""
import hashlib,json,os,shutil,subprocess,uuid
from pathlib import Path
import numpy as np
from .bounded_io import capture,confirmed_removed
from .protocol import HEADER,META_LIMIT,decode_frame
from .output import load_output

IMAGE='sha256:ce00f6dc2eea43dc508b90b548f624646d643a92d363aa715fdb5a3b4f49290a'


def digest(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def command_result(cmd):
    try:p=subprocess.run(cmd,capture_output=True,text=True,timeout=15)
    except subprocess.TimeoutExpired:return {'command':cmd,'exit':None,'stdout':'','stderr':'control command timeout','timed_out':True}
    return {'command':cmd,'exit':p.returncode,'stdout':p.stdout[:16384],'stderr':p.stderr[:4096]}


def validate_inputs(a,b):
    if a.dtype!=np.float16 or b.dtype!=np.float16 or a.ndim!=2 or b.ndim!=2 or not a.flags.c_contiguous or not b.flags.c_contiguous:raise ValueError('FP16 contiguous rank-2 inputs required')
    m,k=a.shape;kb,n=b.shape
    if k!=kb or not all(1<=d<=4096 for d in [m,n,k]):raise ValueError('unsupported dimensions')
    if not np.isfinite(a).all() or not np.isfinite(b).all() or np.any(np.abs(a)>4) or np.any(np.abs(b)>4):raise ValueError('finite inputs within [-4,4] required by contract')


class DockerRunner:
    def __init__(self,binary,sha256):
        self.binary=Path(binary).resolve();self.sha256=sha256
        if digest(self.binary)!=sha256:raise ValueError('artifact hash mismatch')

    def run(self,a,b,work,*,case='cublas',host_private_path='/not-mounted',host_pid=None):
        # This API only enables reviewed fixed fixtures. Generated source remains disabled.
        allowed={'cublas','template','boundaries','constant','pollution','crash','timeout','stdout_flood','stderr_flood','metadata_nested','metadata_forged','bad_frame','memory_oom','pids_limit','disk_limit'}
        if case not in allowed:raise ValueError('unsupported reviewed fixture')
        validate_inputs(a,b)
        m,k=a.shape;n=b.shape[1]
        work=Path(work).resolve();work.mkdir(parents=True,exist_ok=False);work.chmod(0o700)
        artifact=work/'artifact';artifact.mkdir();inputs=work/'inputs';inputs.mkdir()
        # Docker mounts bypass parent permissions; every exposed source itself is public/read-only.
        for name in ['pipe_worker.py','protocol.py']:shutil.copyfile(Path(__file__).with_name(name),artifact/name)
        shutil.copyfile(self.binary,artifact/'nka_gemm_v1.so')
        if digest(artifact/'nka_gemm_v1.so')!=self.sha256:raise ValueError('staged artifact mismatch')
        (artifact/'readonly-guard').write_text('unchanged');(artifact/'readonly-guard').chmod(0o666)
        np.save(inputs/'a.npy',a,allow_pickle=False);np.save(inputs/'b.npy',b,allow_pickle=False)
        req={'case':case,'binary_sha256':self.sha256,'host_private_path':host_private_path,'host_pid':host_pid or os.getpid()}
        (inputs/'request.json').write_text(json.dumps(req));(inputs/'request.json').chmod(0o666)
        for folder in [artifact,inputs]:folder.chmod(0o755)
        before={str(p.relative_to(work)):digest(p) for folder in [artifact,inputs] for p in folder.iterdir()}
        name='nka-bounded-'+uuid.uuid4().hex[:12];memory='256m' if case=='memory_oom' else '2g'
        cmd=['docker','run','--name',name,'--pull=never','--network=none','--sig-proxy=false','--read-only','--log-driver=none','--cap-drop=ALL','--security-opt=no-new-privileges','--user','65534:65534','--init',
             '--pids-limit=32','--memory='+memory,'--memory-swap='+memory,'--cpus=2','--ulimit','fsize=16777216:16777216',
             '--env','HOME=/tmp','--env','PYTHONDONTWRITEBYTECODE=1','--env','OMP_NUM_THREADS=1','--env','MKL_NUM_THREADS=1','--env','OPENBLAS_NUM_THREADS=1',
             '--tmpfs','/tmp:rw,nosuid,nodev,noexec,size=128m','--tmpfs','/outputs:rw,nosuid,nodev,noexec,size=8m,mode=1777',
             '--mount',f'type=bind,src={artifact},dst=/artifact,readonly','--mount',f'type=bind,src={inputs},dst=/inputs,readonly']
        gpu=case not in {'stdout_flood','stderr_flood','metadata_nested','metadata_forged','bad_frame','memory_oom','pids_limit','disk_limit'}
        if gpu:cmd+=['--gpus','device=0']
        cmd+=['--entrypoint','/opt/venv/bin/python',IMAGE,'/artifact/pipe_worker.py','--request','/inputs/request.json']
        def cleanup(reason):
            info={'inspect':command_result(['docker','inspect','--format','{{json .State}}',name])}
            if reason:info['processes_before_cleanup']=command_result(['docker','top',name,'-eo','pid,ppid,args'])
            info['remove']=command_result(['docker','rm','-f',name])
            after=command_result(['docker','inspect',name]);info['inspect_after_cleanup']=after
            info['removed']=confirmed_removed(after)
            if reason:
                pids=[]
                top=info['processes_before_cleanup']
                if top['exit']==0:
                    for line in top['stdout'].splitlines()[1:]:
                        try:pids.append(int(line.split()[0]))
                        except (ValueError,IndexError):pass
                info['observed_host_pids']=pids
                info['host_pids_absent_after_cleanup']=all(not Path('/proc',str(pid)).exists() for pid in pids)
            return info
        payload_limit=m*n*4+16384
        record=capture(cmd,work/'capture',timeout=8 if case=='timeout' else 45,stdout_limit=HEADER.size+META_LIMIT+payload_limit,stderr_limit=65536,cleanup=cleanup)
        record.update(command=cmd,case=case,shape_mnk=[m,n,k],container=name,gpu_exposed=gpu,binary_sha256=self.sha256,
                      protected_files_unchanged=all(digest(work/rel)==sha for rel,sha in before.items()),protected_sha256=before,status='REJECT')
        actual=None
        if record['exit']==0 and record['reason'] is None:
            try:
                meta,payload=decode_frame(work/'capture/stdout.bin',payload_limit);record['metadata']=meta
                output=work/'result.npy';output.write_bytes(payload)
                actual=load_output(output,(m,n))
                if not np.isfinite(actual).all():raise ValueError('nonfinite output')
                record['status']='OUTPUT_VALID'
            except (OSError,ValueError,EOFError) as error:record['rejection']=str(error)[:512]
        else:record['rejection']=record['reason'] or 'worker_failure'
        if not record['cleanup']['removed'] or not record['protected_files_unchanged']:
            record['status']='REJECT';record['rejection']='lifecycle/protected data failure';actual=None
        (work/'record.json').write_text(json.dumps(record,indent=2)+'\n')
        return actual,record
