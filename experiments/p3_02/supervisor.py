"""Docker lifecycle adapter; trusted evaluator stays outside candidate mount/namespace."""
import argparse,hashlib,json,os,shutil,stat,subprocess,time,uuid
from pathlib import Path
import numpy as np
import torch

IMAGE='sha256:ce00f6dc2eea43dc508b90b548f624646d643a92d363aa715fdb5a3b4f49290a'
BINARY='/tmp/nka-p1-02-contract-eeyx9qg9/extensions/nka_gemm_v1/nka_gemm_v1.so'
EXPECTED='93fb2f4344521b7e9b246b54d1c30d4b1a8facf731e6385e18f6119e9c49c72c'


def run(cmd,**kw):
    p=subprocess.run(cmd,capture_output=True,text=True,**kw);return {'command':cmd,'exit':p.returncode,'stdout':p.stdout,'stderr':p.stderr}


def load_output(path,shape):
    # Never follow a candidate-created symlink or let a forged header allocate unbounded host memory.
    fd=os.open(path,os.O_RDONLY|os.O_NOFOLLOW|os.O_NONBLOCK)
    with os.fdopen(fd,'rb') as file:
        info=os.fstat(file.fileno())
        if not stat.S_ISREG(info.st_mode) or info.st_size>int(np.prod(shape))*4+16384:raise ValueError('output type/size invalid')
        version=np.lib.format.read_magic(file)
        if version==(1,0):observed,fortran,dtype=np.lib.format.read_array_header_1_0(file,max_header_size=10000)
        elif version==(2,0):observed,fortran,dtype=np.lib.format.read_array_header_2_0(file,max_header_size=10000)
        else:raise ValueError('unsupported numpy format')
        if observed!=shape or fortran or dtype!=np.dtype('float32') or dtype.hasobject:raise ValueError('output shape/dtype/order invalid')
        if info.st_size-file.tell()!=int(np.prod(shape))*4:raise ValueError('output payload size mismatch')
        file.seek(0);return np.load(file,allow_pickle=False,max_header_size=10000)


def main():
    p=argparse.ArgumentParser();p.add_argument('--work',type=Path,required=True);a=p.parse_args();work=a.work
    torch.set_num_threads(1);work.mkdir(exist_ok=True);work.chmod(0o755)
    artifact=work/'artifact';artifact.mkdir();artifact.chmod(0o755)
    shutil.copyfile(Path(__file__).with_name('worker.py'),artifact/'worker.py')
    shutil.copyfile(BINARY,artifact/'nka_gemm_v1.so')
    if hashlib.sha256((artifact/'nka_gemm_v1.so').read_bytes()).hexdigest()!=EXPECTED:raise ValueError('artifact mismatch')
    (artifact/'readonly-guard').write_text('untouched');(artifact/'readonly-guard').chmod(0o666)
    control=work/'control';control.mkdir();control.chmod(0o700)
    (control/'secret').write_text('decoy-only-'+uuid.uuid4().hex);state={'backend':'cuBLAS','candidate':None,'published':False}
    (control/'state.json').write_text(json.dumps(state));before=hashlib.sha256((control/'state.json').read_bytes()).hexdigest()
    os.environ['NKA_CONTROL_CANARY']='decoy-not-a-real-key'
    rows=[]
    shapes=[(17,19,23),(512,512,512)]
    def job(case,shape):
        tag=case+'-'+uuid.uuid4().hex[:8];jobdir=work/tag;inputs=jobdir/'inputs';outputs=jobdir/'outputs'
        inputs.mkdir(parents=True);outputs.mkdir();jobdir.chmod(0o755);inputs.chmod(0o755);outputs.chmod(0o777)
        m,n,k=shape;g=torch.Generator().manual_seed(171)
        x=torch.randn(m,k,generator=g).mul_(.25).half();y=torch.randn(k,n,generator=g).mul_(.25).half()
        expected=(x.double()@y.double()).float().numpy()
        np.save(inputs/'a.npy',x.numpy(),allow_pickle=False);np.save(inputs/'b.npy',y.numpy(),allow_pickle=False)
        hashes={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in inputs.glob('*.npy')}
        req={'case':case,'binary_sha256':EXPECTED,'host_private_path':str(control/'secret'),'host_pid':os.getpid()}
        (inputs/'request.json').write_text(json.dumps(req))
        name='nka-worker-'+uuid.uuid4().hex[:12]
        cmd=['docker','run','--name',name,'--rm','--pull=never','--network=none','--read-only','--cap-drop=ALL','--security-opt=no-new-privileges','--user','65534:65534','--init',
             '--pids-limit=128','--memory=2g','--cpus=2','--gpus','device=0','--env','HOME=/tmp','--env','PYTHONDONTWRITEBYTECODE=1','--env','OMP_NUM_THREADS=1','--env','MKL_NUM_THREADS=1',
             '--tmpfs','/tmp:rw,nosuid,nodev,size=128m','--mount',f'type=bind,src={artifact},dst=/artifact,readonly',
             '--mount',f'type=bind,src={inputs},dst=/inputs,readonly','--mount',f'type=bind,src={outputs},dst=/outputs',
             '--entrypoint','/opt/venv/bin/python',IMAGE,'/artifact/worker.py','--request','/inputs/request.json']
        started=time.monotonic();record={'case':case,'shape_mnk':list(shape),'command':cmd,'container':name}
        with (jobdir/'stdout.log').open('w') as out,(jobdir/'stderr.log').open('w') as err:
            proc=subprocess.Popen(cmd,stdout=out,stderr=err)
            try:
                if case=='timeout':
                    deadline=time.monotonic()+45
                    while proc.poll() is None and not (outputs/'ready.json').exists() and time.monotonic()<deadline:time.sleep(.05)
                    if not (outputs/'ready.json').exists():raise RuntimeError('timeout fixture did not reach finite CUDA READY')
                    record['ready']=json.loads((outputs/'ready.json').read_text());record['processes_before_cleanup']=run(['docker','top',name,'-eo','pid,ppid,args'])
                    time.sleep(.3);record['forced_cleanup']=run(['docker','rm','-f',name]);proc.wait(timeout=10);record['timed_out']=True
                else:proc.wait(timeout=60)
            except (subprocess.TimeoutExpired,RuntimeError) as error:
                record['supervisor_error']=str(error);record['forced_cleanup']=run(['docker','rm','-f',name])
                try:proc.wait(timeout=10)
                except subprocess.TimeoutExpired:proc.kill();proc.wait(timeout=5)
        record['exit']=proc.returncode;record['wall_seconds']=time.monotonic()-started
        inspected=run(['docker','inspect',name]);record['container_removed']=inspected['exit']!=0
        record['inputs_unchanged']=all(hashlib.sha256((inputs/name).read_bytes()).hexdigest()==value for name,value in hashes.items())
        record['control_state_unchanged']=hashlib.sha256((control/'state.json').read_bytes()).hexdigest()==before
        record['accepted']=False
        if proc.returncode==0 and not record.get('timed_out'):
            try:
                actual=load_output(outputs/'result.npy',(m,n))
                if not np.isfinite(actual).all():raise ValueError('nonfinite output')
                delta=np.abs(actual.astype(np.float64)-expected.astype(np.float64));ok=np.all(delta<=3e-5+1e-4*np.abs(expected))
                record['max_abs_error']=float(delta.max())
                if not ok:raise ValueError('independent CPU FP64 reference mismatch')
                record['accepted']=True
            except (ValueError,OSError,EOFError) as error:record['rejection']=str(error)
        else:record['rejection']='host_timeout' if record.get('timed_out') else 'worker_failure'
        if (outputs/'worker.json').is_file():record['worker_metadata']=json.loads((outputs/'worker.json').read_text())
        rows.append(record);(work/'result.partial.json').write_text(json.dumps(rows,indent=2)+'\n')
        return record
    for case in ['cublas','template']:
        for shape in shapes:job(case,shape)
    job('boundaries',shapes[0])
    for fault in ['constant','pollution','crash','symlink','timeout']:
        job(fault,shapes[0]);job('cublas',shapes[0])
    failures=[]
    for row in rows:
        desired=row['case'] in ['cublas','template','boundaries']
        if row['accepted']!=desired or not row['container_removed'] or not row['inputs_unchanged'] or not row['control_state_unchanged']:failures.append(row['case'])
    boundary=next(r for r in rows if r['case']=='boundaries').get('worker_metadata',{})
    checks=['capabilities_zero','no_new_privs','readonly_write_denied','external_connect_denied']
    if not all(boundary.get(k) for k in checks) or boundary.get('uid')!=65534 or boundary.get('interfaces')!=['lo'] or any(boundary.get(k,True) for k in ['docker_socket_visible','control_canary_in_env','host_private_visible','host_pid_visible']):failures.append('boundary')
    result={'status':'PASS' if not failures else 'FAIL','scope':'trusted GEMM runtime/fixtures in existing image; independent host reference; not unrestricted source execution or release',
            'image_id':IMAGE,'binary_sha256':EXPECTED,'cases':rows,'failures':failures,'control_state_unchanged':hashlib.sha256((control/'state.json').read_bytes()).hexdigest()==before,
            'limits':'CPU memory/pids/cpus configured; no GPU memory quota or infinite GPU-kernel cancellation claim','heldout_executed':False}
    (work/'result.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps({'status':result['status'],'cases':len(rows),'failures':failures}))


if __name__=='__main__':main()
