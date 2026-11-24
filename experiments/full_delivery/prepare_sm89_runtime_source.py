"""Stage the reviewed deployment delta and prove one low-privilege CUDA call before wheel freeze."""
import base64,hashlib,json
from tools.ssh_server_gpu import ROOT,execute_python


def main():
    out=ROOT/'evidence/20261123-sm89-runtime-source';out.mkdir(exist_ok=False)
    files={str(p.relative_to(ROOT)):p.read_bytes() for p in (ROOT/'kernel_agent').rglob('*') if p.is_file() and '__pycache__' not in p.parts and p.name!='.DS_Store'}
    files['pyproject.toml']=(ROOT/'pyproject.toml').read_bytes()
    for name in ['test_registry.py','test_deployment_policy.py','test_qualified_channel.py','test_sm89_runtime.py']:files['tests/'+name]=(ROOT/'tests'/name).read_bytes()
    (out/'source-hashes.json').write_text(json.dumps({n:hashlib.sha256(v).hexdigest() for n,v in files.items()},indent=2))
    runtime=json.loads((ROOT/'evidence/20261123-sm89-worker-runtime/result.json').read_text());assert runtime['status']=='PUBLIC_RUNTIME_READY';runtime['gpu_identity']='GPU-a1a52305-25e4-75a4-cbb8-0d6d8e934138, NVIDIA GeForce RTX 4090, 570.86.10'
    payload='FILES='+repr({n:base64.b64encode(v).decode() for n,v in files.items()})+'\nRUNTIME='+repr(runtime)+'\n'+r'''
import base64,json,os,subprocess,tempfile
from pathlib import Path
root=Path(tempfile.mkdtemp(prefix='nka-sm89-install-',dir='/data'));root.chmod(0o700)
for n,v in FILES.items():p=root/'source'/n;p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes(base64.b64decode(v))
(root/'worker-runtime.json').write_text(json.dumps(RUNTIME,indent=2));(root/'worker-runtime.json').chmod(0o600)
value={'remote_root':str(root),'source':str(root/'source'),'runtime_config':str(root/'worker-runtime.json'),'python':'/tmp/nka-full-server-dqnrbkzs/venv/bin/python','status':'STAGED_SOURCE_NOT_WHEEL'};(root/'location.json').write_text(json.dumps(value,indent=2));print(json.dumps(value))
'''
    (out/'prepare-driver.py').write_text(payload);r=execute_python(payload,120,connection_path=ROOT/'.private/server-gpu-next-connection.json');(out/'prepare.stderr').write_text(r.stderr)
    if r.returncode:raise RuntimeError('Inspect existing source stage; no automatic restaging')
    location=json.loads(r.stdout);(out/'location.json').write_text(json.dumps(location,indent=2));print(json.dumps(location),flush=True)
    code=r'''
import fcntl,json,os,sys,time,subprocess
from pathlib import Path
import numpy as np
from kernel_agent.sm89.catalog import registry
from kernel_agent.sm89.runner import NativeRunner
root=Path(sys.argv[1]);lock=open('/tmp/nka-reviewed-gpu-execution.lock','a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
admission=[]
for _ in range(3):
 r=subprocess.run(['nvidia-smi','--query-gpu=utilization.gpu','--format=csv,noheader,nounits'],capture_output=True,text=True,timeout=15);admission.append(r.stdout);assert r.returncode==0 and int(r.stdout.strip())==0;time.sleep(1)
store=registry(root/'smoke-store');store.initialize();binary,meta,_=store.select();x=(np.arange(17*32).reshape(17,32)%13/16).astype(np.float16);y=(np.arange(32*19).reshape(32,19)%11/16).astype(np.float16);expected=x.astype(np.float64)@y.astype(np.float64)
runner=NativeRunner(binary,meta['binary_sha256'],runtime=root/'worker-runtime.json');actual,record=runner.run(x,y,root/'source-smoke',case='boundaries');ok=actual is not None and np.all(np.abs(actual.astype(np.float64)-expected)<=3e-5+1e-4*np.abs(expected))
value={'status':'PASS' if ok else 'FAIL','gpu_attempts':1,'max_abs_error':float(np.abs(actual.astype(np.float64)-expected).max()) if actual is not None else None,'record':record,'admission':admission,'new_model_calls':0,'production_release':False};(root/'source-smoke-result.json').write_text(json.dumps(value,indent=2));print(json.dumps({'status':value['status'],'record_status':record['status'],'rejection':record.get('rejection'),'metadata':record.get('metadata'),'stderr':(root/'source-smoke/capture/stderr.bin').read_text(errors='replace')[-5000:],'remote_root':str(root)}))
'''
    payload='ROOT='+repr(location['remote_root'])+'\nCODE='+repr(code)+'\n'+r'''
import json,os,subprocess
from pathlib import Path
root=Path(ROOT);script=root/'source-smoke-driver.py';script.write_text(CODE);env={k:v for k,v in os.environ.items() if k in ['HOME','LANG','LD_LIBRARY_PATH']};env.update(PATH='/usr/local/cuda-12.6/bin:/usr/local/bin:/usr/bin:/bin',PYTHONPATH=str(root/'source'),OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',NKA_CONTROL_CANARY='host-only')
r=subprocess.run(['/tmp/nka-full-server-dqnrbkzs/venv/bin/python',str(script),str(root)],capture_output=True,text=True,env=env,cwd=root,timeout=180);(root/'source-smoke.stdout').write_text(r.stdout);(root/'source-smoke.stderr').write_text(r.stderr);print(json.dumps({'exit':r.returncode,'stdout':r.stdout,'stderr':r.stderr}))
'''
    (out/'smoke-driver.py').write_text(payload);r=execute_python(payload,240,connection_path=ROOT/'.private/server-gpu-next-connection.json');(out/'smoke-transport.stderr').write_text(r.stderr)
    if r.returncode:raise RuntimeError('Inspect source smoke state')
    value=json.loads(r.stdout);(out/'smoke.json').write_text(json.dumps(value,indent=2));print(json.dumps(value))


if __name__=='__main__':main()
