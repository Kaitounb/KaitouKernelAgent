"""Stage public fixed sources and build only; no credentials, heldout data or GPU benchmark."""
import base64,hashlib,json
from tools.ssh_server_gpu import ROOT,execute_python


def main():
    output=ROOT/'evidence/20261123-sm89-cublas-v2';output.mkdir(exist_ok=False)
    wheel=ROOT/'artifacts/wheels/nvidia_kernel_agent-0.1.0.dev6-py3-none-any.whl'
    assert hashlib.sha256(wheel.read_bytes()).hexdigest()=='d5f4bbf83dc050271af66baf35738f3800d02c2592b6fec101457122edc4fb38'
    names=['kernels/gemm/cublas_sm86.cu','kernels/gemm/gemm_contract.cuh','workloads/gemm_deployment_stability_v2.json','experiments/full_delivery/build_sm89_cublas.py','experiments/full_delivery/cublas_sustained_qualification_host.py']
    files={n:(ROOT/n).read_bytes() for n in names}
    files['dev6.whl']=wheel.read_bytes()
    kb=ROOT/'provenance/upstream-reviews/20261121-p1-02/ScalingIntelligence__KernelBench/src/kernelbench/eval.py'
    files['upstream/kernelbench/eval.py']=kb.read_bytes()
    (output/'source-hashes.json').write_text(json.dumps({n:hashlib.sha256(v).hexdigest() for n,v in files.items()},indent=2))
    for n,v in files.items():
        p=output/'sources'/n;p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes(v)
    payload='FILES='+repr({n:base64.b64encode(v).decode() for n,v in files.items()})+'\n'+r'''
import base64,fcntl,hashlib,json,os,subprocess,tempfile,zipfile
from pathlib import Path
base=Path('/data/nka-cublas-v2');base.mkdir(mode=0o700,exist_ok=True)
root=Path(tempfile.mkdtemp(prefix='sm89-',dir=base));root.chmod(0o700)
for n,v in FILES.items():
 p=root/n;p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes(base64.b64decode(v))
assert hashlib.sha256((root/'dev6.whl').read_bytes()).hexdigest()=='d5f4bbf83dc050271af66baf35738f3800d02c2592b6fec101457122edc4fb38'
prefix=root/'prefix';prefix.mkdir()
with zipfile.ZipFile(root/'dev6.whl') as z:
 for item in z.infolist():
  p=Path(item.filename);assert not p.is_absolute() and '..' not in p.parts
  if item.is_dir():continue
  target=prefix/p;target.parent.mkdir(parents=True,exist_ok=True);target.write_bytes(z.read(item))
result={'status':'STAGED','remote_root':str(root),'prefix':str(prefix),'python':'/tmp/nka-full-server-dqnrbkzs/venv/bin/python','new_model_calls':0,'production_release':False}
(root/'staged.json').write_text(json.dumps(result,indent=2));print(json.dumps(result))
'''
    (output/'prepare-driver.py').write_text(payload)
    r=execute_python(payload,120,connection_path=ROOT/'.private/server-gpu-next-connection.json')
    (output/'prepare.stderr').write_text(r.stderr)
    if r.returncode:raise RuntimeError('Staging failed: inspect reserved directory')
    location=json.loads(r.stdout);(output/'location.json').write_text(json.dumps(location,indent=2));print(json.dumps(location),flush=True)
    payload='ROOT='+repr(location['remote_root'])+'\n'+r'''
import os,json,subprocess
from pathlib import Path
root=Path(ROOT);py='/tmp/nka-full-server-dqnrbkzs/venv/bin/python'
env={k:v for k,v in os.environ.items() if k in ['HOME','LANG','LD_LIBRARY_PATH']};env.update(PATH='/usr/local/cuda-12.6/bin:'+str(Path(py).parent)+':/usr/local/bin:/usr/bin:/bin',OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1')
with (root/'build.stdout').open('w') as out,(root/'build.stderr').open('w') as err:r=subprocess.run([py,str(root/'experiments/full_delivery/build_sm89_cublas.py'),'--root',str(root)],cwd=root,env=env,stdout=out,stderr=err,timeout=1500)
print(json.dumps({'remote_root':str(root),'exit':r.returncode,'result':json.loads((root/'build-result.json').read_text()) if (root/'build-result.json').exists() else None}))
'''
    (output/'build-driver.py').write_text(payload);r=execute_python(payload,1700,connection_path=ROOT/'.private/server-gpu-next-connection.json');(output/'build-transport.stderr').write_text(r.stderr);(output/'build-transport.stdout').write_text(r.stdout)
    if r.returncode:raise RuntimeError('Build transport failed; inspect current identity, no automatic rebuild')
    value=json.loads(r.stdout);(output/'build.json').write_text(json.dumps(value,indent=2));print(json.dumps(value))


if __name__=='__main__':main()
