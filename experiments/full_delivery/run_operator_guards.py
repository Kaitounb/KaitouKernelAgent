"""Run reviewed public guard program natively and in both sanitizers; no model/heldout data."""
import argparse,base64,hashlib,json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]


def main():
    p=argparse.ArgumentParser();p.add_argument('--target',choices=['3070','server'],required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--jit-base',required=True);p.add_argument('--sdk-prefix');a=p.parse_args();a.output.mkdir(exist_ok=False)
    names=['kernel_agent/operators/__init__.py','kernel_agent/operators/references.py','kernel_agent/operators/flashinfer_backend.py','kernel_agent/operators/partition.py','experiments/full_delivery/operator_guards.py'];files={n:base64.b64encode((ROOT/n).read_bytes()).decode() for n in names};files['kernel_agent/__init__.py']='';py='/home/dministrator/cuda-lab/.venv/bin/python' if a.target=='3070' else '/tmp/nka-full-server-dqnrbkzs/venv/bin/python';cuda='/usr/local/cuda-13.0' if a.target=='3070' else '/usr/local/cuda-12.6'
    payload='FILES='+repr(files)+'\nPY='+repr(py)+'\nCUDA='+repr(cuda)+'\nSDK='+repr(a.sdk_prefix)+'\nJIT='+repr(a.jit_base)+'\nARCH='+repr('8.6' if a.target=='3070' else '8.9')+'\n'+r'''
import base64,json,os,subprocess,tempfile
from pathlib import Path
root=Path(tempfile.mkdtemp(prefix='nka-operator-guards-'))
for name,data in FILES.items():
 p=root/name;p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes(base64.b64decode(data))
env={k:v for k,v in os.environ.items() if k in ['HOME','LANG','LD_LIBRARY_PATH']};env.update(PATH=str(Path(PY).parent)+':'+CUDA+'/bin:/usr/lib/wsl/lib:/usr/local/bin:/usr/bin:/bin',PYTHONPATH=str(root)+(os.pathsep+SDK if SDK else ''),CUDA_HOME=CUDA,TORCH_CUDA_ARCH_LIST=ARCH,FLASHINFER_WORKSPACE_BASE=JIT,OMP_NUM_THREADS='1',PYTORCH_NO_CUDA_MEMORY_CACHING='1');base=[PY,str(root/'experiments/full_delivery/operator_guards.py')];rows=[]
for tool in [None,'memcheck','synccheck']:
 cmd=base if tool is None else [CUDA+'/bin/compute-sanitizer','--tool',tool,'--error-exitcode','99']+base
 r=subprocess.run(cmd,cwd=root,env=env,capture_output=True,text=True,timeout=600);rows.append({'tool':tool or 'native','command':cmd,'exit':r.returncode,'stdout':r.stdout,'stderr':r.stderr});(root/'result.json').write_text(json.dumps(rows,indent=2))
 if r.returncode:break
print(json.dumps({'root':str(root),'rows':rows,'status':'PASS' if len(rows)==3 and all(x['exit']==0 for x in rows) else 'FAIL'}))
'''
    (a.output/'driver.py').write_text(payload)
    if a.target=='3070':from tools.ssh_3070 import execute_python
    else:from tools.ssh_server_gpu import execute_python
    r=execute_python(payload,2000);(a.output/'transport.stdout').write_text(r.stdout);(a.output/'transport.stderr').write_text(r.stderr)
    if r.returncode:raise RuntimeError('guard transport failed, raw retained')
    (a.output/'result.json').write_text(r.stdout);(a.output/'sources.json').write_text(json.dumps({n:hashlib.sha256((ROOT/n).read_bytes()).hexdigest() for n in names},indent=2)+'\n');v=json.loads(r.stdout);print(json.dumps({'status':v['status'],'tools':[{'tool':x['tool'],'exit':x['exit'],'stderr_tail':x['stderr'][-500:]} for x in v['rows']]}))


if __name__=='__main__':main()
