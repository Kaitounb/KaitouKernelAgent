"""Build the staged immutable operator RC once, install offline, validate real rollback."""
import json
from tools.ssh_3070 import ROOT,execute_python

def main():
 out=ROOT/'evidence/20261123-sm86-operators-rc1';root=json.loads((out/'location.json').read_text())['remote_root']
 payload='ROOT='+repr(root)+'\n'+r'''
import hashlib,json,os,subprocess,time
from pathlib import Path
root=Path(ROOT);py='/home/dministrator/cuda-lab/.venv/bin/python';assert json.loads((root/'smoke-flat-witness/result.json').read_text())['status']=='PASS';dist=root/'dist';dist.mkdir(exist_ok=False)
env={k:v for k,v in os.environ.items() if k in ['HOME','LANG','PATH','LD_LIBRARY_PATH']};env.update(PIP_NO_INDEX='1',PYTHONNOUSERSITE='1',PYTHONDONTWRITEBYTECODE='1',OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1');records=[]
def run(label,cmd,cwd,timeout=300):
 start=time.monotonic()
 with (root/(label+'.stdout')).open('w') as o,(root/(label+'.stderr')).open('w') as e:r=subprocess.run(cmd,cwd=cwd,env=env,stdout=o,stderr=e,timeout=timeout)
 records.append({'label':label,'exit':r.returncode,'wall_seconds':time.monotonic()-start,'command':cmd});(root/'install-progress.json').write_text(json.dumps(records,indent=2));return r.returncode
code=run('wheel',[py,'-m','pip','wheel','--no-deps','--no-build-isolation','--wheel-dir',str(dist),'.'],root/'source')
wheel=next(dist.glob('*.whl'),None)
if code==0:
 assert wheel.name=='nvidia_kernel_agent-0.1.1rc1-py3-none-any.whl'
 code=run('install',[py,'-m','pip','install','--no-index','--no-deps','--target',str(root/'prefix'),str(wheel)],root)
if code==0:
 env['PYTHONPATH']=str(root/'prefix');code=run('installed-validation',[py,'-B',str(root/'source/experiments/full_delivery/sm86_operator_release_validate.py'),'--root',str(root),'--phase','installed'],root,900)
result={'status':'PASS' if code==0 else 'FAIL','remote_root':str(root),'prefix':str(root/'prefix'),'wheel':str(wheel) if wheel else None,'wheel_sha256':hashlib.sha256(wheel.read_bytes()).hexdigest() if wheel else None,'records':records,'validation':json.loads((root/'installed/result.json').read_text()) if (root/'installed/result.json').exists() else None,'new_model_calls':0,'new_event_samples':0,'production_release':False};(root/'install-result.json').write_text(json.dumps(result,indent=2));print(json.dumps({k:v for k,v in result.items() if k!='validation'}))
'''
 (out/'install-driver.py').write_text(payload);r=execute_python(payload,1200);(out/'install-transport.stderr').write_text(r.stderr);(out/'install-transport.stdout').write_text(r.stdout);assert r.returncode==0;v=json.loads(r.stdout);(out/'install-result.json').write_text(json.dumps(v,indent=2));print(json.dumps(v))
if __name__=='__main__':main()
