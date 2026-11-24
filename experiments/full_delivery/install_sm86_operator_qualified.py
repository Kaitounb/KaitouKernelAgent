"""Offline single-build final package, real qualified channel, persistent launcher and rollback."""
import base64,hashlib,json
from pathlib import Path
from tools.ssh_3070 import ROOT,execute_python


def main():
 out=ROOT/'evidence/20261123-sm86-operators-qualified-installed';out.mkdir(exist_ok=False);assert not (ROOT/'artifacts/wheels/nvidia_kernel_agent-0.1.1-py3-none-any.whl').exists();assert (ROOT/'kernel_agent/operators/qualification.json').exists()
 preview=json.loads((ROOT/'evidence/20261123-sm86-operators-rc2/location.json').read_text())['remote_root'];names=[p for p in (ROOT/'kernel_agent').rglob('*') if p.is_file() and '__pycache__' not in p.parts and p.name!='.DS_Store']+[ROOT/'pyproject.toml',ROOT/'experiments/full_delivery/sm86_operator_qualified_validate.py',ROOT/'experiments/full_delivery/sm86_operator_release_validate.py'];names +=[ROOT/'tests'/n for n in ['test_registry.py','test_deployment_policy.py','test_qualified_channel.py','test_sm89_runtime.py','test_sm89_qualified.py','test_operator_qualification.py']];files={str(p.relative_to(ROOT)):p.read_bytes() for p in names};(out/'source-hashes.json').write_text(json.dumps({n:hashlib.sha256(b).hexdigest() for n,b in files.items()},indent=2))
 payload='FILES='+repr({n:base64.b64encode(b).decode() for n,b in files.items()})+'\nPREVIEW='+repr(preview)+'\n'+r'''
import base64,hashlib,json,os,shutil,subprocess,tempfile,time
from pathlib import Path
root=Path(tempfile.mkdtemp(prefix='operators-0.1.1-',dir='/home/dministrator/cuda-lab/nka-persistent'));root.chmod(0o700);preview=Path(PREVIEW);py='/home/dministrator/cuda-lab/.venv/bin/python'
for n,b in FILES.items():p=root/'source'/n;p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes(base64.b64decode(b))
shutil.copytree(preview/'sdk',root/'sdk');launcher=root/'kernel-agent-operators';launcher.write_text('#!/bin/sh\nexport PYTHONPATH='+str(root/'current')+'\nexec '+py+' -B -m kernel_agent.operators.cli "$@"\n');launcher.chmod(0o700);location={'remote_root':str(root),'prefix':str(root/'prefix'),'bundle':str(root/'sdk'),'launcher':str(launcher),'current':str(root/'current'),'preview_prefix':str(preview/'prefix')};(root/'location.json').write_text(json.dumps(location,indent=2));records=[];env={k:v for k,v in os.environ.items() if k in ['HOME','PATH','LANG','LD_LIBRARY_PATH']};env.update(PIP_NO_INDEX='1',PYTHONDONTWRITEBYTECODE='1',PYTHONNOUSERSITE='1',OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1')
def run(label,args,cwd,timeout=300):
 start=time.monotonic()
 with (root/(label+'.stdout')).open('w') as o,(root/(label+'.stderr')).open('w') as e:r=subprocess.run(args,cwd=cwd,env=env,stdout=o,stderr=e,timeout=timeout)
 records.append({'label':label,'exit':r.returncode,'wall_seconds':time.monotonic()-start,'command':args});(root/'progress.json').write_text(json.dumps(records,indent=2));return r.returncode
code=run('wheel',[py,'-m','pip','wheel','--no-deps','--no-build-isolation','--wheel-dir',str(root/'dist'),'.'],root/'source');wheel=next((root/'dist').glob('*.whl'),None)
if code==0:
 assert wheel.name=='nvidia_kernel_agent-0.1.1-py3-none-any.whl';code=run('install',[py,'-m','pip','install','--no-index','--no-deps','--target',str(root/'prefix'),str(wheel)],root)
if code==0:
 env['PYTHONPATH']=str(root/'prefix')+':'+str(root/'source');code=run('validation',[py,'-B',str(root/'source/experiments/full_delivery/sm86_operator_qualified_validate.py'),'--root',str(root),'--preview',str(preview/'prefix')],root,1200)
validation=json.loads((root/'validation/result.json').read_text()) if (root/'validation/result.json').exists() else None
result={'status':'PASS' if code==0 else 'FAIL','location':location,'wheel':str(wheel) if wheel else None,'wheel_sha256':hashlib.sha256(wheel.read_bytes()).hexdigest() if wheel else None,'records':records,'validation':validation,'new_model_calls':0,'new_event_samples':0,'production_release':code==0,'scope':'SM86_REVIEWED_FIXED_ATTENTION_MLA'};(root/'result.json').write_text(json.dumps(result,indent=2));print(json.dumps(result))
'''
 (out/'driver.py').write_text(payload);r=execute_python(payload,1800);(out/'transport.stderr').write_text(r.stderr);(out/'transport.stdout').write_text(r.stdout)
 if r.returncode:raise RuntimeError('Inspect existing final wheel/install job; never rebuild the same version')
 v=json.loads(r.stdout);(out/'result.json').write_text(json.dumps(v,indent=2));(out/'location.json').write_text(json.dumps(v['location'],indent=2));print(json.dumps({k:x for k,x in v.items() if k not in ['validation','records']}))
if __name__=='__main__':main()
