"""Single-build preview/final GEMM release, real recovery and package rollback on SM86."""
import argparse,base64,hashlib,json
from pathlib import Path
from tools.ssh_3070 import ROOT,execute_python


def main():
 p=argparse.ArgumentParser();p.add_argument('--phase',choices=['preview','qualified'],required=True);a=p.parse_args();version='0.1.2rc1' if a.phase=='preview' else '0.1.2';out=ROOT/('evidence/20261124-sm86-gemm-'+version+'-installed');out.mkdir(exist_ok=False);assert not (ROOT/'artifacts/wheels'/('nvidia_kernel_agent-'+version+'-py3-none-any.whl')).exists();previous=json.loads((ROOT/('evidence/20261123-sm86-operators-qualified-installed/location.json' if a.phase=='preview' else 'evidence/20261124-sm86-gemm-0.1.2rc1-installed/location.json')).read_text())['prefix'];ops=json.loads((ROOT/'evidence/20261123-sm86-operators-qualified-installed/location.json').read_text())['bundle']
 names=[p for p in (ROOT/'kernel_agent').rglob('*') if p.is_file() and '__pycache__' not in p.parts and p.name!='.DS_Store']+[ROOT/'pyproject.toml',ROOT/'examples/sm86_qualified_operators.py',ROOT/'experiments/full_delivery/gemm_release_validate.py',ROOT/'workloads/gemm_deployment_stability_v2.json',ROOT/'experiments/full_delivery/gemm_deployment_gate.py'];names +=[ROOT/'tests'/n for n in ['test_registry.py','test_deployment_policy.py','test_qualified_channel.py','test_sm89_runtime.py','test_sm89_qualified.py','test_operator_qualification.py','test_gemm_deployment_policy.py','test_gemm_target_binding.py']];files={str(p.relative_to(ROOT)):p.read_bytes() for p in names};(out/'source-hashes.json').write_text(json.dumps({n:hashlib.sha256(b).hexdigest() for n,b in files.items()},indent=2))
 payload='FILES='+repr({n:base64.b64encode(b).decode() for n,b in files.items()})+'\nVERSION='+repr(version)+'\nPHASE='+repr(a.phase)+'\nPREVIOUS='+repr(previous)+'\nOPS='+repr(ops)+'\n'+r'''
import base64,hashlib,json,os,subprocess,tempfile,time
from pathlib import Path
root=Path(tempfile.mkdtemp(prefix='gemm-'+VERSION+'-',dir='/home/dministrator/cuda-lab/nka-persistent'));root.chmod(0o700);py='/home/dministrator/cuda-lab/.venv/bin/python'
for n,b in FILES.items():p=root/'source'/n;p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes(base64.b64decode(b))
env={k:v for k,v in os.environ.items() if k in ['HOME','PATH','LANG','LD_LIBRARY_PATH']};env.update(PIP_NO_INDEX='1',PYTHONDONTWRITEBYTECODE='1',PYTHONNOUSERSITE='1',OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1');records=[]
def run(label,args,cwd=root,timeout=300):
 start=time.monotonic()
 with (root/(label+'.stdout')).open('w') as o,(root/(label+'.stderr')).open('w') as e:r=subprocess.run(args,cwd=cwd,env=env,stdout=o,stderr=e,timeout=timeout)
 records.append({'label':label,'exit':r.returncode,'wall_seconds':time.monotonic()-start,'command':args});(root/'progress.json').write_text(json.dumps(records,indent=2));return r.returncode
code=run('wheel',[py,'-m','pip','wheel','--no-deps','--no-build-isolation','--wheel-dir',str(root/'dist'),'.'],root/'source');wheel=next((root/'dist').glob('*.whl'),None)
if code==0:
 assert wheel.name=='nvidia_kernel_agent-'+VERSION+'-py3-none-any.whl';code=run('install',[py,'-m','pip','install','--no-index','--no-deps','--target',str(root/'prefix'),str(wheel)])
if code==0:
 env['PYTHONPATH']=str(root/'prefix');code=run('validation',[py,'-B',str(root/'source/experiments/full_delivery/gemm_release_validate.py'),'--root',str(root),'--phase',PHASE,'--previous-prefix',PREVIOUS,'--ops-bundle',OPS],timeout=5400)
launcher=root/'kernel-agent';launcher.write_text('#!/bin/sh\nexport PYTHONPATH='+str(root/'current')+'\nexec '+py+' -B -m kernel_agent.cli "$@"\n');launcher.chmod(0o700)
location={'remote_root':str(root),'prefix':str(root/'prefix'),'launcher':str(launcher),'current':str(root/'current'),'store':str(root/'validation/store'),'previous_prefix':PREVIOUS,'ops_bundle':OPS};(root/'location.json').write_text(json.dumps(location,indent=2));result={'status':'PASS' if code==0 else 'FAIL','version':VERSION,'phase':PHASE,'location':location,'wheel':str(wheel) if wheel else None,'wheel_sha256':hashlib.sha256(wheel.read_bytes()).hexdigest() if wheel else None,'validation':json.loads((root/'validation/result.json').read_text()) if (root/'validation/result.json').exists() else None,'records':records,'production_release':code==0 and PHASE=='qualified','new_model_calls':0,'new_event_samples':0};(root/'result.json').write_text(json.dumps(result,indent=2));print(json.dumps(result))
'''
 (out/'driver.py').write_text(payload);r=execute_python(payload,6000);(out/'transport.stderr').write_text(r.stderr);(out/'transport.stdout').write_text(r.stdout)
 if r.returncode:raise RuntimeError('Inspect existing immutable wheel/install, do not rebuild/replay')
 v=json.loads(r.stdout);(out/'result.json').write_text(json.dumps(v,indent=2));(out/'location.json').write_text(json.dumps(v['location'],indent=2));print(json.dumps({k:y for k,y in v.items() if k not in ['validation','records']}))
if __name__=='__main__':main()
