"""Test strict SDK directory admission in an actual installed preview before final certification."""
import base64,hashlib,json
from tools.ssh_3070 import ROOT,execute_python


def main():
 out=ROOT/'evidence/20261123-sm86-operators-rc2';out.mkdir(exist_ok=False);parent=json.loads((ROOT/'evidence/20261123-sm86-operators-rc1/location.json').read_text())['remote_root'];public=json.loads((ROOT/'evidence/20261123-sm86-operators-rc1/finish-result.json').read_text())['persistent_root'];names=['kernel_agent/operators/cli.py','kernel_agent/operators/runner.py','kernel_agent/__init__.py','pyproject.toml','tests/test_operator_qualification.py','experiments/full_delivery/sm86_operator_release_validate.py'];files={n:(ROOT/n).read_bytes() for n in names};(out/'source-delta.json').write_text(json.dumps({n:hashlib.sha256(b).hexdigest() for n,b in files.items()},indent=2))
 payload='FILES='+repr({n:base64.b64encode(b).decode() for n,b in files.items()})+'\nPARENT='+repr(parent)+'\nPUBLIC='+repr(public)+'\n'+r'''
import base64,hashlib,json,os,shutil,subprocess,tempfile,time
from pathlib import Path
root=Path(tempfile.mkdtemp(prefix='operators-0.1.1rc2-',dir='/home/dministrator/cuda-lab/nka-persistent'));root.chmod(0o700);shutil.copytree(Path(PARENT)/'source',root/'source',ignore=shutil.ignore_patterns('build','*.egg-info','__pycache__'));(root/'sdk').symlink_to(Path(PUBLIC)/'sdk',target_is_directory=True)
for n,b in FILES.items():p=root/'source'/n;p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes(base64.b64decode(b))
py='/home/dministrator/cuda-lab/.venv/bin/python';env={k:v for k,v in os.environ.items() if k in ['HOME','PATH','LANG','LD_LIBRARY_PATH']};env.update(PIP_NO_INDEX='1',PYTHONDONTWRITEBYTECODE='1',PYTHONNOUSERSITE='1',OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1');records=[]
def run(label,cmd,cwd=root,ok=True,timeout=300):
 start=time.monotonic();r=subprocess.run(cmd,cwd=cwd,env=env,capture_output=True,text=True,timeout=timeout);(root/(label+'.stdout')).write_text(r.stdout);(root/(label+'.stderr')).write_text(r.stderr);records.append({'label':label,'exit':r.returncode,'wall_seconds':time.monotonic()-start,'command':cmd});(root/'progress.json').write_text(json.dumps(records,indent=2));assert (r.returncode==0)==ok,label;return r
status='FAIL';checks=[];wheel=None
try:
 run('wheel',[py,'-m','pip','wheel','--no-deps','--no-build-isolation','--wheel-dir',str(root/'dist'),'.'],root/'source');wheel=next((root/'dist').glob('*.whl'));assert wheel.name=='nvidia_kernel_agent-0.1.1rc2-py3-none-any.whl';run('install',[py,'-m','pip','install','--no-index','--no-deps','--target',str(root/'prefix'),str(wheel)]);env['PYTHONPATH']=str(root/'prefix');modules=['test_registry','test_deployment_policy','test_qualified_channel','test_sm89_runtime','test_sm89_qualified','test_operator_qualification'];run('cpu-tests',[py,'-B','-c',"import sys,unittest;sys.path.append("+repr(str(root/'source/tests'))+");r=unittest.TextTestRunner(verbosity=2).run(unittest.TestLoader().loadTestsFromNames("+repr(modules)+"));raise SystemExit(not r.wasSuccessful())"])
 run('smoke',[py,'-B',str(root/'source/experiments/full_delivery/sm86_operator_release_validate.py'),'--root',str(root),'--phase','smoke','--label','installed-smoke','--installed-smoke'],timeout=300)
 for label,name in [('startup','sitecustomize.py'),('bytecode','__pycache__/unknown.cpython-312.pyc'),('symlink','unlisted-link')]:
  sdk=root/('invalid-sdk-'+label);shutil.copytree(root/'sdk',sdk)
  for p in [sdk,*sdk.rglob('*')]:
   if p.is_dir():p.chmod(0o755)
  extra=sdk/name;extra.parent.mkdir(exist_ok=True)
  if label=='symlink':extra.symlink_to(sdk/'aot-manifest.json')
  else:extra.write_text('raise RuntimeError("unlisted fixture must never run")\n')
  r=run('reject-'+label,[py,'-B','-m','kernel_agent.operators.cli','inspect','--bundle',str(sdk)],ok=False);assert ('unlisted' in r.stderr or 'filesystem entry' in r.stderr);checks.append(label)
 sdk=root/'invalid-manifest';sdk.mkdir();(sdk/'aot-manifest.json').write_text('[]');r=run('reject-nonobject',[py,'-B','-m','kernel_agent.operators.cli','inspect','--bundle',str(sdk)],ok=False);assert 'must be an object' in r.stderr;checks.append('nonobject')
 status='PASS'
finally:
 result={'status':status,'remote_root':str(root),'prefix':str(root/'prefix'),'bundle':str((root/'sdk').resolve()),'wheel':str(wheel) if wheel else None,'wheel_sha256':hashlib.sha256(wheel.read_bytes()).hexdigest() if wheel else None,'gpu':json.loads((root/'installed-smoke/result.json').read_text()) if (root/'installed-smoke/result.json').exists() else None,'rejections':checks,'records':records,'production_release':False,'new_model_calls':0,'new_event_samples':0};(root/'result.json').write_text(json.dumps(result,indent=2));print(json.dumps(result))
'''
 (out/'driver.py').write_text(payload);r=execute_python(payload,900);(out/'transport.stderr').write_text(r.stderr);(out/'transport.stdout').write_text(r.stdout);assert r.returncode==0;v=json.loads(r.stdout);(out/'result.json').write_text(json.dumps(v,indent=2));(out/'location.json').write_text(json.dumps({k:v[k] for k in ['remote_root','prefix','bundle']},indent=2));print(json.dumps({k:x for k,x in v.items() if k not in ['gpu','records']}))
if __name__=='__main__':main()
