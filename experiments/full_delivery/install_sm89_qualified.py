"""Build/install the first evidence-backed qualified SM89 runtime, then verify real operations."""
import base64,hashlib,json
from tools.ssh_server_gpu import ROOT,execute_python


def main():
    output=ROOT/'evidence/20261123-sm89-qualified-installed';output.mkdir(exist_ok=False);assert not (ROOT/'artifacts/wheels/nvidia_kernel_agent-0.1.0-py3-none-any.whl').exists()
    names=[p for p in (ROOT/'kernel_agent').rglob('*') if p.is_file() and '__pycache__' not in p.parts and p.name!='.DS_Store']+[ROOT/'pyproject.toml',ROOT/'experiments/full_delivery/sm89_qualified_validate.py'];names +=[ROOT/'tests'/n for n in ['test_registry.py','test_deployment_policy.py','test_qualified_channel.py','test_sm89_runtime.py','test_sm89_qualified.py']]
    files={str(p.relative_to(ROOT)):p.read_bytes() for p in names};(output/'source-hashes.json').write_text(json.dumps({n:hashlib.sha256(v).hexdigest() for n,v in files.items()},indent=2))
    preview=json.loads((ROOT/'evidence/20261123-sm89-runtime-source/location.json').read_text())['remote_root']
    payload='FILES='+repr({n:base64.b64encode(v).decode() for n,v in files.items()})+'\nPREVIEW='+repr(preview)+'\n'+r'''
import base64,hashlib,json,os,shutil,subprocess,tempfile,time
from pathlib import Path
root=Path(tempfile.mkdtemp(prefix='nka-sm89-qualified-',dir='/data'));root.chmod(0o700)
for n,v in FILES.items():p=root/'source'/n;p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes(base64.b64decode(v))
shutil.copyfile(Path(PREVIEW)/'worker-runtime.json',root/'worker-runtime.json');(root/'worker-runtime.json').chmod(0o600);runtime=json.loads((root/'worker-runtime.json').read_text());py=runtime['python'];tools=root/'tools';tools.mkdir();shutil.copyfile('/tmp/nka-full-server-dqnrbkzs/tools-v2/bin/uv',tools/'uv');(tools/'uv').chmod(0o700)
(root/'preview-location.json').write_text(json.dumps({'prefix':PREVIEW+'/prefix-rc2','store':PREVIEW+'/installed-validation/store'}));location={'remote_root':str(root),'prefix':str(root/'prefix'),'controller_python':py,'worker_runtime':str(root/'worker-runtime.json')};(root/'location.json').write_text(json.dumps(location,indent=2))
env={k:v for k,v in os.environ.items() if k in ['HOME','LANG','LD_LIBRARY_PATH']};env.update(PATH='/usr/bin:/bin',PYTHONNOUSERSITE='1',PYTHONDONTWRITEBYTECODE='1',UV_CACHE_DIR=str(root/'uv-cache'),OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1');records=[]
def run(label,cmd,cwd,timeout=300):
 start=time.monotonic()
 with (root/(label+'.stdout')).open('w') as out,(root/(label+'.stderr')).open('w') as err:r=subprocess.run(cmd,cwd=cwd,env=env,stdout=out,stderr=err,timeout=timeout)
 records.append({'label':label,'exit':r.returncode,'wall_seconds':time.monotonic()-start,'command':cmd});(root/'progress.json').write_text(json.dumps(records,indent=2));return r.returncode
dist=root/'dist';dist.mkdir();assert run('wheel',[py,'-B','-c','from setuptools.build_meta import build_wheel; print(build_wheel('+repr(str(dist))+'))'],root/'source')==0;wheel=next(dist.glob('*.whl'));assert wheel.name=='nvidia_kernel_agent-0.1.0-py3-none-any.whl';assert run('install',[str(tools/'uv'),'pip','install','--python',py,'--target',str(root/'prefix'),'--no-index','--no-deps',str(wheel)],root)==0
code=run('validation',[py,'-B',str(root/'source/experiments/full_delivery/sm89_qualified_validate.py'),'--root',str(root)],root,900)
result={'status':'PASS' if code==0 else 'FAIL','location':location,'wheel_sha256':hashlib.sha256(wheel.read_bytes()).hexdigest(),'wheel_path':str(wheel),'records':records,'validation':json.loads((root/'validation/result.json').read_text()) if (root/'validation/result.json').exists() else None,'new_model_calls':0,'new_event_samples':0,'production_release':code==0,'scope':'SM89_REVIEWED_FIXED_CUBLAS'};(root/'result.json').write_text(json.dumps(result,indent=2));print(json.dumps(result))
'''
    (output/'driver.py').write_text(payload);r=execute_python(payload,1500,connection_path=ROOT/'.private/server-gpu-next-connection.json');(output/'transport.stderr').write_text(r.stderr);(output/'transport.stdout').write_text(r.stdout)
    if r.returncode:raise RuntimeError('Inspect existing final wheel/install; no automatic rebuild')
    value=json.loads(r.stdout);(output/'result.json').write_text(json.dumps(value,indent=2));(output/'location.json').write_text(json.dumps(value['location'],indent=2));print(json.dumps({k:value[k] for k in ['status','location','wheel_sha256','scope','production_release']}))


if __name__=='__main__':main()
