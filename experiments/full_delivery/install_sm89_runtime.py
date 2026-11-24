"""Build immutable dev7/rc2 previews and validate a real package rollback on the target."""
import base64,hashlib,json
from tools.ssh_server_gpu import ROOT,execute_python


def main():
    output=ROOT/'evidence/20261123-sm89-runtime-installed';output.mkdir(exist_ok=False)
    for name in ['nvidia_kernel_agent-0.1.0.dev7-py3-none-any.whl','nvidia_kernel_agent-0.1.0rc2-py3-none-any.whl']:assert not (ROOT/'artifacts/wheels'/name).exists()
    location=json.loads((ROOT/'evidence/20261123-sm89-runtime-source/location.json').read_text());root=location['remote_root']
    smoke=json.loads((ROOT/'evidence/20261123-sm89-runtime-source/smoke-v3.json').read_text());assert smoke['exit']==0 and json.loads(smoke['stdout'])['status']=='PASS'
    names=['experiments/full_delivery/sm89_installed_validate.py','tests/test_sm89_runtime.py'];files={n:(ROOT/n).read_bytes() for n in names}
    hashes={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in (ROOT/'kernel_agent').rglob('*') if p.is_file() and '__pycache__' not in p.parts and p.name!='.DS_Store'};hashes['pyproject.toml']=hashlib.sha256((ROOT/'pyproject.toml').read_bytes()).hexdigest();(output/'source-hashes.json').write_text(json.dumps(hashes,indent=2))
    payload='ROOT='+repr(root)+'\nFILES='+repr({n:base64.b64encode(v).decode() for n,v in files.items()})+'\nHASHES='+repr(hashes)+'\n'+r'''
import base64,hashlib,json,os,shutil,subprocess,time
from pathlib import Path
root=Path(ROOT);source=root/'source';py='/tmp/nka-full-server-dqnrbkzs/venv/bin/python';uv='/tmp/nka-full-server-dqnrbkzs/tools-v2/bin/uv';assert Path(uv).is_file()
for n,v in FILES.items():p=source/n;p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes(base64.b64decode(v))
assert all(hashlib.sha256((source/n).read_bytes()).hexdigest()==h for n,h in HASHES.items())
env={k:v for k,v in os.environ.items() if k in ['HOME','LANG','LD_LIBRARY_PATH']};env.update(PATH='/usr/local/cuda-12.6/bin:/usr/local/bin:/usr/bin:/bin',UV_CACHE_DIR=str(root/'uv-install-cache'),PYTHONNOUSERSITE='1',OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1');records=[];wheels={}
def run(label,cmd,cwd,timeout=300):
 start=time.monotonic()
 with (root/(label+'.stdout')).open('w') as out,(root/(label+'.stderr')).open('w') as err:r=subprocess.run(cmd,cwd=cwd,env=env,stdout=out,stderr=err,timeout=timeout)
 records.append({'label':label,'exit':r.returncode,'wall_seconds':time.monotonic()-start,'command':cmd});(root/'install-progress.json').write_text(json.dumps(records,indent=2));return r.returncode
for tag,version in [('dev7','0.1.0.dev7'),('rc2','0.1.0rc2')]:
 staged=root/('build-'+tag);shutil.copytree(source,staged,ignore=shutil.ignore_patterns('__pycache__','*.pyc'));project=staged/'pyproject.toml';project.write_text(project.read_text().replace('version = "0.1.0rc2"','version = "'+version+'"'));init=staged/'kernel_agent/__init__.py';init.write_text(init.read_text().replace("__version__ = '0.1.0rc2'","__version__ = '"+version+"'"))
 dist=root/('dist-'+tag);dist.mkdir();code='from setuptools.build_meta import build_wheel; print(build_wheel('+repr(str(dist))+'))';assert run('wheel-'+tag,[py,'-c',code],staged)==0
 wheel=next(dist.glob('*.whl'));assert wheel.name=='nvidia_kernel_agent-'+version+'-py3-none-any.whl';prefix=root/('prefix-'+tag);assert run('install-'+tag,[uv,'pip','install','--python',py,'--target',str(prefix),'--no-index','--no-deps',str(wheel)],root)==0;wheels[tag]={'path':str(wheel),'sha256':hashlib.sha256(wheel.read_bytes()).hexdigest(),'version':version}
code=run('installed-validation',[py,str(source/'experiments/full_delivery/sm89_installed_validate.py'),'--root',str(root)],root,900)
result={'status':'PASS' if code==0 else 'FAIL','remote_root':str(root),'records':records,'wheels':wheels,'validation':json.loads((root/'installed-validation/result.json').read_text()) if (root/'installed-validation/result.json').exists() else None,'new_model_calls':0,'new_event_samples':0,'production_release':False};(root/'install-result.json').write_text(json.dumps(result,indent=2));print(json.dumps(result))
'''
    (output/'driver.py').write_text(payload);r=execute_python(payload,1500,connection_path=ROOT/'.private/server-gpu-next-connection.json');(output/'transport.stderr').write_text(r.stderr);(output/'transport.stdout').write_text(r.stdout)
    if r.returncode:raise RuntimeError('Installation transport failed; inspect reserved version artifacts, no rebuilding')
    value=json.loads(r.stdout);(output/'result.json').write_text(json.dumps(value,indent=2));print(json.dumps({k:value[k] for k in ['status','remote_root','wheels','production_release']}))


if __name__=='__main__':main()
