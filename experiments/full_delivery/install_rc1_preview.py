"""Build a distinct rc1 wheel and validate from an empty cwd; dev6 archive is untouched."""
import argparse,base64,hashlib,json
from pathlib import Path
from tools.ssh_3070 import ROOT,execute_python


def main():
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);a=p.parse_args();a.output.mkdir();assert not (ROOT/'kernel_agent/data/gemm_qualification.json').exists();paths=[p for p in (ROOT/'kernel_agent').rglob('*') if p.is_file() and '__pycache__' not in p.parts and p.name!='.DS_Store']+[ROOT/'pyproject.toml',ROOT/'experiments/full_delivery/rc1_installed_validate.py'];paths +=[ROOT/'tests'/n for n in ['test_registry.py','test_deployment_policy.py','test_qualified_channel.py']];files={str(p.relative_to(ROOT)):p.read_bytes() for p in paths};(a.output/'source-hashes.json').write_text(json.dumps({n:hashlib.sha256(v).hexdigest() for n,v in files.items()},indent=2))
    payload='FILES='+repr({n:base64.b64encode(v).decode() for n,v in files.items()})+'\n'+r'''
import base64,json,os,subprocess,tempfile,time,hashlib
from pathlib import Path
root=Path(tempfile.mkdtemp(prefix='nka-rc1-preview-'));source=root/'source';source.mkdir();prefix=root/'prefix';py='/home/dministrator/cuda-lab/.venv/bin/python';env={k:v for k,v in os.environ.items() if k in ['HOME','LANG','LD_LIBRARY_PATH']};env.update(PATH='/usr/local/cuda-13.0/bin:/usr/lib/wsl/lib:/usr/local/bin:/usr/bin:/bin',PIP_NO_INDEX='1',PYTHONNOUSERSITE='1',OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1');records=[]
for n,v in FILES.items():
 p=source/n;p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes(base64.b64decode(v))
def run(cmd,label,cwd):
 start=time.monotonic()
 with (root/(label+'.stdout')).open('w') as out,(root/(label+'.stderr')).open('w') as err:r=subprocess.run(cmd,cwd=cwd,env=env,stdout=out,stderr=err,timeout=1800)
 records.append({'label':label,'exit':r.returncode,'wall_seconds':time.monotonic()-start,'command':cmd});return r.returncode
code=run([py,'-m','pip','wheel','--no-deps','--no-build-isolation','--wheel-dir',str(root/'dist'),'.'],'wheel',source)
if code==0:
 wheel=next((root/'dist').glob('*.whl'));assert wheel.name=='nvidia_kernel_agent-0.1.0rc1-py3-none-any.whl';code=run([py,'-m','pip','install','--no-index','--no-deps','--target',str(prefix),str(wheel)],'install',root)
if code==0:
 tests=root/'installed-tests';tests.mkdir()
 for p in (source/'tests').glob('*.py'):(tests/p.name).write_bytes(p.read_bytes())
 env['PYTHONPATH']=str(prefix);code=run([py,str(source/'experiments/full_delivery/rc1_installed_validate.py'),'--prefix',str(prefix),'--work',str(root/'validation')],'validation',root)
result={'status':'PASS' if code==0 else 'FAIL','remote_root':str(root),'prefix':str(prefix),'records':records,'wheel_sha256':hashlib.sha256(wheel.read_bytes()).hexdigest() if code==0 else None,'new_model_calls':0,'performance_sampling':False,'production_release':False};(root/'result.json').write_text(json.dumps(result,indent=2));print(json.dumps(result))
'''
    (a.output/'driver.py').write_text(payload);r=execute_python(payload,2400);(a.output/'transport.stdout').write_text(r.stdout);(a.output/'transport.stderr').write_text(r.stderr)
    if r.returncode:raise RuntimeError('preview install transport failed; existing job only, no automatic retry')
    value=json.loads(r.stdout);(a.output/'result.json').write_text(json.dumps(value,indent=2));print(json.dumps(value))


if __name__=='__main__':main()
