"""Build dev6 without downloads, install into a fresh prefix, run empty-cwd CLI checks."""
import argparse,base64,hashlib,json
from pathlib import Path
from tools.ssh_3070 import execute_python
ROOT=Path(__file__).resolve().parents[2]


def main():
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);a=p.parse_args();a.output.mkdir(exist_ok=False)
    paths=[p for p in (ROOT/'kernel_agent').rglob('*') if p.is_file() and '__pycache__' not in p.parts and p.name!='.DS_Store']+[ROOT/'pyproject.toml',ROOT/'experiments/full_delivery/operator_installed_validate.py'];files={str(p.relative_to(ROOT)):p.read_bytes() for p in paths};(a.output/'source-hashes.json').write_text(json.dumps({n:hashlib.sha256(v).hexdigest() for n,v in files.items()},indent=2)+'\n')
    payload='FILES='+repr({n:base64.b64encode(v).decode() for n,v in files.items()})+'\n'+r'''
import base64,fcntl,hashlib,json,os,subprocess,tempfile,time
from pathlib import Path
root=Path(tempfile.mkdtemp(prefix='nka-operators-dev6-'));source=root/'source';source.mkdir();py='/home/dministrator/cuda-lab/.venv/bin/python';prefix=root/'prefix';records=[];env={k:v for k,v in os.environ.items() if k in ['HOME','LANG','LD_LIBRARY_PATH']};env.update(PATH='/usr/local/cuda-13.0/bin:/usr/lib/wsl/lib:/usr/local/bin:/usr/bin:/bin',PIP_NO_INDEX='1',PYTHONNOUSERSITE='1',OMP_NUM_THREADS='1')
for n,v in FILES.items():
 p=source/n;p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes(base64.b64decode(v))
def run(cmd,label,cwd):
 start=time.monotonic()
 with (root/(label+'.stdout')).open('w') as out,(root/(label+'.stderr')).open('w') as err:r=subprocess.run(cmd,cwd=cwd,env=env,stdout=out,stderr=err,timeout=900)
 records.append({'label':label,'exit':r.returncode,'command':cmd,'wall_seconds':time.monotonic()-start});return r.returncode
code=run([py,'-m','pip','wheel','--no-deps','--no-build-isolation','--wheel-dir',str(root/'dist'),'.'],'wheel',source)
if code==0:code=run([py,'-m','pip','install','--no-index','--no-deps','--target',str(prefix),str(next((root/'dist').glob('*.whl')))],'install',root)
if code==0:
 gpu_lock=open('/tmp/nka-reviewed-gpu-execution.lock','a');fcntl.flock(gpu_lock,fcntl.LOCK_EX)
 env['PYTHONPATH']=str(prefix);code=run([py,str(source/'experiments/full_delivery/operator_installed_validate.py'),'--prefix',str(prefix),'--bundle','/tmp/nka-sdk-aot-licensed-l9fahjmi/bundle','--work',str(root/'validation')],'validation',root)
result={'status':'PASS' if code==0 else 'FAIL','remote_root':str(root),'prefix':str(prefix),'records':records,'new_model_calls':0,'performance_sampling':False,'production_release':False};(root/'build-result.json').write_text(json.dumps(result,indent=2));print(json.dumps(result))
'''
    # Driver contains public sources/binaries only; original credentials stay in transport env.
    (a.output/'driver.py').write_text(payload);r=execute_python(payload,1800);(a.output/'transport.stderr').write_text(r.stderr);(a.output/'transport.stdout').write_text(r.stdout)
    if r.returncode:raise RuntimeError('install transport failed; do not repeat GPU validation')
    result=json.loads(r.stdout);(a.output/'result.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result))


if __name__=='__main__':main()
