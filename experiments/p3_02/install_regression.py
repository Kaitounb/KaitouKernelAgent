"""Build and install an offline wheel in a fresh target prefix, then run installed CLI recovery."""
import argparse,base64,hashlib,json
from pathlib import Path
from tools.ssh_3070 import execute_python


def main():
    p=argparse.ArgumentParser();p.add_argument('--evidence',type=Path,required=True);a=p.parse_args()
    if a.evidence.exists():p.error('Use a new evidence directory')
    a.evidence.mkdir(parents=True);root=Path(__file__).resolve().parents[2]
    paths=list((root/'kernel_agent').glob('*.py'))+list((root/'kernel_agent/data').rglob('*'))+[root/'pyproject.toml',root/'experiments/p3_02/installed_regression.py',root/'workloads/gemm_development_v1.json',root/'tests/test_deployment_policy.py']
    sources={str(path.relative_to(root)):path.read_bytes() for path in paths if path.is_file() and '__pycache__' not in path.parts}
    for rel,data in sources.items():
        if rel.endswith('.so'):continue
        target=a.evidence/'sources'/rel;target.parent.mkdir(parents=True,exist_ok=True);target.write_bytes(data)
    encoded={rel:base64.b64encode(data).decode() for rel,data in sources.items()}
    payload='FILES='+repr(encoded)+'\n'+r'''
import base64,hashlib,importlib.metadata,json,os,stat,subprocess,tempfile,time,zlib
from pathlib import Path
root=Path(tempfile.mkdtemp(prefix='nka-p3-02-install-'));source=root/'source';source.mkdir()
for rel,data in FILES.items():
 p=source/rel;p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes(base64.b64decode(data))
records=[];env={k:v for k,v in os.environ.items() if k in ['HOME','PATH','LANG','LD_LIBRARY_PATH']};env.update(PIP_NO_INDEX='1',PYTHONNOUSERSITE='1',OPENBLAS_NUM_THREADS='1')
python='/home/dministrator/cuda-lab/.venv/bin/python'
def run(cmd,tag,cwd):
 start=time.monotonic()
 with (root/(tag+'.stdout.log')).open('w') as out,(root/(tag+'.stderr.log')).open('w') as err:p=subprocess.run(cmd,cwd=cwd,env=env,stdout=out,stderr=err,timeout=900)
 records.append({'tag':tag,'command':cmd,'exit':p.returncode,'wall_seconds':time.monotonic()-start});return p.returncode
versions={}
for name in ['pip','setuptools','wheel','numpy']:
 try:versions[name]=importlib.metadata.version(name)
 except importlib.metadata.PackageNotFoundError:versions[name]=None
(root/'existing-tools.json').write_text(json.dumps(versions,indent=2))
dist=root/'dist';prefix=root/'installed'
code=run([python,'-m','pip','wheel','--no-deps','--no-build-isolation','--wheel-dir',str(dist),'.'],'wheel',source)
if code==0:
 wheel=next(dist.glob('*.whl'))
 code=run([python,'-m','pip','install','--no-index','--no-deps','--target',str(prefix),str(wheel)],'install',root)
 if code==0:code=run([python,str(source/'experiments/p3_02/installed_regression.py'),'--prefix',str(prefix),'--work',str(root/'run'),'--development',str(source/'workloads/gemm_development_v1.json')],'regression',root)
if code==0:
 env['PYTHONPATH']=str(prefix)
 code=run([python,str(source/'tests/test_deployment_policy.py'),'-v'],'policy-tests',root/'run/empty-cwd')
 (root/'cpu-policy-tests.json').write_text(json.dumps({'exit':code,'stdout':(root/'policy-tests.stdout.log').read_text(),'stderr':(root/'policy-tests.stderr.log').read_text(),'scope':'4 CPU tests with mocked workers; not a real Docker/GPU outage'},indent=2)+'\n')
(root/'commands.json').write_text(json.dumps(records,indent=2)+'\n')
files={};manifest={}
for item in root.rglob('*'):
 rel=item.relative_to(root)
 if any(part in ['source','installed','artifact','versions','__pycache__'] for part in rel.parts) or item.name=='fallback.so':continue
 info=item.lstat()
 if not stat.S_ISREG(info.st_mode):continue
 if info.st_size>12*1024*1024:raise ValueError('archive file too large')
 data=item.read_bytes();files[str(rel)]=base64.b64encode(data).decode();manifest[str(rel)]={'sha256':hashlib.sha256(data).hexdigest(),'bytes':len(data)}
result={'remote_work':str(root),'exit':code,'files':files,'manifest':manifest}
print(json.dumps({'encoding':'zlib-base64','data':base64.b64encode(zlib.compress(json.dumps(result).encode(),6)).decode()}))
'''
    (a.evidence/'source-hashes.json').write_text(json.dumps({rel:hashlib.sha256(data).hexdigest() for rel,data in sources.items()},indent=2)+'\n')
    driver=payload
    for rel,value in encoded.items():
        if rel.endswith('.so'):driver=driver.replace(repr(value),repr('<archived-binary-'+hashlib.sha256(sources[rel]).hexdigest()+'>'))
    (a.evidence/'remote-driver.py').write_text(driver)
    r=execute_python(payload,timeout=2000);(a.evidence/'transport.log').write_text(r.stdout+r.stderr)
    if r.returncode:raise RuntimeError('transport failed; raw log retained')
    import zlib
    wire=json.loads(r.stdout);record=json.loads(zlib.decompress(base64.b64decode(wire['data'])))
    for rel,data in record.pop('files').items():
        path=Path(rel)
        if path.is_absolute() or '..' in path.parts:raise ValueError('invalid archive path')
        target=a.evidence/'remote'/path;target.parent.mkdir(parents=True,exist_ok=True);target.write_bytes(base64.b64decode(data))
    (a.evidence/'execution.json').write_text(json.dumps(record,indent=2)+'\n');print(json.dumps({k:record[k] for k in ['remote_work','exit']}))
    report=a.evidence/'remote/run/result.json'
    if report.exists():
        result=json.loads(report.read_text());print(json.dumps({k:result[k] for k in ['status','independent_correctness_calls','failures','installed_module']}))
    if record['exit']:raise RuntimeError('offline wheel/installed regression failed; evidence retained')


if __name__=='__main__':main()
