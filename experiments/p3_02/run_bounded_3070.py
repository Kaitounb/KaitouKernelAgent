"""Stage reviewed bounded-runtime source/artifact, run on authorized 3070, preserve bounded evidence."""
import argparse,base64,hashlib,json
from pathlib import Path
from tools.ssh_3070 import execute_python


def main():
    p=argparse.ArgumentParser();p.add_argument('--evidence',type=Path,required=True);a=p.parse_args()
    if a.evidence.exists():p.error('Use a new evidence directory')
    a.evidence.mkdir(parents=True);root=Path(__file__).resolve().parents[2]
    paths=list((root/'kernel_agent').glob('*.py'))+[root/'experiments/p3_02/bounded_regression.py']
    sources={str(path.relative_to(root)):path.read_bytes() for path in paths}
    binary=root/'evidence/20261122-p3-02-build-fixed-tool-path/artifacts/nka_gemm_v1.so';expected='66fcc80f9122c919f92c72e37a01794a792c91a58b71d5705b160e0bc979dfc3'
    if hashlib.sha256(binary.read_bytes()).hexdigest()!=expected:raise ValueError('archived artifact mismatch')
    for rel,data in sources.items():
        target=a.evidence/'sources'/rel;target.parent.mkdir(parents=True,exist_ok=True);target.write_bytes(data)
    encoded={rel:base64.b64encode(data).decode() for rel,data in sources.items()};encoded['trusted.so']=base64.b64encode(binary.read_bytes()).decode()
    payload='FILES='+repr(encoded)+'\n'+r'''
import base64,hashlib,json,os,stat,subprocess,tempfile
from pathlib import Path
root=Path(tempfile.mkdtemp(prefix='nka-p3-02-bounded-'));code=root/'source';code.mkdir()
for rel,data in FILES.items():
 target=code/rel;target.parent.mkdir(parents=True,exist_ok=True);target.write_bytes(base64.b64decode(data))
sha=hashlib.sha256((code/'trusted.so').read_bytes()).hexdigest()
env={k:v for k,v in os.environ.items() if k in ['HOME','PATH','LANG','LD_LIBRARY_PATH']};env['PYTHONPATH']=str(code);env['OPENBLAS_NUM_THREADS']='1'
cmd=['/home/dministrator/cuda-lab/.venv/bin/python',str(code/'experiments/p3_02/bounded_regression.py'),'--work',str(root/'run'),'--binary',str(code/'trusted.so'),'--sha256',sha]
with (root/'stdout.log').open('w') as out,(root/'stderr.log').open('w') as err:
 p=subprocess.run(cmd,cwd=code,env=env,stdout=out,stderr=err,timeout=1000)
files={};manifest={}
for item in root.rglob('*'):
 rel=item.relative_to(root)
 if any(part in ['source','control','artifact','__pycache__'] for part in rel.parts):continue
 info=item.lstat()
 if not stat.S_ISREG(info.st_mode):continue
 if info.st_size>8*1024*1024:raise ValueError('unbounded evidence')
 data=item.read_bytes();files[str(rel)]=base64.b64encode(data).decode();manifest[str(rel)]={'sha256':hashlib.sha256(data).hexdigest(),'bytes':len(data)}
print(json.dumps({'remote_work':str(root),'command':cmd,'exit':p.returncode,'files':files,'manifest':manifest,'binary_sha256':sha}))
'''
    # Source and driver are evidence; the binary is already archived separately by hash.
    (a.evidence/'source-hashes.json').write_text(json.dumps({rel:hashlib.sha256(data).hexdigest() for rel,data in sources.items()},indent=2)+'\n')
    (a.evidence/'remote-driver.py').write_text(payload.replace(repr(encoded['trusted.so']),repr('<archived-binary-'+expected+'>')))
    r=execute_python(payload,timeout=1100);(a.evidence/'transport.log').write_text(r.stdout+r.stderr)
    if r.returncode:raise RuntimeError('transport failed; see preserved log')
    data=json.loads(r.stdout)
    for rel,encoded_file in data.pop('files').items():
        path=Path(rel)
        if path.is_absolute() or '..' in path.parts:raise ValueError('invalid archive path')
        target=a.evidence/'remote'/path;target.parent.mkdir(parents=True,exist_ok=True);target.write_bytes(base64.b64decode(encoded_file))
    (a.evidence/'execution.json').write_text(json.dumps(data,indent=2)+'\n');print(json.dumps({k:data[k] for k in ['remote_work','exit','binary_sha256']}))
    if (a.evidence/'remote/run/result.json').exists():
        result=json.loads((a.evidence/'remote/run/result.json').read_text());print(json.dumps({k:result[k] for k in ['status','runtime_cases','faults','recoveries','failures']}))
    if data['exit']:raise RuntimeError('regression failed; raw evidence preserved')


if __name__=='__main__':main()
