"""Stage only public reviewed sources/dependencies, run bounded builds, archive content-addressed evidence."""
import argparse,base64,hashlib,io,json,zipfile
from pathlib import Path
from tools.ssh_3070 import execute_python


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--evidence',type=Path,required=True);a=parser.parse_args()
    if a.evidence.exists():parser.error('Use a new evidence directory')
    a.evidence.mkdir(parents=True);root=Path(__file__).resolve().parents[2]
    paths=list((root/'kernel_agent').glob('*.py'))+[root/p for p in ['evaluators/gemm_regression.py','kernels/gemm/cublas_sm86.cu','kernels/gemm/wmma_sm86.cu','kernels/gemm/gemm_contract.cuh','workloads/gemm_development_v1.json','experiments/p3_02/builder_regression.py']]
    sources={str(path.relative_to(root)):path.read_bytes() for path in paths}
    upstream=root/'provenance/upstream-reviews/20261121-p1-02/ScalingIntelligence__KernelBench/src/kernelbench'
    for path in upstream.glob('*.py'):sources[str(path.relative_to(root))]=path.read_bytes()
    for rel,data in sources.items():
        target=a.evidence/'sources'/rel;target.parent.mkdir(parents=True,exist_ok=True);target.write_bytes(data)
    binary=root/'evidence/20261122-p3-02-build-fixed-tool-path/artifacts/nka_gemm_v1.so';expected='66fcc80f9122c919f92c72e37a01794a792c91a58b71d5705b160e0bc979dfc3';assert hashlib.sha256(binary.read_bytes()).hexdigest()==expected
    encoded={rel:base64.b64encode(data).decode() for rel,data in sources.items()};encoded['fallback.so']=base64.b64encode(binary.read_bytes()).decode()
    payload='FILES='+repr(encoded)+'\n'+r'''
import base64,hashlib,io,json,os,shutil,stat,subprocess,tempfile,zipfile
from pathlib import Path
root=Path(tempfile.mkdtemp(prefix='nka-p3-02-builder-'));project=root/'project';project.mkdir()
for rel,data in FILES.items():
 target=project/rel;target.parent.mkdir(parents=True,exist_ok=True);target.write_bytes(base64.b64decode(data))
deps=root/'deps';shutil.copytree('/tmp/nka-p1-02-pydeps-hw9k37vu',deps,ignore=shutil.ignore_patterns('__pycache__'));deps.chmod(0o755)
for path in deps.rglob('*'):path.chmod(0o755 if path.is_dir() else 0o644)
import ninja
(deps/'bin').mkdir();shutil.copyfile(Path(ninja.BIN_DIR)/'ninja',deps/'bin/ninja');(deps/'bin/ninja').chmod(0o755)
(deps/'readonly-guard').write_text('unchanged');(deps/'readonly-guard').chmod(0o666)
env={k:v for k,v in os.environ.items() if k in ['HOME','PATH','LANG','LD_LIBRARY_PATH']};env.update(PYTHONPATH=str(project),OPENBLAS_NUM_THREADS='1')
cmd=['/home/dministrator/cuda-lab/.venv/bin/python',str(project/'experiments/p3_02/builder_regression.py'),'--project',str(project),'--deps',str(deps),'--work',str(root/'run'),'--fallback',str(project/'fallback.so')]
with (root/'stdout.log').open('w') as out,(root/'stderr.log').open('w') as err:proc=subprocess.run(cmd,cwd=project,env=env,stdout=out,stderr=err,timeout=3600)
# Preserve every distinct original byte sequence once; repeated paths bind to the same SHA.
manifest={};blobs={}
for item in root.rglob('*'):
 rel=item.relative_to(root)
 if any(part in ['project','deps','control','artifact','__pycache__'] for part in rel.parts):continue
 info=item.lstat()
 if not stat.S_ISREG(info.st_mode):continue
 if info.st_size>12*1024*1024:raise ValueError('unbounded evidence file')
 data=item.read_bytes();sha=hashlib.sha256(data).hexdigest();manifest[str(rel)]={'sha256':sha,'bytes':len(data)};blobs.setdefault(sha,data)
meta={'remote_work':str(root),'command':cmd,'exit':proc.returncode,'manifest':manifest}
raw=io.BytesIO()
with zipfile.ZipFile(raw,'w',compression=zipfile.ZIP_DEFLATED) as archive:
 archive.writestr('execution.json',json.dumps(meta))
 for sha,data in blobs.items():archive.writestr('blobs/'+sha,data)
print(json.dumps({'encoding':'zip-base64','data':base64.b64encode(raw.getvalue()).decode()}))
'''
    (a.evidence/'source-hashes.json').write_text(json.dumps({rel:hashlib.sha256(data).hexdigest() for rel,data in sources.items()},indent=2)+'\n')
    (a.evidence/'remote-driver.py').write_text(payload.replace(repr(encoded['fallback.so']),repr('<archived-fallback-'+expected+'>')))
    result=execute_python(payload,timeout=3800)
    # Raw transport is a lossless ZIP envelope; store it directly without duplicate base64.
    if result.returncode:
        (a.evidence/'transport-error.log').write_text(result.stdout+result.stderr);raise RuntimeError('transport failed')
    wire=json.loads(result.stdout);data=base64.b64decode(wire['data']);(a.evidence/'transport.zip').write_bytes(data)
    with zipfile.ZipFile(io.BytesIO(data)) as archive:
        execution=json.loads(archive.read('execution.json'));blobroot=a.evidence/'blobs';blobroot.mkdir();outroot=a.evidence/'remote';outroot.mkdir()
        for rel,meta in execution['manifest'].items():
            path=Path(rel)
            if path.is_absolute() or '..' in path.parts:raise ValueError('invalid evidence path')
            sha=meta['sha256'];blob=blobroot/sha
            if not blob.exists():
                content=archive.read('blobs/'+sha)
                if len(content)!=meta['bytes'] or hashlib.sha256(content).hexdigest()!=sha:raise ValueError('archive hash mismatch')
                blob.write_bytes(content)
            target=outroot/path;target.parent.mkdir(parents=True,exist_ok=True)
            import os
            os.link(blob,target)
    (a.evidence/'execution.json').write_text(json.dumps(execution,indent=2)+'\n');print(json.dumps({k:execution[k] for k in ['remote_work','exit']}))
    report=a.evidence/'remote/run/result.json'
    if report.exists():
        d=json.loads(report.read_text());print(json.dumps({'status':d['status'],'builds':len(d['builds']),'gpu_runs':len(d['runs']),'failures':d['failures']}))
    if execution['exit']:raise RuntimeError('builder regression failed; evidence preserved')


if __name__=='__main__':main()
