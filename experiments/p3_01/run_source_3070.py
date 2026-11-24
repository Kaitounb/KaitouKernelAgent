"""Run a source-reviewed actual model proposal through the accepted bounded CUDA path."""
import argparse,base64,hashlib,io,json,os,re,zipfile
from pathlib import Path
from tools.ssh_3070 import execute_python
from evaluators.gemm_decision import ROOT,identity,sha,canonical,protected_hashes


def main():
    p=argparse.ArgumentParser();p.add_argument('--proposal',type=Path,required=True);p.add_argument('--evidence',type=Path,required=True);p.add_argument('--resume-remote');a=p.parse_args()
    if a.resume_remote and not re.fullmatch(r'/tmp/nka-p3-01-source-[a-z0-9_]+',a.resume_remote):p.error('resume path must identify an existing owned source run')
    if a.evidence.exists():p.error('Use a new evidence directory')
    a.evidence.mkdir(parents=True);proposal=json.loads((a.proposal/'proposal.json').read_text());assert proposal['actual_parent']==identity(4)
    before=protected_hashes();candidate=identity(4);candidate['files']=candidate['files'].copy();candidate['files']['kernels/gemm/wmma_sm86.cu']=sha(a.proposal/'source/kernels/gemm/wmma_sm86.cu');candidate.pop('id');candidate['id']=hashlib.sha256(canonical(candidate)).hexdigest()
    binding={'parent_id':identity(4)['id'],'candidate':candidate,'raw_response_sha256':proposal['raw_response_sha256'],'source_origin':'one actual DeepSeek response; git --recount metadata recovery only','reviewed':True}
    source_paths=list((ROOT/'kernel_agent').glob('*.py'))+[ROOT/'evaluators/gemm_regression.py',ROOT/'evaluators/gemm_decision.py',ROOT/'workloads/gemm_development_v1.json']+list((ROOT/'experiments/p3_01').glob('*.py'))
    upstream=ROOT/'provenance/upstream-reviews/20261121-p1-02/ScalingIntelligence__KernelBench/src/kernelbench';source_paths+=list(upstream.glob('*.py'))
    files={str(path.relative_to(ROOT)):path.read_bytes() for path in source_paths}
    for name in ['wmma_sm86.cu','cublas_sm86.cu','gemm_contract.cuh']:
        files['kernels/gemm/'+name]=(ROOT/'kernels/gemm'/name).read_bytes();files['candidate/'+name]=(a.proposal/'source/kernels/gemm'/name).read_bytes()
    files['source-binding.json']=json.dumps(binding,indent=2).encode()
    for rel,data in files.items():
        target=a.evidence/'sources'/rel;target.parent.mkdir(parents=True,exist_ok=True);target.write_bytes(data)
    (a.evidence/'source-review.json').write_text(json.dumps({'status':'REVIEWED_FOR_ISOLATED_EXECUTION','candidate':binding,'checks':['license/includes and host launcher byte-identical','cuBLAS and contract header unchanged','no new host code/barrier/dependency','two 16-wide tiles maintain ascending FP32 accumulation order','for condition bounds second tile; multiple-of-16 K leaves at most one tail','new source evidence required; no reuse of old performance']},indent=2)+'\n')
    encoded={rel:base64.b64encode(data).decode() for rel,data in files.items()}
    payload='FILES='+repr(encoded)+'\nPREVIOUS='+repr(a.resume_remote)+'\n'+r'''
import base64,hashlib,io,json,os,shutil,stat,subprocess,tempfile,zipfile
from pathlib import Path
root=Path(tempfile.mkdtemp(prefix='nka-p3-01-source-'));project=root/'project';project.mkdir()
for rel,data in FILES.items():
 p=project/rel;p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes(base64.b64decode(data))
deps=root/'deps';shutil.copytree('/tmp/nka-p1-02-pydeps-hw9k37vu',deps,ignore=shutil.ignore_patterns('__pycache__'));deps.chmod(0o755)
for p in deps.rglob('*'):p.chmod(0o755 if p.is_dir() else 0o644)
import ninja
(deps/'bin').mkdir();shutil.copyfile(Path(ninja.BIN_DIR)/'ninja',deps/'bin/ninja');(deps/'bin/ninja').chmod(0o755)
(deps/'readonly-guard').write_text('unchanged');(deps/'readonly-guard').chmod(0o666)
env={k:v for k,v in os.environ.items() if k in ['HOME','PATH','LANG','LD_LIBRARY_PATH']};env.update(PYTHONPATH=str(project),OPENBLAS_NUM_THREADS='1')
if PREVIOUS:
 cmd=['/home/dministrator/cuda-lab/.venv/bin/python',str(project/'experiments/p3_01/supplement.py'),'--previous',PREVIOUS,'--work',str(root/'run')]
else:
 cmd=['/home/dministrator/cuda-lab/.venv/bin/python',str(project/'experiments/p3_01/source_pipeline.py'),'--project',str(project),'--candidate',str(project/'candidate'),'--deps',str(deps),'--work',str(root/'run')]
with (root/'stdout.log').open('w') as out,(root/'stderr.log').open('w') as err:proc=subprocess.run(cmd,cwd=project,env=env,stdout=out,stderr=err,timeout=3000)
manifest={};blobs={}
for item in root.rglob('*'):
 rel=item.relative_to(root)
 if any(part in ['project','deps','artifact','__pycache__'] for part in rel.parts):continue
 info=item.lstat()
 if not stat.S_ISREG(info.st_mode):continue
 if info.st_size>128*1024*1024:raise ValueError('unbounded source evidence')
 data=item.read_bytes();digest=hashlib.sha256(data).hexdigest();manifest[str(rel)]={'sha256':digest,'bytes':len(data)};blobs.setdefault(digest,data)
meta={'remote_work':str(root),'command':cmd,'exit':proc.returncode,'manifest':manifest}
raw=io.BytesIO()
with zipfile.ZipFile(raw,'w',compression=zipfile.ZIP_DEFLATED) as archive:
 archive.writestr('execution.json',json.dumps(meta))
 for digest,data in blobs.items():archive.writestr('blobs/'+digest,data)
print(json.dumps({'encoding':'zip-base64','data':base64.b64encode(raw.getvalue()).decode()}))
'''
    (a.evidence/'remote-driver.py').write_text(payload);(a.evidence/'source-hashes.json').write_text(json.dumps({rel:hashlib.sha256(data).hexdigest() for rel,data in files.items()},indent=2)+'\n')
    r=execute_python(payload,timeout=3400)
    if r.returncode:(a.evidence/'transport-error.log').write_text(r.stdout+r.stderr);raise RuntimeError('source evaluation transport failed')
    wire=json.loads(r.stdout);data=base64.b64decode(wire['data']);(a.evidence/'transport.zip').write_bytes(data)
    with zipfile.ZipFile(io.BytesIO(data)) as archive:
        record=json.loads(archive.read('execution.json'));blobroot=a.evidence/'blobs';blobroot.mkdir();outroot=a.evidence/'remote';outroot.mkdir()
        for rel,meta in record['manifest'].items():
            path=Path(rel)
            if path.is_absolute() or '..' in path.parts:raise ValueError('invalid archive path')
            digest=meta['sha256'];blob=blobroot/digest
            if not blob.exists():
                value=archive.read('blobs/'+digest)
                if hashlib.sha256(value).hexdigest()!=digest or len(value)!=meta['bytes']:raise ValueError('source archive corrupted')
                blob.write_bytes(value);blob.chmod(0o444)
            target=outroot/path;target.parent.mkdir(parents=True,exist_ok=True);os.link(blob,target)
    (a.evidence/'execution.json').write_text(json.dumps(record,indent=2)+'\n');print(json.dumps({k:record[k] for k in ['remote_work','exit']}))
    if protected_hashes()!=before:raise RuntimeError('protected source changed')
    if (a.evidence/'remote/run/result.json').exists():
        result=json.loads((a.evidence/'remote/run/result.json').read_text());print(json.dumps({k:result[k] for k in ['status','qa_checks','aa_status','verdict','reason']}))
    if record['exit']:raise RuntimeError('source evaluation failed; evidence preserved')


if __name__=='__main__':main()
