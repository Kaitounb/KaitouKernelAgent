"""Run the reviewed B pilot source via the unchanged installed dev5 optimizer and boundary.
Transport reuses the proven lossless SHA-segment encoder; no extra model calls.
"""
import argparse,base64,hashlib,io,json,os,re,uuid,zipfile
from pathlib import Path
from tools.ssh_3070 import ROOT,execute_python


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path,default=ROOT/'evidence/20261122-feedback-B-installed-evaluation');parser.add_argument('--proposal-dir',type=Path);parser.add_argument('--review-file',type=Path);args=parser.parse_args();output=args.output;output.mkdir(exist_ok=False);pilot=ROOT/'evidence/20261122-feedback-live-pilot';proposal=args.proposal_dir or pilot/'B';review=json.loads(args.review_file.read_text()) if args.review_file else json.loads((pilot/'source-review.json').read_text())['B'];assert review['status']=='REVIEWED_PENDING_FRESH_BUILD_AND_GPU_QA';assert hashlib.sha256((proposal/'wmma_sm86.cu').read_bytes()).hexdigest()==review['source_sha256'];(output/'controller.py').write_bytes(Path(__file__).read_bytes())
    known={}
    for folder in (ROOT/'evidence').glob('*/blobs'):
        for blob in folder.iterdir():
            if re.fullmatch('[0-9a-f]{64}',blob.name):known.setdefault(blob.name,blob)
    for blob in (ROOT/'kernel_agent/data/artifacts').glob('*.so'):known[hashlib.sha256(blob.read_bytes()).hexdigest()]=blob
    remote='/tmp/nka-feedback-B-'+uuid.uuid4().hex[:12];(output/'reservation.json').write_text(json.dumps({'remote_work':remote,'status':'RESERVED_NO_AUTOMATIC_RETRY','source_sha256':review['source_sha256']},indent=2)+'\n')
    files={'response.json':(proposal/'response.json').read_bytes(),'archive_parts.py':(ROOT/'experiments/gemm_rel01/archive_parts.py').read_bytes()}
    payload='FILES='+repr({k:base64.b64encode(v).decode() for k,v in files.items()})+'\nREMOTE='+repr(remote)+'\nREVIEW='+repr(review['source_sha256'])+'\nKNOWN='+repr(list(known))+'\n'+r'''
import base64,fcntl,hashlib,importlib.util,io,json,os,stat,subprocess,sys,zipfile
from pathlib import Path
gpu_lock=open('/tmp/nka-reviewed-gpu-execution.lock','a');fcntl.flock(gpu_lock,fcntl.LOCK_EX)
root=Path(REMOTE);root.mkdir()
for name,data in FILES.items():(root/name).write_bytes(base64.b64decode(data))
prefix=Path('/tmp/nka-gemm-rel01-500z3apq/run/prefix');deps=Path('/tmp/nka-gemm-rel01-f3ekx0tu/run/deps');assert (deps/'bin/ninja').is_file()
env={k:v for k,v in os.environ.items() if k in ['HOME','PATH','LANG','LD_LIBRARY_PATH']};env.update(PYTHONPATH=str(prefix),OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1');cmd=[str(prefix/'bin/kernel-agent'),'optimize','--run-dir',str(root/'run'),'--response',str(root/'response.json'),'--recount','--review-sha',REVIEW,'--deps',str(deps)]
with (root/'controller.stdout').open('w') as out,(root/'controller.stderr').open('w') as err:r=subprocess.run(cmd,cwd=root,env=env,stdout=out,stderr=err,timeout=2400)
sys.path.insert(0,str(prefix));spec=importlib.util.spec_from_file_location('original_archive_parts',root/'archive_parts.py');module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module);manifest={};chunks={};known=set(KNOWN)
for path in root.rglob('*'):
 rel=path.relative_to(root)
 if any(x in ['artifact','__pycache__'] for x in rel.parts):continue
 info=path.lstat()
 if not stat.S_ISREG(info.st_mode):continue
 if info.st_size>128*1024*1024:raise ValueError('unbounded evidence file')
 value=path.read_bytes();segments=[]
 for data in module.parts(value):
  digest=hashlib.sha256(data).hexdigest();segments.append({'sha256':digest,'bytes':len(data)})
  if digest not in known:chunks.setdefault(digest,data)
 manifest[str(rel)]={'sha256':hashlib.sha256(value).hexdigest(),'bytes':len(value),'segments':segments}
record={'remote_work':str(root),'command':cmd,'exit':r.returncode,'manifest':manifest,'encoding':'lossless-sha-segments-v1'};stream=io.BytesIO()
with zipfile.ZipFile(stream,'w',compression=zipfile.ZIP_DEFLATED) as archive:
 archive.writestr('execution.json',json.dumps(record))
 for digest,value in chunks.items():archive.writestr('blobs/'+digest,value)
print(json.dumps({'archive':base64.b64encode(stream.getvalue()).decode()}))
'''
    (output/'driver.py').write_text(payload);r=execute_python(payload,3000);(output/'transport.stderr').write_text(r.stderr)
    if r.returncode:(output/'transport.stdout').write_text(r.stdout);raise RuntimeError('transport failed; reservation retained, inspect existing run only')
    data=base64.b64decode(json.loads(r.stdout)['archive']);(output/'raw.zip').write_bytes(data);blobs=output/'blobs';blobs.mkdir()
    with zipfile.ZipFile(io.BytesIO(data)) as archive:
        execution=json.loads(archive.read('execution.json'))
        for rel,meta in execution['manifest'].items():
            name=Path(rel);assert not name.is_absolute() and '..' not in name.parts;parts=[]
            for segment in meta['segments']:
                digest=segment['sha256'];value=known[digest].read_bytes() if digest in known else archive.read('blobs/'+digest);assert len(value)==segment['bytes'] and hashlib.sha256(value).hexdigest()==digest;parts.append(value)
            value=b''.join(parts);assert len(value)==meta['bytes'] and hashlib.sha256(value).hexdigest()==meta['sha256'];blob=blobs/meta['sha256']
            if not blob.exists():
                if meta['sha256'] in known:os.link(known[meta['sha256']],blob)
                else:blob.write_bytes(value);blob.chmod(0o444)
            target=output/'remote'/name;target.parent.mkdir(parents=True,exist_ok=True);os.link(blob,target)
    (output/'execution.json').write_text(json.dumps(execution,indent=2)+'\n');status=json.loads((output/'remote/run/run.json').read_text());print(json.dumps({k:status.get(k) for k in ['status','verdict','reason','qa_checks','aa_status','candidate_id','new_external_model_calls']}))


if __name__=='__main__':main()
