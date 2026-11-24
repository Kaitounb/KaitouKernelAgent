"""Stage public sources, install dev wheel, exercise CLI; collect lossless segmented evidence."""
import argparse,base64,hashlib,io,json,os,re,zipfile
from pathlib import Path
from tools.ssh_3070 import ROOT,execute_python
from evaluators.gemm_decision import protected_hashes


def main():
    p=argparse.ArgumentParser();p.add_argument('--evidence',type=Path,required=True);p.add_argument('--previous-remote');p.add_argument('--dual-test',action='store_true');a=p.parse_args()
    if a.previous_remote and not re.fullmatch('/tmp/nka-gemm-rel01-[a-z0-9_]+',a.previous_remote):p.error('expected owned prior run')
    a.evidence.mkdir(parents=True,exist_ok=False);before=protected_hashes()
    paths=[ROOT/'pyproject.toml']+[p for p in (ROOT/'kernel_agent').rglob('*') if p.is_file() and '__pycache__' not in p.parts and p.suffix!='.pyc' and not p.name.startswith('.')]
    paths += [ROOT/'experiments/gemm_rel01/validate_install.py',ROOT/'experiments/gemm_rel01/archive_parts.py',ROOT/'evidence/20261122-p3-01-deepseek-source/response.json']
    if a.previous_remote:paths += [ROOT/'experiments/gemm_rel01/validate_delta.py',ROOT/'tests/test_optimizer.py',*[ROOT/f'experiments/p3_01/{n}.py' for n in ['fixtures','source_timing','evaluate','patches']]]
    if a.dual_test:paths += [ROOT/'experiments/gemm_rel02/validate_install.py',*[ROOT/f'tests/{n}.py' for n in ['test_optimizer','test_pair_decision','pair_fixtures','test_gemm_decision']],*[ROOT/f'experiments/p3_01/{n}.py' for n in ['fixtures','source_timing','evaluate','patches']]]
    files={str(p.relative_to(ROOT)):p.read_bytes() for p in paths};hashes={p:hashlib.sha256(v).hexdigest() for p,v in files.items()}
    if a.dual_test:
        wrong=ROOT/'evidence/20261122-p3-02-bounded-builder/remote/run/build-wrong-output/artifacts/nka_gemm_v1.so';files['diagnostics/wrong-output.so']=wrong.read_bytes();hashes['diagnostics/wrong-output.so']=hashlib.sha256(wrong.read_bytes()).hexdigest()
    for rel,data in files.items():
        out=a.evidence/'sources'/rel;out.parent.mkdir(parents=True,exist_ok=True);out.write_bytes(data)
    (a.evidence/'source-hashes.json').write_text(json.dumps(hashes,indent=2)+'\n')
    known={}
    for folder in ['evidence/20261122-p3-01-source-gpu/blobs','evidence/20261122-p3-01-source-sanitizer-channel/blobs','evidence/20261122-p3-02-bounded-builder/blobs']:
        for blob in (ROOT/folder).glob('*'):
            if re.fullmatch('[0-9a-f]{64}',blob.name):known[blob.name]=blob
    for blob in (ROOT/'kernel_agent/data/artifacts').glob('*.so'):known[hashlib.sha256(blob.read_bytes()).hexdigest()]=blob
    encoded={p:base64.b64encode(v).decode() for p,v in files.items()}
    payload='FILES='+repr(encoded)+'\nKNOWN='+repr(list(known))+'\nPREVIOUS='+repr(a.previous_remote)+'\nDUAL='+repr(a.dual_test)+'\n'+r'''
import base64,hashlib,io,json,os,stat,subprocess,sys,tempfile,zipfile
from pathlib import Path
root=Path(tempfile.mkdtemp(prefix='nka-gemm-rel01-'));project=root/'project';project.mkdir()
for rel,encoded in FILES.items():
 p=project/rel;p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes(base64.b64decode(encoded))
env={k:v for k,v in os.environ.items() if k in ['HOME','PATH','LANG','LD_LIBRARY_PATH']};env.update(OPENBLAS_NUM_THREADS='1')
cmd=['/home/dministrator/cuda-lab/.venv/bin/python',str(project/'experiments/gemm_rel01/validate_install.py'),'--project',str(project),'--work',str(root/'run')]
if PREVIOUS:cmd=['/home/dministrator/cuda-lab/.venv/bin/python',str(project/'experiments/gemm_rel01/validate_delta.py'),'--project',str(project),'--work',str(root/'run'),'--previous',PREVIOUS]
if DUAL:cmd=['/home/dministrator/cuda-lab/.venv/bin/python',str(project/'experiments/gemm_rel02/validate_install.py'),'--project',str(project),'--work',str(root/'run')]
with (root/'stdout.log').open('w') as out,(root/'stderr.log').open('w') as err:proc=subprocess.run(cmd,cwd=project,env=env,stdout=out,stderr=err,timeout=2400)
sys.path.insert(0,str(project));from experiments.gemm_rel01.archive_parts import parts
manifest={};chunks={};known=set(KNOWN)
for item in root.rglob('*'):
 rel=item.relative_to(root)
 if any(x in ['project','prefix','deps','empty','artifact','__pycache__'] for x in rel.parts):continue
 info=item.lstat()
 if not stat.S_ISREG(info.st_mode):continue
 if info.st_size>128*1024*1024:raise ValueError('unbounded evidence file')
 data=item.read_bytes();segments=[]
 for value in parts(data):
  digest=hashlib.sha256(value).hexdigest();segments.append({'sha256':digest,'bytes':len(value)})
  if digest not in known:chunks.setdefault(digest,value)
 manifest[str(rel)]={'sha256':hashlib.sha256(data).hexdigest(),'bytes':len(data),'segments':segments}
record={'remote_work':str(root),'command':cmd,'exit':proc.returncode,'manifest':manifest,'encoding':'lossless-sha-segments-v1'}
stream=io.BytesIO()
with zipfile.ZipFile(stream,'w',compression=zipfile.ZIP_DEFLATED) as archive:
 archive.writestr('execution.json',json.dumps(record))
 for digest,value in chunks.items():archive.writestr('blobs/'+digest,value)
print(json.dumps({'data':base64.b64encode(stream.getvalue()).decode()}))
'''
    (a.evidence/'remote-driver.py').write_text(payload)
    response=execute_python(payload,timeout=3000)
    if response.returncode:
        (a.evidence/'transport-error.log').write_text(response.stdout+response.stderr);raise RuntimeError('transport failed; retained')
    data=base64.b64decode(json.loads(response.stdout)['data']);(a.evidence/'transport.zip').write_bytes(data);blobs=a.evidence/'blobs';blobs.mkdir();remote=a.evidence/'remote';remote.mkdir();reused=set()
    with zipfile.ZipFile(io.BytesIO(data)) as archive:
        execution=json.loads(archive.read('execution.json'))
        for rel,meta in execution['manifest'].items():
            name=Path(rel)
            if name.is_absolute() or '..' in name.parts:raise ValueError('invalid evidence path')
            values=[]
            for segment in meta['segments']:
                digest=segment['sha256']
                if digest in known:
                    value=known[digest].read_bytes();reused.add(digest)
                else:value=archive.read('blobs/'+digest)
                if len(value)!=segment['bytes'] or hashlib.sha256(value).hexdigest()!=digest:raise ValueError('segment changed')
                values.append(value)
            value=b''.join(values)
            if len(value)!=meta['bytes'] or hashlib.sha256(value).hexdigest()!=meta['sha256']:raise ValueError('whole evidence file changed')
            blob=blobs/meta['sha256']
            if not blob.exists():
                if meta['sha256'] in known:os.link(known[meta['sha256']],blob)
                else:blob.write_bytes(value);blob.chmod(0o444)
            target=remote/name;target.parent.mkdir(parents=True,exist_ok=True);os.link(blob,target)
    (a.evidence/'execution.json').write_text(json.dumps(execution,indent=2)+'\n');(a.evidence/'transport-reuse.json').write_text(json.dumps({'wire_bytes':len(data),'whole_files_verified':len(execution['manifest']),'reused_sha_blobs':{x:str(known[x].relative_to(ROOT)) for x in sorted(reused)},'raw_bytes_reconstructed':sum(m['bytes'] for m in execution['manifest'].values()),'candidate_evidence_generated_in_this_run':True},indent=2)+'\n')
    if protected_hashes()!=before:raise RuntimeError('protected harness/source changed')
    print(json.dumps({k:execution[k] for k in ['remote_work','exit']}))
    if (remote/'run/result.json').exists():
        result=json.loads((remote/'run/result.json').read_text());print(json.dumps({k:result[k] for k in ['status','version','installed_qa_outputs','new_model_calls','aa_segments','diagnostic_status','diagnostic_raw_segments'] if k in result}))
    if execution['exit']:raise RuntimeError('installed integration failed; all evidence retained')


if __name__=='__main__':main()
