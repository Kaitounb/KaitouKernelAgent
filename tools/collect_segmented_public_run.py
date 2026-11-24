"""Read-only cuBLAS qualification recovery using the proven lossless SHA transport."""
import argparse,base64,hashlib,io,json,os,re,zipfile
from pathlib import Path
from tools.ssh_3070 import ROOT,execute_python


def main():
    p=argparse.ArgumentParser();p.add_argument('--remote-root',required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--installed-prefix',default='/tmp/nka-operators-dev6-izkosqud/prefix');a=p.parse_args();assert re.fullmatch('/tmp/nka-cublas-final-[A-Za-z0-9_-]+',a.remote_root);assert Path(a.installed_prefix).is_absolute() and '..' not in Path(a.installed_prefix).parts;a.output.mkdir(exist_ok=False);known={}
    for folder in (ROOT/'evidence').glob('*/blobs'):
        for path in folder.iterdir():
            if re.fullmatch('[0-9a-f]{64}',path.name):known.setdefault(path.name,path)
    for path in (ROOT/'kernel_agent/data/artifacts').glob('*.so'):known[hashlib.sha256(path.read_bytes()).hexdigest()]=path
    source=(ROOT/'experiments/gemm_rel01/archive_parts.py').read_text();plan={'remote_root':a.remote_root,'installed_prefix':a.installed_prefix,'parts_source_sha256':hashlib.sha256(source.encode()).hexdigest(),'encoding':'lossless-sha-segments-v1','read_only':True,'gpu_execution':False,'model_calls':0,'file_limit_bytes':128*1024*1024,'transport_deadline_seconds':600};(a.output/'plan.json').write_text(json.dumps(plan,indent=2));payload='REMOTE='+repr(a.remote_root)+'\nPREFIX='+repr(a.installed_prefix)+'\nSOURCE='+repr(source)+'\nKNOWN='+repr(list(known))+'\n'+r'''
import hashlib,io,json,os,stat,sys,zipfile,base64
from pathlib import Path
sys.path.insert(0,PREFIX);namespace={};exec(compile(SOURCE,'original_archive_parts.py','exec'),namespace);parts=namespace['parts'];root=Path(REMOTE);known=set(KNOWN);manifest={};chunks={}
for path in root.rglob('*'):
 if '__pycache__' in path.parts:continue
 info=path.lstat()
 if not stat.S_ISREG(info.st_mode):continue
 if info.st_size>128*1024*1024:raise ValueError('per-file evidence cap exceeded')
 data=path.read_bytes();segments=[]
 for piece in parts(data):
  h=hashlib.sha256(piece).hexdigest();segments.append({'sha256':h,'bytes':len(piece)})
  if h not in known:chunks.setdefault(h,piece)
 manifest[str(path.relative_to(root))]={'sha256':hashlib.sha256(data).hexdigest(),'bytes':len(data),'segments':segments}
stream=io.BytesIO()
with zipfile.ZipFile(stream,'w',compression=zipfile.ZIP_DEFLATED) as z:
 z.writestr('execution.json',json.dumps({'remote_root':str(root),'manifest':manifest,'encoding':'lossless-sha-segments-v1'}))
 for h,v in chunks.items():z.writestr('blobs/'+h,v)
raw=stream.getvalue();print(json.dumps({'archive_sha256':hashlib.sha256(raw).hexdigest(),'archive':base64.b64encode(raw).decode()}))
'''
    (a.output/'driver.py').write_text(payload);r=execute_python(payload,600);(a.output/'transport.stderr').write_text(r.stderr)
    if r.returncode:raise RuntimeError('read-only segmented collection failed; GPU not rerun')
    wire=json.loads(r.stdout);raw=base64.b64decode(wire['archive']);assert hashlib.sha256(raw).hexdigest()==wire['archive_sha256'];(a.output/'raw.zip').write_bytes(raw);blobs=a.output/'blobs';blobs.mkdir()
    with zipfile.ZipFile(io.BytesIO(raw)) as z:
        execution=json.loads(z.read('execution.json'))
        for rel,meta in execution['manifest'].items():
            name=Path(rel);assert not name.is_absolute() and '..' not in name.parts;pieces=[]
            for s in meta['segments']:
                h=s['sha256'];value=known[h].read_bytes() if h in known else z.read('blobs/'+h);assert len(value)==s['bytes'] and hashlib.sha256(value).hexdigest()==h;pieces.append(value)
            value=b''.join(pieces);assert len(value)==meta['bytes'] and hashlib.sha256(value).hexdigest()==meta['sha256'];blob=blobs/meta['sha256']
            if not blob.exists():
                if meta['sha256'] in known:os.link(known[meta['sha256']],blob)
                else:blob.write_bytes(value);blob.chmod(0o444)
            target=a.output/'remote'/name;target.parent.mkdir(parents=True,exist_ok=True);os.link(blob,target)
    (a.output/'execution.json').write_text(json.dumps(execution,indent=2));result={'status':'COLLECTED','remote_root':a.remote_root,'raw_files_verified':len(execution['manifest']),'archive_sha256':wire['archive_sha256'],'gpu_execution':False,'new_model_calls':0};(a.output/'collection.json').write_text(json.dumps(result,indent=2));print(json.dumps(result))


if __name__=='__main__':main()
