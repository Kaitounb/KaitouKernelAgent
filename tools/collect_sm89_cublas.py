"""Read-only public evidence collection; reuses the established lossless SHA segmentation."""
import argparse,base64,hashlib,io,json,os,re,zipfile
from pathlib import Path
from tools.ssh_server_gpu import ROOT,execute_python


def collect(output,*,target='cublas'):
    locations={'cublas':('evidence/20261123-sm89-cublas-v2/location.json','/data/nka-cublas-v2/sm89-[a-zA-Z0-9_-]+'),'runtime':('evidence/20261123-sm89-runtime-source/location.json','/data/nka-sm89-install-[a-zA-Z0-9_-]+'),'qualified':('evidence/20261123-sm89-qualified-installed/location.json','/data/nka-sm89-qualified-[a-zA-Z0-9_-]+')}
    location_path,pattern=locations[target];output=Path(output);output.mkdir(exist_ok=False)
    location=json.loads((ROOT/location_path).read_text());remote=location['remote_root'];assert re.fullmatch(pattern,remote)
    known={}
    for folder in (ROOT/'evidence').glob('*/blobs'):
        for path in folder.iterdir():
            if re.fullmatch('[0-9a-f]{64}',path.name):known.setdefault(path.name,path)
    for path in (ROOT/'kernel_agent/data/artifacts').glob('*.so'):known.setdefault(hashlib.sha256(path.read_bytes()).hexdigest(),path)
    source=(ROOT/'experiments/gemm_rel01/archive_parts.py').read_text()
    plan={'remote_root':remote,'parts_source_sha256':hashlib.sha256(source.encode()).hexdigest(),'encoding':'lossless-sha-segments-v1','read_only':True,'gpu_execution':False,'new_model_calls':0,'production_release':False};(output/'plan.json').write_text(json.dumps(plan,indent=2))
    payload='REMOTE='+repr(remote)+'\nSOURCE='+repr(source)+'\nKNOWN='+repr(list(known))+'\n'+r'''
import hashlib,io,json,os,stat,sys,zipfile,base64
from pathlib import Path
root=Path(REMOTE);sys.path.insert(0,str(root/('prefix' if (root/'prefix').exists() else 'prefix-rc2')));namespace={};exec(compile(SOURCE,'original_archive_parts.py','exec'),namespace);parts=namespace['parts'];known=set(KNOWN);manifest={};chunks={}
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
    (output/'driver.py').write_text(payload);r=execute_python(payload,900,connection_path=ROOT/'.private/server-gpu-next-connection.json');(output/'transport.stderr').write_text(r.stderr)
    if r.returncode:raise RuntimeError('Read-only collection failed; no GPU replay')
    wire=json.loads(r.stdout);raw=base64.b64decode(wire['archive']);assert hashlib.sha256(raw).hexdigest()==wire['archive_sha256'];(output/'raw.zip').write_bytes(raw);blobs=output/'blobs';blobs.mkdir()
    with zipfile.ZipFile(io.BytesIO(raw)) as z:
        execution=json.loads(z.read('execution.json'))
        for rel,meta in execution['manifest'].items():
            name=Path(rel);assert not name.is_absolute() and '..' not in name.parts;pieces=[]
            for part in meta['segments']:
                h=part['sha256'];value=known[h].read_bytes() if h in known else z.read('blobs/'+h);assert len(value)==part['bytes'] and hashlib.sha256(value).hexdigest()==h;pieces.append(value)
            value=b''.join(pieces);assert len(value)==meta['bytes'] and hashlib.sha256(value).hexdigest()==meta['sha256'];blob=blobs/meta['sha256']
            if not blob.exists():
                if meta['sha256'] in known:os.link(known[meta['sha256']],blob)
                else:blob.write_bytes(value);blob.chmod(0o444)
            target=output/'remote'/name;target.parent.mkdir(parents=True,exist_ok=True);os.link(blob,target)
    (output/'execution.json').write_text(json.dumps(execution,indent=2));value={'status':'COLLECTED','remote_root':remote,'raw_files_verified':len(execution['manifest']),'archive_sha256':wire['archive_sha256'],'gpu_execution':False,'new_model_calls':0};(output/'collection.json').write_text(json.dumps(value,indent=2));print(json.dumps(value))


def main():
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);p.add_argument('--target',choices=['cublas','runtime','qualified'],default='cublas');a=p.parse_args();collect(a.output,target=a.target)


if __name__=='__main__':main()
