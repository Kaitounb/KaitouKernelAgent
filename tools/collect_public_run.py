"""Read-only recovery of a known public task directory; never reruns its workload."""
import argparse,base64,hashlib,io,json,re,zipfile
from pathlib import Path


def main():
    p=argparse.ArgumentParser();p.add_argument('--target',choices=['3070','server'],required=True);p.add_argument('--remote-root',required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    if not (a.remote_root.startswith('/tmp/nka-') or re.fullmatch(r'/home/dministrator/cuda-lab/nka-persistent/(?:operators-0\.1\.1(?:rc2)?|gemm-0\.1\.2(?:rc1)?)-[A-Za-z0-9_-]+',a.remote_root)) or '..' in Path(a.remote_root).parts:raise ValueError('only exact known public task roots')
    a.output.mkdir(exist_ok=False);payload='ROOT='+repr(a.remote_root)+'\n'+r'''
import base64,hashlib,io,json,zipfile
from pathlib import Path
root=Path(ROOT);manifest={};blobs={}
for path in root.rglob('*'):
 if path.is_file() and not path.is_symlink() and '__pycache__' not in path.parts:
  value=path.read_bytes();digest=hashlib.sha256(value).hexdigest();manifest[str(path.relative_to(root))]={'sha256':digest,'bytes':len(value)};blobs.setdefault(digest,value)
raw=io.BytesIO()
with zipfile.ZipFile(raw,'w',compression=zipfile.ZIP_DEFLATED) as z:
 z.writestr('manifest.json',json.dumps(manifest))
 for digest,value in blobs.items():z.writestr('blobs/'+digest,value)
print(json.dumps({'archive_sha256':hashlib.sha256(raw.getvalue()).hexdigest(),'archive':base64.b64encode(raw.getvalue()).decode()}))
'''
    if a.target=='3070':from tools.ssh_3070 import execute_python
    else:from tools.ssh_server_gpu import execute_python
    r=execute_python(payload,120);(a.output/'transport.stderr').write_text(r.stderr)
    if r.returncode:raise RuntimeError('read-only collection failed')
    wire=json.loads(r.stdout);raw=base64.b64decode(wire['archive']);assert hashlib.sha256(raw).hexdigest()==wire['archive_sha256'];(a.output/'raw.zip').write_bytes(raw)
    with zipfile.ZipFile(io.BytesIO(raw)) as z:
        manifest=json.loads(z.read('manifest.json'))
        for rel,meta in manifest.items():
            name=Path(rel);assert not name.is_absolute() and '..' not in name.parts;data=z.read('blobs/'+meta['sha256']);assert hashlib.sha256(data).hexdigest()==meta['sha256'] and len(data)==meta['bytes'];path=a.output/'remote'/name;path.parent.mkdir(parents=True,exist_ok=True);path.write_bytes(data)
    result={'status':'COLLECTED','remote_root':a.remote_root,'files_verified':len(manifest),'archive_sha256':wire['archive_sha256'],'gpu_execution':False,'new_model_calls':0};(a.output/'collection.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result))


if __name__=='__main__':main()
