"""Execute/collect the reserved new Attention acceptance once; exact data stays private."""
import base64,hashlib,io,json,zipfile
from pathlib import Path
from tools.ssh_3070 import ROOT,execute_python


def main():
 out=ROOT/'evidence/20261123-attention-sm86-independent';private=ROOT/'.private/attention-sm86-rc1-independent-v1';location=json.loads((private/'location.json').read_text());stability=json.loads((ROOT/'evidence/20261123-sm86-operator-deployment-stability/location.json').read_text())['remote_root'];reservation=private/'local-execution-reservation.json'
 with reservation.open('x') as f:json.dump({'status':'RESERVED_DO_NOT_REEXECUTE','manifest_sha256':json.loads((out/'freeze.json').read_text())['manifest_sha256']},f)
 reservation.chmod(0o600)
 payload='LOCATION='+repr(location)+'\nSTABILITY='+repr(stability)+'\n'+r'''
import base64,hashlib,io,json,subprocess,zipfile
from pathlib import Path
assert (Path(STABILITY)/'result.json').is_file();root=Path(LOCATION['remote_private_root']);script=LOCATION['remote_host_script'];installed=LOCATION['installed_root'];cmd=['/home/dministrator/cuda-lab/.venv/bin/python','-B',script,'--root',str(root),'--install-root',installed,'--phase','run'];r=subprocess.run(cmd,capture_output=True,text=True,timeout=1200);(root/'host-stdout').write_text(r.stdout);(root/'host-stderr').write_text(r.stderr)
summary=json.loads((root/'summary.json').read_text()) if (root/'summary.json').exists() else {'status':'FAIL_PRIVATE_EVALUATOR','exit':r.returncode};manifest={};blobs={}
for p in root.rglob('*'):
 if p.is_file() and not p.is_symlink() and '__pycache__' not in p.parts:
  b=p.read_bytes();h=hashlib.sha256(b).hexdigest();manifest[str(p.relative_to(root))]={'sha256':h,'bytes':len(b)};blobs.setdefault(h,b)
stream=io.BytesIO()
with zipfile.ZipFile(stream,'w',compression=zipfile.ZIP_DEFLATED) as z:
 z.writestr('archive-manifest.json',json.dumps(manifest))
 for h,b in blobs.items():z.writestr('blobs/'+h,b)
b=stream.getvalue();print(json.dumps({'summary':summary,'archive_sha256':hashlib.sha256(b).hexdigest(),'archive':base64.b64encode(b).decode()}))
'''
 (private/'run-driver.py').write_text(payload);(private/'run-driver.py').chmod(0o600);r=execute_python(payload,1260);(private/'transport.stderr').write_text(r.stderr);(private/'transport.stderr').chmod(0o600)
 if r.returncode:raise RuntimeError('Inspect existing private one-shot job; do not rerun')
 v=json.loads(r.stdout);b=base64.b64decode(v.pop('archive'));assert hashlib.sha256(b).hexdigest()==v['archive_sha256'];p=private/'raw.zip';p.write_bytes(b);p.chmod(0o600)
 with zipfile.ZipFile(io.BytesIO(b)) as z:
  manifest=json.loads(z.read('archive-manifest.json'))
  for name,row in manifest.items():
   n=Path(name);assert not n.is_absolute() and '..' not in n.parts;b=z.read('blobs/'+row['sha256']);assert hashlib.sha256(b).hexdigest()==row['sha256'] and len(b)==row['bytes'];p=private/'remote'/n;p.parent.mkdir(parents=True,exist_ok=True,mode=0o700);p.write_bytes(b);p.chmod(0o600)
 summary={**v['summary'],'archive_sha256':v['archive_sha256'],'private_files_collected':len(manifest)};(out/'summary.json').write_text(json.dumps(summary,indent=2));print(json.dumps(summary))
if __name__=='__main__':main()
