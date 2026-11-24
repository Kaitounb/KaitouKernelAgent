"""One-shot committed MLA acceptance transport; only aggregate evidence is public."""
import base64,hashlib,io,json,uuid,zipfile
from pathlib import Path
from tools.ssh_3070 import ROOT,execute_python


def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def main():
    release=ROOT/'releases/operators-sm86-rc1';seal=json.loads((release/'protocol.freeze.json').read_text());assert sha(release/'freeze.json')==seal['candidate_freeze_sha256'] and sha(release/'protocol.json')==seal['protocol_sha256'];assert all(sha(ROOT/n)==h for n,h in seal['trusted_evaluator_sha256'].items());freeze=json.loads((release/'freeze.json').read_text());protocol=json.loads((release/'protocol.json').read_text());wheel=ROOT/'artifacts/wheels'/freeze['wheel_filename'];assert sha(wheel)==freeze['wheel_sha256']
    private=ROOT/'.private/operators-sm86-rc1-evidence';private.mkdir(mode=0o700,exist_ok=False);remote='/tmp/nka-sealed-mla-'+uuid.uuid4().hex[:16];reservation={'status':'RESERVED_NO_AUTOMATIC_RETRY','remote_work':remote,'freeze_sha256':sha(release/'freeze.json'),'authorization_scope':'user requested completion of complete project using existing 3070; no new paid resource'};(private/'reservation.json').write_text(json.dumps(reservation,indent=2));(private/'reservation.json').chmod(0o600)
    raw=(ROOT/'.private/mla_acceptance_v1.json').read_bytes();assert hashlib.sha256(raw).hexdigest()==protocol['dataset_commitment_sha256'];files={'manifest.json':raw,'freeze.json':(release/'freeze.json').read_bytes(),'protocol.json':(release/'protocol.json').read_bytes(),wheel.name:wheel.read_bytes(),'host.py':(ROOT/'experiments/full_delivery/sealed_mla_host.py').read_bytes()}
    payload='FILES='+repr({n:base64.b64encode(v).decode() for n,v in files.items()})+'\nREMOTE='+repr(remote)+'\n'+r'''
import base64,hashlib,io,json,os,stat,subprocess,zipfile
from pathlib import Path
root=Path(REMOTE);root.mkdir(mode=0o700)
for n,v in FILES.items():
 p=root/n;p.write_bytes(base64.b64decode(v));p.chmod(0o600)
env={k:v for k,v in os.environ.items() if k in ['HOME','PATH','LANG','LD_LIBRARY_PATH']};env.update(OPENBLAS_NUM_THREADS='1')
r=subprocess.run(['/home/dministrator/cuda-lab/.venv/bin/python',str(root/'host.py'),'--root',str(root)],cwd=root,env=env,capture_output=True,text=True,timeout=3600);(root/'controller.stdout').write_text(r.stdout);(root/'controller.stderr').write_text(r.stderr);summary=json.loads((root/'summary.json').read_text()) if (root/'summary.json').exists() else {'status':'FAIL','error':'private host emitted no summary','production_release':False};manifest={};blobs={}
for p in root.rglob('*'):
 rel=p.relative_to(root)
 if 'prefix' in rel.parts or '__pycache__' in rel.parts:continue
 info=p.lstat()
 if not stat.S_ISREG(info.st_mode):continue
 if info.st_size>128*1024*1024:raise ValueError('private evidence file exceeds cap')
 value=p.read_bytes();h=hashlib.sha256(value).hexdigest();manifest[str(rel)]={'sha256':h,'bytes':len(value)};blobs.setdefault(h,value)
stream=io.BytesIO()
with zipfile.ZipFile(stream,'w',compression=zipfile.ZIP_DEFLATED) as z:
 z.writestr('execution.json',json.dumps({'remote_root':str(root),'exit':r.returncode,'manifest':manifest}))
 for h,v in blobs.items():z.writestr('blobs/'+h,v)
raw=stream.getvalue();print(json.dumps({'summary':summary,'archive_sha256':hashlib.sha256(raw).hexdigest(),'archive':base64.b64encode(raw).decode()}))
'''
    (private/'driver.py').write_text(payload);(private/'driver.py').chmod(0o600);r=execute_python(payload,4200);(private/'transport.stderr').write_text(r.stderr);(private/'transport.stderr').chmod(0o600)
    if r.returncode:
        (private/'transport.stdout').write_text(r.stdout);(private/'transport.stdout').chmod(0o600);raise RuntimeError('private transport failed; existing reservation retained, no automatic retry')
    wire=json.loads(r.stdout);raw=base64.b64decode(wire['archive']);assert hashlib.sha256(raw).hexdigest()==wire['archive_sha256'];(private/'archive.zip').write_bytes(raw);(private/'archive.zip').chmod(0o600)
    with zipfile.ZipFile(io.BytesIO(raw)) as z:
        execution=json.loads(z.read('execution.json'))
        for n,m in execution['manifest'].items():
            p=Path(n);assert not p.is_absolute() and '..' not in p.parts;v=z.read('blobs/'+m['sha256']);assert len(v)==m['bytes'] and hashlib.sha256(v).hexdigest()==m['sha256'];target=private/'raw'/p;target.parent.mkdir(mode=0o700,parents=True,exist_ok=True);target.write_bytes(v);target.chmod(0o600)
    public=ROOT/'evidence/20261123-mla-independent-acceptance';public.mkdir(exist_ok=False);summary={**wire['summary'],'private_raw_evidence_sha256':wire['archive_sha256'],'private_evidence_preserved':True};(public/'summary.json').write_text(json.dumps(summary,indent=2));reservation.update(status='COMPLETE',private_archive_sha256=wire['archive_sha256'],summary_sha256=sha(public/'summary.json'));(private/'reservation.json').write_text(json.dumps(reservation,indent=2));print(json.dumps(summary))


if __name__=='__main__':main()
