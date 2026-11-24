"""Explicitly authorized one-shot private transport; exact workloads never enter tool output."""
import base64,hashlib,io,json,uuid,zipfile
from pathlib import Path
from tools.ssh_3070 import ROOT,execute_python
from tools.verify_gemm_release_freeze import verify


def main():
    folder=ROOT/'releases/gemm-sm86-rc1';verified=verify(folder);seal=json.loads((folder/'protocol.freeze.json').read_text());sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
    assert sha(folder/'freeze.json')==seal['candidate_freeze_sha256'] and sha(folder/'protocol.json')==seal['protocol_sha256']
    for path,digest in seal['trusted_evaluator_sha256'].items():assert sha(ROOT/path)==digest
    private=ROOT/'.private/gemm-sm86-rc1-evidence';private.mkdir(mode=0o700,exist_ok=False);reservation=private/'reservation.json';remote='/tmp/nka-sealed-gemm-'+uuid.uuid4().hex[:16]
    reservation.write_text(json.dumps({'status':'RESERVED_NO_AUTOMATIC_RETRY','remote_work':remote,'freeze_id':verified['freeze_id'],'explicit_user_transfer_execution_approval':True},indent=2)+'\n');reservation.chmod(0o600)
    wheel=ROOT/'evidence/20261122-gemm-rel02-dual-installed-fixed-test-import/remote/run/dist/nvidia_kernel_agent-0.1.0.dev5-py3-none-any.whl';manifest=(ROOT/'.private/gemm_acceptance_v1.json').read_bytes();protocol=json.loads((folder/'protocol.json').read_text());assert hashlib.sha256(manifest).hexdigest()==protocol['dataset_commitment_sha256']
    files={'manifest.json':manifest,'freeze.json':(folder/'freeze.json').read_bytes(),'protocol.json':(folder/'protocol.json').read_bytes(),wheel.name:wheel.read_bytes(),'sealed_worker.py':(ROOT/'experiments/full_delivery/sealed_worker.py').read_bytes(),'host.py':(ROOT/'experiments/full_delivery/sealed_gemm_host.py').read_bytes()}
    payload='FILES='+repr({p:base64.b64encode(v).decode() for p,v in files.items()})+'\nREMOTE='+repr(remote)+'\n'+r'''
import base64,hashlib,io,json,os,stat,subprocess,zipfile
from pathlib import Path
root=Path(REMOTE);root.mkdir(mode=0o700)
for name,data in FILES.items():
 p=root/name;p.write_bytes(base64.b64decode(data));p.chmod(0o600)
env={k:v for k,v in os.environ.items() if k in ['HOME','PATH','LANG','LD_LIBRARY_PATH']};env.update(OPENBLAS_NUM_THREADS='1')
r=subprocess.run(['/home/dministrator/cuda-lab/.venv/bin/python',str(root/'host.py'),'--root',str(root)],env=env,cwd=root,capture_output=True,text=True,timeout=1800)
(root/'controller.stdout').write_text(r.stdout);(root/'controller.stderr').write_text(r.stderr)
summary=json.loads((root/'summary.json').read_text()) if (root/'summary.json').exists() else {'status':'FAIL','error':'private evaluator did not emit summary','production_release':False}
metadata={};blobs={}
for p in root.rglob('*'):
 rel=p.relative_to(root)
 if 'prefix' in rel.parts or '__pycache__' in rel.parts:continue
 info=p.lstat()
 if not stat.S_ISREG(info.st_mode):continue
 if info.st_size>128*1024*1024:raise ValueError('private evidence exceeds per-file cap')
 value=p.read_bytes();digest=hashlib.sha256(value).hexdigest();metadata[str(rel)]={'sha256':digest,'bytes':len(value)};blobs.setdefault(digest,value)
raw=io.BytesIO()
with zipfile.ZipFile(raw,'w',compression=zipfile.ZIP_DEFLATED) as archive:
 archive.writestr('execution.json',json.dumps({'remote_work':str(root),'exit':r.returncode,'manifest':metadata}))
 for digest,value in blobs.items():archive.writestr('blobs/'+digest,value)
encoded=raw.getvalue();print(json.dumps({'summary':summary,'private_archive_sha256':hashlib.sha256(encoded).hexdigest(),'private_archive':base64.b64encode(encoded).decode()}))
'''
    (private/'remote-driver.py').write_text(payload);(private/'remote-driver.py').chmod(0o600);r=execute_python(payload,timeout=2400);(private/'transport.stderr').write_text(r.stderr)
    if r.returncode:
        (private/'transport.stdout').write_text(r.stdout);raise RuntimeError('private evaluator transport failed; reservation retained; no retry')
    wire=json.loads(r.stdout);data=base64.b64decode(wire['private_archive']);assert hashlib.sha256(data).hexdigest()==wire['private_archive_sha256'];(private/'archive.zip').write_bytes(data);(private/'archive.zip').chmod(0o600)
    with zipfile.ZipFile(io.BytesIO(data)) as archive:
        execution=json.loads(archive.read('execution.json'))
        for rel,meta in execution['manifest'].items():
            path=Path(rel);assert not path.is_absolute() and '..' not in path.parts;value=archive.read('blobs/'+meta['sha256']);assert hashlib.sha256(value).hexdigest()==meta['sha256'] and len(value)==meta['bytes']
            out=private/'raw'/path;out.parent.mkdir(parents=True,exist_ok=True,mode=0o700);out.write_bytes(value);out.chmod(0o600)
    summary=wire['summary'];summary.update(private_raw_evidence_sha256=wire['private_archive_sha256'],private_evidence_preserved=True)
    public=ROOT/'evidence/20261122-gemm-independent-acceptance';public.mkdir(exist_ok=False);(public/'summary.json').write_text(json.dumps(summary,indent=2)+'\n');reservation.write_text(json.dumps({'status':'COMPLETE','remote_work':remote,'summary_sha256':sha(public/'summary.json'),'private_archive_sha256':wire['private_archive_sha256']},indent=2)+'\n');print(json.dumps(summary))


if __name__=='__main__':main()
