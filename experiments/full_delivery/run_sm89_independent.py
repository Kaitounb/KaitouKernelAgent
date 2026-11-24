"""One new server-local synthetic list after wheel freeze; no old private-data transfer."""
import base64,hashlib,io,json,uuid,zipfile
from tools.ssh_server_gpu import ROOT,execute_python


def main():
    public=ROOT/'evidence/20261123-sm89-independent';public.mkdir(exist_ok=False);private=ROOT/'.private/sm89-rc2-independent-evidence';private.mkdir(mode=0o700,exist_ok=False)
    result=json.loads((ROOT/'evidence/20261123-sm89-runtime-installed/verification.json').read_text());assert result['status']=='AUDIT_PASS'
    remote='/data/nka-sm89-independent-'+uuid.uuid4().hex[:12];install=json.loads((ROOT/'evidence/20261123-sm89-runtime-source/location.json').read_text())['remote_root'];host=ROOT/'experiments/full_delivery/sm89_independent_host.py'
    location={'remote_root':remote,'install_root':install,'new_server_local_synthetic_manifest':True,'old_private_manifest_read':False,'automatic_retry':False};(private/'location.json').write_text(json.dumps(location,indent=2));(private/'location.json').chmod(0o600)
    payload='REMOTE='+repr(remote)+'\nINSTALL='+repr(install)+'\nHOST='+repr(base64.b64encode(host.read_bytes()).decode())+'\n'+r'''
import base64,json,os,subprocess
from pathlib import Path
remote=Path(REMOTE);script=Path(INSTALL)/('independent-host-'+remote.name+'.py');script.write_bytes(base64.b64decode(HOST));env={'PATH':'/usr/local/cuda-12.6/bin:/usr/local/bin:/usr/bin:/bin','HOME':'/root','OMP_NUM_THREADS':'1','OPENBLAS_NUM_THREADS':'1'}
r=subprocess.run(['/tmp/nka-full-server-dqnrbkzs/venv/bin/python',str(script),'--root',str(remote),'--install-root',INSTALL,'--phase','prepare'],capture_output=True,text=True,env=env,timeout=60)
if r.returncode:raise RuntimeError('Fresh manifest preparation failed; inspect reserved path without printing values')
print(json.dumps({'freeze':json.loads(r.stdout),'protocol':json.loads((remote/'protocol.json').read_text()),'host_path':str(script)}))
'''
    (public/'prepare-driver.py').write_text(payload);r=execute_python(payload,120,connection_path=ROOT/'.private/server-gpu-next-connection.json');(public/'prepare.stderr').write_text(r.stderr)
    if r.returncode:raise RuntimeError('Inspect existing new manifest reservation; never reselect dataset')
    frozen=json.loads(r.stdout);(public/'freeze.json').write_text(json.dumps(frozen,indent=2));print(json.dumps(frozen['freeze']),flush=True)
    payload='REMOTE='+repr(remote)+'\nINSTALL='+repr(install)+'\nHOST_PATH='+repr(frozen['host_path'])+'\n'+r'''
import base64,hashlib,io,json,os,stat,subprocess,zipfile
from pathlib import Path
root=Path(REMOTE);env={'PATH':'/usr/local/cuda-12.6/bin:/usr/local/bin:/usr/bin:/bin','HOME':'/root','OMP_NUM_THREADS':'1','OPENBLAS_NUM_THREADS':'1'}
r=subprocess.run(['/tmp/nka-full-server-dqnrbkzs/venv/bin/python',HOST_PATH,'--root',REMOTE,'--install-root',INSTALL,'--phase','run'],capture_output=True,text=True,env=env,timeout=600);(root/'controller.stdout').write_text(r.stdout);(root/'controller.stderr').write_text(r.stderr)
summary=json.loads((root/'summary.json').read_text()) if (root/'summary.json').exists() else {'status':'FAIL','production_release':False};metadata={};blobs={}
for p in root.rglob('*'):
 if '__pycache__' in p.parts or not stat.S_ISREG(p.lstat().st_mode):continue
 if p.stat().st_size>128*1024*1024:raise ValueError('private per-file cap')
 data=p.read_bytes();h=hashlib.sha256(data).hexdigest();metadata[str(p.relative_to(root))]={'sha256':h,'bytes':len(data)};blobs.setdefault(h,data)
stream=io.BytesIO()
with zipfile.ZipFile(stream,'w',compression=zipfile.ZIP_DEFLATED) as z:
 z.writestr('execution.json',json.dumps({'remote_root':REMOTE,'manifest':metadata,'exit':r.returncode}))
 for h,data in blobs.items():z.writestr('blobs/'+h,data)
data=stream.getvalue();print(json.dumps({'summary':summary,'archive_sha256':hashlib.sha256(data).hexdigest(),'archive':base64.b64encode(data).decode()}))
'''
    (private/'driver.py').write_text(payload);(private/'driver.py').chmod(0o600);r=execute_python(payload,900,connection_path=ROOT/'.private/server-gpu-next-connection.json');(private/'transport.stderr').write_text(r.stderr);(private/'transport.stderr').chmod(0o600)
    if r.returncode:raise RuntimeError('Inspect existing independent execution; no retry')
    wire=json.loads(r.stdout);data=base64.b64decode(wire['archive']);assert hashlib.sha256(data).hexdigest()==wire['archive_sha256'];archive=private/'archive.zip';archive.write_bytes(data);archive.chmod(0o600)
    with zipfile.ZipFile(io.BytesIO(data)) as z:
        execution=json.loads(z.read('execution.json'))
        for n,m in execution['manifest'].items():
            from pathlib import Path
            p=Path(n);assert not p.is_absolute() and '..' not in p.parts;target=private/'remote'/p;target.parent.mkdir(mode=0o700,parents=True,exist_ok=True);raw=z.read('blobs/'+m['sha256']);assert len(raw)==m['bytes'] and hashlib.sha256(raw).hexdigest()==m['sha256'];target.write_bytes(raw);target.chmod(0o600)
    (private/'execution.json').write_text(json.dumps(execution,indent=2));(private/'execution.json').chmod(0o600);summary={**wire['summary'],'private_archive_sha256':wire['archive_sha256'],'private_files_preserved':len(execution['manifest'])};(public/'summary.json').write_text(json.dumps(summary,indent=2));print(json.dumps(summary))


if __name__=='__main__':main()
