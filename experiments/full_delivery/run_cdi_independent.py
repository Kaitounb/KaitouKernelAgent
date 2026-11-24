"""Single new3060 acceptance list/16calls; no old private data or resampling."""
import base64,hashlib,io,json,uuid,zipfile
from pathlib import Path
from tools.ssh_3060 import ROOT,execute_python

def main():
 installed=ROOT/'evidence/20261124-3060-0.1.3rc2-installed';verified=json.loads((installed/'verification.json').read_text());assert verified['status']=='AUDIT_PASS_PREVIEW';loc=json.loads((installed/'location.json').read_text());public=ROOT/'evidence/20261124-3060-next-independent';public.mkdir(exist_ok=False);private=ROOT/'.private/3060-rc2-independent-evidence';private.mkdir(mode=0o700,exist_ok=False);remote='/home/featurize/nka-persistent/gemm-independent-'+uuid.uuid4().hex[:12];source=(ROOT/'experiments/full_delivery/cdi_independent_host.py').read_bytes();location={'remote_root':remote,'install_root':loc['root'],'new_server_local_synthetic_manifest':True,'old_private_manifest_read':False,'automatic_retry':False};(private/'location.json').write_text(json.dumps(location,indent=2));(private/'location.json').chmod(0o600)
 common='REMOTE='+repr(remote)+'\nLOCATION='+repr(loc)+'\n'+r'''
import base64,hashlib,io,json,os,shutil,stat,subprocess,time,zipfile
from pathlib import Path
root=Path(REMOTE);loc=LOCATION;installed=Path(loc['root'])
def command(phase):
 name='nka-cdi-independent-'+phase+'-'+root.name[-8:];cmd=['docker','run','--name',name,'--rm','--pull=never','--network=none','--read-only','--cap-drop=ALL','--security-opt=no-new-privileges','--init','--user',str(loc['uid'])+':'+str(loc['gid']),'--group-add',str(loc['docker_gid']),'--memory=4g','--cpus=2','--pids-limit=256','--runtime=runc','--env','NVIDIA_VISIBLE_DEVICES=void','--env','HOME=/tmp','--env','PYTHONDONTWRITEBYTECODE=1','--env','OMP_NUM_THREADS=1','--env','OPENBLAS_NUM_THREADS=1','--tmpfs','/tmp:rw,nosuid,nodev,size=512m','--mount','type=bind,src='+str(root)+',dst='+str(root),'--mount','type=bind,src='+str(installed)+',dst='+str(installed)+',readonly']
 if phase=='run':cmd+=['--device','nvidia.com/gpu=0','--mount','type=bind,src=/var/run/docker.sock,dst=/var/run/docker.sock','--mount','type=bind,src='+shutil.which('docker')+',dst=/usr/bin/docker,readonly','--mount','type=bind,src=/tmp/nka-reviewed-gpu-execution.lock,dst=/tmp/nka-reviewed-gpu-execution.lock']
 return cmd+['--entrypoint','/opt/venv/bin/python',loc['image_id'],'-B',str(root/'host.py'),'--root',str(root),'--install-root',str(installed),'--phase',phase]
'''
 payload=common+'\nHOST='+repr(base64.b64encode(source).decode())+'\n'+r'''
root.mkdir(mode=0o700,exist_ok=False);(root/'host.py').write_bytes(base64.b64decode(HOST));cmd=command('prepare');(root/'prepare-command.json').write_text(json.dumps(cmd,indent=2));r=subprocess.run(cmd,capture_output=True,text=True,timeout=90);(root/'prepare.stdout').write_text(r.stdout);(root/'prepare.stderr').write_text(r.stderr);assert r.returncode==0,'prepare failed; inspect private root without selecting another list';print(json.dumps({'freeze':json.loads(r.stdout),'protocol':json.loads((root/'protocol.json').read_text())}))
'''
 (public/'prepare-driver.py').write_text(payload);r=execute_python(payload,120);(public/'prepare.stderr').write_text(r.stderr);assert r.returncode==0;freeze=json.loads(r.stdout);(public/'freeze.json').write_text(json.dumps(freeze,indent=2));print(json.dumps(freeze['freeze']),flush=True)
 payload=common+r'''
cmd=command('run');(root/'run-command.json').write_text(json.dumps(cmd,indent=2));r=subprocess.run(cmd,capture_output=True,text=True,timeout=600);(root/'controller.stdout').write_text(r.stdout);(root/'controller.stderr').write_text(r.stderr);summary=json.loads((root/'summary.json').read_text()) if (root/'summary.json').exists() else {'status':'FAIL','production_release':False};metadata={};blobs={}
for p in root.rglob('*'):
 if '__pycache__' in p.parts or not stat.S_ISREG(p.lstat().st_mode):continue
 assert p.stat().st_size<=128*1024*1024;data=p.read_bytes();h=hashlib.sha256(data).hexdigest();metadata[str(p.relative_to(root))]={'sha256':h,'bytes':len(data)};blobs.setdefault(h,data)
stream=io.BytesIO()
with zipfile.ZipFile(stream,'w',compression=zipfile.ZIP_DEFLATED) as z:
 z.writestr('execution.json',json.dumps({'remote_root':REMOTE,'manifest':metadata,'exit':r.returncode}))
 for h,data in blobs.items():z.writestr('blobs/'+h,data)
data=stream.getvalue();print(json.dumps({'summary':summary,'archive_sha256':hashlib.sha256(data).hexdigest(),'archive':base64.b64encode(data).decode()}))
'''
 (private/'driver.py').write_text(payload);(private/'driver.py').chmod(0o600);r=execute_python(payload,900);(private/'transport.stderr').write_text(r.stderr);(private/'transport.stderr').chmod(0o600);assert r.returncode==0;wire=json.loads(r.stdout);data=base64.b64decode(wire['archive']);assert hashlib.sha256(data).hexdigest()==wire['archive_sha256'];(private/'archive.zip').write_bytes(data);(private/'archive.zip').chmod(0o600)
 with zipfile.ZipFile(io.BytesIO(data)) as z:
  execution=json.loads(z.read('execution.json'))
  for n,m in execution['manifest'].items():
   name=Path(n);assert not name.is_absolute() and '..' not in name.parts;target=private/'remote'/name;target.parent.mkdir(mode=0o700,parents=True,exist_ok=True);raw=z.read('blobs/'+m['sha256']);assert len(raw)==m['bytes'] and hashlib.sha256(raw).hexdigest()==m['sha256'];target.write_bytes(raw);target.chmod(0o600)
 (private/'execution.json').write_text(json.dumps(execution,indent=2));(private/'execution.json').chmod(0o600);v={**wire['summary'],'private_archive_sha256':wire['archive_sha256'],'private_files_preserved':len(execution['manifest'])};(public/'summary.json').write_text(json.dumps(v,indent=2));print(json.dumps(v))
if __name__=='__main__':main()
