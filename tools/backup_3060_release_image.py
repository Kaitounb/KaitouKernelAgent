"""Stream the accepted public runtime image to a local immutable recovery archive, no credentials in argv."""
import gzip,hashlib,json,os,shutil,subprocess,tarfile,threading,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
def main():
 accepted=json.loads((ROOT/'evidence/20261124-3060-0.1.3-installed/verification.json').read_text());assert accepted['status']=='AUDIT_PASS_QUALIFIED_INSTALL';build=json.loads((ROOT/'evidence/20261124-3060-next-runtime-build/result.json').read_text());image=build['image_id'];assert shutil.disk_usage(ROOT).free>2*build['size_bytes']+1024**3,'insufficient recovery archive diskspace';out=ROOT/'evidence/20261124-3060-runtime-image-backup';out.mkdir(exist_ok=False);folder=ROOT/'artifacts/runtime-images';folder.mkdir(exist_ok=True);target=folder/('rtx3060-'+image.split(':')[1][:16]+'.docker.tar.gz');assert not target.exists();partial=target.with_suffix(target.suffix+'.partial');assert not partial.exists()
 payload='IMAGE='+repr(image)+'\nEXPORT_ROOT='+repr(build['root'])+'\n'+r'''
import gzip,hashlib,json,subprocess,sys,tempfile
from pathlib import Path
archive_dir=Path(EXPORT_ROOT)/'recovery-export';archive_dir.mkdir(exist_ok=False);archive=archive_dir/'runtime.docker.tar.gz';copy=archive.open('xb')
r=subprocess.run(['nvidia-smi','--query-gpu=utilization.gpu','--format=csv,noheader,nounits'],capture_output=True,text=True,timeout=15);assert r.returncode==0 and int(r.stdout.strip())==0
class Writer:
 def __init__(self):self.hash=hashlib.sha256();self.count=0
 def write(self,data):
  n=sys.stdout.buffer.write(data);assert n==len(data);copy.write(data);self.hash.update(data);self.count+=n;return n
 def flush(self):sys.stdout.buffer.flush()
writer=Writer()
with tempfile.TemporaryFile() as err:
 p=subprocess.Popen(['docker','image','save',IMAGE],stdout=subprocess.PIPE,stderr=err)
 with gzip.GzipFile(fileobj=writer,mode='wb',compresslevel=1,mtime=0,filename='') as zipped:
  while True:
   data=p.stdout.read(1024*1024)
   if not data:break
   zipped.write(data)
 p.stdout.close();code=p.wait();assert code==0
writer.flush();copy.close();loaded=subprocess.run(['docker','image','load','--input',str(archive)],capture_output=True,text=True,timeout=1200);(archive_dir/'load.stdout').write_text(loaded.stdout);(archive_dir/'load.stderr').write_text(loaded.stderr);assert loaded.returncode==0;checked=subprocess.run(['docker','image','inspect',IMAGE,'--format','{{.Id}}'],capture_output=True,text=True,timeout=30);assert checked.returncode==0 and checked.stdout.strip()==IMAGE
sys.stderr.write('NKA_EXPORT_RESULT '+json.dumps({'image_id':IMAGE,'bytes':writer.count,'sha256':writer.hash.hexdigest(),'docker_save_exit':code,'docker_load_exit':loaded.returncode,'loaded_image_lookup':checked.stdout.strip(),'remote_archive':str(archive)})+'\n')
'''
 (out/'driver.py').write_text(payload);connection=json.loads((ROOT/'.private/server-gpu-3060-next-connection.json').read_text());env={k:v for k,v in os.environ.items() if 'API_KEY' not in k.upper() and 'AUTH_TOKEN' not in k.upper()};env['SSHPASS']=connection['password'];cmd=['/opt/homebrew/bin/sshpass','-e','ssh','-T','-p',str(connection['port']),'-o','StrictHostKeyChecking=yes','-o','ConnectTimeout=15','-o','ServerAliveInterval=20','-o','ServerAliveCountMax=3','-o','ControlMaster=auto','-o','ControlPersist=600','-o','ControlPath='+str(ROOT/'.private/ssh-3060-next.sock'),connection['user']+'@'+connection['host'],'python3 -'];started=time.monotonic();p=subprocess.Popen(cmd,stdin=subprocess.PIPE,stdout=subprocess.PIPE,stderr=subprocess.PIPE,env=env);p.stdin.write(payload.encode());p.stdin.close();stderr=[]
 def drain():
  while True:
   data=p.stderr.read(8192)
   if not data:break
   stderr.append(data)
 thread=threading.Thread(target=drain);thread.start();h=hashlib.sha256();count=0;last=started
 with partial.open('xb') as f:
  while True:
   data=p.stdout.read(1024*1024)
   if not data:break
   f.write(data);h.update(data);count+=len(data)
   if time.monotonic()-last>=10:
    (out/'progress.json').write_text(json.dumps({'bytes':count,'seconds':time.monotonic()-started}));last=time.monotonic()
 code=p.wait();thread.join();error=b''.join(stderr);(out/'transport.stderr').write_bytes(error);assert code==0;line=next(x for x in error.decode().splitlines() if x.startswith('NKA_EXPORT_RESULT '));remote=json.loads(line.split(' ',1)[1]);assert remote['sha256']==h.hexdigest() and remote['bytes']==count and remote['image_id']==image
 from tools.verify_docker_image_archive import verify
 integrity=verify(partial,image)
 partial.rename(target);target.chmod(0o444);v={'status':'IMAGE_BACKUP_SHA_CONFIG_AND_ALL_LAYER_DIFFIDS_PASS','image_id':image,'archive':str(target),'archive_sha256':h.hexdigest(),'bytes':count,'tar_members_verified':integrity['tar_members_verified'],'layers_verified':integrity['layers_verified'],'image_graph_verification':integrity,'remote_restore_test':remote,'seconds':time.monotonic()-started,'gpu_execution':False,'new_model_calls':0};(out/'result.json').write_text(json.dumps(v,indent=2));print(json.dumps(v))
if __name__=='__main__':main()
