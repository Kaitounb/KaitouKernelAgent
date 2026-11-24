"""Stream only the existing public fixed Docker image, credential-free data, no Mac archive copy."""
import base64,hashlib,json,os,re,subprocess
from pathlib import Path
from tools.ssh_3070 import ROOT,execute_python as source_execute
from tools.ssh_server_gpu import execute_python as target_execute
IMAGE='sha256:ce00f6dc2eea43dc508b90b548f624646d643a92d363aa715fdb5a3b4f49290a'
CONNECTION=ROOT/'.private/server-gpu-3060-connection.json'
SEND=r'''
import base64,gzip,hashlib,json,subprocess,sys
from pathlib import Path
root=Path(__file__).parent;image='sha256:ce00f6dc2eea43dc508b90b548f624646d643a92d363aa715fdb5a3b4f49290a'
print(json.dumps({'format':'nka-public-docker-image-v1','image_id':image}),flush=True)
class Writer:
 def __init__(self):self.buffer=bytearray();self.digest=hashlib.sha256();self.offset=0
 def write(self,data):
  self.buffer.extend(data)
  while len(self.buffer)>=524288:self.emit(524288)
  return len(data)
 def emit(self,n):
  value=bytes(self.buffer[:n]);del self.buffer[:n];self.digest.update(value);print(json.dumps({'offset':self.offset,'data':base64.b64encode(value).decode()}),flush=True);self.offset+=n
 def finish(self):
  if self.buffer:self.emit(len(self.buffer))
writer=Writer()
with (root/'save.stderr').open('w') as e:
 p=subprocess.Popen(['docker','image','save',image],stdout=subprocess.PIPE,stderr=e)
 with gzip.GzipFile(fileobj=writer,mode='wb',compresslevel=1,mtime=0) as out:
  while True:
   value=p.stdout.read(1024*1024)
   if not value:break
   out.write(value)
 code=p.wait()
writer.finish();receipt={'complete':code==0,'bytes':writer.offset,'sha256':writer.digest.hexdigest(),'docker_save_exit':code};(root/'source-receipt.json').write_text(json.dumps(receipt,indent=2));print(json.dumps(receipt),flush=True)
if code:raise SystemExit(code)
'''
RECEIVE=r'''
import base64,hashlib,json,os,subprocess,sys,time
from pathlib import Path
root=Path(__file__).parent;image='sha256:ce00f6dc2eea43dc508b90b548f624646d643a92d363aa715fdb5a3b4f49290a';raw=sys.stdin.buffer;header=json.loads(raw.readline(2*1024*1024));assert header=={'format':'nka-public-docker-image-v1','image_id':image};digest=hashlib.sha256();offset=0;started=time.monotonic();last=-1;complete=None;part=root/'image.tar.gz.part'
with part.open('xb') as f:
 os.chmod(part,0o600)
 while True:
  line=raw.readline(2*1024*1024)
  if not line:raise EOFError('incomplete image transfer')
  value=json.loads(line)
  if 'complete' in value:complete=value;break
  assert set(value)=={'offset','data'} and value['offset']==offset
  data=base64.b64decode(value['data'],validate=True);assert 0<len(data)<=524288 and offset+len(data)<=30*1024**3;f.write(data);digest.update(data);offset+=len(data)
  if offset//(16*1024**2)!=last:
   last=offset//(16*1024**2);(root/'transfer-progress.json').write_text(json.dumps({'status':'RECEIVING','bytes':offset,'seconds':time.monotonic()-started}))
 f.flush();os.fsync(f.fileno())
assert complete['complete'] and complete['docker_save_exit']==0 and complete['bytes']==offset and complete['sha256']==digest.hexdigest();archive=root/'image.tar.gz';os.replace(part,archive);(root/'transfer-receipt.json').write_text(json.dumps(complete,indent=2))
with (root/'load.stdout').open('w') as o,(root/'load.stderr').open('w') as e:r=subprocess.run(['docker','image','load','--input',str(archive)],stdout=o,stderr=e,timeout=1200)
assert r.returncode==0,'docker load failed';r=subprocess.run(['docker','image','inspect',image,'--format','{{.Id}}'],capture_output=True,text=True,timeout=30);assert r.returncode==0 and r.stdout.strip()==image
v={'status':'LOADED_EXACT_IMAGE','image_id':image,'archive_bytes':offset,'archive_sha256':digest.hexdigest(),'seconds':time.monotonic()-started,'root':str(root),'new_gpu_runs':0,'new_model_calls':0};(root/'result.json').write_text(json.dumps(v,indent=2));print(json.dumps(v),flush=True)
'''
def stage(execute,source,prefix,*,connection=None):
 payload='SOURCE='+repr(base64.b64encode(source.encode()).decode())+'\nPREFIX='+repr(prefix)+'\n'+r'''
import base64,json,tempfile
from pathlib import Path
if PREFIX=='source':root=Path(tempfile.mkdtemp(prefix='nka-public-image-send-'))
else:
 parent=Path.home()/'nka-persistent';parent.mkdir(exist_ok=True);root=Path(tempfile.mkdtemp(prefix='image-import-',dir=parent))
root.chmod(0o700);(root/'stream.py').write_bytes(base64.b64decode(SOURCE));print(json.dumps({'root':str(root)}))
'''
 r=execute(payload,60,**({'connection_path':connection} if connection else {}));assert r.returncode==0,r.stderr;return json.loads(r.stdout)['root']
def main():
 out=ROOT/'evidence/20261124-sm86-3060-image-transfer';out.mkdir(exist_ok=False);(out/'send.py').write_text(SEND);(out/'receive.py').write_text(RECEIVE)
 source_root=stage(source_execute,SEND,'source');target_root=stage(target_execute,RECEIVE,'target',connection=CONNECTION);location={'source_root':source_root,'target_root':target_root,'image_id':IMAGE,'scope':'only immutable Docker image; no project private files, manifests or credentials transferred'};(out/'location.json').write_text(json.dumps(location,indent=2))
 lines=(ROOT/'.private/3070-connection.txt').read_text().splitlines();win=next(x for x in lines if 'Windows' in x and '密码' in x);user=re.search(r'账号\s*([^，,\s]+)',win).group(1);password=re.search(r'密码[：:]\s*(\S+)',win).group(1);target=json.loads(CONNECTION.read_text());base={k:v for k,v in os.environ.items() if 'API_KEY' not in k.upper() and 'AUTH_TOKEN' not in k.upper()};env_source={**base,'SSHPASS':password};env_target={**base,'SSHPASS':target['password']};sshpass='/opt/homebrew/bin/sshpass'
 a=[sshpass,'-e','ssh','-T','-p','21022','-o','StrictHostKeyChecking=yes','-o','ServerAliveInterval=20','-o','ServerAliveCountMax=3',user+'@127.0.0.1','wsl -d Ubuntu -u root -- /home/dministrator/cuda-lab/.venv/bin/python -u '+source_root+'/stream.py']
 b=[sshpass,'-e','ssh','-T','-p',str(target['port']),'-o','StrictHostKeyChecking=yes','-o','ServerAliveInterval=20','-o','ServerAliveCountMax=3',target['user']+'@'+target['host'],'python3 -u '+target_root+'/stream.py']
 with (out/'source-transport.stderr').open('w') as ea,(out/'target-transport.stderr').open('w') as eb,(out/'target-transport.stdout').open('w') as ob:
  src=subprocess.Popen(a,env=env_source,stdout=subprocess.PIPE,stderr=ea)
  try:
   dst=subprocess.Popen(b,env=env_target,stdin=src.stdout,stdout=ob,stderr=eb);src.stdout.close();dst_code=dst.wait(timeout=7200);src_code=src.wait(timeout=60)
  except BaseException:
   if src.poll() is None:src.terminate()
   raise
 receipt={'source_exit':src_code,'target_exit':dst_code,'image_id':IMAGE};(out/'transport-result.json').write_text(json.dumps(receipt,indent=2));assert src_code==dst_code==0;v=json.loads((out/'target-transport.stdout').read_text());(out/'result.json').write_text(json.dumps(v,indent=2));print(json.dumps(v))
if __name__=='__main__':main()
