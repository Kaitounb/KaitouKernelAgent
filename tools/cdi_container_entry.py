"""Host stdlib launcher; same reviewed trusted-parent mounts as the3060 acceptance controller."""
import argparse,fcntl,hashlib,json,os,shutil,subprocess,sys,time,uuid
from pathlib import Path
IMAGE='sha256:6ce2012265c68b8553adf8d7807dcf47972a9156d1c7f0a2a64db11f9827e55a'
def command(root,args,name):
 root=root.resolve();loc=json.loads((root/'location.json').read_text());assert loc['image_id']==IMAGE;prefix=(root/'current').resolve(strict=True);approved={Path(loc['prefix']).resolve()}
 if loc.get('previous_prefix'):approved.add(Path(loc['previous_prefix']).resolve())
 if prefix not in approved:raise ValueError('application pointer is outside accepted package prefixes')
 workspace=root/'workspace';workspace.mkdir(mode=0o700,exist_ok=True)
 if not args or args[0] not in ['init','status','activate','rollback','run']:raise ValueError('reviewed GEMM command required')
 if not any(x=='--channel' or x.startswith('--channel=') for x in args):args=[*args,'--channel','qualified']
 return ['docker','run','--name',name,'--rm','--pull=never','--network=none','--read-only','--cap-drop=ALL','--security-opt=no-new-privileges','--init','--user',str(loc['uid'])+':'+str(loc['gid']),'--group-add',str(loc['docker_gid']),'--memory=4g','--cpus=2','--pids-limit=256','--runtime=runc','--device','nvidia.com/gpu=0','--env','NVIDIA_VISIBLE_DEVICES=void','--env','HOME=/tmp','--env','PYTHONDONTWRITEBYTECODE=1','--env','PYTHONNOUSERSITE=1','--env','OMP_NUM_THREADS=1','--env','OPENBLAS_NUM_THREADS=1','--env','PYTHONPATH='+str(prefix),'--tmpfs','/tmp:rw,nosuid,nodev,size=512m','--workdir',str(workspace),'--mount','type=bind,src='+str(root)+',dst='+str(root),'--mount','type=bind,src='+str(prefix)+',dst='+str(prefix)+',readonly','--mount','type=bind,src=/var/run/docker.sock,dst=/var/run/docker.sock','--mount','type=bind,src='+shutil.which('docker')+',dst=/usr/bin/docker,readonly','--entrypoint','/opt/venv/bin/python',IMAGE,'-B','-m','kernel_agent.cdi_cli',*args]
def main():
 p=argparse.ArgumentParser();p.add_argument('--deployment-root',type=Path,required=True);p.add_argument('arguments',nargs=argparse.REMAINDER);a=p.parse_args();args=a.arguments[1:] if a.arguments[:1]==['--'] else a.arguments;root=a.deployment_root.resolve();name='nka-cdi-client-'+uuid.uuid4().hex[:12];lock=open('/tmp/nka-reviewed-gpu-execution.lock','a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
 trace=root/'launcher-records';trace.mkdir(mode=0o700,exist_ok=True);cmd=command(root,args,name);prefix=(root/'current').resolve();record={'command':cmd,'prefix':str(prefix),'package_init_sha256':hashlib.sha256((prefix/'kernel_agent/__init__.py').read_bytes()).hexdigest(),'started_wall':time.time(),'exit':None};path=trace/(name+'.json');path.write_text(json.dumps(record,indent=2))
 try:r=subprocess.run(cmd);code=r.returncode
 finally:
  record.update(exit=locals().get('code'),ended_wall=time.time());path.write_text(json.dumps(record,indent=2))
  subprocess.run(['docker','rm','-f',name],capture_output=True,text=True,timeout=30)
  q=subprocess.run(['docker','ps','-aq','--filter','name=nka-bounded-'],capture_output=True,text=True,timeout=30)
  for cid in q.stdout.split():
   item=subprocess.run(['docker','inspect',cid],capture_output=True,text=True,timeout=30)
   if item.returncode==0 and any(m.get('Source','').startswith(str(root)+'/workspace/') and m.get('Source','').endswith('/artifact') for m in json.loads(item.stdout)[0].get('Mounts',[])):subprocess.run(['docker','rm','-f',cid],capture_output=True,text=True,timeout=30)
 raise SystemExit(code)
if __name__=='__main__':
 try:main()
 except (OSError,ValueError,RuntimeError,AssertionError) as error:
  print(json.dumps({'status':'BUSY' if isinstance(error,BlockingIOError) else 'ERROR','error':str(error)[:512],'production_release':False}),file=sys.stderr);raise SystemExit(2 if isinstance(error,BlockingIOError) else 1)
