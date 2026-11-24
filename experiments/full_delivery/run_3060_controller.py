"""Trusted container parent for known public probes/precheck/qualification; worker socket-free."""
import argparse,json
from pathlib import Path
from tools.ssh_3060 import ROOT,execute_python

def main():
 p=argparse.ArgumentParser();p.add_argument('--phase',choices=['probe','probe-cdi','precheck','qualification'],required=True);a=p.parse_args();out=ROOT/'evidence/20261124-sm86-3060-counterpart';loc=json.loads((out/'location.json').read_text());phase=a.phase
 payload='LOCATION='+repr(loc)+'\nPHASE='+repr(phase)+'\n'+r'''
import hashlib,json,os,shutil,subprocess,time
from pathlib import Path
loc=LOCATION;root=Path(loc['root']);phase=PHASE;reservation=root/(phase+'-reservation.json')
with reservation.open('x') as f:json.dump({'phase':phase,'automatic_retry':False},f)
binding=json.loads((root/'runtime-binding.json').read_text());smi=subprocess.run(['nvidia-smi','--query-gpu=uuid,driver_version','--format=csv,noheader'],capture_output=True,text=True,timeout=15);assert smi.returncode==0 and [x.strip() for x in smi.stdout.strip().split(',')]==[binding['target_uuid'],binding['target_driver']]
if phase in ['probe','probe-cdi']:(root/(phase+'-host.json')).write_text(json.dumps({'pid':os.getpid()}))
name='nka-counterpart-parent-'+phase+'-'+root.name[-8:];cmd=['docker','run','--name',name,'--rm','--pull=never','--network=none','--read-only','--cap-drop=ALL','--security-opt=no-new-privileges','--init','--user',str(loc['uid'])+':'+str(loc['gid']),'--group-add',str(loc['docker_gid']),'--pids-limit=256','--memory=4g','--cpus=2','--runtime=runc','--device','nvidia.com/gpu=0','--env','NVIDIA_VISIBLE_DEVICES=void','--tmpfs','/tmp:rw,nosuid,nodev,size=512m','--env','HOME=/tmp','--env','PYTHONDONTWRITEBYTECODE=1','--env','OMP_NUM_THREADS=1','--env','OPENBLAS_NUM_THREADS=1','--env','MKL_NUM_THREADS=1','--env','PYTHONPATH='+loc['prefix'],'--env','NKA_CONTROL_CANARY=public-3060-probe','--mount','type=bind,src='+str(root)+',dst='+str(root),'--mount','type=bind,src='+loc['prefix']+',dst='+loc['prefix']+',readonly','--mount','type=bind,src=/var/run/docker.sock,dst=/var/run/docker.sock','--mount','type=bind,src='+shutil.which('docker')+',dst=/usr/bin/docker,readonly','--mount','type=bind,src=/tmp/nka-reviewed-gpu-execution.lock,dst=/tmp/nka-reviewed-gpu-execution.lock','--entrypoint','/opt/venv/bin/python',loc['image_id'],'-B']
cmd+=([str(root/'gpu_probe.py'),str(root)] if phase in ['probe','probe-cdi'] else [str(root/'host.py'),'--root',str(root),'--phase',phase]);(root/(phase+'-parent-command.json')).write_text(json.dumps(cmd,indent=2));started=time.monotonic();code=None
try:
 with (root/(phase+'-parent.stdout')).open('w') as o,(root/(phase+'-parent.stderr')).open('w') as e:r=subprocess.run(cmd,stdout=o,stderr=e,timeout=5400 if phase=='qualification' else 1000);code=r.returncode
finally:
 subprocess.run(['docker','rm','-f',name],capture_output=True,text=True,timeout=30)
 # Only workers carrying this exact public run ID; never other users/tasks.
 q=subprocess.run(['docker','ps','-aq','--filter','label=nka.counterpart='+root.name],capture_output=True,text=True,timeout=30);owned=q.stdout.split()
 for cid in owned:subprocess.run(['docker','rm','-f',cid],capture_output=True,text=True,timeout=30)
 q=subprocess.run(['docker','ps','-aq','--filter','label=nka.counterpart='+root.name],capture_output=True,text=True,timeout=30);assert q.returncode==0 and not q.stdout.strip()
 if phase in ['probe','probe-cdi']:
  q=subprocess.run(['docker','ps','-aq','--filter','name=nka-bounded-'],capture_output=True,text=True,timeout=30)
  for cid in q.stdout.split():
   details=subprocess.run(['docker','inspect',cid],capture_output=True,text=True,timeout=30)
   if details.returncode==0:
    mounts=json.loads(details.stdout)[0].get('Mounts',[])
    if any(x.get('Source')==str(root/('gpu-probe-cdi/artifact' if phase=='probe-cdi' else 'gpu-probe/artifact')) for x in mounts):subprocess.run(['docker','rm','-f',cid],capture_output=True,text=True,timeout=30)
filename='gpu-probe-result.json' if phase in ['probe','probe-cdi'] else phase+'-result.json';result=json.loads((root/filename).read_text()) if (root/filename).exists() else None;v={'phase':phase,'exit':code,'seconds':time.monotonic()-started,'result':result,'owned_workers_remaining':False,'parent_stderr_tail':(root/(phase+'-parent.stderr')).read_text()[-3000:]};(root/(phase+'-launch-result.json')).write_text(json.dumps(v,indent=2));print(json.dumps(v))
'''
 with (out/(phase+'-launch-driver.py')).open('x') as f:f.write(payload)
 r=execute_python(payload,5700 if phase=='qualification' else 1200);(out/(phase+'-transport.stderr')).write_text(r.stderr);(out/(phase+'-transport.stdout')).write_text(r.stdout);assert r.returncode==0;v=json.loads(r.stdout);(out/(phase+'-launch-result.json')).write_text(json.dumps(v,indent=2));result=v.get('result') or {};print(json.dumps({'phase':phase,'exit':v['exit'],'status':result.get('status'),'qa_checks':result.get('qa_checks'),'aa_sessions':result.get('aa_sessions'),'ab_sessions':result.get('ab_sessions'),'parent_stderr_tail':v['parent_stderr_tail']}))
if __name__=='__main__':main()
