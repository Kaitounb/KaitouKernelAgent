"""Install one immutable3060 wheel and run its bounded real validation exactly once."""
import argparse,base64,hashlib,json
from tools.ssh_3060 import ROOT,execute_python

def main():
 p=argparse.ArgumentParser();p.add_argument('--phase',choices=['preview','qualified'],required=True);a=p.parse_args();version='0.1.3rc2' if a.phase=='preview' else '0.1.3';out=ROOT/('evidence/20261124-3060-'+version+'-installed');out.mkdir(exist_ok=False);build=json.loads((ROOT/('evidence/20261124-3060-build-'+version+'/result.json')).read_text());wheel=ROOT/'artifacts/wheels'/('nvidia_kernel_agent-'+version+'-py3-none-any.whl');assert hashlib.sha256(wheel.read_bytes()).hexdigest()==build['wheel_sha256'];prior=json.loads((ROOT/'evidence/20261124-3060-0.1.3rc2-installed/location.json').read_text())['prefix'] if a.phase=='qualified' else None
 files={'dist/'+wheel.name:wheel.read_bytes(),'source/experiments/full_delivery/cdi_release_validate.py':(ROOT/'experiments/full_delivery/cdi_release_validate.py').read_bytes()}
 for path in (ROOT/'kernel_agent').rglob('*'):
  if path.is_file() and '__pycache__' not in path.parts and path.name!='.DS_Store':files['source/'+str(path.relative_to(ROOT))]=path.read_bytes()
 for path in (ROOT/'tests').glob('*.py'):files['source/tests/'+path.name]=path.read_bytes()
 files['source/workloads/gemm_deployment_stability_v2.json']=(ROOT/'workloads/gemm_deployment_stability_v2.json').read_bytes();files['source/experiments/full_delivery/gemm_deployment_gate.py']=(ROOT/'experiments/full_delivery/gemm_deployment_gate.py').read_bytes();(out/'source-hashes.json').write_text(json.dumps({n:hashlib.sha256(b).hexdigest() for n,b in files.items()},indent=2))
 payload='FILES='+repr({n:base64.b64encode(b).decode() for n,b in files.items()})+'\nVERSION='+repr(version)+'\nPHASE='+repr(a.phase)+'\nPREVIOUS='+repr(prior)+'\nWHEEL_SHA='+repr(build['wheel_sha256'])+'\n'+r'''
import base64,grp,hashlib,json,os,shutil,subprocess,tempfile,time
from pathlib import Path
root=Path(tempfile.mkdtemp(prefix='gemm-release-'+VERSION+'-',dir=Path.home()/'nka-persistent'));root.chmod(0o700);image='sha256:6ce2012265c68b8553adf8d7807dcf47972a9156d1c7f0a2a64db11f9827e55a';uid=os.getuid();gid=os.getgid();docker_gid=grp.getgrnam('docker').gr_gid
for n,b in FILES.items():p=root/n;p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes(base64.b64decode(b))
wheel=root/'dist'/('nvidia_kernel_agent-'+VERSION+'-py3-none-any.whl');assert hashlib.sha256(wheel.read_bytes()).hexdigest()==WHEEL_SHA
loc={'root':str(root),'remote_root':str(root),'prefix':str(root/'prefix'),'image_id':image,'uid':uid,'gid':gid,'docker_gid':docker_gid,'version':VERSION,'wheel_sha256':WHEEL_SHA,'previous_prefix':PREVIOUS};(root/'location.json').write_text(json.dumps(loc,indent=2));records=[]
def base(name,gpu=False):
 cmd=['docker','run','--name',name,'--rm','--pull=never','--network=none','--read-only','--cap-drop=ALL','--security-opt=no-new-privileges','--init','--user',str(uid)+':'+str(gid),'--group-add',str(docker_gid),'--memory=4g','--cpus=2','--pids-limit=256','--runtime=runc','--env','NVIDIA_VISIBLE_DEVICES=void','--env','HOME=/tmp','--env','PYTHONDONTWRITEBYTECODE=1','--env','OMP_NUM_THREADS=1','--env','OPENBLAS_NUM_THREADS=1','--tmpfs','/tmp:rw,nosuid,nodev,size=512m','--mount','type=bind,src='+str(root)+',dst='+str(root)]
 if gpu:
  cmd+=['--device','nvidia.com/gpu=0','--mount','type=bind,src=/var/run/docker.sock,dst=/var/run/docker.sock','--mount','type=bind,src='+shutil.which('docker')+',dst=/usr/bin/docker,readonly','--mount','type=bind,src=/tmp/nka-reviewed-gpu-execution.lock,dst=/tmp/nka-reviewed-gpu-execution.lock','--env','PYTHONPATH='+str(root/'prefix')]
  if PREVIOUS:cmd+=['--mount','type=bind,src='+PREVIOUS+',dst='+PREVIOUS+',readonly']
 return cmd+['--entrypoint','/opt/venv/bin/python',image,'-B']
def run(label,cmd,timeout):
 with (root/(label+'-started.json')).open('x') as f:json.dump({'once':True},f)
 start=time.monotonic()
 with (root/(label+'.stdout')).open('w') as o,(root/(label+'.stderr')).open('w') as e:r=subprocess.run(cmd,stdout=o,stderr=e,timeout=timeout)
 records.append({'label':label,'command':cmd,'exit':r.returncode,'seconds':time.monotonic()-start});(root/'progress.json').write_text(json.dumps(records,indent=2));return r.returncode
code=run('install',base('nka-cdi-install-'+root.name[-8:])+['-m','pip','install','--no-index','--no-deps','--target',str(root/'prefix'),str(wheel)],180)
name='nka-cdi-validate-'+root.name[-8:]
try:
 if code==0:
  samples=[]
  for _ in range(3):
   r=subprocess.run(['nvidia-smi','--query-gpu=uuid,driver_version,utilization.gpu,power.limit','--format=csv,noheader,nounits'],capture_output=True,text=True,timeout=15);samples.append({'exit':r.returncode,'stdout':r.stdout,'stderr':r.stderr});f=[x.strip() for x in r.stdout.strip().split(',')];assert r.returncode==0 and f[:2]==['GPU-0140b895-e4e9-7c67-aba1-c7a953df1847','580.95.05'] and int(f[2])==0 and float(f[3])==170;time.sleep(1)
  (root/'gpu-admission.json').write_text(json.dumps(samples,indent=2));cmd=base(name,True)+[str(root/'source/experiments/full_delivery/cdi_release_validate.py'),'--root',str(root),'--phase',PHASE,'--host-pid',str(os.getpid())]
  if PREVIOUS:cmd+=['--previous-prefix',PREVIOUS]
  code=run('validation',cmd,1200)
finally:
 subprocess.run(['docker','rm','-f',name],capture_output=True,text=True,timeout=30)
 q=subprocess.run(['docker','ps','-aq','--filter','name=nka-bounded-'],capture_output=True,text=True,timeout=30)
 for cid in q.stdout.split():
  info=subprocess.run(['docker','inspect',cid],capture_output=True,text=True,timeout=30)
  if info.returncode==0 and any(m.get('Source','').startswith(str(root)+'/validation/') for m in json.loads(info.stdout)[0].get('Mounts',[])):subprocess.run(['docker','rm','-f',cid],capture_output=True,text=True,timeout=30)
value={'status':'PASS' if code==0 else 'FAIL','phase':PHASE,'version':VERSION,'location':loc,'wheel_sha256':WHEEL_SHA,'validation':json.loads((root/'validation/result.json').read_text()) if (root/'validation/result.json').exists() else None,'records':records,'new_model_calls':0,'new_event_samples':0,'production_release':PHASE=='qualified' and code==0};(root/'result.json').write_text(json.dumps(value,indent=2));print(json.dumps(value))
'''
 (out/'driver.py').write_text(payload);r=execute_python(payload,1500);(out/'transport.stderr').write_text(r.stderr);(out/'transport.stdout').write_text(r.stdout);assert r.returncode==0;v=json.loads(r.stdout);(out/'result.json').write_text(json.dumps(v,indent=2));(out/'location.json').write_text(json.dumps(v['location'],indent=2));print(json.dumps({k:v[k] for k in ['status','phase','version','location','production_release']}))
if __name__=='__main__':main()
