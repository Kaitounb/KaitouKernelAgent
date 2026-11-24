"""Publish and exercise the installed qualified3060 launcher, including real preview package recovery."""
import base64,hashlib,json
from tools.ssh_3060 import ROOT,execute_python

def main():
 installed=ROOT/'evidence/20261124-3060-0.1.3-installed';v=json.loads((installed/'verification.json').read_text());assert v['status']=='AUDIT_PASS_QUALIFIED_INSTALL';loc=json.loads((installed/'location.json').read_text());out=ROOT/'evidence/20261124-3060-active-entry';out.mkdir(exist_ok=False);source=(ROOT/'tools/cdi_container_entry.py').read_bytes();(out/'launcher-source-sha256.json').write_text(json.dumps({'tools/cdi_container_entry.py':hashlib.sha256(source).hexdigest()}))
 payload='LOCATION='+repr(loc)+'\nSOURCE='+repr(base64.b64encode(source).decode())+'\nSWITCH_SOURCE='+repr(base64.b64encode((ROOT/'tools/switch_cdi_package.py').read_bytes()).decode())+'\n'+r'''
import base64,hashlib,json,os,subprocess,sys,time
from pathlib import Path
loc=LOCATION;root=Path(loc['root']);work=root/'activation';work.mkdir(exist_ok=False);tools=root/'tools';tools.mkdir(exist_ok=True);script=tools/'cdi_container_entry.py';script.write_bytes(base64.b64decode(SOURCE));launcher=root/'kernel-agent-3060';launcher.write_text('#!/bin/sh\nexec '+sys.executable+' -B '+str(script)+' --deployment-root '+str(root)+' -- "$@"\n');launcher.chmod(0o700);switch_script=tools/'switch_cdi_package.py';switch_script.write_bytes(base64.b64decode(SWITCH_SOURCE));switcher=root/'kernel-agent-3060-switch';switcher.write_text('#!/bin/sh\nexec '+sys.executable+' -B '+str(switch_script)+' --deployment-root '+str(root)+' "$@"\n');switcher.chmod(0o700);space=root/'workspace';space.mkdir(mode=0o700,exist_ok=True);records=[]
def run(label,cmd,success=True):
 started=time.monotonic();r=subprocess.run(cmd,capture_output=True,text=True,timeout=240);(work/(label+'.stdout')).write_text(r.stdout);(work/(label+'.stderr')).write_text(r.stderr);records.append({'label':label,'command':cmd,'exit':r.returncode,'seconds':time.monotonic()-started});(work/'commands.json').write_text(json.dumps(records,indent=2));assert (r.returncode==0)==success,label;return r
code="import numpy as np;from pathlib import Path;p=Path("+repr(str(space))+");r=np.random.default_rng(3060130);a=(r.standard_normal((17,23))*.25).astype(np.float16);b=(r.standard_normal((23,19))*.25).astype(np.float16);np.save(p/'a.npy',a);np.save(p/'b.npy',b);np.save(p/'reference.npy',a.astype(np.float64)@b.astype(np.float64))"
cmd=['docker','run','--rm','--pull=never','--network=none','--read-only','--cap-drop=ALL','--security-opt=no-new-privileges','--user',str(loc['uid'])+':'+str(loc['gid']),'--runtime=runc','--env','NVIDIA_VISIBLE_DEVICES=void','--env','OPENBLAS_NUM_THREADS=1','--mount','type=bind,src='+str(space)+',dst='+str(space),'--entrypoint','/opt/venv/bin/python',loc['image_id'],'-B','-c',code];run('create-public-inputs',cmd)
store=space/'production-store';run('init',[str(launcher),'init','--root',str(store)]);outputs=[]
def calculate(label,selected_store,channel=None):
 args=[str(launcher),'run','--root',str(selected_store),'--a',str(space/'a.npy'),'--b',str(space/'b.npy'),'--output',str(space/(label+'.npy'))]
 if channel:args+=['--channel',channel]
 r=run(label,args);value=json.loads(r.stdout);assert value['status']=='PASS' and value['production_release'] is (channel is None);outputs.append({'label':label,'execution':value});(work/'outputs.json').write_text(json.dumps(outputs,indent=2))
status='FAIL'
try:
 calculate('qualified-default',store)
 run('switch-preview',[str(switcher),'preview']);preview=space/'preview-recovery-store';run('preview-init',[str(launcher),'init','--root',str(preview),'--channel','development']);calculate('package-preview',preview,'development')
finally:
 run('switch-release',[str(switcher),'release'])
calculate('package-restored',store);s=json.loads(run('qualified-status',[str(launcher),'status','--root',str(store)]).stdout);assert s['production_release'] is True and s['selected_version']=='sm86-cublas-container'
# Actual different power limit must be refused before a CUDA worker; restore170W finally.
power=[]
def limit(watts):
 r=subprocess.run(['sudo','-n','nvidia-smi','-pl',str(watts)],capture_output=True,text=True,timeout=30);power.append({'watts':watts,'exit':r.returncode,'stdout':r.stdout,'stderr':r.stderr});assert r.returncode==0
 for _ in range(15):
  q=subprocess.run(['nvidia-smi','--query-gpu=power.limit','--format=csv,noheader,nounits'],capture_output=True,text=True,timeout=15)
  if q.returncode==0 and float(q.stdout.strip())==watts:return
  time.sleep(.2)
 raise ValueError('power limit did not settle')
try:
 limit(160);r=run('wrong-power-rejected',[str(launcher),'status','--root',str(store)],False);assert 'requires170W' in r.stderr
finally:
 limit(170);(work/'power-control.json').write_text(json.dumps(power,indent=2))
q=subprocess.run(['nvidia-smi','--query-gpu=uuid,driver_version,power.limit,utilization.gpu,temperature.gpu,memory.used','--format=csv,noheader,nounits'],capture_output=True,text=True,timeout=30);post={'exit':q.returncode,'stdout':q.stdout,'stderr':q.stderr};(work/'postflight.json').write_text(json.dumps(post,indent=2));assert q.returncode==0 and float(q.stdout.split(',')[2].strip())==170
value={'status':'ACTIVE_ENTRY_PASS','root':str(root),'launcher':str(launcher),'package_switcher':str(switcher),'store':str(store),'workspace':str(space),'current_prefix':str((root/'current').resolve()),'launcher_source_sha256':hashlib.sha256(script.read_bytes()).hexdigest(),'gpu_outputs':len(outputs),'gpu_attempts':sum(len(x['execution']['attempts']) for x in outputs),'actual_package_switch_pass':True,'wrong_power_actual_negative_pass':True,'power_restored_watts':170,'production_release':True,'new_model_calls':0,'new_event_samples':0};(work/'result.json').write_text(json.dumps(value,indent=2));print(json.dumps(value))
'''
 (out/'driver.py').write_text(payload);r=execute_python(payload,900);(out/'transport.stderr').write_text(r.stderr);(out/'transport.stdout').write_text(r.stdout);assert r.returncode==0;v=json.loads(r.stdout);(out/'result.json').write_text(json.dumps(v,indent=2));print(json.dumps(v))
if __name__=='__main__':main()
