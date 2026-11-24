"""Check released GPU/owned containers after the once-only counterpart; no CUDA work."""
import json
from tools.ssh_3060 import ROOT,execute_python

def main():
 out=ROOT/'evidence/20261124-sm86-3060-counterpart';loc=json.loads((out/'location.json').read_text());finished=json.loads((out/'qualification-launch-result.json').read_text());assert finished['exit']==0
 payload='LOCATION='+repr(loc)+'\n'+r'''
import fcntl,json,subprocess,time
from pathlib import Path
loc=LOCATION;root=Path(loc['root']);lock=open('/tmp/nka-reviewed-gpu-execution.lock','a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);records=[]
for args in [['docker','ps','-aq','--filter','label=nka.counterpart='+root.name],['docker','ps','-aq','--filter','name=nka-counterpart-parent-'],['nvidia-smi','--query-gpu=uuid,name,driver_version,utilization.gpu,memory.used,temperature.gpu,power.limit','--format=csv,noheader,nounits'],['nvidia-smi','--query-compute-apps=pid,process_name,used_memory','--format=csv,noheader']]:
 r=subprocess.run(args,capture_output=True,text=True,timeout=30);records.append({'command':args,'exit':r.returncode,'stdout':r.stdout,'stderr':r.stderr})
assert all(x['exit']==0 for x in records) and all(not x['stdout'].strip() for x in records[:2]);assert not records[-1]['stdout'].strip();f=records[2]['stdout'].strip().split(',');assert f[0].strip()=='GPU-d464a726-39c2-9ad5-d856-f5d00c1fa870' and int(f[3].strip())==0 and float(f[-1].strip())==170
v={'status':'POSTFLIGHT_IDLE_NO_OWNED_CONTAINERS','wall':time.time(),'records':records,'gpu_lock_acquired':True,'power_limit_unchanged_watts':170,'production_release':False,'new_model_calls':0};(root/'postflight.json').write_text(json.dumps(v,indent=2));print(json.dumps(v))
'''
 with (out/'postflight-driver.py').open('x') as f:f.write(payload)
 r=execute_python(payload,150);(out/'postflight.stderr').write_text(r.stderr);assert r.returncode==0;v=json.loads(r.stdout);(out/'postflight.json').write_text(json.dumps(v,indent=2));print(json.dumps(v))
if __name__=='__main__':main()
