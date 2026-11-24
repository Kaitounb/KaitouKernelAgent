"""Final read-only production pointer/admission/idle check after runtime image recovery verification."""
import json,stat
from tools.ssh_3060 import ROOT,execute_python

def main():
 backup=json.loads((ROOT/'evidence/20261124-3060-runtime-image-backup/result.json').read_text());assert backup['status']=='IMAGE_BACKUP_SHA_CONFIG_AND_ALL_LAYER_DIFFIDS_PASS';loc=json.loads((ROOT/'evidence/20261124-3060-0.1.3-installed/location.json').read_text());out=ROOT/'evidence/20261124-3060-final-postflight';out.mkdir(exist_ok=False)
 payload='LOCATION='+repr(loc)+'\n'+r'''
import fcntl,hashlib,json,subprocess,time
from pathlib import Path
loc=LOCATION;root=Path(loc['root']);store=root/'workspace/production-store';r=subprocess.run([str(root/'kernel-agent-3060'),'status','--root',str(store)],capture_output=True,text=True,timeout=90);assert r.returncode==0;s=json.loads(r.stdout);assert s['production_release'] is True and s['selected_version']=='sm86-cublas-container';lock=open('/tmp/nka-reviewed-gpu-execution.lock','a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);commands=[['docker','ps','-a','--format','{{json .}}'],['nvidia-smi','--query-gpu=uuid,driver_version,power.limit,utilization.gpu,memory.used,temperature.gpu','--format=csv,noheader,nounits'],['nvidia-smi','--query-compute-apps=pid,process_name,used_memory','--format=csv,noheader']];records=[]
for cmd in commands:
 p=subprocess.run(cmd,capture_output=True,text=True,timeout=30);assert p.returncode==0;records.append({'command':cmd,'exit':p.returncode,'stdout':p.stdout,'stderr':p.stderr})
assert not any(json.loads(row)['Names'].startswith('nka-') for row in records[0]['stdout'].splitlines());fields=[v.strip() for v in records[1]['stdout'].split(',')];assert fields[:2]==['GPU-0140b895-e4e9-7c67-aba1-c7a953df1847','580.95.05'] and float(fields[2])==170 and int(fields[3])==0 and not records[2]['stdout'].strip();assert (root/'current').resolve()==Path(loc['prefix']);certificate=root/'prefix/kernel_agent/data/cdi/gemm_qualification.json';v={'status':'PRODUCTION_ENTRY_QUALIFIED_IDLE_AND_RESTORED_PASS','wall':time.time(),'root':str(root),'current_prefix':str((root/'current').resolve()),'qualification_sha256':hashlib.sha256(certificate.read_bytes()).hexdigest(),'admission_status':s,'records':records,'gpu_lock_acquired':True,'no_owned_containers':True,'power_limit_watts':170,'new_model_calls':0,'new_gpu_outputs':0,'production_release':True};print(json.dumps(v))
'''
 (out/'driver.py').write_text(payload);r=execute_python(payload,180);(out/'transport.stderr').write_text(r.stderr);assert r.returncode==0;v=json.loads(r.stdout)
 for path in [ROOT/'.private/server-gpu-3060-next-connection.json',ROOT/'.private/3060-rc2-independent-evidence']:
  assert stat.S_IMODE(path.stat().st_mode)==(0o700 if path.is_dir() else 0o600)
 v['local_private_permissions_pass']=True;(out/'result.json').write_text(json.dumps(v,indent=2));print(json.dumps({k:v[k] for k in ['status','current_prefix','qualification_sha256','no_owned_containers','power_limit_watts','local_private_permissions_pass','production_release']}))
if __name__=='__main__':main()
