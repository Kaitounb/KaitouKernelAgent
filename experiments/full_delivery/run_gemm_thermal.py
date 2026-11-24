"""One bounded minimum-power thermal diagnostic with finally restoration."""
import base64,hashlib,json
from tools.ssh_3070 import ROOT,execute_python

def main():
 out=ROOT/'evidence/20261124-sm86-gemm-thermal';assert json.loads((out/'capability.json').read_text())['status']=='CONTROL_SUPPORTED';reservation=out/'diagnostic-reservation.json';source=(ROOT/'experiments/full_delivery/gemm_thermal_target.py').read_bytes();prefix=json.loads((ROOT/'evidence/20261123-sm86-operators-qualified-installed/location.json').read_text())['prefix']
 with reservation.open('x') as f:json.dump({'scope':'one90s100Wpublic1024cube thermal diagnostic; noAA','target_sha256':hashlib.sha256(source).hexdigest(),'no_retry':True},f,indent=2)
 payload='SOURCE='+repr(base64.b64encode(source).decode())+'\nPREFIX='+repr(prefix)+'\n'+r'''
import base64,fcntl,json,subprocess,tempfile,threading,time
from pathlib import Path
root=Path(tempfile.mkdtemp(prefix='nka-gemm-thermal-'));root.chmod(0o755);(root/'target.py').write_bytes(base64.b64decode(SOURCE));lock=open('/tmp/nka-reviewed-gpu-execution.lock','a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
smi='/mnt/c/Windows/System32/nvidia-smi.exe';records=[];samples=[];stop=threading.Event()
def run(args,timeout=30):
 r=subprocess.run(args,capture_output=True,text=True,timeout=timeout);v={'args':args,'exit':r.returncode,'stdout':r.stdout,'stderr':r.stderr};records.append(v);(root/'commands.json').write_text(json.dumps(records,indent=2));return r
def monitor():
 while not stop.is_set():
  r=subprocess.run([smi,'--query-gpu=timestamp,utilization.gpu,clocks.sm,clocks.mem,temperature.gpu,power.draw,power.limit','--format=csv,noheader,nounits'],capture_output=True,text=True,timeout=15);samples.append({'wall':time.time(),'exit':r.returncode,'stdout':r.stdout,'stderr':r.stderr});stop.wait(1)
before=run([smi,'--query-gpu=power.limit,utilization.gpu','--format=csv,noheader,nounits']);assert before.returncode==0 and [x.strip() for x in before.stdout.strip().split(',')]==['240.00','0']
thread=threading.Thread(target=monitor,daemon=True);status='FAIL'
try:
 assert run([smi,'-pl','100']).returncode==0
 run([smi,'-q','-d','POWER,TEMPERATURE,PERFORMANCE,CLOCK']);thread.start()
 image='sha256:ce00f6dc2eea43dc508b90b548f624646d643a92d363aa715fdb5a3b4f49290a'
 cmd=['docker','run','--rm','--gpus','all','--network','none','--read-only','--cap-drop','ALL','--security-opt','no-new-privileges','--user','65534:65534','--pids-limit','128','--tmpfs','/tmp:rw,nosuid,nodev,size=512m','--env','HOME=/tmp','--env','OMP_NUM_THREADS=1','--env','OPENBLAS_NUM_THREADS=1','--mount','type=bind,src='+PREFIX+',dst=/package,readonly','--mount','type=bind,src='+str(root)+',dst=/diag,readonly','--entrypoint','python',image,'/diag/target.py']
 r=run(cmd,150);status='DIAGNOSTIC_COMPLETE' if r.returncode==0 else 'DIAGNOSTIC_FAILED'
 run([smi,'-q','-d','POWER,TEMPERATURE,PERFORMANCE,CLOCK'])
finally:
 stop.set()
 if thread.ident:thread.join(20)
 (root/'telemetry.json').write_text(json.dumps(samples,indent=2))
 restored=run([smi,'-pl','240']);final=run([smi,'--query-gpu=power.limit','--format=csv,noheader,nounits']);restored_ok=final.returncode==0 and float(final.stdout.strip())==240
result={'status':status,'remote_root':str(root),'commands':records,'telemetry':samples,'restored':restored_ok,'qualification':False,'production_release':False};(root/'result.json').write_text(json.dumps(result,indent=2));print(json.dumps(result))
'''
 (out/'diagnostic-driver.py').write_text(payload);r=execute_python(payload,240);(out/'diagnostic.stderr').write_text(r.stderr);(out/'diagnostic.stdout').write_text(r.stdout);assert r.returncode==0;v=json.loads(r.stdout);(out/'diagnostic.json').write_text(json.dumps(v,indent=2));assert v['restored'];print(json.dumps({k:v[k] for k in ['status','remote_root','restored','qualification']}))
if __name__=='__main__':main()
