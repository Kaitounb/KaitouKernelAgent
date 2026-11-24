"""Finite trusted activity probe, not a latency qualification or candidate evaluation."""
import json
from tools.ssh_3070 import ROOT, execute_python


def main():
    output = ROOT / 'evidence/20261123-hags-postboot-clock-probe'
    output.mkdir(exist_ok=False)
    payload = r'''
import fcntl,json,subprocess,time,threading
from pathlib import Path
lock=open('/tmp/nka-reviewed-gpu-execution.lock','a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
smi='/usr/lib/wsl/lib/nvidia-smi';native='/mnt/c/Windows/System32/nvidia-smi.exe'
def command(args):
 r=subprocess.run(args,capture_output=True,text=True,timeout=30)
 return {'args':args,'exit':r.returncode,'stdout':r.stdout,'stderr':r.stderr}
admission=[]
for _ in range(3):
 r=command([smi,'--query-gpu=utilization.gpu','--format=csv,noheader,nounits']);admission.append(r)
 assert r['exit']==0 and int(r['stdout'].strip())==0,'GPU busy: no probe launched'
 time.sleep(1)
commands=[];samples=[];stop=threading.Event();thread=None;error=None
def monitor():
 while not stop.is_set():
  samples.append(command([smi,'--query-gpu=utilization.gpu,clocks.sm,clocks.mem,temperature.gpu','--format=csv,noheader,nounits']));stop.wait(.5)
try:
 for option in ['--lock-gpu-clocks=1200,1200','--lock-memory-clocks=7001,7001']:
  r=command([native,'-i','0',option]);commands.append(r);assert r['exit']==0
 import torch
 torch.set_num_threads(1)
 a=torch.ones((1024,1024),device='cuda',dtype=torch.float16);b=torch.ones_like(a);out=torch.empty_like(a)
 thread=threading.Thread(target=monitor,daemon=True);thread.start();deadline=time.monotonic()+8
 while time.monotonic()<deadline:
  for _ in range(64):torch.mm(a,b,out=out)
  torch.cuda.synchronize()
except Exception as e:error=repr(e)
finally:
 stop.set()
 if thread:thread.join(15)
 for option in ['--reset-gpu-clocks','--reset-memory-clocks']:commands.append(command([native,'-i','0',option]))
loaded=[]
for r in samples:
 assert r['exit']==0
 values=[float(x.strip()) for x in r['stdout'].strip().split(',')]
 if values[0]>=90:loaded.append(values)
ok=error is None and len(loaded)>=3 and all(r['exit']==0 for r in commands) and all(1185<=r[1]<=1215 and 6798<=r[2]<=6802 and r[3]<80 for r in loaded)
print(json.dumps({'status':'PASS' if ok else 'FAIL','admission':admission,'commands':commands,'telemetry':samples,'loaded_samples':len(loaded),'loaded_sm_mhz':sorted(set(r[1] for r in loaded)),'loaded_memory_mhz':sorted(set(r[2] for r in loaded)),'filesystem_permissions_modified':False,'error':error,'performance_samples':0,'heldout_executed':False,'new_model_calls':0,'production_release':False}))
'''
    (output / 'driver.py').write_text(payload)
    r = execute_python(payload, 180)
    (output / 'transport.stderr').write_text(r.stderr)
    if r.returncode:
        raise RuntimeError('Probe transport failed: inspect clock restoration before continuing')
    result = json.loads(r.stdout)
    (output / 'result.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps({k:v for k,v in result.items() if k not in ['commands','admission','telemetry']}))
    if result['status'] != 'PASS':
        raise SystemExit(2)


if __name__ == '__main__':
    main()
