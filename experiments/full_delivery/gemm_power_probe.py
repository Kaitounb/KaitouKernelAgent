"""One-shot reversible NVSMI power-control capability check; no GPU timing."""
import json
from pathlib import Path
from tools.ssh_3070 import ROOT,execute_python

def main():
 out=ROOT/'evidence/20261124-sm86-gemm-thermal';out.mkdir(exist_ok=False)
 payload=r"""
import fcntl,json,subprocess
lock=open('/tmp/nka-reviewed-gpu-execution.lock','a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
exe='/mnt/c/Windows/System32/nvidia-smi.exe';records=[]
def run(args):
 r=subprocess.run([exe]+args,capture_output=True,text=True,timeout=30);v={'args':args,'exit':r.returncode,'stdout':r.stdout,'stderr':r.stderr};records.append(v);return r
before=run(['--query-gpu=uuid,power.limit,power.min_limit,power.max_limit,utilization.gpu','--format=csv,noheader,nounits']);assert before.returncode==0
fields=[v.strip() for v in before.stdout.strip().split(',')];assert fields[0]=='GPU-859dd34e-59f1-959a-2546-ca027062f9bd' and float(fields[1])==240 and float(fields[2])==100 and int(fields[4])==0
try:
 set_result=run(['-pl','100'])
 read=run(['--query-gpu=power.limit','--format=csv,noheader,nounits'])
 supported=set_result.returncode==0 and read.returncode==0 and float(read.stdout.strip())==100
finally:
 restored=run(['-pl','240']);final=run(['--query-gpu=power.limit','--format=csv,noheader,nounits'])
print(json.dumps({'status':'CONTROL_SUPPORTED' if supported else 'CONTROL_UNSUPPORTED','records':records,'restored':final.returncode==0 and float(final.stdout.strip())==240,'new_gpu_timing_segments':0}))
"""
 (out/'capability-driver.py').write_text(payload);r=execute_python(payload,150);(out/'capability.stderr').write_text(r.stderr);assert r.returncode==0;v=json.loads(r.stdout);(out/'capability.json').write_text(json.dumps(v,indent=2));assert v['restored'];print(json.dumps(v))
if __name__=='__main__':main()
