"""Freeze and run one bounded scheduling trace of the trusted cuBLAS default."""
import argparse,base64,hashlib,json
from pathlib import Path
from tools.ssh_3070 import ROOT,execute_python

def main():
 p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);a=p.parse_args();a.output.mkdir(exist_ok=False);source=(ROOT/'experiments/full_delivery/cublas_timeline_target.py').read_bytes();plan={'scope':'One diagnostic only; profiler data cannot qualify or replace old AA failures','public_shapes':[[1,4096,512],[8,1024,1024],[128,2048,512]],'inputs':16,'repetitions':3,'pairs':9,'segments':162,'target_segment_ms':250,'native_clock_requests_mhz':[1200,7001],'source_sha256':hashlib.sha256(source).hexdigest(),'no_private_data':True,'new_model_calls':0,'new_cuda_source':False,'production_release':False};(a.output/'plan.json').write_text(json.dumps(plan,indent=2));(a.output/'target.py').write_bytes(source)
 payload='SOURCE='+repr(base64.b64encode(source).decode())+'\n'+r'''
import base64,fcntl,json,os,subprocess,tempfile,time
from pathlib import Path
root=Path(tempfile.mkdtemp(prefix='nka-cublas-timeline-'));(root/'target.py').write_bytes(base64.b64decode(SOURCE));lock=open('/tmp/nka-reviewed-gpu-execution.lock','a');fcntl.flock(lock,fcntl.LOCK_EX);records=[]
def run(args,timeout):
 start=time.monotonic();r=subprocess.run(args,capture_output=True,text=True,timeout=timeout);records.append({'command':args,'exit':r.returncode,'wall_seconds':time.monotonic()-start,'stdout':r.stdout,'stderr':r.stderr});(root/'commands.json').write_text(json.dumps(records,indent=2));return r
smi='/mnt/c/Windows/System32/nvidia-smi.exe'
try:
 assert run([smi,'-i','0','--lock-gpu-clocks=1200,1200'],30).returncode==0
 assert run([smi,'-i','0','--lock-memory-clocks=7001,7001'],30).returncode==0
 command=['nsys','profile','--trace=cuda,nvtx','--cuda-graph-trace=graph:host-only','--sample=none','--cpuctxsw=none','--capture-range=cudaProfilerApi','--capture-range-end=stop','--output',str(root/'trace'),'/home/dministrator/cuda-lab/.venv/bin/python',str(root/'target.py'),'--root',str(root)]
 r=run(command,240)
 if r.returncode==0:run(['nsys','export','--type=sqlite','--output',str(root/'trace.sqlite'),str(root/'trace.nsys-rep')],120)
finally:
 run([smi,'-i','0','--reset-gpu-clocks'],30);run([smi,'-i','0','--reset-memory-clocks'],30)
files={p.name:{'bytes':p.stat().st_size} for p in root.iterdir() if p.is_file()};print(json.dumps({'remote_root':str(root),'diagnostic_completed':(root/'diagnostic-segments.json').exists(),'files':files,'new_model_calls':0,'production_release':False}))
'''
 (a.output/'driver.py').write_text(payload);r=execute_python(payload,600);(a.output/'transport.stderr').write_text(r.stderr)
 if r.returncode:(a.output/'transport.stdout').write_text(r.stdout);raise RuntimeError('Diagnostic transport failed; inspect existing job, do not relaunch')
 v=json.loads(r.stdout);(a.output/'result.json').write_text(json.dumps(v,indent=2));print(json.dumps(v))
if __name__=='__main__':main()
