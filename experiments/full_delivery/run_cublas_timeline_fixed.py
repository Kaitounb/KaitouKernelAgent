"""One isolated-config correction for the documented WSL Nsight timestamp issue."""
import argparse,base64,hashlib,json
from pathlib import Path
from tools.ssh_3070 import ROOT,execute_python

def main():
 p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);a=p.parse_args();a.output.mkdir(exist_ok=False);source=(ROOT/'experiments/full_delivery/cublas_timeline_target.py').read_bytes();(a.output/'target.py').write_bytes(source);plan={'scope':'One isolated Nsight WSL timestamp repair; short graph activity smoke then conditional original162 diagnostic segments, no qualification','target_sha256':hashlib.sha256(source).hexdigest(),'original_incomplete_evidence':'evidence/20261123-cublas-timeline-diagnostic','nsys_config':'CuptiUseRawGpuTimestamps=false','global_config_changed':False,'new_model_calls':0,'production_release':False};(a.output/'plan.json').write_text(json.dumps(plan,indent=2))
 payload='SOURCE='+repr(base64.b64encode(source).decode())+'\n'+r'''
import base64,fcntl,json,os,sqlite3,subprocess,tempfile,time
from pathlib import Path
root=Path(tempfile.mkdtemp(prefix='nka-cublas-timeline-fixed-'));(root/'target.py').write_bytes(base64.b64decode(SOURCE));lock=open('/tmp/nka-reviewed-gpu-execution.lock','a');fcntl.flock(lock,fcntl.LOCK_EX);records=[];home=root/'home';home.mkdir();config=home/'.config/NVIDIA Corporation/nsys-config.ini';config.parent.mkdir(parents=True);config.write_text('CuptiUseRawGpuTimestamps=false\n');env={k:v for k,v in os.environ.items() if k in ['LD_LIBRARY_PATH','LANG','WSL_INTEROP','WSL_DISTRO_NAME']};env.update(HOME=str(home),PATH='/usr/local/bin:/usr/bin:/bin:/usr/lib/wsl/lib',OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1')
def run(args,timeout,child_env=None):
 start=time.monotonic();r=subprocess.run(args,capture_output=True,text=True,timeout=timeout,env=child_env);records.append({'command':args,'exit':r.returncode,'wall_seconds':time.monotonic()-start,'stdout':r.stdout,'stderr':r.stderr});(root/'commands.json').write_text(json.dumps(records,indent=2));return r
py='/home/dministrator/cuda-lab/.venv/bin/python';smoke=root/'smoke.py';smoke.write_text("import torch\ntorch.set_num_threads(1)\ntorch.cuda.cudart().cudaProfilerStart()\na=torch.ones((256,256),device='cuda');b=a.clone();out=torch.empty_like(a)\nfor _ in range(4):torch.mm(a,b,out=out)\ng=torch.cuda.CUDAGraph()\nwith torch.cuda.graph(g):\n for _ in range(4):torch.mm(a,b,out=out)\nfor _ in range(4):g.replay()\ntorch.cuda.synchronize()\ntorch.cuda.cudart().cudaProfilerStop()\n")
def profile(script,label,args):
 r=run(['nsys','profile','--trace=cuda,nvtx','--cuda-graph-trace=graph:host-only','--sample=none','--cpuctxsw=none','--capture-range=cudaProfilerApi','--capture-range-end=stop','--flush-on-cudaprofilerstop=true','--output',str(root/label),py,str(script),*args],240,env)
 if r.returncode==0:return run(['nsys','export','--type=sqlite','--output',str(root/(label+'.sqlite')),str(root/(label+'.nsys-rep'))],120,env).returncode==0
 return False
result={'status':'TIMESTAMP_REPAIR_UNVERIFIED','production_release':False,'new_model_calls':0,'qualification':False}
smi='/mnt/c/Windows/System32/nvidia-smi.exe'
try:
 assert run([smi,'-i','0','--lock-gpu-clocks=1200,1200'],30).returncode==0;assert run([smi,'-i','0','--lock-memory-clocks=7001,7001'],30).returncode==0
 if profile(smoke,'smoke',[]):
  c=sqlite3.connect('file:'+str(root/'smoke.sqlite')+'?mode=ro',uri=True);tables=[x[0] for x in c.execute("SELECT name FROM sqlite_master WHERE type='table'")];counts={n:c.execute('SELECT COUNT(*) FROM "'+n+'"').fetchone()[0] for n in tables if n.startswith('CUPTI_ACTIVITY_KIND_')};c.close();result['smoke_activity_counts']=counts
  if counts.get('CUPTI_ACTIVITY_KIND_GRAPH_TRACE',0)>0:
   result['status']='GPU_GRAPH_SMOKE_PASS'
   if profile(root/'target.py','trace',['--root',str(root)]):result['status']='DIAGNOSTIC_COLLECTED_TIMESTAMP_CORRECTED'
finally:
 result['reset_gpu_exit']=run([smi,'-i','0','--reset-gpu-clocks'],30).returncode;result['reset_memory_exit']=run([smi,'-i','0','--reset-memory-clocks'],30).returncode
(root/'result.json').write_text(json.dumps(result,indent=2));print(json.dumps({'remote_root':str(root),'result':result}))
'''
 (a.output/'driver.py').write_text(payload);r=execute_python(payload,900);(a.output/'transport.stderr').write_text(r.stderr)
 if r.returncode:(a.output/'transport.stdout').write_text(r.stdout);raise RuntimeError('Existing diagnostic reservation kept; no automatic retry')
 v=json.loads(r.stdout);(a.output/'result.json').write_text(json.dumps(v,indent=2));print(json.dumps(v))
if __name__=='__main__':main()
