"""One isolated-config timeline of the newly observed small-M failure, no qualification."""
import base64,hashlib,json
from tools.ssh_3070 import ROOT,execute_python

def main():
 out=ROOT/'evidence/20261124-sm86-gemm-power-timeline';out.mkdir(exist_ok=False);source=(ROOT/'experiments/full_delivery/gemm_power_timeline_target.py').read_bytes();prefix=json.loads((ROOT/'evidence/20261123-sm86-operators-qualified-installed/location.json').read_text())['prefix'];plan={'scope':'one profiled native100W small-M54segment scheduling diagnostic; not newAA, no certificate','target_sha256':hashlib.sha256(source).hexdigest(),'original_rejection_protocol':'10e2c52428ce8a551533f917451eccabd490a371857ab905d9d55077ceb8af22','power_limit_watts':100,'restore_watts':240,'cpu_limits_distinction':'native host, not Docker2CPU; profile overhead means no unprofiled latency claim','new_model_calls':0,'heldout_read':False};(out/'plan.json').write_text(json.dumps(plan,indent=2))
 payload='SOURCE='+repr(base64.b64encode(source).decode())+'\nPREFIX='+repr(prefix)+'\n'+r'''
import base64,fcntl,json,os,subprocess,tempfile,threading,time
from pathlib import Path
root=Path(tempfile.mkdtemp(prefix='nka-gemm-power-timeline-'));root.chmod(0o700);(root/'target.py').write_bytes(base64.b64decode(SOURCE));home=root/'home';config=home/'.config/NVIDIA Corporation/nsys-config.ini';config.parent.mkdir(parents=True);config.write_text('CuptiUseRawGpuTimestamps=false\n');lock=open('/tmp/nka-reviewed-gpu-execution.lock','a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);records=[];samples=[];stop=threading.Event();smi='/mnt/c/Windows/System32/nvidia-smi.exe';env={k:v for k,v in os.environ.items() if k in ['LD_LIBRARY_PATH','LANG','WSL_INTEROP','WSL_DISTRO_NAME']};env.update(HOME=str(home),PATH='/usr/local/bin:/usr/bin:/bin:/usr/lib/wsl/lib',OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1')
def run(args,timeout=30,child_env=None):
 start=time.monotonic();r=subprocess.run(args,capture_output=True,text=True,timeout=timeout,env=child_env);records.append({'args':args,'exit':r.returncode,'wall_seconds':time.monotonic()-start,'stdout':r.stdout,'stderr':r.stderr});(root/'commands.json').write_text(json.dumps(records,indent=2));return r
assert run([smi,'--query-gpu=power.limit,utilization.gpu','--format=csv,noheader,nounits']).stdout.strip()=='240.00, 0'
def monitor():
 while not stop.is_set():
  r=subprocess.run([smi,'--query-gpu=timestamp,utilization.gpu,clocks.sm,clocks.mem,temperature.gpu,power.draw,power.limit','--format=csv,noheader,nounits'],capture_output=True,text=True,timeout=15);samples.append({'wall':time.time(),'exit':r.returncode,'stdout':r.stdout,'stderr':r.stderr});stop.wait(.5)
thread=threading.Thread(target=monitor,daemon=True);status='FAIL'
try:
 assert run([smi,'-pl','100']).returncode==0
 time.sleep(1);assert float(run([smi,'--query-gpu=power.limit','--format=csv,noheader,nounits']).stdout.strip())==100;thread.start()
 py='/home/dministrator/cuda-lab/.venv/bin/python';cmd=['nsys','profile','--trace=cuda,nvtx','--cuda-graph-trace=graph:host-only','--sample=none','--cpuctxsw=none','--capture-range=cudaProfilerApi','--capture-range-end=stop','--flush-on-cudaprofilerstop=true','--output',str(root/'trace'),py,str(root/'target.py'),'--root',str(root),'--prefix',PREFIX]
 r=run(cmd,240,env)
 if r.returncode==0:
  r=run(['nsys','export','--type=sqlite','--output',str(root/'trace.sqlite'),str(root/'trace.nsys-rep')],120,env)
  if r.returncode==0:status='DIAGNOSTIC_COLLECTED'
finally:
 stop.set()
 if thread.ident:thread.join(20)
 (root/'telemetry.json').write_text(json.dumps(samples,indent=2));run([smi,'-pl','240']);final=run([smi,'--query-gpu=power.limit','--format=csv,noheader,nounits']);restored=final.returncode==0 and float(final.stdout.strip())==240
v={'status':status,'remote_root':str(root),'power_restored':restored,'production_release':False,'new_model_calls':0};(root/'result.json').write_text(json.dumps(v,indent=2));print(json.dumps(v))
'''
 (out/'driver.py').write_text(payload);r=execute_python(payload,480);(out/'transport.stdout').write_text(r.stdout);(out/'transport.stderr').write_text(r.stderr);assert r.returncode==0;v=json.loads(r.stdout);(out/'result.json').write_text(json.dumps(v,indent=2));assert v['power_restored'];print(json.dumps(v))
if __name__=='__main__':main()
