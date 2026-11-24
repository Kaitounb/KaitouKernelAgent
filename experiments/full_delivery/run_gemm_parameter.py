"""One first-time warp-1 A-control batch, never any old rejected identity."""
import argparse,base64,hashlib,json
from pathlib import Path
from tools.ssh_3070 import execute_python
ROOT=Path(__file__).resolve().parents[2]


def main():
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);a=p.parse_args();a.output.mkdir(exist_ok=False)
    names=['experiments/full_delivery/gemm_parameter.py','evaluators/gemm_candidate.py','evaluators/gemm_decision.py','evaluators/gemm_regression.py','kernel_agent/operators/measurement.py','kernel_agent/optimization/identity.py','workloads/gemm_development_v1.json','workloads/gemm_contract_v1.json','workloads/gemm_measurement_v1.json','kernels/gemm/wmma_sm86.cu','kernels/gemm/cublas_sm86.cu','kernels/gemm/gemm_contract.cuh'];files={n:(ROOT/n).read_bytes() for n in names};files['kernel_agent/__init__.py']=b'';files['kernelbench/__init__.py']=b''
    kb=ROOT/'kernel_agent/optimization/assets/provenance/upstream-reviews/20261121-p1-02/ScalingIntelligence__KernelBench/src/kernelbench'
    for n in ['eval.py','dataset.py','timing.py']:files['kernelbench/'+n]=(kb/n).read_bytes()
    (a.output/'source-hashes.json').write_text(json.dumps({n:hashlib.sha256(v).hexdigest() for n,v in files.items()},indent=2)+'\n')
    payload='FILES='+repr({n:base64.b64encode(v).decode() for n,v in files.items()})+'\n'+r'''
import base64,fcntl,hashlib,json,os,subprocess,tempfile,time
from pathlib import Path
root=Path(tempfile.mkdtemp(prefix='nka-gemm-warp1-'));lock=open('/tmp/nka-reviewed-gpu-execution.lock','a');fcntl.flock(lock,fcntl.LOCK_EX);py='/home/dministrator/cuda-lab/.venv/bin/python';env={k:v for k,v in os.environ.items() if k in ['HOME','LANG','LD_LIBRARY_PATH']};env.update(PYTHONPATH=str(root)+':/tmp/nka-gemm-rel01-f3ekx0tu/run/deps',PATH='/usr/local/cuda-13.0/bin:/usr/lib/wsl/lib:/usr/local/bin:/usr/bin:/bin',OMP_NUM_THREADS='1');records=[];samples=[]
for n,v in FILES.items():
 p=root/n;p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes(base64.b64decode(v))
before={n:hashlib.sha256((root/n).read_bytes()).hexdigest() for n in FILES}
def idle():
 count=0
 for _ in range(10):
  r=subprocess.run(['nvidia-smi','--query-gpu=timestamp,utilization.gpu,memory.used,temperature.gpu,clocks.sm,power.draw','--format=csv,noheader'],env=env,capture_output=True,text=True,timeout=10);samples.append({'time':time.time(),'exit':r.returncode,'stdout':r.stdout,'stderr':r.stderr})
  try:good=r.returncode==0 and int(r.stdout.split(',')[1].strip().split()[0])==0
  except (ValueError,IndexError):good=False
  count=count+1 if good else 0
  if count==3:return True
  time.sleep(1)
 return False
def run(mode,session=None):
 args=[py,'-m','experiments.full_delivery.gemm_parameter','--root',str(root)];label='decision'
 if mode=='decide':args.append('--decide')
 else:args+=['--mode',mode,'--session',str(session),'--warps','1'];label=mode+'-'+str(session)
 started=time.time();start=time.monotonic()
 with (root/(label+'.stdout')).open('w') as out,(root/(label+'.stderr')).open('w') as err:r=subprocess.run(args,env=env,cwd=root,stdout=out,stderr=err,timeout=600)
 records.append({'label':label,'exit':r.returncode,'started_at':started,'finished_at':time.time(),'wall_seconds':time.monotonic()-start});return r.returncode
status='PASS';decision=None
for mode in ['aa','ab']:
 for session in range(3):
  if not idle():status='GPU_BUSY_UNSAMPLED_REMAINDER';break
  if run(mode,session)!=0:status='FAIL';break
 if status!='PASS':break
 assert run('decide')==0;decision=json.loads((root/'decision.json').read_text())
 if mode=='aa' and decision['reason']!='AA_PASS_AB_NOT_RUN':status='AA_REJECTED';break
after={n:hashlib.sha256((root/n).read_bytes()).hexdigest() for n in FILES};assert before==after;result={'status':status,'decision':decision,'records':records,'sources_before':before,'sources_after':after,'warp_control':1,'original_wmmas4_resampled':False,'new_model_calls':0,'production_release':False};(root/'result.json').write_text(json.dumps(result,indent=2));(root/'gpu-admission.json').write_text(json.dumps(samples,indent=2));print(json.dumps({'remote_root':str(root),'result':result}))
'''
    (a.output/'driver.py').write_text(payload);r=execute_python(payload,4000);(a.output/'transport.stderr').write_text(r.stderr);(a.output/'transport.stdout').write_text(r.stdout)
    if r.returncode:raise RuntimeError('transport failed; collect existing job only')
    value=json.loads(r.stdout);(a.output/'result.json').write_text(json.dumps(value,indent=2)+'\n');print(json.dumps({'remote_root':value['remote_root'],'status':value['result']['status'],'decision':value['result']['decision']}))


if __name__=='__main__':main()
