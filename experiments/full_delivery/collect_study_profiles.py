"""Predeclared C-parent profiles for three independent searches, no model calls."""
import argparse,base64,hashlib,json
from pathlib import Path
from tools.ssh_3070 import execute_python
ROOT=Path(__file__).resolve().parents[2]


def main():
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);p.add_argument('--only-mla',action='store_true');p.add_argument('--sdk-prefix');a=p.parse_args();a.output.mkdir(exist_ok=False)
    names=['kernel_agent/__init__.py','kernel_agent/bounded_io.py','kernel_agent/protocol.py','kernel_agent/runner.py','kernel_agent/output.py','kernel_agent/optimization/__init__.py','kernel_agent/optimization/profile_runner.py','kernel_agent/optimization/profile_worker.py','kernel_agent/optimization/profile_target.py','kernel_agent/optimization/source_worker.py','kernel_agent/optimization/source_packets.py','kernel_agent/optimization/fixtures.py','kernel_agent/operators/__init__.py','kernel_agent/operators/flashinfer_backend.py','kernel_agent/operators/references.py','kernel_agent/operators/partition.py','experiments/full_delivery/study_mla_profile_target.py','experiments/p1_04/ncu_adapter.py','workloads/gemm_development_v1.json','workloads/mla_development_v1.json','provenance/upstream-reviews/20261118/meta-pytorch__KernelAgent/files/kernel_perf_agent/kernel_opt/profiler/ncu_profiler.py','provenance/upstream-reviews/20261118/meta-pytorch__KernelAgent/files/kernel_perf_agent/kernel_opt/diagnose_prompt/metric_schema.py'];files={n:(ROOT/n).read_bytes() for n in names};plan={'scope':'C initial parent profiles, three separate collections per target, no candidate comparison','targets':{'gemm':['dev-01','dev-02','dev-06'],'mla':['mla-dev-02','mla-dev-04','mla-dev-08']},'repetitions':[0,1,2],'maximum_profile_calls':18,'B_profile_reads':0,'new_model_calls':0,'input_seed':171,'source_hashes':{n:hashlib.sha256(v).hexdigest() for n,v in files.items()}};plan['targets']={'mla':plan['targets']['mla']} if a.only_mla else plan['targets'];plan['maximum_profile_calls']=sum(len(x) for x in plan['targets'].values())*3;plan['sdk_prefix']=a.sdk_prefix
    (a.output/'plan.json').write_text(json.dumps(plan,indent=2)+'\n');files['plan.json']=(a.output/'plan.json').read_bytes()
    payload='FILES='+repr({n:base64.b64encode(v).decode() for n,v in files.items()})+'\nSDK_PREFIX='+repr(a.sdk_prefix)+'\n'+r'''
import base64,fcntl,hashlib,json,os,subprocess,tempfile
from pathlib import Path
root=Path(tempfile.mkdtemp(prefix='nka-study-profiles-'));lock=open('/tmp/nka-reviewed-gpu-execution.lock','a');fcntl.flock(lock,fcntl.LOCK_EX)
for n,v in FILES.items():
 p=root/n;p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes(base64.b64decode(v))
if SDK_PREFIX:
 manifest=json.loads((Path(SDK_PREFIX)/'aot-manifest.json').read_text());body={k:v for k,v in manifest.items() if k!='bundle_id'};assert hashlib.sha256(json.dumps(body,sort_keys=True,separators=(',',':')).encode()).hexdigest()==manifest['bundle_id']=='3e97a2b632ea2e189da2ef4000b3a6d0750ec3ef852206e13dcfc71adf3eab32';assert all(hashlib.sha256((Path(SDK_PREFIX)/n).read_bytes()).hexdigest()==h for n,h in body['files'].items())
code=r"""
import hashlib,json,os,statistics,subprocess,sys,time
from pathlib import Path
from kernel_agent.optimization.profile_runner import run
from experiments.p1_04.ncu_adapter import HARDWARE_METRICS,LAUNCH_METRICS,profile_cuda,parse_profile
root=Path.cwd();plan=json.loads((root/'plan.json').read_text());metrics=HARDWARE_METRICS+LAUNCH_METRICS;rows=[];binary=Path('/tmp/nka-gemm-rel01-500z3apq/run/prefix/kernel_agent/data/artifacts/93fb2f4344521b7e9b246b54d1c30d4b1a8facf731e6385e18f6119e9c49c72c.so')
def admit(work):
 samples=[];count=0
 for _ in range(15):
  r=subprocess.run(['nvidia-smi','--query-gpu=timestamp,utilization.gpu,memory.used,temperature.gpu,clocks.sm,power.draw','--format=csv,noheader'],capture_output=True,text=True,timeout=10);samples.append({'time':time.time(),'exit':r.returncode,'stdout':r.stdout,'stderr':r.stderr})
  try:good=r.returncode==0 and int(r.stdout.split(',')[1].strip().split()[0])==0
  except (ValueError,IndexError):good=False
  count=count+1 if good else 0
  if count==3:break
  time.sleep(1)
 (work/'admission.json').write_text(json.dumps(samples,indent=2));return count==3
for op,ids in plan['targets'].items():
 development=json.loads((root/('workloads/'+op+'_development_v1.json')).read_text())
 for wid in ids:
  w=next(x for x in development['workloads'] if x['id']==wid)
  for rep in plan['repetitions']:
   work=root/'profiles'/op/wid/str(rep);work.mkdir(parents=True);entry={'op':op,'workload':wid,'repetition':rep,'status':'NOT_RUN'};rows.append(entry)
   if not admit(work):entry['status']='UNSAMPLED_GPU_BUSY';(root/'progress.json').write_text(json.dumps(rows,indent=2));continue
   try:
    if op=='gemm':
     files,record=run(binary,binary.stem,w['shape_mnk'],metrics,work/'worker',root/'experiments/p1_04/ncu_adapter.py');assert record['cleanup']['removed']
     if files is None:raise RuntimeError('bounded profile unavailable')
     parsed=parse_profile(work/'worker/metrics.csv',metrics,4);execution=json.loads((work/'worker/execution.json').read_text());csv=work/'worker/metrics.csv';native=binary.stem
    else:
     (work/'request.json').write_text(json.dumps({'workload':w}));target=work/'target.py';target.write_text("import sys\nfrom experiments.full_delivery.study_mla_profile_target import main\nsys.argv=[__file__,'--request',"+repr(str(work/'request.json'))+",'--output',"+repr(str(work/'validation.json'))+"]\nmain()\n");profile_cuda(target,work,'native',metrics,'/opt/nvidia/nsight-compute/2026.3.1/ncu',timeout=180,kernel_regex='.*BatchMLAPagedAttentionKernel.*');parsed=parse_profile(work/'native.csv',metrics,None,expected_kernel='BatchMLAPagedAttentionKernel');execution=json.loads((work/'native.execution.json').read_text());csv=work/'native.csv';validation=json.loads((work/'validation.json').read_text());assert validation['status']=='PASS';native=next(iter(validation['native_sha256'].values()));assert native=='cdc4c3ba11ef61b7e2778f93af247bc0ac53027daa8035c55b541e6ea3fd601e'
    assert len(parsed['launches'])==3;facts=[]
    for i,m in enumerate(metrics,1):
     values=[x['metrics'][m] for x in parsed['launches']];observed=all(x['status']=='observed' for x in values);facts.append({'id':'F'+str(i),'metric':m,'value':statistics.median(x['value'] for x in values) if observed else None,'unit':values[0]['unit'],'status':'observed' if observed else 'unavailable'})
    profile={'facts':facts,'provenance':[{'csv_sha256':hashlib.sha256(csv.read_bytes()).hexdigest(),'binary_sha256':native,'upstream_profiler_sha256':parsed['upstream_profiler_sha256']}],'profile_wall_seconds':execution['wall_seconds']};(work/'profile.json').write_text(json.dumps(profile,indent=2));(work/'parsed.json').write_text(json.dumps(parsed,indent=2));entry.update(status='PASS',profile_wall_seconds=execution['wall_seconds'],metric_fields=len(facts),launches=3,binary_sha256=native)
   except (RuntimeError,ValueError,AssertionError,OSError,KeyError) as error:entry.update(status='UNAVAILABLE_OR_FAILED',error=str(error)[:512])
   (root/'progress.json').write_text(json.dumps(rows,indent=2))
result={'status':'PASS' if len(rows)==plan['maximum_profile_calls'] and all(x['status']=='PASS' for x in rows) else 'PARTIAL','rows':rows,'new_model_calls':0,'event_sampling':False,'heldout_read':False,'production_release':False};(root/'result.json').write_text(json.dumps(result,indent=2));print(json.dumps(result))
"""
(root/'collect.py').write_text(code);env={k:v for k,v in os.environ.items() if k in ['HOME','LANG','LD_LIBRARY_PATH']};env.update(PYTHONPATH=str(root)+':'+(SDK_PREFIX or '/tmp/nka-flashinfer-sdk-local-m0583epr/prefix-cxx20')+':/tmp/nka-p1-04-parser-deps',PATH='/usr/local/cuda-13.0/bin:/usr/lib/wsl/lib:/usr/local/bin:/usr/bin:/bin',CUDA_HOME='/usr/local/cuda-13.0',TORCH_CUDA_ARCH_LIST='8.6',FLASHINFER_CUDA_ARCH_LIST='8.6',FLASHINFER_WORKSPACE_BASE='/tmp/nka-mla-measurement-v1-3070-jit',MAX_JOBS='2',OMP_NUM_THREADS='1')
with (root/'controller.stdout').open('w') as out,(root/'controller.stderr').open('w') as err:r=subprocess.run(['/home/dministrator/cuda-lab/.venv/bin/python',str(root/'collect.py')],cwd=root,env=env,stdout=out,stderr=err,timeout=1200)
print(json.dumps({'remote_root':str(root),'exit':r.returncode,'result':json.loads((root/'result.json').read_text()) if (root/'result.json').exists() else None}))
'''
    (a.output/'driver.py').write_text(payload);r=execute_python(payload,1300);(a.output/'transport.stderr').write_text(r.stderr);(a.output/'transport.stdout').write_text(r.stdout)
    if r.returncode:raise RuntimeError('profile preparation transport failed; inspect existing job only')
    result=json.loads(r.stdout);(a.output/'result.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result))


if __name__=='__main__':main()
