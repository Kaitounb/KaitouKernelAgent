"""One preregistered HAGS-On deployment qualification, no failed identity retry."""
import argparse,fcntl,hashlib,json,math,os,subprocess,sys,threading,time,traceback
from pathlib import Path


def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()

def main():
 p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);a=p.parse_args();root=a.root;protocol=json.loads((root/'protocol.json').read_text());assert sha(Path(__file__))==protocol['host_sha256'];prefix=Path(protocol['installed_prefix']);old=Path(protocol['baseline_prefix']);sdk=Path(protocol['sdk_prefix']);sys.path[:0]=[str(root/'source'),str(prefix)];from kernel_agent.operators.measurement import assess
 from experiments.full_delivery.operator_deployment_cases import WORKLOADS
 assert WORKLOADS==protocol['workloads'] and all(sha(prefix/n)==h for n,h in protocol['runtime_files'].items()) and sha(old/'kernel_agent/operators/flashinfer_backend.py')==sha(prefix/'kernel_agent/operators/flashinfer_backend.py')
 manifest=json.loads((sdk/'aot-manifest.json').read_text());assert manifest['bundle_id']==protocol['bundle_id'] and all(sha(sdk/n)==h for n,h in manifest['files'].items());assert all(sha(root/'source'/n)==h for n,h in protocol['source_files'].items())
 env={k:v for k,v in os.environ.items() if k in ['HOME','PATH','LANG','LD_LIBRARY_PATH']};env.update(PYTHONPATH=str(root/'source')+':'+str(prefix)+':'+str(root/'sdk'),PYTHONDONTWRITEBYTECODE='1',OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',FLASHINFER_WORKSPACE_BASE='/tmp/nka-unused-aot-no-jit',TORCH_CUDA_ARCH_LIST='8.6');lock=open('/tmp/nka-reviewed-gpu-execution.lock','a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);samples=[];records=[];aa=[];ab=[];decision=None;status='FAIL';stop=threading.Event()
 def sample(label):
  r=subprocess.run(['nvidia-smi','--query-gpu=timestamp,utilization.gpu,memory.used,temperature.gpu,clocks.sm,power.draw','--format=csv,noheader'],capture_output=True,text=True,timeout=10);v={'wall':time.time(),'phase':label,'exit':r.returncode,'stdout':r.stdout,'stderr':r.stderr};samples.append(v);return v
 def idle():
  consecutive=0
  for _ in range(10):
   v=sample('idle-admission')
   try:good=v['exit']==0 and int(v['stdout'].split(',')[1].strip().split()[0])==0
   except (ValueError,IndexError):good=False
   consecutive=consecutive+1 if good else 0
   if consecutive==3:return True
   time.sleep(1)
  return False
 def monitor():
  while not stop.wait(.5):
   try:sample('active')
   except Exception as e:samples.append({'wall':time.time(),'error':type(e).__name__})
 def validate(report,mode,session):
  assert report['status']=='PASS' and report['mode']==mode and report['session']==session and report['torch']==protocol['torch'] and report['cuda']==protocol['cuda'] and report['capability']==[8,6]
  grid={(w['id'],n) for w in WORKLOADS for n in [1,16]};assert len(report['rows'])==len(grid) and {(r['workload'],r['input_count']) for r in report['rows']}==grid
  for row in report['rows']:
   assert row['shared_input_storage'] is True and row['input_device_ptrs']['incumbent']==row['input_device_ptrs']['candidate']
   assert row['qa_after_pass'] and row['inputs_and_guards_unchanged'] and row['capture_order']==(['incumbent','candidate'] if session%2==0 else ['candidate','incumbent'])
   pilots=row['pilot_elapsed_ms'];assert len(pilots)==3 and all(math.isfinite(v) and v>0 for v in pilots) and row['replays']==max(1,min(8192,math.ceil(4000/sorted(pilots)[1]))) and row['timed_calls_per_segment']==32*row['replays']
   assert row['native_hashes'] and all(manifest['files'].get(n)==h for n,h in row['native_hashes'].items())
   assert len(row['segments'])==18
   for role in ['incumbent','candidate']:assert len(row['input_hashes'][role])==row['input_count'] and len(row['output_hashes'][role])==32 and len(row['qa_before_errors'][role])==row['input_count']
  return report
 t=threading.Thread(target=monitor,daemon=True);t.start()
 try:
  for mode,target in [('aa',aa),('ab',ab)]:
   for session in range(3):
    if not idle():status='GPU_BUSY_UNSAMPLED_REMAINDER';break
    cmd=[sys.executable,'-B','-m','experiments.full_delivery.operator_deployment_shared_paired','--root',str(root),'--mode',mode,'--session',str(session)];start=time.monotonic()
    with (root/(mode+'-'+str(session)+'.stdout')).open('w') as o,(root/(mode+'-'+str(session)+'.stderr')).open('w') as e:r=subprocess.run(cmd,cwd=root,env=env,stdout=o,stderr=e,timeout=650)
    records.append({'mode':mode,'session':session,'exit':r.returncode,'wall_seconds':time.monotonic()-start});(root/'progress.json').write_text(json.dumps(records,indent=2))
    if r.returncode!=0:status='WORKER_FAILED';break
    target.append(validate(json.loads((root/(mode+'-'+str(session)+'.json')).read_text()),mode,session));status='RUNNING'
   if status!='RUNNING':break
   decision=assess(aa,ab,WORKLOADS,[w['id'] for w in WORKLOADS]);(root/'decision.json').write_text(json.dumps(decision,indent=2))
   if mode=='aa' and decision['reason']!='AA_PASS_AB_NOT_RUN':status='AA_REJECTED';break
   if mode=='ab':status='PASS_DEPLOYMENT_STABILITY_NONREGRESSION' if decision.get('aa_status')=='PASS' and len(decision['policies'])==2 and all(min(v['workload_speedups'].values())>=.97 for v in decision['policies']) else 'NONREGRESSION_REJECTED'
 except Exception:(root/'failure.log').write_text(traceback.format_exc());status='FAIL'
 finally:
  stop.set();t.join(15)
  binding=all(sha(prefix/n)==h for n,h in protocol['runtime_files'].items()) and all(sha(sdk/n)==h for n,h in manifest['files'].items()) and all(sha(root/'source'/n)==h for n,h in protocol['source_files'].items())
  if not binding:status='SOURCE_OR_BINARY_CHANGED'
  result={'status':status,'protocol_sha256':sha(root/'protocol.json'),'aa_sessions':len(aa),'ab_sessions':len(ab),'raw_segments':sum(len(row['segments']) for report in aa+ab for row in report['rows']),'records':records,'decision':decision,'source_and_native_unchanged':binding,'new_model_calls':0,'old_sealed_executed':False,'runtime_promotion':False,'production_release':False,'scope':protocol['scope']};(root/'result.json').write_text(json.dumps(result,indent=2));(root/'gpu-monitor.json').write_text(json.dumps(samples,indent=2));print(json.dumps({k:v for k,v in result.items() if k not in ['decision','records']}))

if __name__=='__main__':main()
