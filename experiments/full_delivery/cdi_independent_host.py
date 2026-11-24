"""Fresh server-only synthetic list after frozen preview; no old sealed values are read."""
import argparse,hashlib,json,os,secrets,sys,traceback
from pathlib import Path

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main():
 p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--install-root',type=Path,required=True);p.add_argument('--phase',choices=['prepare','run'],required=True);a=p.parse_args();root=a.root;installed=a.install_root;location=json.loads((installed/'location.json').read_text());wheel=installed/'dist'/('nvidia_kernel_agent-'+location['version']+'-py3-none-any.whl');sys.path.insert(0,str(installed/'prefix'))
 from kernel_agent.cdi_qualification import RUNTIME_FILES,index
 if a.phase=='prepare':
  root.mkdir(mode=0o700,exist_ok=True)
  with (root/'prepare-reservation.json').open('x') as f:json.dump({'one_shot':True},f)
  assert not (root/'manifest.json').exists() and not (root/'protocol.json').exists();assert location['version']=='0.1.3rc2' and sha(wheel)==location['wheel_sha256'];catalog=index()
  def odd(low,high):return secrets.randbelow((high-low)//2+1)*2+low
  shapes=[(1,odd(17,97),odd(17,193)),(odd(17,127),odd(33,255),odd(33,257)),(odd(129,385),odd(129,513),odd(129,513)),(odd(65,257),odd(97,385),odd(65,385))]
  manifest={'id':'rtx3060-cdi-rc2-independent-v1','origin':'fresh server-local synthetic data; no old private manifest read','workloads':[{'shape_mnk':list(shape),'seeds':[secrets.randbelow(2**31) for _ in range(2)],'distribution':'signed4' if i==3 else 'random'} for i,shape in enumerate(shapes)]};(root/'manifest.json').write_text(json.dumps(manifest,indent=2));(root/'manifest.json').chmod(0o600)
  protocol={'scope':'New RTX3060 installed nativeCDI fixedcuBLAS independent correctness','manifest_sha256':sha(root/'manifest.json'),'wheel_sha256':sha(wheel),'host_sha256':sha(Path(__file__)),'runtime_files':{n:sha(installed/'prefix'/n) for n in RUNTIME_FILES},'manifest_count':4,'seeds_per_workload':2,'roles':[catalog['default'],catalog['fallback']],'binary_sha256':{v:m['binary_sha256'] for v,m in catalog['versions'].items()},'expected_gpu_calls':16,'reference':'trusted CPU NumPy FP64 outside UID65534 worker','atol':3e-5,'rtol':1e-4,'automatic_retry':False,'one_shot':True,'worker_access':'per-call inputs only; no full list/reference/modelcredentials','new_model_calls':0,'new_event_samples':0,'production_release':False};(root/'protocol.json').write_text(json.dumps(protocol,indent=2));print(json.dumps({'status':'FRESH_SERVER_LOCAL_MANIFEST_FROZEN','manifest_sha256':protocol['manifest_sha256'],'protocol_sha256':sha(root/'protocol.json'),'wheel_sha256':sha(wheel),'host_sha256':protocol['host_sha256'],'expected_gpu_calls':16,'old_private_manifest_read':False,'production_release':False}));return
 protocol=json.loads((root/'protocol.json').read_text());assert sha(Path(__file__))==protocol['host_sha256'] and sha(root/'manifest.json')==protocol['manifest_sha256'] and sha(wheel)==protocol['wheel_sha256'];assert all(sha(installed/'prefix'/n)==h for n,h in protocol['runtime_files'].items());reservation=root/'execution-started.json'
 with reservation.open('x') as f:json.dump({'one_shot_reserved':True,'automatic_retry':False},f)
 import fcntl,subprocess,time,numpy as np
 from kernel_agent.cdi_qualification import registry
 from kernel_agent.cdi_runner import DockerRunner
 lock=open('/tmp/nka-reviewed-gpu-execution.lock','a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);summary={'status':'FAIL','executed_gpu_calls':0,'correct_outputs':0,'manifest_count':4,'manifest_sha256':protocol['manifest_sha256'],'protocol_sha256':sha(root/'protocol.json'),'wheel_sha256':sha(wheel),'old_private_manifest_read':False,'exact_values_exposed_to_model':False,'new_model_calls':0,'new_event_samples':0,'production_release':False};rows=[]
 try:
  for _ in range(3):
   r=subprocess.run(['nvidia-smi','--query-gpu=utilization.gpu','--format=csv,noheader,nounits'],capture_output=True,text=True,timeout=15);assert r.returncode==0 and int(r.stdout.strip())==0;time.sleep(1)
  store=registry(root/'store');store.initialize();manifest=json.loads((root/'manifest.json').read_text());assert len(manifest['workloads'])==4
  for i,item in enumerate(manifest['workloads']):
   m,n,k=item['shape_mnk']
   for j,seed in enumerate(item['seeds']):
    rng=np.random.default_rng(seed)
    def values(shape):
     if item['distribution']=='signed4':return ((np.arange(np.prod(shape)).reshape(shape)%2*2-1)*4).astype(np.float16)
     return np.clip(rng.normal(0,.25,shape),-4,4).astype(np.float16)
    x,y=values((m,k)),values((k,n));reference=(x.astype(np.float64)@y.astype(np.float64)).astype(np.float32);case=root/('case-'+str(i)+'-'+str(j));case.mkdir(mode=0o700)
    for name,value in [('a',x),('b',y),('reference',reference)]:np.save(case/(name+'.npy'),value)
    for version in protocol['roles']:
     binary,meta=store.release(version);job=case/version;summary['executed_gpu_calls']+=1;reservation.write_text(json.dumps(summary));actual,record=DockerRunner(binary,meta['binary_sha256']).run(x,y,job,case='boundaries',host_private_path=str(root/'manifest.json'),host_pid=os.getpid());good=actual is not None and bool(np.isfinite(actual).all()) and bool(np.all(np.abs(actual.astype(np.float64)-reference.astype(np.float64))<=protocol['atol']+protocol['rtol']*np.abs(reference)));rows.append({'job':str(job.relative_to(root)),'version':version,'binary_sha256':meta['binary_sha256'],'correct':good,'max_abs_error':float(np.abs(actual.astype(np.float64)-reference.astype(np.float64)).max()) if actual is not None else None});(root/'checks.json').write_text(json.dumps(rows,indent=2));assert record['cleanup']['removed'],'cleanup unconfirmed; no further calls'
  summary.update(status='PASS' if len(rows)==16 and all(x['correct'] for x in rows) else 'FAIL',correct_outputs=sum(x['correct'] for x in rows),max_abs_error=max((x['max_abs_error'] for x in rows if x['max_abs_error'] is not None),default=None))
 except Exception:(root/'private-failure.log').write_text(traceback.format_exc());summary['error']='private evaluator failed; exact values remain private'
 (root/'summary.json').write_text(json.dumps(summary,indent=2));print(json.dumps(summary))
if __name__=='__main__':main()
