"""New server-local synthetic Attention acceptance; exact values stay in private control storage."""
import argparse,hashlib,json,secrets,sys,traceback
from pathlib import Path


def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()

def main():
 p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--install-root',type=Path,required=True);p.add_argument('--phase',choices=['prepare','run'],required=True);a=p.parse_args();root=a.root;installed=a.install_root
 if a.phase=='prepare':
  root.mkdir(mode=0o700,exist_ok=False);wheel=installed/'nvidia_kernel_agent-0.1.1rc1-py3-none-any.whl';assert sha(wheel)=='93b228111847c97dbad3e29f718fb5b77a039ca4d2406a5333b84feb32b3dffb';workloads=[]
  for i in range(4):
   sq=secrets.randbelow(47)+1;sk=sq+secrets.randbelow(113);workloads.append({'batch':secrets.randbelow(2)+1,'sq':sq,'sk':sk,'heads':secrets.randbelow(4)+1,'dim':128 if i in [1,2] else 64,'causal':i in [1,3],'mask':i in [2,3],'seeds':[secrets.randbelow(2**31) for _ in range(2)]})
  (root/'manifest.json').write_text(json.dumps({'id':'attention-sm86-rc1-independent-v1','origin':'fresh server-local synthetic; no old sealed data read','workloads':workloads},indent=2));(root/'manifest.json').chmod(0o600)
  protocol={'scope':'SM86 fixed reviewed Attention independent correctness, not performance','manifest_sha256':sha(root/'manifest.json'),'wheel_sha256':sha(wheel),'host_sha256':sha(Path(__file__)),'workload_count':4,'seeds_per_workload':2,'expected_gpu_calls':8,'atol':.002,'rtol':.002,'one_shot':True,'automatic_retry':False,'worker_access':'per-call request/input only; no reference or full manifest mount','old_private_manifest_read':False,'new_model_calls':0,'production_release':False};(root/'protocol.json').write_text(json.dumps(protocol,indent=2));print(json.dumps({'status':'NEW_PRIVATE_SYNTHETIC_MANIFEST_FROZEN','manifest_sha256':protocol['manifest_sha256'],'protocol_sha256':sha(root/'protocol.json'),'host_sha256':protocol['host_sha256'],'wheel_sha256':protocol['wheel_sha256'],'expected_gpu_calls':8,'old_private_manifest_read':False}));return
 protocol=json.loads((root/'protocol.json').read_text());assert sha(root/'manifest.json')==protocol['manifest_sha256'] and sha(Path(__file__))==protocol['host_sha256']
 with (root/'execution-reservation.json').open('x') as f:json.dump({'one_shot_reserved':True,'automatic_retry':False},f)
 import fcntl,os,subprocess,time
 import numpy as np
 import torch
 sys.path.insert(0,str(installed/'prefix'));from kernel_agent.operators import references as ref
 lock=open('/tmp/nka-reviewed-gpu-execution.lock','a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);torch.set_num_threads(1);summary={'status':'FAIL','expected_gpu_calls':8,'executed_gpu_calls':0,'correct_outputs':0,'manifest_sha256':protocol['manifest_sha256'],'protocol_sha256':sha(root/'protocol.json'),'wheel_sha256':protocol['wheel_sha256'],'old_private_manifest_read':False,'exact_values_exposed_to_model':False,'new_model_calls':0,'new_event_samples':0,'production_release':False};checks=[]
 try:
  bundle=installed/'sdk';bid=json.loads((bundle/'aot-manifest.json').read_text())['bundle_id'];env={k:v for k,v in os.environ.items() if k in ['HOME','PATH','LANG','LD_LIBRARY_PATH']};env.update(PYTHONPATH=str(installed/'prefix'),PYTHONDONTWRITEBYTECODE='1',OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1')
  for i,item in enumerate(json.loads((root/'manifest.json').read_text())['workloads']):
   for j,seed in enumerate(item['seeds']):
    case=root/('case-'+str(i)+'-'+str(j));case.mkdir(mode=0o700);inputs=case/'inputs';inputs.mkdir();refs=case/'references';refs.mkdir();rng=np.random.default_rng(seed);b,sq,sk,h,d=[item[k] for k in ['batch','sq','sk','heads','dim']];values=[(rng.standard_normal(shape)*.25).astype(np.float16) for shape in [(b,sq,h,d),(b,sk,h,d),(b,sk,h,d)]];mask=None
    for name,value in zip(['q.npy','k.npy','v.npy'],values):np.save(inputs/name,value,allow_pickle=False)
    if item['mask']:
     mask=rng.random((sq,sk))>.2;mask[0]=False;np.save(inputs/'mask.npy',mask,allow_pickle=False)
    target=ref.attention(*(torch.from_numpy(x).permute(0,2,1,3) for x in values),causal=item['causal'],mask=torch.from_numpy(mask) if mask is not None else None).permute(0,2,1,3).numpy();np.save(refs/'output.npy',target,allow_pickle=False);request={'operation':'attention','bundle_id':bid,'fault':None,**{k:v for k,v in item.items() if k!='seeds'}};(case/'request.json').write_text(json.dumps(request));summary['executed_gpu_calls']+=1;(root/'execution-reservation.json').write_text(json.dumps(summary))
    cmd=[sys.executable,'-B','-m','kernel_agent.operators.cli','run','--bundle',str(bundle),'--request',str(case/'request.json'),'--inputs',str(inputs),'--output',str(case/'outputs'),'--work',str(case/'run')];r=subprocess.run(cmd,env=env,capture_output=True,text=True,timeout=120);(case/'stdout').write_text(r.stdout);(case/'stderr').write_text(r.stderr);assert r.returncode==0
    execution=json.loads((case/'run/execution.json').read_text());assert len(execution['attempts'])==1 and execution['attempts'][0]['cleanup']['removed'];actual=np.load(case/'outputs/output.npy',allow_pickle=False).astype(np.float64);delta=np.abs(actual-target);good=bool(np.isfinite(actual).all() and np.all(delta<=protocol['atol']+protocol['rtol']*np.abs(target)));checks.append({'case':case.name,'correct':good,'max_abs_error':float(delta.max())});(root/'checks.json').write_text(json.dumps(checks,indent=2));assert good
  summary.update(status='PASS',correct_outputs=len(checks),max_abs_error=max(row['max_abs_error'] for row in checks))
 except Exception:(root/'private-failure.log').write_text(traceback.format_exc());summary['error']='private evaluator failed; exact values retained privately'
 (root/'summary.json').write_text(json.dumps(summary,indent=2));print(json.dumps(summary))

if __name__=='__main__':main()
