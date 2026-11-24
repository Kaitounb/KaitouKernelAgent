"""Private trusted evaluator. Exact heldout values never enter public output."""
import argparse,fcntl,hashlib,json,os,subprocess,sys,traceback
from pathlib import Path


def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);a=p.parse_args();root=a.root;private=root/'private';private.mkdir(mode=0o700);summary={'status':'FAIL','manifest_count':4,'executed_gpu_calls':0,'heldout_exposed_to_model':False,'new_model_calls':0,'performance_sampling':False,'production_release':False};checks=[]
    try:
        protocol=json.loads((root/'protocol.json').read_text());freeze=json.loads((root/'freeze.json').read_text());assert sha(root/'manifest.json')==protocol['dataset_commitment_sha256'];manifest=json.loads((root/'manifest.json').read_text());assert manifest['contract_id']=='mla-f16-r512-rope64-v1' and len(manifest['workloads'])==4 and len(manifest['seeds'])==2
        wheel=root/freeze['wheel_filename'];assert sha(wheel)==freeze['wheel_sha256'];prefix=root/'prefix';env={k:v for k,v in os.environ.items() if k in ['HOME','PATH','LANG','LD_LIBRARY_PATH']};env.update(PYTHONPATH=str(prefix),OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1')
        with (private/'install.stdout').open('w') as out,(private/'install.stderr').open('w') as err:r=subprocess.run([sys.executable,'-m','pip','install','--no-index','--no-deps','--target',str(prefix),str(wheel)],env=env,stdout=out,stderr=err,timeout=120)
        assert r.returncode==0;sys.path.insert(0,str(prefix));import numpy as np,torch,kernel_agent
        from kernel_agent.operators import references as ref
        from kernel_agent.operators.runner import OperatorRunner
        assert kernel_agent.__version__=='0.1.0.dev6' and Path(kernel_agent.__file__).is_relative_to(prefix);torch.set_num_threads(1);bundle=Path(freeze['remote_bundle']);runner=OperatorRunner(bundle,freeze['bundle_id']);assert sha(bundle/'aot-manifest.json')==freeze['aot_manifest_sha256'];lock=open('/tmp/nka-reviewed-gpu-execution.lock','a');fcntl.flock(lock,fcntl.LOCK_EX)
        for index,w in enumerate(manifest['workloads']):
            lengths=w['lengths'];h=w['heads'];b=len(lengths);maximum=max(lengths);page=16;ptr=[0]
            for n in lengths:ptr.append(ptr[-1]+(n+page-1)//page)
            for si,seed in enumerate(manifest['seeds']):
                generator=torch.Generator().manual_seed(seed);qa,qr,c,kr=[(torch.randn(shape,generator=generator)*.25).clamp(-4,4).half() for shape in [(b,h,512),(b,h,64),(b,maximum,512),(b,maximum,64)]];indices=torch.randperm(ptr[-1],generator=generator).tolist();pages=torch.zeros(ptr[-1],page,512,dtype=torch.float16);rp=torch.zeros(ptr[-1],page,64,dtype=torch.float16)
                for bi,n in enumerate(lengths):
                    for start in range(0,n,page):physical=indices[ptr[bi]+start//page];count=min(page,n-start);pages[physical,:count]=c[bi,start:start+count];rp[physical,:count]=kr[bi,start:start+count]
                arrays={n:x.numpy() for n,x in zip(['q_abs.npy','q_rope.npy','cache.npy','rope_cache.npy'],[qa,qr,pages,rp])};arrays.update({'indptr.npy':np.array(ptr,np.int32),'indices.npy':np.array(indices,np.int32)});z,lse=ref.mla_core(qa,qr,c,kr,lengths,192**-.5);expected={'z.npy':z.numpy(),'lse.npy':lse.numpy()};request={'operation':'mla','bundle_id':freeze['bundle_id'],'fault':None,'lengths':lengths,'heads':h,'page_size':16,'splits':1,'num_pages':ptr[-1]};job=private/('job-'+str(index)+'-'+str(si));summary['executed_gpu_calls']+=1;(private/'execution-started.json').write_text(json.dumps(summary));actual,record=runner.run(request,arrays,job);errors={};good=actual is not None
                for name,target in expected.items():
                    np.save(job/('reference-'+name),target,allow_pickle=False)
                    if actual is None:continue
                    value=actual[name];finite=np.isfinite(target);delta=np.abs(value.astype(np.float64)[finite]-target[finite]);errors[name]=float(delta.max()) if delta.size else 0.;good=good and bool(np.all(delta<=2e-3+2e-3*np.abs(target[finite]))) and bool(np.array_equal(np.isneginf(value),np.isneginf(target)))
                checks.append({'job':job.name,'correct':bool(good),'max_abs_errors':errors,'record':record});(private/'checks.json').write_text(json.dumps(checks,indent=2))
                if not record.get('cleanup',{}).get('removed'):raise RuntimeError('unconfirmed cleanup; no subsequent GPU invocation')
        summary.update(status='PASS' if all(x['correct'] for x in checks) and len(checks)==8 else 'FAIL',checked_invocations=len(checks),correct_invocations=sum(x['correct'] for x in checks),checked_output_tensors=2*len(checks),max_abs_errors={name:max(x['max_abs_errors'].get(name,float('inf')) for x in checks) for name in ['z.npy','lse.npy']},bundle_id=freeze['bundle_id'],wheel_sha256=freeze['wheel_sha256'],protocol_sha256=sha(root/'protocol.json'),manifest_commitment=sha(root/'manifest.json'),reference_not_mounted=True,model_credentials_not_transferred=True)
    except Exception:
        (private/'failure.log').write_text(traceback.format_exc());summary['error']='trusted_evaluator_failed; exact details preserved privately'
    (root/'summary.json').write_text(json.dumps(summary,indent=2));print(json.dumps(summary))


if __name__=='__main__':main()
