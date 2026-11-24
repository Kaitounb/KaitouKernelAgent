"""Three-process descriptive backend timing using the pinned KernelBench function unchanged."""
import argparse,hashlib,importlib.util,json,os,statistics,subprocess,sys,time,traceback
from pathlib import Path
import torch
import torch.nn.functional as F
from torch.nn.attention import SDPBackend,sdpa_kernel
from kernel_agent.operators import references as ref
from kernel_agent.operators.flashinfer_backend import attention


def child(root,session):
    torch.set_num_threads(1);torch.manual_seed(811+session);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False;rows=[];monitor=[]
    for _ in range(3):
        r=subprocess.run(['nvidia-smi','--query-gpu=utilization.gpu,memory.used,temperature.gpu,clocks.sm,power.draw','--format=csv,noheader'],capture_output=True,text=True,timeout=10);monitor.append({'exit':r.returncode,'stdout':r.stdout,'stderr':r.stderr})
        if r.returncode or int(r.stdout.split(',')[0].strip().split()[0])!=0:
            result={'status':'SKIPPED_GPU_BUSY','session':session,'monitor':monitor,'records':[]};(root/f'session-{session}.json').write_text(json.dumps(result,indent=2)+'\n');return
        time.sleep(1)
    path=root/'upstream/kernelbench_timing.py';spec=importlib.util.spec_from_file_location('original_kernelbench_timing',path);module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    for name,b,sq,sk,h,d,causal in [('prefill',1,128,128,4,64,True),('decode',1,1,1024,8,64,False)]:
        cpu=[(torch.randn(shape)*.25).half() for shape in [(b,sq,h,d),(b,sk,h,d),(b,sk,h,d)]];expected=ref.attention(*(x.permute(0,2,1,3) for x in cpu),causal=causal).permute(0,2,1,3);gpu=[x.cuda() for x in cpu]
        def decomposed(q,k,v):
            scores=q.permute(0,2,1,3).float()@k.permute(0,2,3,1).float()/d**.5
            if causal:scores=scores.masked_fill(~torch.ones(sq,sk,dtype=torch.bool,device=q.device).tril(),-torch.inf)
            p=torch.softmax(scores,-1);return (p@v.permute(0,2,1,3).float()).permute(0,2,1,3).half().contiguous()
        def sdpa(q,k,v,backend):
            with sdpa_kernel(backend):out=F.scaled_dot_product_attention(*(x.permute(0,2,1,3) for x in [q,k,v]),is_causal=causal)
            return out.permute(0,2,1,3).contiguous()
        functions={'decomposed-fp32':decomposed,'flashinfer-fa2':lambda q,k,v:attention(q,k,v,causal=causal),'torch-sdpa-math':lambda q,k,v:sdpa(q,k,v,SDPBackend.MATH),'torch-sdpa-flash':lambda q,k,v:sdpa(q,k,v,SDPBackend.FLASH_ATTENTION)}
        # Rotate backend order by session; each source remains the same across processes.
        names=list(functions);names=names[session%4:]+names[:session%4]
        for backend in names:
            try:
                fn=functions[backend];out=fn(*gpu);actual=out.cpu().double();delta=(actual-expected).abs();assert torch.isfinite(actual).all() and bool((delta<=2e-3+2e-3*expected.abs()).all());assert out.dtype==torch.float16 and out.is_contiguous()
                times=module.time_execution_with_cuda_event(fn,gpu,num_warmup=3,num_trials=10,discard_first=1,verbose=False)
                after=fn(*gpu).cpu().double();assert bool(((after-expected).abs()<=2e-3+2e-3*expected.abs()).all()) and all(torch.equal(x.cpu(),y) for x,y in zip(gpu,cpu))
                row={'label':name+'/'+backend,'status':'PASS','backend':backend,'phase':name,'shape_b_sq_sk_h_d':[b,sq,sk,h,d],'causal':causal,'elapsed_ms':times,'median_ms':statistics.median(times),'max_abs_error':float(delta.max()),'full_score_materialized':backend in ['decomposed-fp32','torch-sdpa-math'],'inputs_unchanged':True}
            except Exception:row={'label':name+'/'+backend,'status':'UNAVAILABLE_OR_FAILED','traceback':traceback.format_exc()}
            rows.append(row);(root/f'session-{session}.json').write_text(json.dumps({'status':'RUNNING','records':rows},indent=2)+'\n')
    result={'status':'PASS' if all(x['status']=='PASS' for x in rows) else 'PARTIAL','session':session,'records':rows,'monitor':monitor,'torch':torch.__version__,'cuda':torch.version.cuda,'capability':torch.cuda.get_device_capability(),'timing_source_sha256':hashlib.sha256(path.read_bytes()).hexdigest(),'performance_claim':False,'promotion_allowed':False};(root/f'session-{session}.json').write_text(json.dumps(result,indent=2)+'\n')


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--session',type=int);a=p.parse_args()
    if a.session is not None:child(a.root,a.session);return
    sessions=[]
    for i in range(3):
        time.sleep(3)  # Let the utilization sampling window of the preceding process expire.
        with (a.root/f'bench-{i}.stdout').open('w') as out,(a.root/f'bench-{i}.stderr').open('w') as err:r=subprocess.run([sys.executable,__file__,'--root',str(a.root),'--session',str(i)],cwd=a.root,env=os.environ.copy(),stdout=out,stderr=err,timeout=300)
        if r.returncode:sessions.append({'status':'FAIL','session':i});break
        sessions.append(json.loads((a.root/f'session-{i}.json').read_text()))
        if sessions[-1]['status']=='SKIPPED_GPU_BUSY':break
    result={'status':'PASS' if len(sessions)==3 and all(x['status']=='PASS' for x in sessions) else 'INCOMPLETE','sessions':sessions,'checks':sum(len(x.get('records',[])) for x in sessions),'passed':sum(r['status']=='PASS' for x in sessions for r in x.get('records',[])),'kind':'descriptive cold-cache backend diagnostic, not paired screening or agent gain','new_model_calls':0,'promotion_allowed':False,'performance_claim':False,'production_release':False};(a.root/'result.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps({k:result[k] for k in ['status','checks','passed','kind']}))


if __name__=='__main__':main()
