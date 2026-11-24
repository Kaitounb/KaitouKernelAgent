"""Frozen public MLA workload regression, memory accounting, no search or heldout access."""
import argparse,hashlib,json,math,time,traceback
from pathlib import Path
import torch
from kernel_agent.operators import references as ref
from kernel_agent.operators.flashinfer_backend import MLAPlan,SplitMLAPlan


def memory_budget(w,page_size,ring=16,splits=4):
    lengths=w['lengths'];b=len(lengths);h=w['heads'];pages=sum((n+page_size-1)//page_size for n in lengths);cache=pages*page_size*576*2;q=b*h*576*2;out=b*h*(512*2+4)
    cpu_reference=b*max(lengths)*576*8+b*h*max(lengths)*8+b*h*576*8+b*h*512*8
    return {'cache_bytes':cache,'query_bytes':q,'output_bytes':out,'ring_sets':ring,'max_explicit_splits':splits,'workspace_bytes':splits*128*1024*1024,'conservative_gpu_bytes_including_1GiB_runtime_and_256MiB_flush':ring*(cache+q+out)*2+splits*128*1024*1024+1280*1024*1024,'trusted_cpu_reference_bytes':cpu_reference}


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--config-grid',action='store_true');a=p.parse_args();root=a.root;torch.set_num_threads(2);development=json.loads((root/'workloads/mla_development_v1.json').read_text());records=[];budgets=[]
    def execute(w,seed,page_size,distribution='random',splits=1):
        torch.manual_seed(seed);lengths=w['lengths'];b=len(lengths);heads=w['heads'];maximum=max(lengths);shapes=[(b,heads,512),(b,heads,64),(b,maximum,512),(b,maximum,64)]
        cpu=[(torch.randn(shape)*.25).clamp_(-4,4).half() for shape in shapes]
        if distribution=='zero':cpu=[torch.zeros_like(x) for x in cpu]
        elif distribution=='signed4':
            cpu[0].fill_(4);cpu[1].fill_(4)
            for i in [2,3]:cpu[i].fill_(4);cpu[i][:,1::2].fill_(-4)
        qa,qr,c,kr=cpu;expected,lse=ref.mla_core(qa,qr,c,kr,lengths,192**-.5);ptr=[0]
        for n in lengths:ptr.append(ptr[-1]+(n+page_size-1)//page_size)
        indices=torch.randperm(ptr[-1]).tolist();pages=torch.zeros(ptr[-1],page_size,512,dtype=torch.float16);rp=torch.zeros(ptr[-1],page_size,64,dtype=torch.float16)
        for bi,n in enumerate(lengths):
            for token in range(n):
                page=indices[ptr[bi]+token//page_size];slot=token%page_size;pages[page,slot]=c[bi,token];rp[page,slot]=kr[bi,token]
        gpu=[x.cuda() for x in [qa,qr,pages,rp]];snapshots=[x.clone() for x in [qa,qr,pages,rp]];options=dict(lengths=lengths,indptr=ptr,indices=indices,num_pages=ptr[-1],page_size=page_size,heads=heads,scale=192**-.5,device=gpu[0].device);plan=SplitMLAPlan(**options,splits=splits) if a.config_grid else MLAPlan(**options);out,actual_lse=plan.run(*gpu);actual=out.cpu().double();actual_lse=actual_lse.cpu().double();delta=(actual-expected).abs();assert torch.isfinite(actual).all() and bool((delta<=2e-3+2e-3*expected.abs()).all());assert torch.equal(torch.isneginf(actual_lse),torch.isneginf(lse));valid=torch.isfinite(lse);lse_delta=(actual_lse[valid]-lse[valid]).abs();assert bool((lse_delta<=2e-3+2e-3*lse[valid].abs()).all());assert all(torch.equal(x.cpu(),y) for x,y in zip(gpu,snapshots));assert out.dtype==torch.float16 and actual_lse.shape==(b,heads)
        return {'max_abs_error':float(delta.max()),'natural_lse_max_error':float(lse_delta.max()) if lse_delta.numel() else 0.,'inputs_unchanged':True,'output_sha256':hashlib.sha256(out.cpu().numpy().tobytes()).hexdigest(),'gpu_peak_allocated_bytes':torch.cuda.max_memory_allocated(),'explicit_splits':splits}
    for w in development['workloads']:
        for page_size in [1,16,64]:
            budget=memory_budget(w,page_size);assert budget['conservative_gpu_bytes_including_1GiB_runtime_and_256MiB_flush']<6*1024**3;budgets.append({'workload':w['id'],'page_size':page_size,**budget})
            for splits in ([1,2,4] if a.config_grid else [1]):
                for seed in development['seeds']:
                    label=f"{w['id']}-page{page_size}-split{splits}-seed{seed}";start=time.monotonic()
                    try:row={'label':label,'status':'PASS','details':execute(w,seed,page_size,splits=splits),'wall_seconds':time.monotonic()-start}
                    except Exception:row={'label':label,'status':'FAIL','traceback':traceback.format_exc(),'wall_seconds':time.monotonic()-start}
                    records.append(row);(root/'checks.json').write_text(json.dumps(records,indent=2)+'\n')
    for distribution in ['zero','signed4']:
        try:row={'label':'boundary-'+distribution,'status':'PASS','details':execute({'lengths':[17],'heads':16},773,16,distribution,splits=4 if a.config_grid else 1)}
        except Exception:row={'label':'boundary-'+distribution,'status':'FAIL','traceback':traceback.format_exc()}
        records.append(row)
    if a.config_grid:
        try:row={'label':'boundary-all-empty','status':'PASS','details':execute({'lengths':[0],'heads':16},773,16,'zero',splits=4)}
        except Exception:row={'label':'boundary-all-empty','status':'FAIL','traceback':traceback.format_exc()}
        records.append(row)
    result={'status':'PASS' if all(x['status']=='PASS' for x in records) else 'FAIL','checks':len(records),'passed':sum(x['status']=='PASS' for x in records),'records':records,'memory_budgets':budgets,'capability':torch.cuda.get_device_capability(),'torch':torch.__version__,'cuda':torch.version.cuda,'independent_workloads_read':False,'new_model_calls':0,'performance_claim':False,'production_release':False}
    (root/'result.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps({k:result[k] for k in ['status','checks','passed','independent_workloads_read']}))


if __name__=='__main__':main()
