"""Trusted reviewed operator build and public CUDA correctness checks, no model calls."""
import argparse,hashlib,json,os,platform,subprocess,time,traceback
from pathlib import Path
import torch
from kernel_agent.operators import references as ref
from kernel_agent.operators.flashinfer_backend import attention,MLAPlan,rotary,append_cache,merge_partials


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--root',type=Path,required=True);parser.add_argument('--only-softmax',action='store_true');a=parser.parse_args();root=a.root;records=[];torch.set_num_threads(2)
    environment={'python':platform.python_version(),'torch':torch.__version__,'cuda':torch.version.cuda,'gpu':torch.cuda.get_device_name(),'capability':torch.cuda.get_device_capability(),'cuda_home':os.environ.get('CUDA_HOME')}
    (root/'environment.json').write_text(json.dumps(environment,indent=2)+'\n')
    def check(label,fn):
        start=time.monotonic()
        try:details=fn();torch.cuda.synchronize();row={'label':label,'status':'PASS','details':details,'wall_seconds':time.monotonic()-start}
        except Exception:row={'label':label,'status':'FAIL','traceback':traceback.format_exc(),'wall_seconds':time.monotonic()-start}
        records.append(row);(root/'checks.json').write_text(json.dumps(records,indent=2)+'\n');return row
    def compare(actual,expected,atol=2e-3,rtol=2e-3):
        actual=actual.detach().cpu().double();expected=expected.double()
        assert actual.shape==expected.shape and torch.isfinite(actual).all()
        difference=(actual-expected).abs();assert bool((difference<=atol+rtol*expected.abs()).all()),f'max_abs_error={float(difference.max())}'
        return float(difference.max())
    def build():
        from torch.utils.cpp_extension import load
        return load(name='nka_torch_persistent_v271',sources=[str(root/'kernels/softmax/torch_persistent.cu')],extra_include_paths=[str(root/'upstream')],extra_cflags=['-O3'],extra_cuda_cflags=['-O3','-lineinfo'],build_directory=str(root/'build'),verbose=True)
    (root/'build').mkdir();module=None
    start=time.monotonic()
    try:module=build();records.append({'label':'softmax-native-build','status':'PASS','wall_seconds':time.monotonic()-start})
    except Exception:records.append({'label':'softmax-native-build','status':'FAIL','traceback':traceback.format_exc()})
    torch.manual_seed(771)
    if module is not None:
        for dtype in [torch.float16,torch.float32]:
            for width in [1,7,31,32,127,129,513,1024,2048,2049,4097]:
                def softmax_test(dtype=dtype,width=width):
                    x=torch.randn(7,width,dtype=torch.float32)*20;x[0]=0;x[1]=10000;x[2]=-10000
                    if width>1:x[3,0]=10000;x[3,1]=-10000
                    x=x.to(dtype);original=x.clone();expected=torch.softmax(x.double(),-1);stream=torch.cuda.Stream()
                    with torch.cuda.stream(stream):
                        gpu=x.cuda();out=module.softmax(gpu);copied=out.cpu()
                    stream.synchronize();error=compare(copied,expected,2e-6,2e-5);assert out.dtype==torch.float32 and torch.equal(gpu.cpu(),original)
                    return {'shape':list(x.shape),'dtype':str(dtype),'backend':'pytorch-persistent' if width<=2048 else 'aten-cuda-fallback','max_abs_error':error,'current_stream':True,'inputs_unchanged':True}
                check(f'softmax-{dtype}-{width}',softmax_test)
    if a.only_softmax:
        result={'status':'PASS' if all(x['status']=='PASS' for x in records) else 'FAIL','checks':len(records),'passed':sum(x['status']=='PASS' for x in records),'environment':environment,'records':records,'scope':'softmax only','new_model_calls':0,'performance_claim':False,'production_release':False}
        (root/'result.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps({k:result[k] for k in ['status','checks','passed','scope']}));return
    for b,sq,sk,h,d,causal in [(1,1,17,2,64,False),(1,17,17,2,64,True),(2,7,31,3,64,True),(1,33,65,4,128,False),(1,9,65,2,128,True)]:
        def attention_test(b=b,sq=sq,sk=sk,h=h,d=d,causal=causal):
            cpu=[(torch.randn(shape)*.25).half() for shape in [(b,sq,h,d),(b,sk,h,d),(b,sk,h,d)]];expected=ref.attention(*(x.permute(0,2,1,3) for x in cpu),causal=causal).permute(0,2,1,3);gpu=[x.cuda() for x in cpu]
            output=attention(*gpu,causal=causal);error=compare(output,expected);assert all(torch.equal(x.cpu(),y) for x,y in zip(gpu,cpu))
            return {'b_sq_sk_h_d':[b,sq,sk,h,d],'causal':causal,'backend':'flashinfer-fa2','max_abs_error':error,'full_scores_materialized':False,'inputs_unchanged':True}
        check(f'attention-{b}-{sq}-{sk}-{h}-{d}-{causal}',attention_test)
    def mask_test():
        q=(torch.randn(1,7,2,64)*.25).half();k=(torch.randn(1,19,2,64)*.25).half();v=torch.randn_like(k)*.25;mask=torch.ones(7,19,dtype=torch.bool);mask[:,3::4]=False
        expected=ref.attention(*(x.permute(0,2,1,3) for x in [q,k,v]),mask=mask).permute(0,2,1,3);out=attention(q.cuda(),k.cuda(),v.cuda(),mask=mask.cuda());return {'max_abs_error':compare(out,expected),'custom_mask':True}
    check('attention-custom-mask',mask_test)
    def rope_test():
        q=(torch.randn(19,3,64)*.25).half();k=(torch.randn(19,1,64)*.25).half();positions=torch.tensor([0,1,2,7,15,16,17,31,32,63,64,127,128,255,511,1023,2047,4095,8191],dtype=torch.int32)
        expected_q=ref.rotary(q,positions);expected_k=ref.rotary(k,positions);out_q,out_k=rotary(q.cuda(),k.cuda(),positions.cuda())
        return {'query_max_abs_error':compare(out_q,expected_q),'key_max_abs_error':compare(out_k,expected_k),'interleaved':True,'positions_max':8191}
    check('rope-explicit-positions',rope_test)
    def append_test():
        lengths=[1,17,35];page_size=16;indptr=[0,1,3,6];indices=[4,1,5,0,3,2];c=(torch.randn(sum(lengths),512)*.25).half();kr=(torch.randn(sum(lengths),64)*.25).half();cache=torch.zeros(6,16,512,device='cuda',dtype=torch.float16);rope_cache=torch.zeros(6,16,64,device='cuda',dtype=torch.float16)
        # Two append phases cross page boundaries; retain existing page identities.
        old=[0,0,0];first=[1,15,16];first_ptr=[0,1,2,3];first_indices=[indices[0],indices[1],indices[3]];offset=[0,1,18]
        part_c=torch.cat([c[o:o+n] for o,n in zip(offset,first)]);part_kr=torch.cat([kr[o:o+n] for o,n in zip(offset,first)])
        append_cache(cache,rope_cache,part_c.cuda(),part_kr.cuda(),old_lengths=old,append_lengths=first,indptr=first_ptr,indices=first_indices)
        remaining=[n-f for n,f in zip(lengths,first)];part_c=torch.cat([c[o+f:o+n] for o,f,n in zip(offset,first,lengths)]);part_kr=torch.cat([kr[o+f:o+n] for o,f,n in zip(offset,first,lengths)])
        updated=append_cache(cache,rope_cache,part_c.cuda(),part_kr.cuda(),old_lengths=first,append_lengths=remaining,indptr=indptr,indices=indices);assert updated==lengths
        actual=cache.cpu();actual_rope=rope_cache.cpu()
        for bi,n in enumerate(lengths):
            for t in range(n):
                page=indices[indptr[bi]+t//16];slot=t%16;assert torch.equal(actual[page,slot],c[offset[bi]+t]) and torch.equal(actual_rope[page,slot],kr[offset[bi]+t])
            if n%16:
                page=indices[indptr[bi+1]-1];assert bool((actual[page,n%16:]==0).all()) and bool((actual_rope[page,n%16:]==0).all())
        return {'two_phase_append':True,'ragged_lengths':lengths,'nonidentity_page_map':True,'tail_untouched':True}
    check('mla-cache-append',append_test)
    for index,(lengths,heads,page_size) in enumerate([([1],16,1),([15],16,16),([17],16,16),([257],32,16),([31,129],16,16),([0,17,255],16,16),([1025],64,64),([4097],128,64)]):
        def mla_test(lengths=lengths,heads=heads,page_size=page_size):
            batch=len(lengths);maximum=max(lengths);c=(torch.randn(batch,maximum,512)*.25).half();kr=(torch.randn(batch,maximum,64)*.25).half();qa=(torch.randn(batch,heads,512)*.25).half();qr=(torch.randn(batch,heads,64)*.25).half();scale=192**-.5
            indptr=[0]
            for n in lengths:indptr.append(indptr[-1]+(n+page_size-1)//page_size)
            total=indptr[-1];indices=torch.randperm(total).tolist();pages=torch.zeros(total,page_size,512,dtype=torch.float16);rope_pages=torch.zeros(total,page_size,64,dtype=torch.float16)
            for bi,n in enumerate(lengths):
                for token in range(n):
                    page=indices[indptr[bi]+token//page_size];slot=token%page_size;pages[page,slot]=c[bi,token];rope_pages[page,slot]=kr[bi,token]
            z,lse=ref.mla_core(qa,qr,c,kr,lengths,scale);gpu=[x.cuda() for x in [qa,qr,pages,rope_pages]]
            t=time.monotonic();plan=MLAPlan(lengths=lengths,indptr=indptr,indices=indices,num_pages=total,page_size=page_size,heads=heads,scale=scale,device=gpu[0].device);planning=time.monotonic()-t
            out,actual_lse=plan.run(*gpu);error=compare(out,z);actual_lse=actual_lse.cpu().double();nonempty=torch.tensor(lengths)>0
            assert torch.equal(torch.isneginf(actual_lse),torch.isneginf(lse));lse_error=compare(actual_lse[nonempty],lse[nonempty]);assert all(torch.equal(x.cpu(),y) for x,y in zip(gpu,[qa,qr,pages,rope_pages]))
            return {'lengths':lengths,'heads':heads,'latent_rank':512,'rope_dim':64,'scale':scale,'page_size':page_size,'random_page_map':True,'backend':'flashinfer-mla-fa2','max_abs_error':error,'natural_lse_error':lse_error,'planning_wall_seconds':planning,'workspace_bytes':128*1024*1024,'full_scores_materialized':False,'inputs_unchanged':True}
        check(f'mla-{index}',mla_test)
    def split_test():
        heads=16;left_lengths=[0,1,17,128];right_lengths=[0,0,2,1];lengths=[a+b for a,b in zip(left_lengths,right_lengths)];c=(torch.randn(4,129,512)*.25).half();kr=(torch.randn(4,129,64)*.25).half();qa=(torch.randn(4,heads,512)*.25).half();qr=(torch.randn(4,heads,64)*.25).half();scale=192**-.5
        def native(part_lengths,starts):
            ptr=[0];pages=[];rope_pages=[]
            for bi,(n,start) in enumerate(zip(part_lengths,starts)):
                count=(n+15)//16;ptr.append(ptr[-1]+count)
                for pi in range(count):
                    page=torch.zeros(16,512,dtype=torch.float16);rp=torch.zeros(16,64,dtype=torch.float16);size=min(16,n-pi*16);page[:size]=c[bi,start+pi*16:start+pi*16+size];rp[:size]=kr[bi,start+pi*16:start+pi*16+size];pages.append(page);rope_pages.append(rp)
            pg=torch.stack(pages).cuda();rp=torch.stack(rope_pages).cuda();plan=MLAPlan(lengths=part_lengths,indptr=ptr,indices=list(range(ptr[-1])),num_pages=ptr[-1],page_size=16,heads=heads,scale=scale,device=pg.device)
            return plan.run(qa.cuda(),qr.cuda(),pg,rp)
        left=native(left_lengths,[0]*4);right=native(right_lengths,left_lengths);actual,lse=merge_partials(left,right,left_lengths=left_lengths,right_lengths=right_lengths);expected,expected_lse=ref.mla_core(qa,qr,c,kr,lengths,scale)
        error=compare(actual,expected);assert torch.equal(torch.isneginf(lse.cpu()),torch.isneginf(expected_lse));lse_error=compare(lse.cpu()[1:],expected_lse[1:]);return {'max_abs_error':error,'natural_lse_error':lse_error,'left_lengths':left_lengths,'right_lengths':right_lengths,'combine_included_in_execution':True,'cuda_sdk_merge':True,'empty_identity':True}
    check('mla-explicit-split-combine',split_test)
    result={'status':'PASS' if all(x['status']=='PASS' for x in records) else 'FAIL','checks':len(records),'passed':sum(x['status']=='PASS' for x in records),'environment':environment,'records':records,'new_model_calls':0,'performance_claim':False,'production_release':False}
    (root/'result.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps({k:result[k] for k in ['status','checks','passed','new_model_calls','performance_claim','production_release']}))


if __name__=='__main__':main()
