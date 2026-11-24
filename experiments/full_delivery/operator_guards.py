"""Public input/offset/stream/empty-mask checks, also run under Compute Sanitizer."""
import json,torch
from kernel_agent.operators import references as ref
from kernel_agent.operators.flashinfer_backend import attention,MLAPlan,SplitMLAPlan


def main():
    torch.set_num_threads(1);torch.manual_seed(772);torch.backends.cuda.matmul.allow_tf32=False;records=[]
    def compare(out,expected):
        out=out.cpu().double();error=(out-expected).abs();assert torch.isfinite(out).all() and bool((error<=2e-3+2e-3*expected.abs()).all());return float(error.max())
    def guarded(cpu):
        raw=torch.full((cpu.numel()+16,),-3.25,dtype=cpu.dtype,device='cuda');view=raw[8:-8].view(cpu.shape);return raw,view
    def guards(raw):return bool((raw[:8]==-3.25).all()) and bool((raw[-8:]==-3.25).all())
    cpu=[(torch.randn(shape)*.25).half() for shape in [(2,7,3,64),(2,19,3,64),(2,19,3,64)]];pairs=[guarded(x) for x in cpu];gpu=[x[1] for x in pairs];mask=torch.ones(7,19,dtype=torch.bool);mask[0]=False;mask[:,3::4]=False;expected=ref.attention(*(x.permute(0,2,1,3) for x in cpu),mask=mask,causal=True).permute(0,2,1,3)
    stream=torch.cuda.Stream();stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        torch.cuda._sleep(50000)
        for x,y in zip(gpu,cpu):x.copy_(y)
        output=attention(*gpu,mask=mask.cuda(),causal=True);consumer=output.clone().cpu()
    stream.synchronize();error=compare(consumer,expected);assert bool((consumer[:,0]==0).all()) and all(guards(raw) and torch.equal(value.cpu(),host) for (raw,value),host in zip(pairs,cpu));assert output.dtype==torch.float16 and output.data_ptr() not in [x.data_ptr() for x in gpu]
    records.append({'case':'attention-offset8-stream-mask-causal','max_abs_error':error,'full_mask_zero':True,'guards_and_inputs':True,'output_contract':True})
    raw=torch.zeros(cpu[0].numel()+1,device='cuda',dtype=torch.float16);bad=raw[1:].view(cpu[0].shape)
    try:attention(bad,gpu[1],gpu[2])
    except ValueError:records.append({'case':'unaligned-attention-input','rejected_before_native':True})
    else:raise AssertionError('unaligned pointer accepted')
    bad_metadata=[{'lengths':[17],'indptr':[0,1],'indices':[0]}, {'lengths':[1],'indptr':[0,1],'indices':[9]}, {'lengths':[32],'indptr':[0,2],'indices':[0,0]}, {'lengths':[True],'indptr':[0,1],'indices':[0]}]
    for entry in bad_metadata:
        try:MLAPlan(**entry,num_pages=2,page_size=16,heads=16,scale=192**-.5,device='cuda')
        except ValueError:continue
        raise AssertionError('invalid page metadata accepted')
    records.append({'case':'mla-page-contract','rejected_before_native':4})
    lengths=[0,17,33];heads=16;maximum=33;host_c=(torch.randn(3,maximum,512)*.25).half();host_r=(torch.randn(3,maximum,64)*.25).half();qa=(torch.randn(3,heads,512)*.25).half();qr=(torch.randn(3,heads,64)*.25).half();ptr=[0,0,2,5];indices=[3,1,4,0,2];pages=torch.zeros(5,16,512,dtype=torch.float16);rp=torch.zeros(5,16,64,dtype=torch.float16)
    for bi,n in enumerate(lengths):
        for token in range(n):
            page=indices[ptr[bi]+token//16];slot=token%16;pages[page,slot]=host_c[bi,token];rp[page,slot]=host_r[bi,token]
    cpu=[qa,qr,pages,rp];pairs=[guarded(x) for x in cpu];gpu=[x[1] for x in pairs];plan=MLAPlan(lengths=lengths,indptr=ptr,indices=indices,num_pages=5,page_size=16,heads=heads,scale=192**-.5,device='cuda')
    stream=torch.cuda.Stream();stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        torch.cuda._sleep(50000)
        for x,y in zip(gpu,cpu):x.copy_(y)
        z,lse=plan.run(*gpu);consumer=z.clone().cpu();lse_cpu=lse.clone().cpu()
    stream.synchronize();expected,expected_lse=ref.mla_core(qa,qr,host_c,host_r,lengths,192**-.5);error=compare(consumer,expected);assert torch.equal(torch.isneginf(lse_cpu),torch.isneginf(expected_lse));lse_error=compare(lse_cpu[1:],expected_lse[1:]);assert all(guards(raw) and torch.equal(value.cpu(),host) for (raw,value),host in zip(pairs,cpu))
    uv=(torch.randn(heads,512,128)*.1).half();projected=torch.einsum('bhr,hrv->bhv',z.float(),uv.cuda().float());expected_projected=torch.einsum('bhr,hrv->bhv',expected,uv.double());projection_error=compare(projected,expected_projected)
    records.append({'case':'mla-offset8-stream-ragged-and-layer-projection','max_abs_error':error,'natural_lse_error':lse_error,'projection_max_abs_error':projection_error,'guards_and_inputs':True,'output_contract':True,'empty_zero_negative_inf':True})
    heads=128;qa=(torch.randn(2,heads,512)*.25).half();qr=(torch.randn(2,heads,64)*.25).half();c=(torch.randn(2,17,512)*.25).half();kr=(torch.randn(2,17,64)*.25).half();pages=torch.zeros(2,16,512,dtype=torch.float16);rp=torch.zeros(2,16,64,dtype=torch.float16);pages[0]=c[1,:16];pages[1,0]=c[1,16];rp[0]=kr[1,:16];rp[1,0]=kr[1,16]
    plan=SplitMLAPlan(lengths=[0,17],indptr=[0,0,2],indices=[0,1],num_pages=2,page_size=16,heads=heads,scale=192**-.5,device='cuda',splits=4);out,lse=plan.run(qa.cuda(),qr.cuda(),pages.cuda(),rp.cuda());expected,expected_lse=ref.mla_core(qa,qr,c,kr,[0,17],192**-.5);error=compare(out,expected);assert torch.equal(torch.isneginf(lse.cpu()),torch.isneginf(expected_lse));lse_error=compare(lse.cpu()[1:],expected_lse[1:]);records.append({'case':'mla-split4-heads128-grouped-native-merge','max_abs_error':error,'natural_lse_error':lse_error,'empty_shards':True,'cuda_algorithm_modified':False})
    print(json.dumps({'status':'PASS','records':records,'performance_sampling':False,'new_model_calls':0}))


if __name__=='__main__':main()
