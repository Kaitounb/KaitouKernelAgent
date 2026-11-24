"""Trusted fixed-SDK parent profile, matching benchmark page permutation/alignment."""
import argparse,hashlib,json
from pathlib import Path
import torch
from kernel_agent.operators.flashinfer_backend import MLAPlan
from kernel_agent.operators.references import mla_core


def th(x):return hashlib.sha256(x.detach().cpu().contiguous().numpy().tobytes()).hexdigest()


def main():
    p=argparse.ArgumentParser();p.add_argument('--request',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();cfg=json.loads(a.request.read_text());w=cfg['workload'];lengths=w['lengths'];h=w['heads'];b=len(lengths);maximum=max(lengths);torch.set_num_threads(1);g=torch.Generator().manual_seed(171);qa,qr,c,kr=[(torch.randn(shape,generator=g)*.25).clamp_(-4,4).half() for shape in [(b,h,512),(b,h,64),(b,maximum,512),(b,maximum,64)]];expected,lse=mla_core(qa,qr,c,kr,lengths,192**-.5);ptr=[0]
    for n in lengths:ptr.append(ptr[-1]+(n+15)//16)
    indices=torch.randperm(ptr[-1],generator=torch.Generator().manual_seed(811)).tolist();cache=torch.zeros(ptr[-1],16,512,dtype=torch.float16);rope=torch.zeros(ptr[-1],16,64,dtype=torch.float16)
    for i,n in enumerate(lengths):
        for t in range(n):physical=indices[ptr[i]+t//16];cache[physical,t%16]=c[i,t];rope[physical,t%16]=kr[i,t]
    cpu=[qa,qr,cache,rope];stream=torch.cuda.Stream();stream.wait_stream(torch.cuda.current_stream());gpu=[];guards=[];outs=[]
    with torch.no_grad(),torch.cuda.stream(stream):
        for x in cpu:
            raw=torch.full((x.numel()+16,),7,dtype=torch.float16,device='cuda');value=raw[8:-8].view(x.shape);value.copy_(x);gpu.append(value);guards.append(raw)
        plan=MLAPlan(lengths=lengths,indptr=ptr,indices=indices,num_pages=ptr[-1],page_size=16,heads=h,scale=192**-.5,device='cuda')
        for _ in range(10):plan.run(*gpu)
        stream.synchronize();native={str(p):hashlib.sha256(Path(p).read_bytes()).hexdigest() for p in torch.ops.loaded_libraries if str(p).endswith('.so') and 'batch_mla_attention_' in str(p)};assert len(native)==1;torch.cuda.profiler.start()
        try:
            for _ in range(3):outs.append(plan.run(*gpu))
            stream.synchronize()
        finally:torch.cuda.profiler.stop()
        errors=[]
        for z,s in outs:
            z=z.cpu().double();s=s.cpu().double();delta=(z-expected).abs();assert bool((delta<=2e-3+2e-3*expected.abs()).all()) and bool(((s-lse).abs()<=2e-3+2e-3*lse.abs()).all());errors.append(float(delta.max()))
        assert [th(x) for x in gpu]==[th(x) for x in cpu] and all(bool((x[:8]==7).all()) and bool((x[-8:]==7).all()) for x in guards)
    a.output.write_text(json.dumps({'status':'PASS','workload':w,'target_launches':3,'native_sha256':native,'max_abs_error':max(errors),'inputs_and_guards_unchanged':True,'page_size':16,'splits':1,'page_permutation_seed':811,'offset_elements':8,'profile_scope':'native fa2 only, not full wrapper/Graph latency','performance_claim':False},indent=2)+'\n')


if __name__=='__main__':main()
