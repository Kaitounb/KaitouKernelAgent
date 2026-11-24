"""Reviewed SDK MLA native profile-only entry, no model source or Event scoring."""
import argparse,hashlib,json,os,shutil
from pathlib import Path
import torch
from kernel_agent.operators.flashinfer_backend import MLAPlan
from kernel_agent.operators.references import mla_core


def main():
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);a=p.parse_args();torch.set_num_threads(1);torch.manual_seed(171);heads=128;length=8192;page_size=16
    qa=(torch.randn(1,heads,512)*.25).half();qr=(torch.randn(1,heads,64)*.25).half();c=(torch.randn(1,length,512)*.25).half();kr=(torch.randn(1,length,64)*.25).half();expected,lse=mla_core(qa,qr,c,kr,[length],192**-.5);gpu=[x.cuda() for x in [qa,qr,c.view(-1,page_size,512),kr.view(-1,page_size,64)]];plan=MLAPlan(lengths=[length],indptr=[0,length//page_size],indices=list(range(length//page_size)),num_pages=length//page_size,page_size=page_size,heads=heads,scale=192**-.5,device=gpu[0].device)
    for _ in range(10):plan.run(*gpu)
    torch.cuda.synchronize();base=Path(os.environ['FLASHINFER_WORKSPACE_BASE']).resolve();native={str(Path(p).resolve().relative_to(base)):hashlib.sha256(Path(p).read_bytes()).hexdigest() for p in torch.ops.loaded_libraries if Path(p).resolve().is_relative_to(base)}
    if not native:raise RuntimeError('loaded native identity missing')
    outputs=[];torch.cuda.profiler.start()
    try:
        for _ in range(3):outputs.append(plan.run(*gpu))
        torch.cuda.synchronize()
    finally:torch.cuda.profiler.stop()
    errors=[]
    for z,s in outputs:
        z=z.cpu().double();s=s.cpu().double();delta=(z-expected).abs();assert torch.isfinite(z).all() and bool((delta<=2e-3+2e-3*expected.abs()).all());assert bool(((s-lse).abs()<=2e-3+2e-3*lse.abs()).all());errors.append(float(delta.max()))
    assert all(torch.equal(x.cpu(),y) for x,y in zip(gpu,[qa,qr,c.view(-1,page_size,512),kr.view(-1,page_size,64)]))
    for name,digest in native.items():
        source=base/name;assert hashlib.sha256(source.read_bytes()).hexdigest()==digest;target=a.output.parent/'native'/source.name;target.parent.mkdir(exist_ok=True);shutil.copyfile(source,target)
    a.output.write_text(json.dumps({'status':'PASS','target_launches':3,'public_workload':'mla-dev-04','max_abs_error':max(errors),'inputs_unchanged':True,'native_sha256':native,'scope':'profile-only native core; wrapper/combine Event cost is measured separately','performance_claim':False},indent=2)+'\n')


if __name__=='__main__':main()
