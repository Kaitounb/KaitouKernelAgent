"""Public CUDA Graph correctness probe, no Event timing or model calls."""
import argparse,json,subprocess,sys,traceback
from pathlib import Path
import torch
from kernel_agent.operators.flashinfer_backend import SplitMLAPlan
from kernel_agent.operators.references import mla_core


def child(splits):
    torch.set_num_threads(1);torch.manual_seed(775);h=128;l=2048;qa=(torch.randn(1,h,512)*.25).half();qr=(torch.randn(1,h,64)*.25).half();c=(torch.randn(1,l,512)*.25).half();kr=(torch.randn(1,l,64)*.25).half();expected,lse=mla_core(qa,qr,c,kr,[l],192**-.5);gpu=[x.cuda() for x in [qa,qr,c.view(-1,16,512),kr.view(-1,16,64)]];plan=SplitMLAPlan(lengths=[l],indptr=[0,l//16],indices=list(range(l//16)),num_pages=l//16,page_size=16,heads=h,scale=192**-.5,device=gpu[0].device,splits=splits);stream=torch.cuda.Stream();stream.wait_stream(torch.cuda.current_stream())
    with torch.no_grad(),torch.cuda.stream(stream):
        for _ in range(3):plan.run(*gpu)
        stream.synchronize();graph=torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph,stream=stream):out,s=plan.run(*gpu)
        graph.replay();stream.synchronize();out=out.cpu().double();s=s.cpu().double();error=(out-expected).abs();assert bool((error<=2e-3+2e-3*expected.abs()).all()) and bool(((s-lse).abs()<=2e-3+2e-3*lse.abs()).all())
    print(json.dumps({'status':'PASS','splits':splits,'max_abs_error':float(error.max()),'graph_replay_correct':True,'performance_sampling':False}))


def main():
    p=argparse.ArgumentParser();p.add_argument('--work',type=Path);p.add_argument('--splits',type=int);a=p.parse_args()
    if a.splits:
        try:child(a.splits)
        except Exception:traceback.print_exc();raise SystemExit(1)
        return
    a.work.mkdir(exist_ok=False);rows=[]
    for splits in [1,2,4]:
        r=subprocess.run([sys.executable,__file__,'--splits',str(splits)],capture_output=True,text=True,timeout=180);rows.append({'splits':splits,'exit':r.returncode,'stdout':r.stdout,'stderr':r.stderr})
    result={'status':'PASS' if all(r['exit']==0 for r in rows) else 'FAIL','rows':rows,'performance_sampling':False,'new_model_calls':0};(a.work/'result.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result))


if __name__=='__main__':main()
