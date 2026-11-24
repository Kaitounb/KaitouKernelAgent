"""Trusted SDK configuration evaluator using the original two-Graph Event ordering.

Only page/split integers enter from candidates. No model-generated host/native source
is executed in this controller; free GEMM source still uses the bounded Docker path.
"""
import argparse,hashlib,json,math,os,subprocess,sys,time,traceback
from pathlib import Path
import torch
from kernel_agent.operators.flashinfer_backend import SplitMLAPlan
from kernel_agent.operators.references import mla_core
from kernel_agent.optimization.identity import group_log_ratio


def tensor_hash(x):return hashlib.sha256(x.detach().cpu().contiguous().numpy().tobytes()).hexdigest()


def native_hashes():
    base=Path(os.environ.get('NKA_FROZEN_SDK_PREFIX') or os.environ['FLASHINFER_WORKSPACE_BASE']).resolve()
    return {str(Path(p).resolve().relative_to(base)):hashlib.sha256(Path(p).read_bytes()).hexdigest() for p in torch.ops.loaded_libraries if Path(p).resolve().is_relative_to(base)}


def compare(actual,expected):
    actual=actual.detach().cpu().double();finite=torch.isfinite(expected);assert torch.equal(torch.isneginf(actual),torch.isneginf(expected));delta=(actual[finite]-expected[finite]).abs();assert bool((delta<=2e-3+2e-3*expected[finite].abs()).all());return float(delta.max()) if delta.numel() else 0.


def paired(workload,count,session,mode,config):
    started_at=time.time();lengths=workload['lengths'];b=len(lengths);h=workload['heads'];maximum=max(lengths);cpu=[];expected=[]
    for slot in range(count):
        generator=torch.Generator().manual_seed(701+slot);values=[(torch.randn(shape,generator=generator)*.25).clamp_(-4,4).half() for shape in [(b,h,512),(b,h,64),(b,maximum,512),(b,maximum,64)]];cpu.append(values);expected.append(mla_core(*values,lengths,192**-.5))
    configs={'incumbent':config if mode=='aa' else {'page_size':16,'splits':1},'candidate':config};order=['incumbent','candidate'] if session%2==0 else ['candidate','incumbent'];stream=torch.cuda.Stream();stream.wait_stream(torch.cuda.current_stream());plans={};rings={};storage={};input_before={};graphs={};outputs={};qa_errors={}
    with torch.no_grad(),torch.cuda.stream(stream):
        for role in order:
            page=configs[role]['page_size'];ptr=[0]
            for n in lengths:ptr.append(ptr[-1]+(n+page-1)//page)
            indices=torch.randperm(ptr[-1],generator=torch.Generator().manual_seed(811)).tolist();plans[role]=SplitMLAPlan(lengths=lengths,indptr=ptr,indices=indices,num_pages=ptr[-1],page_size=page,heads=h,scale=192**-.5,device='cuda',splits=configs[role]['splits']);rings[role]=[];storage[role]=[];qa_errors[role]=[]
            for slot,(qa,qr,c,kr) in enumerate(cpu):
                pages=torch.zeros(ptr[-1],page,512,dtype=torch.float16);rp=torch.zeros(ptr[-1],page,64,dtype=torch.float16)
                for bi,n in enumerate(lengths):
                    for token in range(n):physical=indices[ptr[bi]+token//page];offset=token%page;pages[physical,offset]=c[bi,token];rp[physical,offset]=kr[bi,token]
                values=[]
                for x in [qa,qr,pages,rp]:
                    raw=torch.full((x.numel()+16,),7,dtype=torch.float16,device='cuda');value=raw[8:-8].view(x.shape);value.copy_(x);values.append(value);storage[role].append(raw)
                rings[role].append(values);out,lse=plans[role].run(*values);qa_errors[role].append([compare(out,expected[slot][0]),compare(lse,expected[slot][1])])
            input_before[role]=[[tensor_hash(x) for x in values] for values in rings[role]]
        for role in order:
            graphs[role]=torch.cuda.CUDAGraph();outputs[role]=[]
            with torch.cuda.graph(graphs[role],stream=stream):
                for i in range(32):outputs[role].append(plans[role].run(*rings[role][i%count]))
            graphs[role].replay();stream.synchronize()
            for i,(z,lse) in enumerate(outputs[role]):compare(z,expected[i%count][0]);compare(lse,expected[i%count][1])
        binaries=native_hashes()
        if not binaries:raise RuntimeError('native SDK identity witness missing')
        deadline=time.monotonic()+1.
        while time.monotonic()<deadline:
            for role in order:graphs[role].replay()
            stream.synchronize()
        start,end=[torch.cuda.Event(enable_timing=True) for _ in range(2)];segments=[]
        for pair in range(9):
            sequence=['incumbent','candidate'] if (pair+session)%2==0 else ['candidate','incumbent']
            for position,role in enumerate(sequence):
                start.record(stream)
                for _ in range(32):graphs[role].replay()
                end.record(stream);end.synchronize();elapsed=start.elapsed_time(end)
                if not math.isfinite(elapsed) or elapsed<=0:raise RuntimeError('invalid MLA Event segment')
                segments.append({'pair':pair,'label':role,'order':position,'elapsed_ms':elapsed,'finished_at':time.time()})
        hashes={}
        for role in order:
            hashes[role]=[]
            for i,(z,lse) in enumerate(outputs[role]):compare(z,expected[i%count][0]);compare(lse,expected[i%count][1]);hashes[role].append([tensor_hash(z),tensor_hash(lse)])
            assert [[tensor_hash(x) for x in values] for values in rings[role]]==input_before[role]
            assert all(bool((raw[:8]==7).all()) and bool((raw[-8:]==7).all()) for raw in storage[role])
    assert binaries==native_hashes()
    return {'workload':workload['id'],'input_count':count,'configs':configs,'capture_order':order,'segments':segments,'input_hashes':input_before,'output_hashes':hashes,'qa_before_errors':qa_errors,'qa_after_pass':True,'inputs_and_guards_unchanged':True,'native_hashes':binaries,'started_at':started_at,'finished_at':time.time(),'reference_scope':'trusted controller; no free candidate source','peak_allocated_bytes':torch.cuda.max_memory_allocated()}


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--session',type=int,required=True);p.add_argument('--mode',choices=['aa','ab'],required=True);p.add_argument('--page-size',type=int,choices=[1,16,64],default=16);p.add_argument('--splits',type=int,choices=[1,2,4],default=1);p.add_argument('--anchors-only',action='store_true');a=p.parse_args();torch.set_num_threads(1);torch.backends.cuda.matmul.allow_tf32=False;workloads=json.loads((a.root/'workloads/mla_development_v1.json').read_text());rows=[];status='PASS';failure=None
    for w in workloads['workloads']:
        if a.anchors_only and w['id'] not in workloads['anchors']:continue
        for count in [1,16]:
            try:rows.append(paired(w,count,a.session,a.mode,{'page_size':a.page_size,'splits':a.splits}))
            except Exception:status='FAIL';failure=traceback.format_exc();break
            (a.root/f'{a.mode}-{a.session}.progress.json').write_text(json.dumps({'rows':rows},indent=2)+'\n')
        if status!='PASS':break
    result={'status':status,'mode':a.mode,'session':a.session,'config':{'page_size':a.page_size,'splits':a.splits},'rows':rows,'failure':failure,'graph_launches':32,'replays':32,'torch':torch.__version__,'cuda':torch.version.cuda,'capability':torch.cuda.get_device_capability(),'candidate_space':'reviewed SDK configuration only','performance_claim':False};(a.root/f'{a.mode}-{a.session}.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps({'status':status,'mode':a.mode,'session':a.session,'rows':len(rows),'failure':failure}))


if __name__=='__main__':main()
