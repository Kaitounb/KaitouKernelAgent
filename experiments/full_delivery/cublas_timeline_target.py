"""Finite trusted cuBLAS scheduling diagnostic, never a qualification run."""
import argparse,json,math,sys,time
from pathlib import Path

def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);a=p.parse_args();root=a.root;prefix=Path('/tmp/nka-operators-dev6-izkosqud/prefix');sys.path[:0]=[str(prefix/'kernel_agent/optimization'),str(prefix/'kernel_agent'),str(prefix)]
    import torch
    from fixtures import make_inputs
    from source_worker import load
    torch.set_num_threads(1);digest='66fcc80f9122c919f92c72e37a01794a792c91a58b71d5705b160e0bc979dfc3';module=load(prefix/'kernel_agent/data/artifacts'/(digest+'.so'),digest);model=module.gemm_cublas;rows=[];stream=torch.cuda.Stream();torch.cuda.cudart().cudaProfilerStart()
    with torch.no_grad(),torch.cuda.stream(stream):
        for workload,shape in [('dev-03',[1,4096,512]),('dev-04',[8,1024,1024]),('dev-06',[128,2048,512])]:
            ring=[tuple(t.cuda() for t in make_inputs(shape,701+i)) for i in range(16)]
            for x,y in ring:model(x,y)
            stream.synchronize();graphs={};outputs={}
            for label in ['A','B']:
                graphs[label]=torch.cuda.CUDAGraph();outputs[label]=[]
                with torch.cuda.graph(graphs[label],stream=stream):
                    for i in range(32):outputs[label].append(model(*ring[i%16]))
            start,end=[torch.cuda.Event(enable_timing=True) for _ in range(2)]
            for repetition in range(3):
                deadline=time.monotonic()+1
                while time.monotonic()<deadline:
                    graphs['A'].replay();graphs['B'].replay();stream.synchronize()
                pilot=[]
                for _ in range(3):
                    start.record(stream)
                    for _ in range(4):graphs['A'].replay()
                    end.record(stream);end.synchronize();pilot.append(start.elapsed_time(end))
                replays=max(1,min(2048,math.ceil(1000/sorted(pilot)[1])))
                for pair in range(9):
                    for label in (['A','B'] if (pair+repetition)%2==0 else ['B','A']):
                        name=f'{workload}/repeat{repetition}/pair{pair}/{label}';torch.cuda.nvtx.range_push(name);before=time.perf_counter();start.record(stream)
                        for _ in range(replays):graphs[label].replay()
                        end.record(stream);submitted=time.perf_counter();end.synchronize();finished=time.perf_counter();torch.cuda.nvtx.range_pop()
                        rows.append({'range':name,'workload':workload,'repetition':repetition,'pair':pair,'label':label,'replays':replays,'gpu_event_ms':start.elapsed_time(end),'cpu_enqueue_ms':(submitted-before)*1000,'cpu_wait_ms':(finished-submitted)*1000})
            del outputs,graphs,ring
    torch.cuda.synchronize();torch.cuda.cudart().cudaProfilerStop();(root/'diagnostic-segments.json').write_text(json.dumps({'status':'DIAGNOSTIC_COMPLETE','scope':'Profiled fixed trusted cuBLAS scheduling; not formal AA, no qualification or performance claim','rows':rows,'production_release':False},indent=2))
if __name__=='__main__':main()
