"""Final-only parameterization of the existing P1-03 paired helper.
Only timer-loop change: expose replays instead of fixed 12. No new timing backend.
"""
import math,time
import torch
from evaluators.gemm_regression import make_inputs

def paired_long(incumbent,candidate,shape,count,session,replays):
    """Adapted from the archived two-Graph A/A loop; returns no device objects."""
    cpu=[make_inputs(shape,701+i) for i in range(count)]
    expected=[(a.double() @ b.double()).float() for a,b in cpu]
    stream=torch.cuda.Stream()
    functions={'incumbent':incumbent,'candidate':candidate}
    capture_order=['incumbent','candidate'] if session%2==0 else ['candidate','incumbent']
    with torch.no_grad(),torch.cuda.stream(stream):
        ring=[(a.cuda(),b.cuda()) for a,b in cpu]
        for model in functions.values():
            for a,b in ring:model(a,b)
        stream.synchronize()
        graphs={};outputs={}
        for label in capture_order:
            graphs[label]=torch.cuda.CUDAGraph();outputs[label]=[]
            with torch.cuda.graph(graphs[label],stream=stream):
                for i in range(32):outputs[label].append(functions[label](*ring[i%count]))
        deadline=time.monotonic()+1.0
        while time.monotonic()<deadline:
            for label in capture_order:graphs[label].replay()
            stream.synchronize()
        start,end=[torch.cuda.Event(enable_timing=True) for _ in range(2)]
        segments=[]
        for pair in range(9):
            order=['incumbent','candidate'] if (pair+session)%2==0 else ['candidate','incumbent']
            for position,label in enumerate(order):
                start.record(stream)
                for _ in range(replays):graphs[label].replay()
                end.record(stream);end.synchronize();elapsed=start.elapsed_time(end)
                if not math.isfinite(elapsed) or elapsed<=0:raise RuntimeError('invalid Event segment')
                segments.append({'pair':pair,'label':label,'order':position,'elapsed_ms':elapsed})
        for label in capture_order:
            for i,out in enumerate(outputs[label]):torch.testing.assert_close(out.cpu(),expected[i%count],atol=3e-5,rtol=1e-4)
        for gpu,original in zip(ring,cpu):
            for x,y in zip(gpu,original):torch.testing.assert_close(x.cpu(),y,atol=0,rtol=0)
    return {'shape_mnk':list(shape),'input_count':count,'capture_order':capture_order,'segments':segments,
            'cpu_reference_and_inputs_pass':True,'replays':replays,'calls_per_segment':32*replays}
