"""Original paired timed path; reference-free output/input witnesses validated only by host."""
import math,time
import torch
from fixtures import make_inputs
from source_worker import tensor_hash


def paired_witness(incumbent,candidate,shape,count,session):
    """Adapted from the archived two-Graph A/A loop; returns no device objects."""
    cpu=[make_inputs(shape,701+i) for i in range(count)]
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
                for _ in range(12):graphs[label].replay()
                end.record(stream);end.synchronize();elapsed=start.elapsed_time(end)
                if not math.isfinite(elapsed) or elapsed<=0:raise RuntimeError('invalid Event segment')
                segments.append({'pair':pair,'label':label,'order':position,'elapsed_ms':elapsed})
        output_hashes={label:[tensor_hash(out) for out in outputs[label]] for label in capture_order}
        input_hashes=[[tensor_hash(x) for x in gpu] for gpu in ring]
    return {'shape_mnk':list(shape),'input_count':count,'capture_order':capture_order,'segments':segments,
            'output_hashes':output_hashes,'input_hashes':input_hashes,'witnesses_ready':True}
