"""P1-03 trusted parameter worker; reuses the P1-02 CUDA binary and checks."""
import argparse
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import time
import traceback

import torch
from evaluators.gemm_decision import ROOT,BOUNDARIES,identity,protected_hashes,sha
from evaluators.gemm_regression import check_case,make_inputs,Fault




def checks(model,development):
    rows=[]
    cases=[(w['shape_mnk'],'random',False) for w in development]+BOUNDARIES
    for shape,distribution,offset in cases:
        for seed in (171,172):
            rows.append(check_case(model,shape,seed,distribution,offset))
    return rows


def paired(incumbent,candidate,shape,count,session):
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
                for _ in range(12):graphs[label].replay()
                end.record(stream);end.synchronize();elapsed=start.elapsed_time(end)
                if not math.isfinite(elapsed) or elapsed<=0:raise RuntimeError('invalid Event segment')
                segments.append({'pair':pair,'label':label,'order':position,'elapsed_ms':elapsed})
        for label in capture_order:
            for i,out in enumerate(outputs[label]):torch.testing.assert_close(out.cpu(),expected[i%count],atol=3e-5,rtol=1e-4)
        for gpu,original in zip(ring,cpu):
            for x,y in zip(gpu,original):torch.testing.assert_close(x.cpu(),y,atol=0,rtol=0)
    return {'shape_mnk':list(shape),'input_count':count,'capture_order':capture_order,'segments':segments,
            'cpu_reference_and_inputs_pass':True}


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--binary',type=Path,required=True);p.add_argument('--sha256',required=True)
    p.add_argument('--warps',type=int,choices=[1,2,4,8],required=True)
    p.add_argument('--incumbent-warps',type=int,choices=[1,2,4,8],default=4)
    p.add_argument('--session',type=int,choices=[0,1,2],required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--fault',choices=['constant','pollution','host-error','host-timeout'])
    args=p.parse_args()
    if args.output.exists():p.error('Use a new output')
    report={'status':'FAIL','evidence_mode':'GPU','session':args.session,'graph_launches':32,'replays':12,
            'parent':identity(args.incumbent_warps),'candidate':identity(args.warps),
            'protocol_sha256':sha(ROOT/'workloads/gemm_measurement_v1.json'),'harness_before':protected_hashes(),
            'fault_fixture':args.fault,'formal_performance_claim':False}
    try:
        if sha(args.binary)!=args.sha256:raise RuntimeError('trusted binary hash mismatch')
        spec=importlib.util.spec_from_file_location('nka_gemm_v1',args.binary.resolve())
        extension=importlib.util.module_from_spec(spec);spec.loader.exec_module(extension)
        torch.set_num_threads(1)
        incumbent=lambda a,b:extension.gemm_template(a,b,args.incumbent_warps)
        candidate=lambda a,b:extension.gemm_template(a,b,args.warps)
        report['binary']={'path':str(args.binary),'sha256':args.sha256}
        report['environment']={'device':torch.cuda.get_device_name(0),'capability':list(torch.cuda.get_device_capability(0)),'torch':torch.__version__,'cuda':torch.version.cuda}
        if args.fault=='host-timeout':
            check_case(candidate,(32,48,64),171)
            args.output.with_suffix('.ready.json').write_text(json.dumps({'finite_cuda_work_completed':True}))
            time.sleep(120)
            raise RuntimeError('timeout fixture was not stopped')
        if args.fault:candidate=Fault(args.fault,incumbent)
        development=json.loads((ROOT/'workloads/gemm_development_v1.json').read_text())['workloads']
        if args.fault:
            check_case(candidate,(17,19,23),171,offset=True)
            raise RuntimeError('fault unexpectedly passed')
        report['checks_before']=checks(candidate,development);report['correctness_before_pass']=True
        report['measurements']=[];report['candidate_aa']=[]
        for workload in development:
            for count in (1,16):
                row=paired(candidate,candidate,workload['shape_mnk'],count,args.session)
                row['workload']=workload['id'];report['candidate_aa'].append(row)
                row=paired(incumbent,candidate,workload['shape_mnk'],count,args.session)
                row['workload']=workload['id'];report['measurements'].append(row)
        report['checks_after']=checks(candidate,development);report['correctness_after_pass']=True
        report['status']='PASS'
    except Exception:
        report['error']=traceback.format_exc()
        raise
    finally:
        report['harness_after']=protected_hashes()
        args.output.write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')


if __name__=='__main__':main()
