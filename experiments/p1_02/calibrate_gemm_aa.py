"""Trusted two-Graph GEMM A/A calibration over development anchors/cache policies."""
import argparse
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import statistics
import time

import torch

ANCHORS = [(512,512,512),(1,4096,512),(127,1009,513)]


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--binary',type=Path,required=True)
    ap.add_argument('--sha256',required=True)
    ap.add_argument('--kind',choices=['cublas','template'],default='cublas')
    ap.add_argument('--session',type=int,required=True)
    ap.add_argument('--output',type=Path,required=True)
    args = ap.parse_args()
    if args.output.exists() or args.session < 0:
        ap.error('Use new output and nonnegative session')
    binary = args.binary.resolve()
    if hashlib.sha256(binary.read_bytes()).hexdigest() != args.sha256:
        raise ValueError('trusted binary hash mismatch')
    module_name = 'nka_cublas_gemm' if args.kind == 'cublas' else 'nka_gemm_v1'
    spec = importlib.util.spec_from_file_location(module_name,binary)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    function = module.gemm_cublas if args.kind == 'cublas' else lambda a,b: module.gemm_template(a,b,4)
    torch.set_num_threads(1)
    report = {'status':'PASS','session':args.session,'binary_sha256':args.sha256,
              'kind':args.kind,'device':torch.cuda.get_device_name(0),'scope':'same trusted binary, two independent graphs; no profiler',
              'graph_launches':32,'replays':12,'checks':[]}
    for shape in ANCHORS:
        m,n,k = shape
        for count in (1,16):
            g = torch.Generator().manual_seed(701)
            cpu = [(torch.randn(m,k,generator=g).mul_(.25).half(),torch.randn(k,n,generator=g).mul_(.25).half()) for _ in range(count)]
            expected = [(a.double() @ b.double()).float() for a,b in cpu]
            stream = torch.cuda.Stream()
            with torch.no_grad(),torch.cuda.stream(stream):
                ring = [(a.cuda(),b.cuda()) for a,b in cpu]
                for a,b in ring:
                    function(a,b)
                stream.synchronize()
                graphs,outputs = {},{}
                for label in ('A','A_prime'):
                    graphs[label] = torch.cuda.CUDAGraph()
                    outputs[label] = []
                    with torch.cuda.graph(graphs[label],stream=stream):
                        for i in range(32):
                            outputs[label].append(function(*ring[i % count]))
                deadline = time.monotonic()+1.0
                while time.monotonic() < deadline:
                    for graph in graphs.values():
                        graph.replay()
                    stream.synchronize()
                segments = []
                start,end = [torch.cuda.Event(enable_timing=True) for _ in range(2)]
                for pair in range(9):
                    order = ('A','A_prime') if (pair+args.session)%2==0 else ('A_prime','A')
                    for position,label in enumerate(order):
                        start.record(stream)
                        for _ in range(12):
                            graphs[label].replay()
                        end.record(stream);end.synchronize()
                        elapsed = start.elapsed_time(end)
                        if not math.isfinite(elapsed) or elapsed <= 0:
                            raise RuntimeError('invalid CUDA Event')
                        segments.append({'pair':pair,'order':position,'label':label,'elapsed_ms':elapsed})
                for values in outputs.values():
                    for i,value in enumerate(values):
                        torch.testing.assert_close(value.cpu(),expected[i % count],rtol=1e-4,atol=3e-5)
                for values,original in zip(ring,cpu):
                    for x,y in zip(values,original):
                        torch.testing.assert_close(x.cpu(),y,rtol=0,atol=0)
            ratios=[]
            for i in range(9):
                values={r['label']:r['elapsed_ms'] for r in segments if r['pair']==i}
                ratios.append(values['A']/values['A_prime'])
            logs=[math.log(x) for x in ratios]
            mean=statistics.mean(logs);sem=statistics.stdev(logs)/math.sqrt(9)
            report['checks'].append({'shape_mnk':list(shape),'input_count':count,'cache':'repeated' if count==1 else 'input ring',
                'input_bytes':count*(m*k+k*n)*2,'segments':segments,'ratios':ratios,'geomean':math.exp(mean),
                'max_pair_deviation':max(abs(x-1) for x in ratios),'exploratory_t8_95_interval':[math.exp(mean-2.306*sem),math.exp(mean+2.306*sem)],
                'median_segment_ms':statistics.median(x['elapsed_ms'] for x in segments),'cpu_reference_and_inputs_pass':True})
            # Captured graphs and private allocation pools must be released between cases.
            del outputs,graphs,ring
    with args.output.open('x') as file:
        json.dump(report,file,indent=2,allow_nan=False);file.write('\n')


if __name__ == '__main__':
    main()
