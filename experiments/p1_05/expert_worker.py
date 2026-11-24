"""cuBLAS engineering comparison using the unchanged P1-03 pairing/check helpers."""
import argparse
import importlib.util
import json
from pathlib import Path
import traceback
import torch

from evaluators.gemm_candidate import paired,checks
from evaluators.gemm_decision import ROOT,identity,protected_hashes,sha


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--binary',type=Path,required=True);p.add_argument('--sha256',required=True)
    p.add_argument('--session',type=int,choices=[0,1,2],required=True);p.add_argument('--output',type=Path,required=True)
    args=p.parse_args()
    if args.output.exists():p.error('Use a new output')
    report={'status':'FAIL','evidence_mode':'GPU','session':args.session,'comparison':'cuBLAS_latency / frozen_4warps_latency',
            'scope':'separate engineering comparison, not agent gain','candidate':identity(4),'graph_launches':32,'replays':12,
            'worker_sha256':sha(Path(__file__)),'harness_before':protected_hashes(),'formal_performance_claim':False}
    try:
        if sha(args.binary)!=args.sha256:raise ValueError('trusted binary mismatch')
        spec=importlib.util.spec_from_file_location('nka_gemm_v1',args.binary.resolve());module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
        torch.set_num_threads(1)
        current=lambda a,b:module.gemm_template(a,b,4)
        expert=module.gemm_cublas
        report['binary']={'path':str(args.binary),'sha256':args.sha256}
        report['baseline']={'entry':'gemm_cublas','source_sha256':sha(ROOT/'kernels/gemm/cublas_sm86.cu'),
                            'note':'cuBLAS entry in the SAME combined trusted .so; not the old standalone .so'}
        report['environment']={'device':torch.cuda.get_device_name(0),'capability':list(torch.cuda.get_device_capability(0)),'torch':torch.__version__,'cuda':torch.version.cuda}
        development=json.loads((ROOT/'workloads/gemm_development_v1.json').read_text())['workloads']
        report['checks_before']={name:checks(fn,development) for name,fn in [('cublas',expert),('best',current)]}
        report['measurements']=[]
        for w in development:
            for count in (1,16):
                row=paired(expert,current,w['shape_mnk'],count,args.session);row['workload']=w['id'];report['measurements'].append(row)
        report['checks_after']={name:checks(fn,development) for name,fn in [('cublas',expert),('best',current)]}
        report['status']='PASS'
    except Exception:
        report['error']=traceback.format_exc();raise
    finally:
        report['harness_after']=protected_hashes()
        args.output.write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')


if __name__=='__main__':main()
