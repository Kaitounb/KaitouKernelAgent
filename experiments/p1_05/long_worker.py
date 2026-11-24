"""One frozen same-identity A/A collection with pre-sized long segments."""
import argparse,importlib.util,json,traceback
from pathlib import Path
import torch
from evaluators.gemm_decision import ROOT,identity,protected_hashes,sha
from evaluators.gemm_candidate import checks
from experiments.p1_05.pairing_long import paired_long


def main():
    p=argparse.ArgumentParser();p.add_argument('--binary',type=Path,required=True);p.add_argument('--plan',type=Path,required=True)
    p.add_argument('--session',type=int,choices=[0,1,2],required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args()
    if a.output.exists():p.error('Use a new output')
    plan=json.loads(a.plan.read_text());expected='93fb2f4344521b7e9b246b54d1c30d4b1a8facf731e6385e18f6119e9c49c72c'
    report={'status':'FAIL','mode':plan['mode'],'evidence_mode':'GPU','session':a.session,'plan_sha256':sha(a.plan),
            'initial':identity(4),'best':identity(4),'harness_before':protected_hashes(),
            'worker_sha256':sha(Path(__file__)),'pairing_sha256':sha(ROOT/'experiments/p1_05/pairing_long.py'),
            'same_identity_pair_is_aa_control':True,'graph_launches':32,'formal_performance_claim':False}
    try:
        if sha(a.binary)!=expected:raise ValueError('trusted artifact mismatch')
        spec=importlib.util.spec_from_file_location('nka_gemm_v1',a.binary.resolve());module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
        torch.set_num_threads(1);fn=lambda x,y:module.gemm_template(x,y,4)
        report['binary_sha256']=expected;report['environment']={'device':torch.cuda.get_device_name(0),'capability':list(torch.cuda.get_device_capability(0)),'torch':torch.__version__,'cuda':torch.version.cuda}
        dev=json.loads((ROOT/'workloads/gemm_development_v1.json').read_text())['workloads']
        report['checks_before']=checks(fn,dev);report['measurements']=[]
        for cfg in plan['rows']:
            row=paired_long(fn,fn,cfg['shape_mnk'],cfg['input_count'],a.session,cfg['replays']);row['workload']=cfg['workload'];report['measurements'].append(row)
        report['checks_after']=checks(fn,dev);report['status']='PASS'
    except Exception:
        report['error']=traceback.format_exc();raise
    finally:
        report['harness_after']=protected_hashes();a.output.write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')


if __name__=='__main__':main()
