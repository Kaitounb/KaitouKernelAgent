"""New reviewed warp-1 control, reusing the original frozen GEMM numerical/Event code."""
import argparse,json,traceback
from pathlib import Path
import torch
from evaluators.gemm_candidate import checks,paired
from evaluators.gemm_decision import identity,protected_hashes,sha,ROOT
from kernel_agent.operators.measurement import assess


def child(root,mode,session,warps):
    import importlib.util
    torch.set_num_threads(1);binary=Path('/tmp/nka-gemm-rel01-500z3apq/run/prefix/kernel_agent/data/artifacts/93fb2f4344521b7e9b246b54d1c30d4b1a8facf731e6385e18f6119e9c49c72c.so');assert sha(binary)==binary.stem;spec=importlib.util.spec_from_file_location('nka_gemm_v1',binary);module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module);development=json.loads((ROOT/'workloads/gemm_development_v1.json').read_text());candidate=lambda a,b:module.gemm_template(a,b,warps);incumbent=candidate if mode=='aa' else lambda a,b:module.gemm_template(a,b,4);report={'status':'FAIL','session':session,'mode':mode,'parent':identity(4),'candidate':identity(warps),'protocol_sha256':sha(ROOT/'workloads/gemm_measurement_v1.json'),'harness_before':protected_hashes(),'binary':{'sha256':binary.stem},'environment':{'torch':torch.__version__,'cuda':torch.version.cuda,'capability':list(torch.cuda.get_device_capability())},'graph_launches':32,'replays':12,'evidence_mode':'GPU','rows':[]}
    try:
        assert report['environment']=={'torch':'2.14.0+cu130','cuda':'13.0','capability':[8,6]};report['checks_before']=checks(candidate,development['workloads']);report['correctness_before_pass']=True
        for w in development['workloads']:
            for count in [1,16]:report['rows'].append({**paired(incumbent,candidate,w['shape_mnk'],count,session),'workload':w['id']})
        report['checks_after']=checks(candidate,development['workloads']);report['correctness_after_pass']=True;report['status']='PASS'
    except Exception:report['error']=traceback.format_exc()
    finally:report['harness_after']=protected_hashes();(root/f'{mode}-{session}.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({'status':report['status'],'mode':mode,'session':session,'rows':len(report['rows'])}));return report['status']=='PASS'


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--mode',choices=['aa','ab']);p.add_argument('--session',type=int,choices=[0,1,2]);p.add_argument('--warps',type=int,choices=[1],default=1);p.add_argument('--decide',action='store_true');a=p.parse_args()
    if a.decide:
        d=json.loads((ROOT/'workloads/gemm_development_v1.json').read_text());aa=[json.loads((a.root/f'aa-{i}.json').read_text()) for i in range(3)];ab=[json.loads(x.read_text()) for i in range(3) if (x:=a.root/f'ab-{i}.json').exists()];decision=assess(aa,ab,d['workloads'],d['anchors']);(a.root/'decision.json').write_text(json.dumps(decision,indent=2)+'\n');print(json.dumps(decision))
    elif not child(a.root,a.mode,a.session,a.warps):raise SystemExit(1)
