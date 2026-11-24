"""Bind independently audited outcomes to mini-trial attempts without remeasurement."""
import argparse,hashlib,json
from pathlib import Path
from kernel_agent.registry import atomic_json
ROOT=Path(__file__).resolve().parents[2]


def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def finish(folder,status,reason,feedback,evidence,*,reuse=True,notes=None):
    if (folder/'decision.json').exists():raise ValueError('terminal attempt cannot be overwritten')
    from kernel_agent.optimization.feedback_channels import execution_feedback
    execution_feedback(feedback);record={'status':status,'reason':reason,'execution_feedback':feedback,'evidence':{str(p.relative_to(ROOT)):sha(p) for p in evidence},'exact_result_reused':reuse,'notes':notes,'promotion_performed':False,'production_release':False};atomic_json(folder/'decision.json',record);return record


def resolve(root,op,method,step,evidence=None):
    folder=root/op/method/str(step)
    if (folder/'decision.json').exists():return json.loads((folder/'decision.json').read_text())
    result=json.loads((folder/'result.json').read_text());proposal=json.loads((folder/'proposal.json').read_text());cid=proposal['candidate_id'];feedback={'candidate_id':cid,'compile_status':'PASS','compile_error':None,'correctness':'PASS','latency_ms':[],'evaluation_wall_seconds':0.};refs=[];status='REJECT';reason=None
    if op=='mla':
        config=proposal['config'];config_source={'page_size':16,'splits':2};path=ROOT/'evidence/20261123-mla-live-split2-sm86' if config==config_source else (Path(evidence) if evidence else None)
        if config=={'page_size':16,'splits':1}:
            # Exact starting implementation, not a new performance experiment.
            path=ROOT/'evidence/20261123-mla-live-split2-sm86';status='DUPLICATE';reason='unchanged_initial_sdk_configuration';role='incumbent';mode='ab'
        elif path is not None:role='candidate';mode='ab' if (path/'remote/ab-0.json').exists() else 'aa'
        else:return {'status':'PENDING_NEW_EXACT_CONFIG_EVALUATION','config':config}
        verification=json.loads((path/'verification.json').read_text());assert verification['status']=='PASS';freeze=json.loads((path/'remote/freeze.json').read_text())
        if status!='DUPLICATE':
            assert freeze['config']==config;reason=verification['decision']['reason'];status='SCREENING_PASS' if verification['decision']['verdict']=='KEEP' else 'REJECT'
        task=json.loads((root/'mla-task.json').read_text());source=(path/'remote/kernel_agent/operators/flashinfer_backend.py').read_text()+'\n'+(path/'remote/kernel_agent/operators/partition.py').read_text();assert hashlib.sha256(source.encode()).hexdigest()==task['source_sha256'];report=json.loads((path/f'remote/{mode}-0.json').read_text());row=next(x for x in report['rows'] if x['workload']=='mla-dev-04' and x['input_count']==1);feedback['latency_ms']=[x['elapsed_ms']/1024 for x in row['segments'] if x['label']==role];refs=[path/'verification.json',path/'remote/freeze.json',path/f'remote/{mode}-0.json']
    elif method=='A':
        warps=proposal['config']['warps']
        if warps==1:
            path=ROOT/'evidence/20261123-mini-gemm-A-warp1/archive/remote';value=json.loads((path/'result.json').read_text());assert value['warp_control']==1 and value['sources_before']==value['sources_after'] and value['decision']['reason']=='candidate_aa_unstable';report=json.loads((path/'aa-0.json').read_text());assert report['candidate']['warps']==1 and report['status']=='PASS' and len(report['checks_before'])==len(report['checks_after'])==28 and report['harness_before']==report['harness_after'];row=next(x for x in report['rows'] if x['workload']=='dev-01' and x['input_count']==1);feedback['latency_ms']=[x['elapsed_ms']/384 for x in row['segments'] if x['label']=='candidate'];refs=[path/'result.json',path/'aa-0.json'];reason='candidate_aa_unstable'
        elif warps==2:
            path=ROOT/'evidence/20261122-p1-04-feedback/evaluation/decision.json';value=json.loads(path.read_text());assert value['verdict']=='REJECT' and value['reason']=='insufficient_gain_or_regression';report_path=path.parent/'run/session-0.json';report=json.loads(report_path.read_text());assert report['candidate']['warps']==2;row=next(x for x in report['measurements'] if x['workload']=='dev-01' and x['input_count']==1);feedback['latency_ms']=[x['elapsed_ms']/384 for x in row['segments'] if x['label']=='candidate'];refs=[path,report_path];reason=value['reason']
        elif warps==8:
            from kernel_agent.optimization.identity import identity
            path=ROOT/'evidence/20261121-p1-03-deepseek/evaluation-retry1/decision.json';v=json.loads(path.read_text());report_path=path.parent/'run/session-0.json';report=json.loads(report_path.read_text());assert report['candidate']==identity(8) and v['candidate_id']==identity(8)['id'] and v['verdict']=='REJECT' and v['detail']=='candidate A/A unstable';refs=[path,report_path];reason='previous_warp8_rejected_no_resample'
        else:
            status='DUPLICATE';reason='unchanged_initial_4warp_configuration';refs=[ROOT/'workloads/gemm_measurement_v1.json',ROOT/'evidence/20261122-p1-05-final-long/REPORT.md']
    else:return {'status':'NEEDS_SOURCE_REVIEW_AND_EXACT_GPU_OUTCOME','source_sha256':proposal.get('source_sha256')}
    return finish(folder,status,reason,feedback,refs,notes='Archived timing reused with zero marginal GPU evaluation. Original phase cost remains in cited evidence; this is not a fresh independent performance sample.')


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--op',choices=['gemm','mla'],required=True);p.add_argument('--method',choices=['A','B','C'],required=True);p.add_argument('--step',type=int,required=True);p.add_argument('--evidence',type=Path);a=p.parse_args();print(json.dumps(resolve(a.root,a.op,a.method,a.step,a.evidence)))
