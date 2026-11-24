"""Record a reviewed source's exact audited outcome in one method's private history."""
import argparse,json
from pathlib import Path
from experiments.full_delivery.mini_results import finish
ROOT=Path(__file__).resolve().parents[2]


def record(root,method,step,evidence):
    folder=root/'gemm'/method/str(step);proposal=json.loads((folder/'proposal.json').read_text());verification=json.loads((evidence/'verification.json').read_text());assert verification['status']=='PASS' and verification['source_sha256']==proposal['source_sha256'];run=evidence/'remote/run';state=json.loads((run/'run.json').read_text());assert state['source_sha256']==proposal['source_sha256'];status='SCREENING_PASS' if state['verdict']=='KEEP' else 'REJECT';work=run/'evaluation';mode='ab' if (work/'ab-0/report.json').exists() else 'aa';report=json.loads((work/(mode+'-0')/'report.json').read_text());rows=report['measurements' if mode=='ab' else 'candidate_aa'];row=next(x for x in rows if x['workload']=='dev-01' and x['input_count']==1);feedback={'candidate_id':proposal['candidate_id'],'compile_status':'PASS','compile_error':None,'correctness':'PASS','latency_ms':[x['elapsed_ms']/384 for x in row['segments'] if x['label']=='candidate'],'evaluation_wall_seconds':0.};return finish(folder,status,state['reason'],feedback,[evidence/'verification.json',run/'run.json',work/'result.json'],notes={'physical_candidate_id':verification['candidate_id'],'trial_proposal_id':proposal['candidate_id'],'shared_exact_source_outcome':True,'original_build_wall_seconds':verification['build_wall_seconds'],'original_all_worker_wall_seconds':verification['all_worker_wall_seconds'],'marginal_reuse_wall_seconds':0.})


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--method',choices=['B','C'],required=True);p.add_argument('--step',type=int,required=True);p.add_argument('--evidence',type=Path,required=True);a=p.parse_args();print(json.dumps(record(a.root,a.method,a.step,a.evidence.resolve())))
