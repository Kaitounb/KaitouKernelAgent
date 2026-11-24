"""Offline reconstruction of the bounded V0 trajectory and frozen final retest."""
import argparse,ast,hashlib,json
from pathlib import Path
from evaluators.gemm_decision import ROOT,identity,sha
from experiments.p1_05.summarize import summarize
from experiments.p1_03.replay import replay


def function_source(path,name):
    text=Path(path).read_text();tree=ast.parse(text)
    node=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name==name)
    return '\n'.join(text.splitlines()[node.lineno-1:node.end_lineno])


def audit(folder,long_folder):
    folder=Path(folder);long_folder=Path(long_folder)
    computed=summarize(folder,long_folder)
    if computed!=json.loads((folder/'summary.json').read_text()):raise ValueError('summary differs from raw data')
    original=function_source(ROOT/'evaluators/gemm_candidate.py','paired')
    expected=original.replace('def paired(incumbent,candidate,shape,count,session):','def paired_long(incumbent,candidate,shape,count,session,replays):').replace('for _ in range(12):','for _ in range(replays):')
    expected=expected.replace("'cpu_reference_and_inputs_pass':True}","'cpu_reference_and_inputs_pass':True,'replays':replays,'calls_per_segment':32*replays}")
    if function_source(ROOT/'experiments/p1_05/pairing_long.py','paired_long')!=expected:raise ValueError('long pairing changed more than declared parameter/metadata')
    for base,paths in [(folder,['experiments/p1_05/expert_worker.py']), (long_folder,['experiments/p1_05/long_worker.py','experiments/p1_05/pairing_long.py'])]:
        binding=json.loads((base/'bindings.json').read_text())
        for rel,value in binding['harness'].items():
            if sha(base/'sources'/rel)!=value:raise ValueError('staged numerical evaluator mismatch')
        for rel in paths:
            if sha(base/'sources'/rel)!=sha(ROOT/rel):raise ValueError('staged added worker mismatch')
    ledger=json.loads((folder/'trajectory-index.json').read_text())
    replays=[];calls=0;tokens=0
    for entry in ledger['entries']:
        for key in ['proposal','response','decision','trajectory']:
            path=(ROOT/entry[key]).resolve()
            if not path.is_relative_to(ROOT/'evidence') or sha(path)!=entry[key+'_sha256']:raise ValueError('historical trajectory artifact changed')
        directory=(ROOT/entry['decision']).parent
        r=replay(directory)
        if r['verdict']!='REJECT' or r['incumbent_after']!=identity(4)['id']:raise ValueError('historical state chain changed')
        proposal=json.loads((ROOT/entry['proposal']).read_text())
        response=json.loads((ROOT/entry['response']).read_text())
        if proposal['warps']!=entry['warps'] or response['usage']['total_tokens']!=entry['tokens']:raise ValueError('historical model proposal/usage mismatch')
        calls+=entry['calls'];tokens+=entry['tokens'];replays.append(r)
    if calls!=2 or tokens!=6558 or ledger['promotions']!=0:raise ValueError('V0 model accounting differs')
    commitment=json.loads((ROOT/'workloads/gemm_acceptance_commitment_v1.json').read_text())
    if sha(ROOT/commitment['manifest'])!=commitment['sha256']:raise ValueError('heldout commitment changed')
    execution=json.loads((folder/'run/execution.json').read_text());long_exec=json.loads((long_folder/'run/execution.json').read_text())
    return {'status':'PASS','kind':'CPU-only recomputation; no API, GPU or credentials needed',
            'frozen_best':identity(4)['id'],'independent_final_sessions':3,'long_final_segments':864,
            'short_failed_final_segments_preserved':1728,'separate_expert_segments':864,
            'all_new_raw_segments':3456,'new_correctness_cases_including_calibration':672,
            'initial_short_stability':'FAILED, retained; superseded only for final verification by preregistered long mode',
            'long_stability':computed['final_verification_gate'],'final_timing_accepted':computed['status']=='PASS','p1_05_acceptance':'PENDING_TIMING_STABILITY' if computed['status']!='PASS' else 'PASS','nominal_agent_speedup':1.0,
            'nominal_wins_ties_losses':computed['nominal_wins_ties_losses'],
            'prior_real_model_calls':calls,'prior_reported_tokens':tokens,'new_model_calls':0,'new_profiles':0,
            'historical_trajectory_replays':[{'verdict':r['verdict'],'reason':r['reason'],'incumbent_after':r['incumbent_after']} for r in replays],
            'worker_wall_seconds':sum(r['wall_seconds'] for r in execution['commands'])+sum(r['wall_seconds'] for r in long_exec['commands']),
            'heldout_commitment_preserved':True,'heldout_executed':False,'production_release':False,
            'summary_sha256':sha(folder/'summary.json'),'trajectory_index_sha256':sha(folder/'trajectory-index.json')}


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--evidence',type=Path,required=True);p.add_argument('--long-evidence',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args()
    if a.output.exists():p.error('Use a new audit output')
    result=audit(a.evidence,a.long_evidence);a.output.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))


if __name__=='__main__':main()
