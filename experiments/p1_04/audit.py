"""CPU evidence audit for the P1-04 hardware-feedback round; no API/GPU invocation."""
import argparse
import gzip
import hashlib
import json
from pathlib import Path

from evaluators.gemm_decision import ROOT,identity,protected_hashes,sha
from experiments.p1_03.replay import replay
from experiments.p1_04.propose import make_facts,validate_feedback
from experiments.p1_04.compare import compare


def audit(base):
    baseline=ROOT/'evidence/20261122-p1-04-ncu-baseline';candidate=ROOT/'evidence/20261122-p1-04-ncu-warps2'
    report_import=json.loads((base/'ncu-report-verification.json').read_text())
    reparse=json.loads((base/'parser-replay.json').read_text())
    if report_import['status']!='PASS' or reparse['status']!='PASS':raise ValueError('raw report import/parser replay failed')
    profiles=0;observations=0;valid_zeros=0
    for folder,warps in [(baseline,4),(candidate,2)]:
        execution=json.loads((folder/'run/execution.json').read_text())
        if execution['status']!='PASS' or execution['parser_tests_exit']!=0:raise ValueError('profile or parser fixtures failed')
        selection=json.loads((folder/'run/metric-selection.json').read_text())
        if len(selection['selected'])!=19:raise ValueError('unexpected selected field count')
        if not selection['unavailable_from_query']:raise ValueError('missing unavailability record')
        with gzip.open(folder/'run/query-metrics.txt.gz','rb') as file:
            query_hash=hashlib.sha256(file.read()).hexdigest()
        if query_hash!=selection['query_sha256']:raise ValueError('metric query archive mismatch')
        for size in (512,1024):
            tag=f'warps-{warps}-{size}';path=folder/'run'/(tag+'.metrics.json')
            data=json.loads(path.read_text());config=json.loads((folder/'run'/(tag+'.config.json')).read_text())
            checks=json.loads((folder/'run'/(tag+'.checks.json')).read_text())
            if config['binary_sha256']!='93fb2f4344521b7e9b246b54d1c30d4b1a8facf731e6385e18f6119e9c49c72c' or config['warps']!=warps:raise ValueError('profile artifact/config mismatch')
            if checks['status']!='PASS' or not checks['cpu_fp64_reference'] or not checks['inputs_immutable']:raise ValueError('profile correctness failed')
            if data['csv_sha256']!=sha(folder/'run'/(tag+'.csv')) or len(data['launches'])!=3:raise ValueError('profile source or launch coverage mismatch')
            native=folder/'run'/(tag+'.ncu-rep')
            if not any(r['sha256']==sha(native) and r['exit']==0 and r['target_kernel_in_import'] for r in report_import['reports']):raise ValueError('native report verification missing')
            for rel in ['experiments/p1_04/ncu_adapter.py','experiments/p1_04/profile_entry.py']:
                if sha(folder/'sources'/rel)!=sha(ROOT/rel):raise ValueError('profile source drift')
            for row in data['launches']:
                if len(row['metrics'])!=19:raise ValueError('missing profile fields')
                for cell in row['metrics'].values():
                    if cell['status']=='observed':observations+=1;valid_zeros+=int(cell['value']==0)
                    elif cell['value'] is not None:raise ValueError('unavailable metric got a value')
            profiles+=1
    facts=make_facts(baseline/'run')
    if facts!=json.loads((base/'proposal/facts.json').read_text()):raise ValueError('model facts differ from raw profile summary')
    raw=json.loads((base/'proposal/response.json').read_text());usage=json.loads((base/'proposal/usage.json').read_text())
    if usage['calls_attempted']!=1 or usage['status']!='RECEIVED' or raw['model']!='deepseek-flash' or raw['choices'][0]['finish_reason']!='stop':raise ValueError('real model call evidence incomplete')
    feedback=validate_feedback(raw['choices'][0]['message']['content'],identity(4),facts)
    if feedback!=json.loads((base/'proposal/feedback.json').read_text()) or feedback['proposal']!=json.loads((base/'proposal/proposal.json').read_text()):raise ValueError('model proposal/binding changed')
    if raw['usage']['total_tokens']!=usage['usage']['total_tokens']:raise ValueError('token accounting mismatch')
    comparison=compare(baseline/'run',candidate/'run',feedback)
    if comparison!=json.loads((base/'profile-comparison.json').read_text()):raise ValueError('profile prediction comparison mismatch')
    result=replay(base/'evaluation')
    bindings=json.loads((base/'evaluation/bindings.json').read_text())
    if bindings['harness']!=protected_hashes():raise ValueError('normal evaluator modified')
    if (base/'evaluation/initial-incumbent.json').read_bytes()!=(base/'evaluation/incumbent.json').read_bytes():raise ValueError('rejected candidate changed current')
    for rel,value in bindings['harness'].items():
        if sha(base/'evaluation/sources'/rel)!=value:raise ValueError('normal evaluator snapshot mismatch')
    commitment=json.loads((ROOT/'workloads/gemm_acceptance_commitment_v1.json').read_text())
    if sha(ROOT/commitment['manifest'])!=commitment['sha256']:raise ValueError('heldout commitment changed')
    return {'status':'PASS','kind':'CPU audit of existing actual profile/model/GPU evidence, not a new GPU run',
            'profiles':profiles,'target_launch_records':profiles*3,'fields_per_launch':19,'observed_cells':observations,'actual_zero_cells_preserved':valid_zeros,
            'all_native_reports_import_verified':True,'raw_csv_reparse_verified':True,'real_model_calls':1,'reported_tokens':usage['usage']['total_tokens'],
            'predictions_checked':comparison['predictions_checked'],'descriptive_prediction_direction_matches':comparison['predictions_direction_matched'],
            'independent_event_replay':result,'incumbent_bytes_unchanged':True,'heldout_commitment_preserved_without_execution':True}


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--evidence',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    args=p.parse_args()
    if args.output.exists():p.error('Use a new output')
    result=audit(args.evidence);args.output.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k!='independent_event_replay'},indent=2))


if __name__=='__main__':main()
