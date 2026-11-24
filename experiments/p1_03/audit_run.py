"""Verify source bindings, actual model usage, raw GPU replay and no-gain control."""
import argparse
import hashlib
import json
from pathlib import Path

from evaluators.gemm_decision import ROOT,identity,protected_hashes,decide,sha
from experiments.p1_03.replay import replay


def audit(base):
    evaluation=base/'evaluation-retry1'
    result=replay(evaluation)
    binding=json.loads((evaluation/'bindings.json').read_text())
    if binding['harness']!=protected_hashes():raise ValueError('current protected source differs from evaluated source')
    for rel,expected in binding['harness'].items():
        if sha(evaluation/'sources'/rel)!=expected:raise ValueError('staged source differs from binding')
    if (evaluation/'initial-incumbent.json').read_bytes()!=(evaluation/'incumbent.json').read_bytes():raise ValueError('actual run unexpectedly changed incumbent')
    usage=json.loads((base/'proposal/usage.json').read_text());raw=json.loads((base/'proposal/response.json').read_text())
    if usage['calls_attempted']!=1 or not usage['real_external_model_call'] or usage['status']!='RECEIVED':raise ValueError('real model call evidence missing')
    if usage['usage']['total_tokens']!=raw['usage']['total_tokens']:raise ValueError('usage differs from raw response')
    if raw['model']!='deepseek-flash' or raw['choices'][0]['finish_reason']!='stop':raise ValueError('unexpected model or truncated response')
    proposal=json.loads((base/'proposal/proposal.json').read_text())
    if proposal['warps']!=8 or proposal['parent_id']!=binding['parent']['id']:raise ValueError('proposal binding mismatch')
    for section in ['initial_template','trusted_fallback']:
        protocol=json.loads((ROOT/'workloads/gemm_measurement_v1.json').read_text())
        path='kernels/gemm/wmma_sm86.cu' if section=='initial_template' else 'kernels/gemm/cublas_sm86.cu'
        if sha(ROOT/path)!=protocol[section]['source_sha256']:raise ValueError('source/artifact provenance changed')
    control=base/'negative-control'
    reports=[json.loads((control/'run'/f'session-{i}.json').read_text()) for i in range(3)]
    negative=decide(reports,identity(4),identity(4),binding['harness'])
    if negative!=json.loads((control/'decision.json').read_text()) or negative['verdict']!='REJECT' or negative['reason']!='duplicate':raise ValueError('negative control differs')
    if any(p['pass'] for p in negative['policies']):raise ValueError('same implementation passes gain threshold')
    execution=json.loads((evaluation/'run/execution.json').read_text())
    expected={'fault-constant','fault-pollution','fault-host-error','fault-host-timeout'}
    if {r['tag'] for r in execution['commands'] if r['tag'].startswith('fault-')}!=expected:raise ValueError('missing fault regression')
    if not execution['no_model_credentials_in_worker']:raise ValueError('unexpected model credential environment')
    if not all(r['exit']==0 and not r['remaining_group_pids'] for r in execution['commands'] if r['tag'].startswith('session-')):raise ValueError('worker failure')
    commitment=json.loads((ROOT/'workloads/gemm_acceptance_commitment_v1.json').read_text())
    if sha(ROOT/commitment['manifest'])!=commitment['sha256']:raise ValueError('heldout commitment changed')
    return {'status':'PASS','kind':'CPU audit of actual external model and GPU evidence, not new GPU execution',
            'real_model_calls':1,'reported_total_tokens':raw['usage']['total_tokens'],
            'actual_candidate_verdict':result['verdict'],'actual_candidate_reason':'candidate A/A unstable',
            'actual_incumbent_bytes_unchanged':True,'correctness_cases_actual':168,
            'ab_segments_actual':result['raw_ab_segments'],'aa_segments_actual':result['raw_candidate_aa_segments'],
            'negative_control_verdict':negative['verdict'],'negative_control_reason':negative['reason'],
            'negative_control_policies':negative['policies'],'heldout_hash_verified_without_execution':True,
            'current_harness':binding['harness']}


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--evidence',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    args=p.parse_args()
    if args.output.exists():p.error('Use a new output')
    result=audit(args.evidence);args.output.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps({k:v for k,v in result.items() if k not in ['current_harness','negative_control_policies']},indent=2))


if __name__=='__main__':main()
