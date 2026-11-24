"""Recompute the archived first-round verdict and incumbent from raw trusted evidence."""
import argparse
import json
from pathlib import Path
from evaluators.gemm_decision import decide,identity,validate_proposal,sha


def replay(folder):
    initial=json.loads((folder/'initial-incumbent.json').read_text())
    binding=json.loads((folder/'bindings.json').read_text())
    if initial!=identity(initial['warps']) or initial!=binding['parent']:
        raise ValueError('initial incumbent source binding changed')
    proposal=validate_proposal((folder/'proposal.json').read_text(),initial['id'])
    candidate=identity(proposal['warps'])
    if candidate!=binding['candidate']:raise ValueError('candidate source binding changed')
    reports=[json.loads((folder/'run'/f'session-{i}.json').read_text()) for i in range(3)]
    expected=decide(reports,initial,candidate,binding['harness'])
    execution=json.loads((folder/'run/execution.json').read_text())
    commands=execution['commands'];fixtures_ok=True
    for row in commands:
        if row['tag'].startswith('fault-'):
            rejected=row['timed_out'] if row['tag']=='fault-host-timeout' else row['exit']!=0 and not row['timed_out']
            fixtures_ok=fixtures_ok and rejected and not row['remaining_group_pids']
    recovery=next(row for row in commands if row['tag']=='recovery')
    fixtures_ok=fixtures_ok and recovery['exit']==0
    if not fixtures_ok or not execution['preflight_idle']:
        expected.update(verdict='REJECT',reason='fault_recovery_idle_or_harness_check_failed',incumbent_after=initial['id'])
    actual=json.loads((folder/'decision.json').read_text())
    if expected!=actual:raise ValueError('stored verdict differs from raw-evidence recomputation')
    events=[json.loads(line) for line in (folder/'candidates.jsonl').read_text().splitlines()]
    current=events[0]['incumbent']
    if current!=initial:raise ValueError('trajectory initial state mismatch')
    for event in events[1:]:
        if event['event']=='reject_fault':
            if not event['rejected'] or event['incumbent_after']!=current['id']:raise ValueError('fault corrupted incumbent')
        elif event['event']=='recovery':
            if not event['pass'] or event['incumbent_after']!=current['id']:raise ValueError('recovery corrupted incumbent')
        elif event['event']=='candidate':
            if event['decision']!=expected:raise ValueError('trajectory decision mismatch')
            current=candidate if expected['verdict']=='KEEP' else current
        else:raise ValueError('unknown trajectory event')
    if current!=json.loads((folder/'incumbent.json').read_text()):raise ValueError('final incumbent mismatch')
    return {'status':'PASS','kind':'CPU replay of archived GPU evidence','verdict':expected['verdict'],
            'reason':expected['reason'],'incumbent_after':current['id'],'faults_kept_incumbent':fixtures_ok,
            'sessions':3,'raw_ab_segments':sum(len(r['segments']) for s in reports for r in s['measurements']),
            'raw_candidate_aa_segments':sum(len(r['segments']) for s in reports for r in s['candidate_aa']),
            'decision_sha256':sha(folder/'decision.json'),'policies':expected['policies']}


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--evidence',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    args=p.parse_args()
    if args.output.exists():p.error('Use a new output')
    result=replay(args.evidence)
    args.output.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))


if __name__=='__main__':main()
