"""Recompute frozen V0 final and expert tables from every raw paired segment."""
import argparse,json,math,statistics
from pathlib import Path
from collections import Counter
from evaluators.gemm_decision import ROOT,BOUNDARIES,identity,protected_hashes,sha,group_log_ratio,decide


def interval(values):
    if len(values)!=3 or any(not math.isfinite(v) for v in values):raise ValueError('three finite independent session logs required')
    mean=statistics.mean(values);delta=4.303*statistics.stdev(values)/math.sqrt(3)
    return {'geomean':math.exp(mean),'exploratory_session_interval':[math.exp(mean-delta),math.exp(mean+delta)]}


def summarize_group(groups):
    if len(groups)!=3 or {s for s,_ in groups}!={0,1,2}:raise ValueError('missing/duplicate session')
    means=[];durations={'incumbent':[],'candidate':[]};maxdev=0
    for session,row in sorted(groups):
        mean,deviation=group_log_ratio(row,session);means.append(mean);maxdev=max(maxdev,deviation)
        for segment in row['segments']:durations[segment['label']].append(segment['elapsed_ms']*1000/row.get('calls_per_segment',32*12))
    return {**interval(means),'pairs':27,'raw_segments':54,'baseline_median_us':statistics.median(durations['incumbent']),
            'candidate_median_us':statistics.median(durations['candidate']),'session_mean_logs':means,'max_pair_deviation':maxdev}


def validate_checks(checks,development):
    expected=Counter((tuple(w['shape_mnk']),s,'random',False) for w in development for s in (171,172))
    expected.update((tuple(shape),s,dist,offset) for shape,dist,offset in BOUNDARIES for s in (171,172))
    observed=Counter((tuple(c['shape_mnk']),c['seed'],c['distribution'],c['offset_guard']) for c in checks)
    if observed!=expected or any(not c['input_immutable'] for c in checks):raise ValueError('fresh correctness coverage incomplete')


def validate_long(folder,source,binding,development):
    from experiments.p1_05.plan_long import make_plan
    plan=json.loads((folder/'plan.json').read_text())
    if plan!=make_plan(source,250):raise ValueError('long plan changed from preregistered sizing rule')
    if json.loads((folder/'freeze.json').read_text())!=json.loads((source/'freeze.json').read_text()):raise ValueError('long run changed best')
    execution=json.loads((folder/'run/execution.json').read_text())
    if not execution['preflight_idle'] or len(execution['commands'])!=3 or any(r['exit']!=0 or r['timed_out'] for r in execution['commands']):raise ValueError('long verification processes failed')
    reports=[json.loads((folder/'run'/f'final-{i}.json').read_text()) for i in range(3)]
    expected={(r['workload'],r['input_count']):r for r in plan['rows']}
    worst=0.0;unstable=[]
    for report in reports:
        if report['status']!='PASS' or report['mode']!=plan['mode'] or report['initial']!=identity(4) or report['best']!=identity(4) or report['environment']['capability']!=[8,6]:raise ValueError('wrong long identity/mode/device')
        if report['harness_before']!=binding['harness'] or report['harness_after']!=binding['harness']:raise ValueError('long evaluator drift')
        if report['plan_sha256']!=sha(folder/'plan.json') or report['worker_sha256']!=sha(ROOT/'experiments/p1_05/long_worker.py') or report['pairing_sha256']!=sha(ROOT/'experiments/p1_05/pairing_long.py'):raise ValueError('long code/plan binding changed')
        if report['binary_sha256']!=json.loads((source/'freeze.json').read_text())['binary_sha256'] or not report['same_identity_pair_is_aa_control']:raise ValueError('wrong long artifact/role')
        for phase in ['checks_before','checks_after']:validate_checks(report[phase],development)
        if len(report['measurements'])!=16 or {(r['workload'],r['input_count']) for r in report['measurements']}!=set(expected):raise ValueError('long denominator changed')
        for row in report['measurements']:
            cfg=expected[(row['workload'],row['input_count'])]
            if row['shape_mnk']!=cfg['shape_mnk'] or row['replays']!=cfg['replays'] or row['calls_per_segment']!=cfg['calls_per_segment'] or not row['cpu_reference_and_inputs_pass']:raise ValueError('long group contract changed')
            if row['capture_order']!=(['incumbent','candidate'] if report['session']%2==0 else ['candidate','incumbent']):raise ValueError('long capture order changed')
            mean,dev=group_log_ratio(row,report['session']);worst=max(worst,dev)
            if dev>=plan['max_pair_deviation'] or abs(math.exp(mean)-1)>=plan['max_group_geomean_deviation']:
                unstable.append({'session':report['session'],'workload':row['workload'],'input_count':row['input_count'],'geomean':math.exp(mean),'max_pair_deviation':dev})
    return reports,{'status':'FAIL' if unstable else 'PASS','unstable_groups':unstable,'mode':plan['mode'],'same_identity_no_gain':True,'max_pair_deviation':worst,'plan_sha256':sha(folder/'plan.json'),'scope':'final-only verification; original search promotion protocol unchanged'}


def summarize(folder,long_evidence=None):
    folder=Path(folder);binding=json.loads((folder/'bindings.json').read_text());frozen=json.loads((folder/'freeze.json').read_text())
    if frozen['candidate']!=identity(4) or binding['initial']!=identity(4) or binding['best']!=identity(4):raise ValueError('frozen identity differs from initial')
    if binding['harness']!=protected_hashes():raise ValueError('current evaluator drift')
    protocol=json.loads((ROOT/'workloads/gemm_measurement_v1.json').read_text())
    if frozen['binary_sha256']!=protocol['initial_template']['binary_sha256']:raise ValueError('freeze artifact differs')
    execution=json.loads((folder/'run/execution.json').read_text())
    expected_tags={f'{group}-{s}' for group in ['final','expert'] for s in range(3)}
    if len(execution['commands'])!=6 or {r['tag'] for r in execution['commands']}!=expected_tags or any(r['exit']!=0 or r['timed_out'] for r in execution['commands']):raise ValueError('failed or missing process; keep failures in original denominator')
    if execution['idle_before']!={'final':True,'expert':True}:raise ValueError('not an idle GPU window')
    development=json.loads((ROOT/'workloads/gemm_development_v1.json').read_text())['workloads'];grid={(w['id'],c) for w in development for c in (1,16)}
    reports={group:[json.loads((folder/'run'/f'{group}-{s}.json').read_text()) for s in range(3)] for group in ['final','expert']}
    gate=decide(reports['final'],identity(4),identity(4),binding['harness'])
    short_gate=gate
    if long_evidence is not None:
        reports['final'],gate=validate_long(Path(long_evidence),folder,binding,development)
    elif gate['reason']!='duplicate' or len(gate['policies'])!=2:
        raise ValueError('final verification failed; preserve short batch and use preregistered longer verification: '+json.dumps(gate))
    if any(p.get('pass') for p in gate.get('policies',[])):raise ValueError('same implementation spuriously meets gain gate')
    for report in reports['expert']:
        if report['status']!='PASS' or report['evidence_mode']!='GPU' or report['environment']['capability']!=[8,6]:raise ValueError('expert process failed/unsupported')
        if report['harness_before']!=binding['harness'] or report['harness_after']!=binding['harness']:raise ValueError('expert harness drift')
        if report['candidate']!=identity(4) or report['binary']['sha256']!=frozen['binary_sha256']:raise ValueError('expert compared wrong artifact')
        if report['baseline']['source_sha256']!=protocol['trusted_fallback']['source_sha256'] or report['worker_sha256']!=sha(ROOT/'experiments/p1_05/expert_worker.py'):raise ValueError('expert source binding differs')
        if report['graph_launches']!=32 or report['replays']!=12:raise ValueError('timing boundary changed')
        if len(report['measurements'])!=16 or {(r['workload'],r['input_count']) for r in report['measurements']}!=grid:raise ValueError('expert workload denominator changed')
        for phase in ['checks_before','checks_after']:
            for name in ['cublas','best']:validate_checks(report[phase][name],development)
        for row in report['measurements']:
            if not row['cpu_reference_and_inputs_pass'] or row['capture_order']!=(['incumbent','candidate'] if report['session']%2==0 else ['candidate','incumbent']):raise ValueError('expert graph verification failed')
            if row['shape_mnk']!=next(w['shape_mnk'] for w in development if w['id']==row['workload']):raise ValueError('expert shape mapping differs')
    results={}
    for group,items in reports.items():
        rows=[]
        for w in development:
            for count in (1,16):
                selected=[(r['session'],next(x for x in r['measurements'] if x['workload']==w['id'] and x['input_count']==count)) for r in items]
                rows.append({'workload':w['id'],'shape_mnk':w['shape_mnk'],'input_count':count,**summarize_group(selected)})
        policies=[]
        for count in (1,16):
            logs=[statistics.mean(row['session_mean_logs'][s] for row in rows if row['input_count']==count) for s in range(3)]
            policies.append({'input_count':count,**interval(logs),'workloads':8,'denominator_preserved':True})
        results[group]={'ratio_definition':'initial_latency / frozen_best_latency' if group=='final' else 'cuBLAS_latency / frozen_best_latency',
                        'rows':rows,'all_workload_policies':policies,'raw_measurement_segments':864}
    return {'status':'TIMING_NOT_ACCEPTED' if gate.get('status')=='FAIL' else 'PASS','scope':'Frozen final retest on 8 development workloads; 4 heldout workloads NOT executed',
            'initial_identity':binding['initial']['id'],'best_identity':frozen['candidate']['id'],'binary_sha256':frozen['binary_sha256'],
            'nominal_agent_speedup':1.0,'nominal_wins_ties_losses':{'wins':0,'ties':8,'losses':0,'reason':'unchanged source/config identity, not measured-noise classification'},
            'correctness_cases':{'final':168,'expert':336},'failed_processes':0,'total_raw_segments':2592 if long_evidence is None else 3456,'final_verification_gate':gate,'short_segment_verification':short_gate,
            'final_retest_mode':'original-12-replay' if long_evidence is None else 'frozen-final-long-v2','final_retest_measurement_segments':864,'short_noisy_final_segments_preserved':1728 if long_evidence is not None else 0,
            'final':results['final'],'expert':results['expert'],'new_model_calls':0,'new_profile_runs':0,'positive_optimization_claim':False,
            'statistical_limit':'3-session t intervals are exploratory; no strict post-selection confidence guarantee',
            'timing_scope':'CUDA Graph stream Event window; host setup/allocation/capture outside bracket, residual submit/scheduling/clock noise possible; no deployment eager latency claim'}


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--evidence',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--long-evidence',type=Path)
    args=p.parse_args()
    if args.output.exists():p.error('Use a new output')
    result=summarize(args.evidence,args.long_evidence);args.output.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k not in ['final','expert','final_verification_gate']},indent=2))


if __name__=='__main__':main()
