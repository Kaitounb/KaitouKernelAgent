"""Verbatim pure deployment gate from accepted SM89 protocol, no GPU/model execution."""
import math,statistics
def assess_deployment(aa,ab,workloads,policy):
    from kernel_agent.optimization.identity import group_log_ratio
    ids={w['id'] for w in workloads};grid={(w,n) for w in ids for n in [1,16]}
    value={'verdict':'REJECT','aa_status':'INCOMPLETE','policies':[],'production_release':False}
    if len(aa)!=3 or {r['session'] for r in aa}!={0,1,2}:return {**value,'reason':'three_aa_required'}
    violations=[]
    for report in aa:
        assert report['mode']=='aa' and report['status']=='PASS' and len(report['rows'])==16
        assert {(r['workload'],r['input_count']) for r in report['rows']}==grid
        for row in report['rows']:
            mean,deviation=group_log_ratio(row,report['session'])
            if deviation>=policy['aa_pair_deviation_limit_exclusive'] or abs(math.exp(mean)-1)>=policy['aa_group_geomean_deviation_limit_exclusive']:
                violations.append({'session':report['session'],'workload':row['workload'],'input_count':row['input_count'],'max_pair_deviation':deviation,'geomean':math.exp(mean)})
    if violations:return {**value,'aa_status':'REJECT','reason':'aa_unstable_v2','unstable_groups':violations}
    value['aa_status']='PASS'
    if len(ab)!=3 or {r['session'] for r in ab}!={0,1,2}:return {**value,'reason':'AA_PASS_AB_NOT_RUN'}
    logs={key:[] for key in grid}
    for report in ab:
        assert report['mode']=='ab' and report['status']=='PASS' and len(report['rows'])==16
        assert {(r['workload'],r['input_count']) for r in report['rows']}==grid
        for row in report['rows']:logs[(row['workload'],row['input_count'])].append(group_log_ratio(row,report['session'])[0])
    for count in [1,16]:
        guards={w:math.exp(statistics.mean(logs[(w,count)])) for w in sorted(ids)}
        value['policies'].append({'input_count':count,'workload_speedups':guards,'pass':min(guards.values())>=policy['deployment_nonregression_guard']})
    passed=all(p['pass'] for p in value['policies'])
    return {**value,'verdict':'PASS' if passed else 'REJECT','reason':'deployment_guards_pass' if passed else 'deployment_guard_regression'}
