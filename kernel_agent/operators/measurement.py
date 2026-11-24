"""Numeric paired gate adapted from the frozen GEMM gate; no model or GPU execution."""
import math,statistics
from kernel_agent.optimization.identity import group_log_ratio


def assess(aa_reports,ab_reports,workloads,anchors):
    result={'verdict':'REJECT','reason':'invalid_evidence','policies':[],'performance_claim':False,'production_release':False}
    try:
        if len(aa_reports)!=3 or {r['session'] for r in aa_reports}!={0,1,2}:raise ValueError('three independent AA sessions required')
        ids={w['id'] for w in workloads};grid={(w,n) for w in ids for n in [1,16]};unstable=[]
        if not ids or len(ids)!=len(workloads) or not anchors or len(set(anchors))!=len(anchors) or not set(anchors)<=ids:raise ValueError('invalid workload/anchor set')
        for report in aa_reports:
            if report['status']!='PASS' or report['mode']!='aa' or len(report['rows'])!=len(grid) or {(r['workload'],r['input_count']) for r in report['rows']}!=grid:raise ValueError('incomplete AA evidence')
            for row in report['rows']:
                mean,deviation=group_log_ratio(row,report['session'])
                if deviation>=.02 or abs(math.exp(mean)-1)>=.015:unstable.append({'session':report['session'],'workload':row['workload'],'input_count':row['input_count'],'max_pair_deviation':deviation,'geomean':math.exp(mean)})
        if unstable:return {**result,'reason':'candidate_aa_unstable','unstable_groups':unstable}
        if len(ab_reports)!=3 or {r['session'] for r in ab_reports}!={0,1,2}:return {**result,'reason':'AA_PASS_AB_NOT_RUN','aa_status':'PASS'}
        logs={(w,n):[] for w,n in grid};session_anchors={1:[],16:[]}
        for report in sorted(ab_reports,key=lambda r:r['session']):
            if report['status']!='PASS' or report['mode']!='ab' or len(report['rows'])!=len(grid) or {(r['workload'],r['input_count']) for r in report['rows']}!=grid:raise ValueError('incomplete AB evidence')
            current={}
            for row in report['rows']:
                mean,_=group_log_ratio(row,report['session']);key=(row['workload'],row['input_count']);logs[key].append(mean);current[key]=mean
            for count in [1,16]:session_anchors[count].append(statistics.mean(current[(w,count)] for w in anchors))
        for count,values in session_anchors.items():
            mean=statistics.mean(values);sem=statistics.stdev(values)/math.sqrt(3);gm=math.exp(mean);interval=[math.exp(mean-4.303*sem),math.exp(mean+4.303*sem)];guards={w:math.exp(statistics.mean(logs[(w,count)])) for w in sorted(ids)};passed=gm>=1.05 and interval[0]>1.02 and min(guards.values())>=.97;result['policies'].append({'input_count':count,'anchor_geomean':gm,'exploratory_session_interval':interval,'workload_speedups':guards,'pass':passed})
        result.update(verdict='KEEP' if all(p['pass'] for p in result['policies']) else 'REJECT',reason='frozen_screening_pass' if all(p['pass'] for p in result['policies']) else 'insufficient_gain_or_regression',aa_status='PASS')
    except (ValueError,KeyError,TypeError,OverflowError) as error:result['detail']=str(error)
    return result
