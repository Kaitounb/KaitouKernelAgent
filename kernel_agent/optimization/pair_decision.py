"""Two-artifact adapter around the frozen original KDA-style screening arithmetic."""
import json,math,statistics
from .identity import ROOT,identity
from .cases import make_cases
from .pair_contract import validate_descriptor,pair_id
from .pair_validation import validate_ledgers,validate_report
from .identity import group_log_ratio


def decide_pair(aa,ab,ledgers,descriptor,*,evidence_kind='GPU'):
    parent=identity(4)['id']
    candidate=descriptor.get('candidate',{}).get('id') if isinstance(descriptor,dict) and isinstance(descriptor.get('candidate'),dict) else None
    if not isinstance(candidate,str):candidate=None
    result={'verdict':'REJECT','reason':'invalid_evidence','parent_id':parent,'candidate_id':candidate,'incumbent_after':parent,'pair_sha256':None,'policies':[],
            'evidence_kind':evidence_kind,'formal_performance_claim':False,'runtime_activation':False,'production_release':False,'frozen_best_updated':False}
    try:
        validate_descriptor(descriptor);result['pair_sha256']=pair_id(descriptor);workloads,contract,cases=make_cases(ROOT);validate_ledgers(ledgers,cases,descriptor)
        if evidence_kind not in ['GPU','CPU_SIMULATION']:raise ValueError('invalid evidence kind')
        known=json.loads((ROOT/'known-rejections.json').read_text())
        if candidate in known:result['reason']='known_candidate_aa_unstable';return result
        if candidate==parent or descriptor['binary_sha256']['candidate']==descriptor['binary_sha256']['incumbent']:result['reason']='duplicate';return result
        if len(aa)!=3 or {r['report']['session'] for r in aa}!={0,1,2}:raise ValueError('three A/A sessions required')
        unstable=[]
        for r in aa:unstable+=validate_report(r['report'],r['record'],contract,workloads,ledgers,descriptor,'aa',r['report']['session'])
        if unstable:result.update(reason='candidate_aa_unstable',unstable_groups=unstable);return result
        if len(ab)!=3 or {r['report']['session'] for r in ab}!={0,1,2}:raise ValueError('three A/B sessions required')
        for r in ab:validate_report(r['report'],r['record'],contract,workloads,ledgers,descriptor,'ab',r['report']['session'])
        protocol=json.loads((ROOT/'workloads/gemm_measurement_v1.json').read_text());development=json.loads((ROOT/'workloads/gemm_development_v1.json').read_text());ids={w['id'] for w in workloads};anchors=set(development['anchors'])
        per_policy={1:[],16:[]};workload_logs={(i,c):[] for i in ids for c in [1,16]}
        for session in sorted(ab,key=lambda r:r['report']['session']):
            report=session['report'];current={}
            for row in report['measurements']:
                key=(row['workload'],row['input_count']);mean,_=group_log_ratio(row,report['session'])
                workload_logs[key].append(mean);current[key]=mean
            for policy in (1,16):per_policy[policy].append(statistics.mean(current[(a,policy)] for a in anchors))
        thresholds=protocol['screening'];passes=[]
        for policy,values in per_policy.items():
            mean=statistics.mean(values);sem=statistics.stdev(values)/math.sqrt(3)
            gm=math.exp(mean);ci=[math.exp(mean-4.303*sem),math.exp(mean+4.303*sem)]
            guards={w:math.exp(statistics.mean(workload_logs[(w,policy)])) for w in sorted(ids)}
            passed=gm>=thresholds['minimum_anchor_geomean'] and ci[0]>thresholds['minimum_exploratory_interval_lower'] and min(guards.values())>=thresholds['minimum_each_development_workload_geomean']
            result['policies'].append({'input_count':policy,'anchor_geomean':gm,'exploratory_session_interval':ci,'workload_speedups':guards,'pass':passed});passes.append(passed)
        if all(passes):result.update(verdict='KEEP',reason='frozen_screening_pass',incumbent_after=candidate)
        else:result['reason']='insufficient_gain_or_regression'
    except (KeyError,ValueError,TypeError,OverflowError,IndexError) as error:result['detail']=str(error)
    return result
