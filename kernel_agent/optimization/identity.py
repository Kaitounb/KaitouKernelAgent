"""Task adapter for KDA evidence/promotion; no candidate source execution."""
import hashlib
import json
import math
from pathlib import Path
import statistics
from collections import Counter

ROOT=Path(__file__).resolve().parent/'assets'
BOUNDARIES=[((1,1,1),'random',False),((17,19,23),'random',True),((48,32,80),'random',False),
            ((32,48,64),'random',True),((16,16,16),'zero',False),((32,48,64),'signed4',False)]


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def canonical(value):
    return json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False).encode()


def identity(warps):
    if type(warps) is not int or warps not in (1,2,4,8):
        raise ValueError('warps must be an integer in 1/2/4/8')
    paths=['kernels/gemm/wmma_sm86.cu','kernels/gemm/cublas_sm86.cu','kernels/gemm/gemm_contract.cuh',
           'workloads/gemm_contract_v1.json','workloads/gemm_measurement_v1.json']
    body={'template':'wmma-sm86-v1','warps':warps,'files':{p:sha(ROOT/p) for p in paths}}
    return {**body,'id':hashlib.sha256(canonical(body)).hexdigest()}


def strict_json(text):
    if not isinstance(text,str) or len(text.encode())>16384:
        raise ValueError('proposal must be UTF-8 JSON <=16 KiB')
    def pairs(items):
        d={}
        for k,v in items:
            if k in d:raise ValueError('duplicate JSON key')
            d[k]=v
        return d
    def constant(value):
        raise ValueError('nonfinite JSON constant')
    return json.loads(text,object_pairs_hook=pairs,parse_constant=constant)


def validate_proposal(text,parent_id):
    d=strict_json(text)
    if not isinstance(d,dict) or set(d)!={'schema_version','parent_id','warps','reason'}:
        raise ValueError('only schema_version/parent_id/warps/reason are accepted')
    if type(d['schema_version']) is not int or d['schema_version']!=1:
        raise ValueError('unsupported proposal schema')
    if d['parent_id']!=parent_id:
        raise ValueError('stale or incorrect parent')
    identity(d['warps'])
    if not isinstance(d['reason'],str) or not 1<=len(d['reason'])<=1000:
        raise ValueError('reason must contain 1..1000 characters')
    return d


def protected_hashes():
    paths=['evaluators/gemm_candidate.py','evaluators/gemm_decision.py','evaluators/gemm_regression.py',
           'workloads/gemm_contract_v1.json','workloads/gemm_measurement_v1.json','workloads/gemm_development_v1.json',
           'kernels/gemm/wmma_sm86.cu','kernels/gemm/cublas_sm86.cu','kernels/gemm/gemm_contract.cuh']
    return {p:sha(ROOT/p) for p in paths}


def group_log_ratio(case,session=None):
    if len(case.get('segments',[]))!=18:
        raise ValueError('missing Event segments')
    logs=[]
    for pair in range(9):
        rows=[r for r in case['segments'] if r['pair']==pair]
        if len(rows)!=2 or {r['label'] for r in rows}!={'incumbent','candidate'} or {r['order'] for r in rows}!={0,1}:
            raise ValueError('invalid pair/order/labels')
        if session is not None:
            order=['incumbent','candidate'] if (pair+session)%2==0 else ['candidate','incumbent']
            if [r['label'] for r in sorted(rows,key=lambda r:r['order'])]!=order:
                raise ValueError('paired order does not match session')
        values={r['label']:r['elapsed_ms'] for r in rows}
        if any(type(v) not in (int,float) or not math.isfinite(v) or v<=0 for v in values.values()):
            raise ValueError('nonfinite or nonpositive Event segment')
        logs.append(math.log(values['incumbent']/values['candidate']))
    return statistics.mean(logs),max(abs(math.exp(x)-1) for x in logs)


def decide(reports,parent,candidate,expected_guard):
    """Recompute all ratios from raw evidence; fail closed on missing/mismatched data."""
    protocol=json.loads((ROOT/'workloads/gemm_measurement_v1.json').read_text())
    development=json.loads((ROOT/'workloads/gemm_development_v1.json').read_text())
    ids={r['id'] for r in development['workloads']};anchors=set(development['anchors'])
    result={'verdict':'REJECT','reason':'invalid_evidence','parent_id':parent['id'],'candidate_id':candidate['id'],
            'incumbent_after':parent['id'],'policies':[],'formal_performance_claim':False}
    try:
        if len(reports)!=3 or {r.get('session') for r in reports}!={0,1,2}:
            raise ValueError('exactly three independent sessions required')
        grid={(i,c) for i in ids for c in (1,16)}
        per_policy={1:[],16:[]};workload_logs={key:[] for key in grid}
        for report in sorted(reports,key=lambda r:r['session']):
            if report.get('status')!='PASS' or report.get('evidence_mode')!='GPU':
                raise ValueError('session did not pass on GPU')
            if report['parent']!=parent or report['candidate']!=candidate:
                raise ValueError('candidate/parent identity mismatch')
            if report['harness_before']!=expected_guard or report['harness_after']!=expected_guard:
                raise ValueError('protected evaluator drift')
            if report['protocol_sha256']!=sha(ROOT/'workloads/gemm_measurement_v1.json'):
                raise ValueError('measurement protocol changed')
            if report.get('graph_launches')!=32 or report.get('replays')!=12 or not report.get('environment',{}).get('capability')==[8,6]:
                raise ValueError('unsupported timing mode or hardware')
            if report.get('binary',{}).get('sha256')!=protocol['initial_template']['binary_sha256']:
                raise ValueError('unverified build artifact')
            if not report.get('correctness_before_pass') or not report.get('correctness_after_pass'):
                raise ValueError('fresh correctness missing')
            for key in ('checks_before','checks_after'):
                checks=report.get(key,[])
                if len(checks)!=28 or any(not c.get('input_immutable') for c in checks):
                    raise ValueError('incomplete correctness evidence')
                expected=Counter((tuple(w['shape_mnk']),seed,'random',False) for w in development['workloads'] for seed in (171,172))
                expected.update((tuple(shape),seed,dist,offset) for shape,dist,offset in BOUNDARIES for seed in (171,172))
                observed=Counter((tuple(c['shape_mnk']),c['seed'],c['distribution'],c['offset_guard']) for c in checks)
                if expected!=observed:raise ValueError('correctness coverage changed')
            for collection in ['measurements','candidate_aa']:
                rows=report.get(collection,[])
                if len(rows)!=16 or {(r['workload'],r['input_count']) for r in rows}!=grid:
                    raise ValueError('missing workload or cache policy')
                for row in rows:
                    if not row.get('cpu_reference_and_inputs_pass'):
                        raise ValueError('graph output/input check failed')
                    expected_shape=next(w['shape_mnk'] for w in development['workloads'] if w['id']==row['workload'])
                    expected_capture=['incumbent','candidate'] if report['session']%2==0 else ['candidate','incumbent']
                    if row.get('shape_mnk')!=expected_shape or row.get('capture_order')!=expected_capture:
                        raise ValueError('shape/capture order mismatch')
                    mean,deviation=group_log_ratio(row,report['session'])
                    if collection=='candidate_aa' and (deviation>=.02 or abs(math.exp(mean)-1)>=.015):
                        raise ValueError('candidate A/A unstable')
            current={}
            for row in report['measurements']:
                key=(row['workload'],row['input_count']);mean,_=group_log_ratio(row,report['session'])
                workload_logs[key].append(mean);current[key]=mean
            for policy in (1,16):
                per_policy[policy].append(statistics.mean(current[(a,policy)] for a in anchors))
        thresholds=protocol['screening'];passes=[]
        for policy,values in per_policy.items():
            mean=statistics.mean(values);sem=statistics.stdev(values)/math.sqrt(3)
            gm=math.exp(mean);ci=[math.exp(mean-4.303*sem),math.exp(mean+4.303*sem)]
            guards={w:math.exp(statistics.mean(workload_logs[(w,policy)])) for w in sorted(ids)}
            passed=gm>=thresholds['minimum_anchor_geomean'] and ci[0]>thresholds['minimum_exploratory_interval_lower'] and min(guards.values())>=thresholds['minimum_each_development_workload_geomean']
            result['policies'].append({'input_count':policy,'anchor_geomean':gm,'exploratory_session_interval':ci,'workload_speedups':guards,'pass':passed})
            passes.append(passed)
        if all(passes) and candidate['id']!=parent['id']:
            result.update(verdict='KEEP',reason='frozen_screening_pass',incumbent_after=candidate['id'])
        else:
            result['reason']='duplicate' if candidate['id']==parent['id'] else 'insufficient_gain_or_regression'
    except (KeyError,ValueError,TypeError,OverflowError) as error:
        result['detail']=str(error)
    return result
