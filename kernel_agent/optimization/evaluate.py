"""Trusted parent-side FP64 references, contract/hash witnesses and new-source AA decision."""
import hashlib,json,math,statistics
from pathlib import Path
import numpy as np
import torch
from kernel_agent.output import load_output
from .fixtures import make_inputs
from .cases import case_key
from .identity import group_log_ratio


def input_facts(case):
    cpu=make_inputs(case['shape_mnk'],case['seed'],case['distribution'])
    return cpu,[hashlib.sha256(x.numpy().tobytes()).hexdigest() for x in cpu]


def witness(row,case,ledger):
    for key in ['id','group','kind','shape_mnk','seed','distribution','offset_guard']:
        if row.get(key)!=case[key]:raise ValueError('case witness identity mismatch')
    value=ledger[case_key(case)]
    if row.get('input_hashes')!=value['input_hashes'] or row.get('output_sha256')!=value['output_sha256'] or not row.get('fresh_output') or not row.get('producer_consumer_stream'):raise ValueError('unvalidated output/input/stream witness')
    guards=[hashlib.sha256(np.full(34,7,np.float16).tobytes()).hexdigest()]*2 if case['offset_guard'] else []
    if row.get('guard_hashes')!=guards:raise ValueError('input guard mismatch')


def validate_qa(report,record,cases,expected,job,ledger):
    if report is None or len(report.get('checks',[]))!=len(cases):raise ValueError('QA worker incomplete')
    rows=[]
    for row,case in zip(report['checks'],cases):
        value=load_output(Path(job)/'outputs'/(case['id']+'.npy'),tuple(case['shape_mnk'][:2]));reference,hashes=expected[case['id']]
        if not np.isfinite(value).all():raise ValueError('nonfinite candidate output')
        delta=np.abs(value.astype(np.float64)-reference.astype(np.float64));ok=np.all(delta<=3e-5+1e-4*np.abs(reference))
        if not ok:raise ValueError('independent FP64 mismatch: '+case['id'])
        actual_sha=hashlib.sha256(value.tobytes()).hexdigest()
        if actual_sha!=row['output_sha256'] or hashes!=row['input_hashes']:raise ValueError('QA snapshot digest mismatch')
        ledger[case_key(case)]={'output_sha256':actual_sha,'input_hashes':hashes,'case':case,'binary_sha256':report['binary_sha256'],'max_abs_error':float(delta.max())}
        witness(row,case,ledger);rows.append({'case':case['id'],'max_abs_error':float(delta.max()),'input_immutable':True,'offset_guard':case['offset_guard'],'producer_consumer_stream':True})
    return rows


def expected_before(cases):
    expected={}
    for case in cases:
        cpu,hashes=input_facts(case);expected[case['id']]=((cpu[0].double()@cpu[1].double()).float().numpy(),hashes)
    return expected


def validate_aa(report,record,contract,workloads,ledger,session):
    if report is None or report.get('session')!=session or report.get('graph_launches')!=32 or report.get('replays')!=12:raise ValueError('AA protocol mismatch')
    for collection in ['checks_before','checks_after']:
        if len(report.get(collection,[]))!=len(contract):raise ValueError('AA contract witnesses missing')
        for row,case in zip(report[collection],contract):witness(row,case,ledger)
    expected_grid={(w['id'],c) for w in workloads for c in [1,16]}
    rows=report.get('candidate_aa',[])
    if len(rows)!=16 or {(r['workload'],r['input_count']) for r in rows}!=expected_grid:raise ValueError('AA grid incomplete')
    unstable=[]
    for row in rows:
        shape=next(w['shape_mnk'] for w in workloads if w['id']==row['workload']);count=row['input_count'];order=['incumbent','candidate'] if session%2==0 else ['candidate','incumbent']
        if row['shape_mnk']!=shape or row['capture_order']!=order:raise ValueError('AA shape/capture order differs')
        expected_values=[ledger[case_key({'shape_mnk':shape,'seed':701+i,'distribution':'random','offset_guard':False})] for i in range(count)]
        if row['input_hashes']!=[v['input_hashes'] for v in expected_values]:raise ValueError('graph input snapshot mismatch')
        for label in ['incumbent','candidate']:
            if row['output_hashes'][label]!=[expected_values[i%count]['output_sha256'] for i in range(32)]:raise ValueError('graph output not in independently verified ledger')
        mean,deviation=group_log_ratio(row,session);gm=math.exp(mean)
        if deviation>=.02 or abs(gm-1)>=.015:unstable.append({'session':session,'workload':row['workload'],'input_count':count,'geomean':gm,'max_pair_deviation':deviation})
    return unstable
