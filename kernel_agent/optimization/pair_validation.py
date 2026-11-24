"""Reference-free worker witnesses checked against two host-only FP64-validated ledgers."""
import hashlib,math,struct
from kernel_agent.bounded_io import confirmed_removed
from .cases import case_key
from .identity import group_log_ratio
from .pair_contract import ROLES,pair_id


def require(ok,message):
    if not ok:raise ValueError(message)


def validate_ledgers(ledgers,cases,descriptor):
    require(set(ledgers)==set(ROLES),'two oracle ledgers required')
    wanted={case_key(c) for c in cases}
    for role in ROLES:
        require(set(ledgers[role])==wanted,'oracle cases incomplete')
        for case in cases:
            value=ledgers[role][case_key(case)]
            require(value['case']==case and value['binary_sha256']==descriptor['binary_sha256'][role] and math.isfinite(value['max_abs_error']) and value['max_abs_error']>=0,'oracle source/case/binary binding differs')


def witness(row,case,ledger):
    require(all(row.get(k)==v for k,v in case.items()),'contract case witness mismatch')
    value=ledger[case_key(case)];guards=[hashlib.sha256(struct.pack('<e',7.0)*34).hexdigest()]*2 if case['offset_guard'] else []
    require(row['output_sha256']==value['output_sha256'] and row['input_hashes']==value['input_hashes'] and row['guard_hashes']==guards and row['fresh_output'] is True and row['producer_consumer_stream'] is True,'unvalidated output/input/guard/stream witness')


def validate_report(report,record,contract,workloads,ledgers,descriptor,mode,session):
    require(report is not None and report['status']=='PASS' and report['mode']==mode and report['session']==session and report['graph_launches']==32 and report['replays']==12,'pair session/protocol incomplete')
    require(record['status']=='WITNESSES_READY' and record['exit']==0 and record['reason'] is None and record['protected_unchanged'] is True and record['cleanup']['removed'] is True and confirmed_removed(record['cleanup']['after']),'pair worker/lifecycle rejected')
    hashes=descriptor['binary_sha256'];pid=pair_id(descriptor)
    require(record['source_binary_sha256']==hashes['candidate'] and record['incumbent_sha256']==hashes['incumbent'] and record['pair_sha256']==pid,'record artifact binding differs')
    require(report['binary_sha256']==hashes['candidate'] and report['candidate_binary_sha256']==hashes['candidate'] and report['incumbent_binary_sha256']==hashes['incumbent'] and report['pair_sha256']==pid,'report artifact binding differs')
    require(report['capability']==[8,6] and report['torch']=='2.14.0+cu130' and report['cuda']=='13.0' and report['cpu_reference_available'] is False and report['module_identity_distinct'] is True and report['function_identity_distinct'] is True,'dual import/device/reference boundary invalid')
    require(report['module_paths']=={r:'/artifact/'+r+'/nka_gemm_v1.so' for r in ROLES} and report['module_names']=={r:'_nka_'+r+'.nka_gemm_v1' for r in ROLES},'loader namespace/path differs')
    require(report['backends']=={'incumbent':'template','candidate':'template'},'diagnostic/different-backend evidence cannot screen a source candidate')
    expected={'artifact/'+name:digest for name,digest in descriptor['worker_sha256'].items()}
    expected.update({'artifact/'+r+'/nka_gemm_v1.so':hashes[r] for r in ROLES})
    protected=record['protected_hashes'];require(set(protected)==set(expected)|{'inputs/request.json'} and all(protected[p]==h for p,h in expected.items()),'mounted worker/source changed')
    for phase in ['checks_before','checks_after']:
        collections={'candidate':report[phase]} if mode=='aa' else report[phase]
        require(set(collections)==({'candidate'} if mode=='aa' else set(ROLES)),'missing two-sided checks')
        for role,rows in collections.items():
            require(len(rows)==len(contract),'contract coverage incomplete')
            for row,case in zip(rows,contract):witness(row,case,ledgers[role])
    rows=report['candidate_aa' if mode=='aa' else 'measurements'];grid={(w['id'],count) for w in workloads for count in [1,16]}
    require(len(rows)==len(grid) and {(r['workload'],r['input_count']) for r in rows}==grid,'pair workload/cache coverage incomplete')
    unstable=[]
    for row in rows:
        shape=next(w['shape_mnk'] for w in workloads if w['id']==row['workload']);count=row['input_count'];order=['incumbent','candidate'] if session%2==0 else ['candidate','incumbent']
        require(row['shape_mnk']==shape and row['capture_order']==order,'pair capture order/shape differs')
        for label in ROLES:
            role='candidate' if mode=='aa' else label
            values=[ledgers[role][case_key({'shape_mnk':shape,'seed':701+i,'distribution':'random','offset_guard':False})] for i in range(count)]
            require(row['input_hashes']==[v['input_hashes'] for v in values] and row['output_hashes'][label]==[values[i%count]['output_sha256'] for i in range(32)],'graph witnesses not in role-specific validated ledger')
        mean,deviation=group_log_ratio(row,session)
        if mode=='aa' and (deviation>=.02 or abs(math.exp(mean)-1)>=.015):unstable.append({'session':session,'workload':row['workload'],'input_count':count,'geomean':math.exp(mean),'max_pair_deviation':deviation})
    return unstable
