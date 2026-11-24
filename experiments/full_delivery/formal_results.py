"""Trusted ingestion outside the frozen proposal controller; never runs CUDA."""
import argparse,json,os,time,re
from pathlib import Path
from kernel_agent.registry import atomic_json
from experiments.full_delivery.formal_study import ROOT,verify,complete,sha
from experiments.full_delivery.study_cache import validate_entry,_latencies


def reserve(root,folder,evidence):
    verify(root);proposal=json.loads((folder/'proposal.json').read_text());cid=proposal['descriptor']['id']
    path=root/'evaluation-reservations'/(cid+'.json')
    value={'candidate_id':cid,'proposal':str(folder.relative_to(ROOT)),'evidence':str(evidence.relative_to(ROOT)),'reserved_at':time.time(),'automatic_retry':False}
    with path.open('x') as f:json.dump(value,f,indent=2);f.flush();os.fsync(f.fileno())
    return value


def ingest_gemm(root,folder,evidence):
    verify(root);proposal=json.loads((folder/'proposal.json').read_text());v=json.loads((evidence/'verification.json').read_text());assert v['status']=='PASS' and v['candidate_id']==proposal['descriptor']['id'] and v['source_sha256']==proposal['source_change']['source_sha256']
    run=evidence/'remote/run';state=json.loads((run/'run.json').read_text());work=run/'evaluation';result=json.loads((work/'result.json').read_text());assert result['reason']==v['decision']['reason'] and state['candidate_id']==v['candidate_id']
    mode='ab' if (work/'ab-0/report.json').exists() else 'aa';r=json.loads((work/(mode+'-0')/'report.json').read_text());reason={'candidate_aa_unstable':'aa_unstable','insufficient_gain_or_regression':'insufficient_gain','frozen_screening_pass':'screening_pass'}.get(result['reason'],result['reason']);assert reason in ['aa_unstable','insufficient_gain','screening_pass']
    entry={'descriptor':proposal['descriptor'],'verdict':'SCREENING_PASS' if result['verdict']=='KEEP' else 'REJECT','reason':reason,'compile_status':'PASS','correctness':'PASS','latencies':_latencies(r,'measurements' if mode=='ab' else 'candidate_aa',384),'proofs':{str(p.relative_to(ROOT)):sha(p) for p in [evidence/'verification.json',run/'run.json',work/'result.json']},'physical_binary_sha256':v['binary_sha256'],'physical_binary':str(work/'build/artifacts/nka_gemm_v1.so'),'evaluation_wall_seconds':v['all_worker_wall_seconds'],'build_wall_seconds':v['build_wall_seconds'],'raw_gate_reason':result['reason']}
    return store(root,folder,entry)


def store(root,folder,entry,scope='fresh'):
    verify(root);validate_entry(entry);path=root/'cache-updates'/(entry['descriptor']['id']+'.json')
    if path.exists():assert json.loads(path.read_text())==entry
    else:atomic_json(path,entry)
    p=json.loads((folder/'proposal.json').read_text());return complete(root,p['op'],p['workload'],p['repetition'],p['method'],p['step'],entry,scope)


def identifier_only(root,folder,reference,mapping):
    """Exact whole-source proof, not a heuristic semantic-equivalence classifier."""
    source=folder/'wmma_sm86.cu';assert mapping and all(re.fullmatch(r'[A-Za-z_]\w*',x) for pair in mapping.items() for x in pair);assert len(set(mapping.values()))==len(mapping)
    transformed=re.sub(r'\b(?:'+'|'.join(re.escape(x) for x in mapping)+r')\b',lambda m:mapping[m[0]],source.read_text());assert transformed==reference.read_text()
    review={'verdict':'REJECT','reason':'static_invalid','source_sha256':sha(source),'reference_source':str(reference.relative_to(ROOT)),'reference_sha256':sha(reference),'identifier_mapping':mapping,'verification':'Replacing only these complete identifier tokens makes the entire source byte-for-byte equal to reference. Frozen objective excludes local-identifier-only variants; no performance or QA inheritance.','compile_status':'NOT_RUN','correctness':'NOT_RUN'};atomic_json(folder/'static-review.json',review);proposal=json.loads((folder/'proposal.json').read_text());return store(root,folder,{'descriptor':proposal['descriptor'],'verdict':'REJECT','reason':'static_invalid','compile_status':'NOT_RUN','correctness':'NOT_RUN','latencies':{},'proofs':{str((folder/'static-review.json').relative_to(ROOT)):sha(folder/'static-review.json'),str(reference.relative_to(ROOT)):sha(reference)}},'static')


def resolve_pending(root):
    verify(root);cache=json.loads((root/'cache.initial.json').read_text());cache.update({p.stem:json.loads(p.read_text()) for p in (root/'cache-updates').glob('*.json')});resolved=[]
    for p in sorted((root/'searches').rglob('proposal.json')):
        if (p.parent/'decision.json').exists():continue
        d=json.loads(p.read_text());entry=cache.get(d['descriptor']['id'])
        if entry:resolved.append(complete(root,d['op'],d['workload'],d['repetition'],d['method'],d['step'],entry))
    return resolved


def evaluation_evidence(root,cid):
    reservation=json.loads((root/'evaluation-reservations'/(cid+'.json')).read_text());correction=root/'evaluation-reservations'/(cid+'.transport-correction.json')
    if correction.exists():
        d=json.loads(correction.read_text());assert d['candidate_id']==cid and d['old_evidence']==reservation['evidence'] and d['prior_execution_confirmed_absent'] is True and all(sha(ROOT/n)==h for n,h in d['proofs'].items());return ROOT/d['new_evidence']
    return ROOT/reservation['evidence']


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=['reserve','ingest-gemm','resolve']);p.add_argument('--root',type=Path,required=True);p.add_argument('--folder',type=Path);p.add_argument('--evidence',type=Path);a=p.parse_args();root=a.root.resolve();folder=a.folder.resolve() if a.folder else None;evidence=a.evidence.resolve() if a.evidence else None
    print(json.dumps(reserve(root,folder,evidence) if a.action=='reserve' else ingest_gemm(root,folder,evidence) if a.action=='ingest-gemm' else resolve_pending(root)))
