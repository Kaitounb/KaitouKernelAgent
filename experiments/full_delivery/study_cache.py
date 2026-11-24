"""Read only exact pre-study outcomes; no model history or hardware metrics in cache."""
import hashlib,json
from pathlib import Path
from kernel_agent.optimization.identity import canonical,identity
ROOT=Path(__file__).resolve().parents[2]


def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def descriptor(op,*,config=None,source_sha=None):
    if op=='gemm':
        value=identity(config['warps'] if config else 4)
        if source_sha:value['files']['kernels/gemm/wmma_sm86.cu']=source_sha;value['id']=hashlib.sha256(canonical({k:v for k,v in value.items() if k!='id'})).hexdigest()
        return value
    body={'template':'fixed-flashinfer-mla-v1','config':config,'source_sha256':{n:sha(ROOT/n) for n in ['kernel_agent/operators/flashinfer_backend.py','kernel_agent/operators/partition.py','workloads/mla_contract_v1.json','workloads/mla_measurement_v2.json']},'native_sha256':['cdc4c3ba11ef61b7e2778f93af247bc0ac53027daa8035c55b541e6ea3fd601e','31be5df9f2aa84fc721827f4097fc16a4d6f500e364d951b8bad844b4f0c3603'],'environment':{'capability':[8,6],'torch':'2.14.0+cu130','cuda':'13.0'}};return {**body,'id':hashlib.sha256(canonical(body)).hexdigest()}


def _latencies(report,key,divisor,role='candidate'):
    return {r['workload']:[x['elapsed_ms']/divisor for x in r['segments'] if x['label']==role] for r in report[key] if r['input_count']==1}


def initial_entries():
    entries={}
    def add(desc,reason,proofs,latencies=None,**extra):
        entries[desc['id']]={'descriptor':desc,'verdict':'DUPLICATE' if reason=='no_change' else 'REJECT','reason':reason,'compile_status':'PASS','correctness':'PASS','latencies':latencies or {},'proofs':{str(p.relative_to(ROOT)):sha(p) for p in proofs},**extra}
    for warps in [1,2,4,8]:
        desc=descriptor('gemm',config={'warps':warps})
        if warps==1:
            p=ROOT/'evidence/20261123-mini-gemm-A-warp1/archive/remote';result=json.loads((p/'result.json').read_text());assert result['decision']['reason']=='candidate_aa_unstable';r=json.loads((p/'aa-0.json').read_text());assert r['candidate']==desc;add(desc,'aa_unstable',[p/'result.json',p/'aa-0.json'],_latencies(r,'rows',384))
        elif warps==2:
            p=ROOT/'evidence/20261122-p1-04-feedback/evaluation';r=json.loads((p/'run/session-0.json').read_text());d=json.loads((p/'decision.json').read_text());assert r['candidate']==desc and d['verdict']=='REJECT';add(desc,'insufficient_gain',[p/'decision.json',p/'run/session-0.json'],_latencies(r,'measurements',384))
        elif warps==8:
            p=ROOT/'evidence/20261121-p1-03-deepseek/evaluation-retry1';r=json.loads((p/'run/session-0.json').read_text());d=json.loads((p/'decision.json').read_text());assert r['candidate']==desc and d['detail']=='candidate A/A unstable';add(desc,'aa_unstable',[p/'decision.json',p/'run/session-0.json'],_latencies(r,'candidate_aa',384))
        else:
            p=ROOT/'evidence/20261122-p1-04-feedback/evaluation/run/session-0.json';r=json.loads(p.read_text());assert r['parent']==desc;add(desc,'no_change',[p],_latencies(r,'measurements',384,'incumbent'))
    source='239802123f6e01ccd229840aa595dfa1b4c8644db2534e246b535ee2cf89b787';p=ROOT/'evidence/20261123-mini-gemm-pointer-source';v=json.loads((p/'verification.json').read_text());desc=descriptor('gemm',source_sha=source);assert desc['id']==v['candidate_id'] and v['decision']['reason']=='candidate_aa_unstable';r=json.loads((p/'remote/run/evaluation/aa-0/report.json').read_text());add(desc,'aa_unstable',[p/'verification.json',p/'remote/run/run.json'],_latencies(r,'candidate_aa',384),physical_binary_sha256=v['binary_sha256'],physical_binary=str(p/'remote/run/evaluation/build/artifacts/nka_gemm_v1.so'))
    # Reviewed prefetch source's completed deferred timing supplements original busy batch.
    p=ROOT/'evidence/20261122-feedback-B-installed-fixed-deps/remote/run';state=json.loads((p/'run.json').read_text());source=state['source_sha256'];desc=descriptor('gemm',source_sha=source);assert desc['id']==state['candidate_id'];q=ROOT/'evidence/20261123-feedback-B-deferred-timing';v=json.loads((q/'verification.json').read_text());
    if v['status']=='PASS_CLOSED_LOOP_REJECT':
        deferred=next((q/'remote').glob('**/ab-0/report.json'),None)
        if deferred is not None:add(desc,'insufficient_gain',[q/'verification.json',p/'run.json'],_latencies(json.loads(deferred.read_text()),'measurements',384),physical_binary_sha256=state.get('binary_sha256','690c6c2f5312e1074b206f3cc20fc70d20ec615ce91ff10956e850ae37bba138'),physical_binary=str(p/'evaluation/build/artifacts/nka_gemm_v1.so'))
    # Original K2 transformation is an immutable, known unstable source identity.
    p=ROOT/'evidence/20261122-p3-01-source-sanitizer-channel';review=json.loads((p/'source-review.json').read_text());source=review['candidate']['candidate']['files']['kernels/gemm/wmma_sm86.cu']
    if source:
        desc=descriptor('gemm',source_sha=source);decision=json.loads((p/'decision.json').read_text());assert decision['verdict']=='REJECT' and desc==review['candidate']['candidate'];r=json.loads((p/'remote/run/aa-0/report.json').read_text());add(desc,'aa_unstable',[p/'verification.json',p/'decision.json'],_latencies(r,'candidate_aa',384))
    for config,path,reason in [({'page_size':16,'splits':1},'20261123-mla-live-split2-sm86','no_change'),({'page_size':16,'splits':2},'20261123-mla-live-split2-sm86','insufficient_gain'),({'page_size':1,'splits':1},'20261123-mini-mla-page1-sm86','aa_unstable'),({'page_size':64,'splits':1},'20261123-mini-mla-page64-sm86','aa_unstable')]:
        p=ROOT/'evidence'/path;v=json.loads((p/'verification.json').read_text());assert v['status']=='PASS';mode='ab' if (p/'remote/ab-0.json').exists() else 'aa';r=json.loads((p/f'remote/{mode}-0.json').read_text());role='incumbent' if reason=='no_change' else 'candidate';add(descriptor('mla',config=config),reason,[p/'verification.json',p/f'remote/{mode}-0.json'],_latencies(r,'rows',1024,role))
    return entries


def validate_entry(entry):
    assert entry['descriptor']['id']==hashlib.sha256(canonical({k:v for k,v in entry['descriptor'].items() if k!='id'})).hexdigest();assert all(sha(ROOT/n)==h for n,h in entry['proofs'].items())
    if entry.get('physical_binary'):assert sha(entry['physical_binary'])==entry['physical_binary_sha256']
    return entry
