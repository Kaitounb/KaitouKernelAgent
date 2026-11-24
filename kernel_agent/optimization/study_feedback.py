"""Formal-study context: explicit safe outcomes and own patches, no B hardware channel."""
import hashlib,json,math,re
from .feedback_channels import prepare_channel,execution_feedback

REASONS={'invalid_json','invalid_patch','static_invalid','duplicate','no_change','compile_failed','correctness_failed','sanitizer_failed','timed_out','cleanup_unconfirmed','aa_unstable','insufficient_gain','screening_pass','gpu_busy_unsampled','not_supported','model_call_failed'}


def _sha(value):
    if not isinstance(value,str) or not re.fullmatch('[0-9a-f]{64}',value):raise ValueError('study source identity invalid')


def outcome(value,op):
    if not isinstance(value,dict) or set(value)!={'step','candidate_id','verdict','reason','source_change','evaluation_scope'}:raise ValueError('study outcome fields outside allowlist')
    if type(value['step']) is not int or not 0<=value['step']<12 or value['verdict'] not in ['REJECT','DUPLICATE','SCREENING_PASS'] or value['reason'] not in REASONS or value['evaluation_scope'] not in ['fresh','exact_cache','static']:raise ValueError('invalid study outcome')
    _sha(value['candidate_id']);change=value['source_change']
    if op=='gemm':
        if not isinstance(change,dict) or set(change)!={'source_sha256','patch'} or not isinstance(change['patch'],str) or len(change['patch'].encode())>32768 or '\x00' in change['patch']:raise ValueError('invalid own CUDA patch history')
        _sha(change['source_sha256'])
    elif op=='mla':
        if not isinstance(change,dict) or set(change)!={'page_size','splits'} or type(change['page_size']) is not int or change['page_size'] not in [1,16,64] or type(change['splits']) is not int or change['splits'] not in [1,2,4]:raise ValueError('invalid own SDK configuration history')
    else:raise ValueError('unknown study operator')
    return json.loads(json.dumps(value,allow_nan=False))


def prepare(method,op,task,baseline,executions,outcomes,*,profile_loader=None,profile_context_loader=None):
    if len(executions)!=len(outcomes) or len(outcomes)>12:raise ValueError('history coverage differs')
    history=[outcome(x,op) for x in outcomes]
    if [x['step'] for x in history]!=list(range(len(history))):raise ValueError('history not contiguous')
    if any(x['candidate_id']!=y['candidate_id'] for x,y in zip(history,executions)):raise ValueError('outcome/execution candidate differs')
    base=execution_feedback(baseline);channel=prepare_channel(method,task,executions,profile_loader=profile_loader);channel['payload']['feedback'].update(baseline_execution=base,outcomes=history)
    if method=='C' and profile_context_loader is not None:
        contexts=profile_context_loader();ids={x['id'] for x in channel['payload']['feedback']['profile']['facts']}
        if not isinstance(contexts,list) or not contexts:raise ValueError('profile context required')
        covered=[]
        binaries={x['binary_sha256'] for x in channel['payload']['feedback']['profile']['provenance']};sources={task['source_sha256']}|{x['source_change']['source_sha256'] for x in history if 'source_sha256' in x['source_change']}
        for item in contexts:
            if not isinstance(item,dict) or set(item)!={'kind','source_sha256','binary_sha256','fact_ids','kernel','scope'} or item['kind'] not in ['parent','candidate']:raise ValueError('profile context fields differ')
            _sha(item['source_sha256']);_sha(item['binary_sha256'])
            if item['binary_sha256'] not in binaries or item['source_sha256'] not in sources or item['kind']=='parent' and item['source_sha256']!=task['source_sha256']:raise ValueError('profile context source/binary differs')
            if not isinstance(item['fact_ids'],list) or not item['fact_ids'] or any(x not in ids for x in item['fact_ids']):raise ValueError('profile context fact binding differs')
            if any(not isinstance(item[k],str) or not 1<=len(item[k])<=4096 or any(c in item[k] for c in ['\x00','\n','\r']) for k in ['kernel','scope']):raise ValueError('invalid profile scope')
            covered+=item['fact_ids']
        if set(covered)!=ids or len(set(covered))!=len(covered):raise ValueError('profile context coverage differs')
        channel['payload']['feedback']['profile_contexts']=json.loads(json.dumps(contexts,allow_nan=False))
    channel['payload']['objective']={'goal':'reduce complete core latency relative to the fixed initial parent, subject to numerical/sanitizer/AA/guard gates','parent_policy':'Every patch applies to the supplied frozen initial source; use your own prior patches/outcomes to avoid repeats. A passed build/QA is not a passed timing gate.','keep_parent':'An explicit keep_parent action consumes one proposal and is a duplicate, not an error or gain.','stop_policy':'three consecutive duplicate/keep_parent or invalid attempts, first screening pass, at most12 proposals','cache_policy':'Exact validated identities reuse prior rejection including AA failures; local-identifier-only variants are static duplicates; never resample rejected identity.'}
    # The original common task hash deliberately remains independent of profile and history.
    channel['payload_sha256']=hashlib.sha256(json.dumps(channel['payload'],sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest();return channel
