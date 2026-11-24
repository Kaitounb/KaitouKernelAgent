"""Allowlisted experiment input adapter. A/B never invoke a profile loader."""
import hashlib,json,math,re


def canonical(value):return json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False)


def _exact(value,keys):
    if not isinstance(value,dict) or set(value)!=set(keys):raise ValueError('feedback fields differ from allowlist')


def _number(value,minimum=0):
    if type(value) not in [int,float] or not math.isfinite(value) or value<minimum:raise ValueError('invalid numeric feedback')
    return value


def execution_feedback(value):
    _exact(value,['candidate_id','compile_status','compile_error','correctness','latency_ms','evaluation_wall_seconds'])
    if not re.fullmatch('[0-9a-f]{64}',value['candidate_id']):raise ValueError('candidate binding invalid')
    if value['compile_status'] not in ['PASS','FAIL','NOT_RUN'] or value['compile_error'] not in [None,'syntax','type','resource_limit','timeout','link','other']:raise ValueError('compiler feedback must be classified; raw resource logs excluded')
    if value['correctness'] not in ['PASS','FAIL','NOT_RUN']:raise ValueError('invalid correctness feedback')
    if not isinstance(value['latency_ms'],list) or len(value['latency_ms'])>128:raise ValueError('invalid latency feedback')
    for item in value['latency_ms']:_number(item,1e-12)
    _number(value['evaluation_wall_seconds'])
    return json.loads(canonical(value))


def profile_feedback(value):
    _exact(value,['facts','provenance','profile_wall_seconds'])
    if not isinstance(value['facts'],list) or not 1<=len(value['facts'])<=256:raise ValueError('profile facts required')
    if not isinstance(value['provenance'],list) or not value['provenance']:raise ValueError('profile provenance required')
    ids=set()
    for fact in value['facts']:
        _exact(fact,['id','metric','value','unit','status'])
        if not re.fullmatch('F[1-9][0-9]*',fact['id']) or fact['id'] in ids:raise ValueError('invalid or duplicate fact ID')
        ids.add(fact['id'])
        if not re.fullmatch('(sm|smsp|dram|lts|gpu|launch)__[A-Za-z0-9_.]+',fact['metric']):raise ValueError('unknown metric namespace')
        if fact['status'] not in ['observed','unavailable'] or not isinstance(fact['unit'],str) or len(fact['unit'])>64:raise ValueError('invalid metric status/unit')
        if fact['status']=='observed':_number(fact['value'])
        elif fact['value'] is not None:raise ValueError('unavailable metric must not be zero-filled')
    for row in value['provenance']:
        _exact(row,['csv_sha256','binary_sha256','upstream_profiler_sha256'])
        if any(not isinstance(d,str) or not re.fullmatch('[0-9a-f]{64}',d) for d in row.values()):raise ValueError('profile provenance hash missing')
    # Legacy profiles can lack wall-clock accounting. Preserve unknown, never zero-fill.
    if value['profile_wall_seconds'] is not None:_number(value['profile_wall_seconds'])
    return json.loads(canonical(value))


def prepare_channel(method,task,executions,*,profile_loader=None):
    if method not in ['A','B','C']:raise ValueError('unknown experiment method')
    _exact(task,['source','source_sha256','contract','gpu','allowed_changes','model','temperature','max_tokens'])
    if not isinstance(task['source'],str) or hashlib.sha256(task['source'].encode()).hexdigest()!=task['source_sha256']:raise ValueError('source binding differs')
    if task['model']!='deepseek-flash' or task['temperature']!=.2 or task['max_tokens']!=768:raise ValueError('shared model budget changed')
    for name in ['contract','gpu','allowed_changes']:
        if not isinstance(task[name],str) or not 0<len(task[name])<=8192:raise ValueError('invalid common task')
    common=json.loads(canonical(task));history=[execution_feedback(x) for x in executions]
    payload={'task':common,'feedback':{'executions':history}}
    profile_reads=0;profile_seconds=0
    if method=='C':
        if profile_loader is None:raise ValueError('C requires profile loader')
        profile=profile_feedback(profile_loader());profile_reads=1;profile_seconds=profile['profile_wall_seconds'];payload['feedback']['profile']=profile
    return {'method':method,'payload':payload,'common_sha256':hashlib.sha256(canonical(common).encode()).hexdigest(),'payload_sha256':hashlib.sha256(canonical(payload).encode()).hexdigest(),'profile_loader_calls':profile_reads,'profile_wall_seconds':profile_seconds,'new_model_calls':0}
