"""Stateful small trial proposals; GPU evaluation remains in established trusted tools."""
import argparse,hashlib,json
from pathlib import Path
from kernel_agent.optimization.identity import canonical,identity,strict_json
from kernel_agent.optimization.feedback_channels import prepare_channel,execution_feedback
from kernel_agent.optimization.call_budget import CallBudget
from kernel_agent.optimization.live_call import dispatch
from kernel_agent.optimization.patches import apply_reviewable_patch
from kernel_agent.registry import atomic_json
ROOT=Path(__file__).resolve().parents[2]
TERMINAL={'REJECT','DUPLICATE','SCREENING_PASS'}


def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def prepare(root):
    root.mkdir(exist_ok=False);profiles={'gemm':ROOT/'evidence/20261123-gemm-mini-baseline-profile/recovered/remote/profile-recovered.json','mla':ROOT/'evidence/20261123-mla-live-config/profile.json'}
    sources={'gemm':(ROOT/'kernels/gemm/wmma_sm86.cu').read_text(),'mla':(ROOT/'kernel_agent/operators/flashinfer_backend.py').read_text()+'\n'+(ROOT/'kernel_agent/operators/partition.py').read_text()};tasks={}
    for op in ['gemm','mla']:
        source=sources[op];profile=json.loads(profiles[op].read_text());assert profile['profile_wall_seconds']>0
        task={'source':source,'source_sha256':hashlib.sha256(source.encode()).hexdigest(),'contract':('FP16 row-major finite +/-4 -> FP32 GEMM accumulation/output. Public anchor M=N=K=512 (dev-01), fixed original 4-warps WMMA source. All eight frozen dev workloads and tails remain protection checks. Existing original AA and K2-source AA failed; do not assume stable gain.' if op=='gemm' else 'FP16 finite +/-4 MLA core Z/natural LSE, R512/Drope64 original scale1/sqrt192. Public anchor B1 H128 L8192 (mla-dev-04), initial page16/split1 with SDK internal split planner. All8 frozen dev workloads protected. Initial AA failed SM86 noise; exact new page16/split2 QA/AA passed but AB GM0.569/0.573, rejected. Other task results/heldout remain unavailable.'),'gpu':'RTX3070 SM86, Torch2.14.0+cu130, CUDA13','allowed_changes':('Only one small structural CUDA device-body change inside supplied gemm_wmma template. Preserve complete host launcher/includes/license/warp mapping/FP32 MMA accumulation order, no evaluator/contract/tolerance/dependency edits, no asm/printf/host effects/device allocation/loops without finite input bounds. Return one fixed-file Git unified diff. The patch MUST begin with exactly diff --git a/kernels/gemm/wmma_sm86.cu b/kernels/gemm/wmma_sm86.cu, then --- a/kernels/gemm/wmma_sm86.cu and +++ b/kernels/gemm/wmma_sm86.cu, then real @@ hunks. Omission is rejected before compilation. parent_id current. Hypothesis only, never claim unmeasured gain.' if op=='gemm' else 'Only page_size in [1,16,64], splits in [1,2,4], reason<=1000chars. This maps to fixed SDK native CUDA and explicit split/combine graph. No free host/native source. Planning/layout/H2D excluded, full split+combine included. SDK already internally splits; extra splits can add launches. Same transform space for B/C, cite only supplied fact IDs.'),'model':'deepseek-flash','temperature':.2,'max_tokens':768}
        tasks[op]=task;atomic_json(root/(op+'-profile.json'),profile);atomic_json(root/(op+'-task.json'),task)
        for method in ['A','B','C']:(root/op/method).mkdir(parents=True)
    names=[Path(__file__),ROOT/'kernel_agent/optimization/live_call.py',ROOT/'kernel_agent/optimization/feedback_channels.py',ROOT/'kernel_agent/optimization/call_budget.py',ROOT/'kernel_agent/optimization/patches.py',ROOT/'kernel_agent/optimization/identity.py',ROOT/'workloads/mla_measurement_v2.json',ROOT/'workloads/gemm_measurement_v1.json']
    for p in names:
        t=root/'sources'/p.relative_to(ROOT);t.parent.mkdir(parents=True,exist_ok=True);t.write_bytes(p.read_bytes())
    protocol={'scope':'P3-04 corrected v2 small trial, not formal P6','anchors':{'gemm':'dev-01','mla':'mla-dev-04'},'methods':['A','B','C'],'independent_searches_per_cell':1,'proposals_per_cell':6,'total_attempts':36,'maximum_new_api_calls':24,'model':'deepseek-flash','temperature':.2,'max_tokens':768,'tasks':{op:sha(root/(op+'-task.json')) for op in tasks},'profiles':{op:sha(root/(op+'-profile.json')) for op in tasks},'controls':{str(p.relative_to(ROOT)):sha(p) for p in names},'search_parent_policy':'fixed frozen initial parent, all candidate evidence compared against same start; select best externally after six attempts','A_sequences':{'gemm':[1,2,8,4,1,2],'mla':[1,64,16,1,64,16]},'duplicate_policy':'consume attempt, exact software/source/config/hardware/protocol identity can reuse audited result including rejection; never resample known rejected identities','failure_policy':'invalid/compile/QA/AA/AB failures consume attempt; uncertain API has no retry; next step needs terminal external result','C_profile_policy':'initial parent profile only, actual source/device/binary-bound metrics and wall cost; no hardware feedback in B','candidate_execution':'GEMM source requires trusted semantic review SHA and existing bounded Docker tools; MLA reviewed SDK configurations only','heldout_read':False,'activation':False,'production_release':False};atomic_json(root/'protocol.json',protocol);print(json.dumps({'status':'PREPARED','attempts':36,'maximum_new_api_calls':24,'new_model_calls':0}))


def verify(root):
    p=json.loads((root/'protocol.json').read_text());assert all(sha(ROOT/n)==h for n,h in p['controls'].items());assert all(sha(root/(op+'-task.json'))==h for op,h in p['tasks'].items());assert all(sha(root/(op+'-profile.json'))==h for op,h in p['profiles'].items());return p


def propose(root,op,method,step):
    protocol=verify(root)
    if not 0<=step<6:raise ValueError('outside frozen six-attempt budget')
    branch=root/op/method;history=[]
    for i in range(step):
        previous=json.loads((branch/str(i)/'decision.json').read_text())
        if previous['status'] not in TERMINAL:raise ValueError('previous step not terminal')
        history.append(execution_feedback(previous['execution_feedback']))
    folder=branch/str(step);folder.mkdir(exist_ok=False);task=json.loads((root/(op+'-task.json')).read_text());base={'op':op,'method':method,'step':step,'parent_source_sha256':task['source_sha256'],'new_model_calls':0,'status':'PROPOSED','production_release':False}
    if method=='A':
        config={'warps':protocol['A_sequences'][op][step]} if op=='gemm' else {'page_size':protocol['A_sequences'][op][step],'splits':1};proposal={'config':config,'candidate_id':hashlib.sha256(canonical({'op':op,'source':task['source_sha256'],'config':config})).hexdigest()};atomic_json(folder/'proposal.json',proposal);atomic_json(folder/'result.json',{**base,'status':'CONTROLLED_PARAMETER_PROPOSAL',**proposal});print(json.dumps({**base,**proposal}));return
    channel=prepare_channel(method,task,history,profile_loader=lambda:json.loads((root/(op+'-profile.json')).read_text()));schema={'schema_version':1,'parent_id':'current','patch':'diff --git a/kernels/gemm/wmma_sm86.cu b/kernels/gemm/wmma_sm86.cu\n--- a/kernels/gemm/wmma_sm86.cu\n+++ b/kernels/gemm/wmma_sm86.cu\n@@ -1,1 +1,1 @@\n...','reason':'one testable hypothesis <=1000chars'} if op=='gemm' else {'schema_version':1,'parent_id':'current','page_size':16,'splits':1,'reason':'one hypothesis <=1000chars'};request={'model':'deepseek-flash','temperature':.2,'max_tokens':768,'messages':[{'role':'system','content':'Return only JSON with exactly this schema: '+json.dumps(schema)},{'role':'user','content':json.dumps(channel['payload'])}]};atomic_json(folder/'channel.json',channel);atomic_json(folder/'request.json',request);budget=CallBudget(ROOT/'evidence/20261122-full-project-model-calls',ROOT/'evidence/20261122-full-project-readiness/model-budget.json');response,cost=dispatch(request,folder,budget,'mini-v2-'+op+'-'+method+'-'+str(step),sha(folder/'request.json'),ROOT/'.private/deepseek-api-key');base.update(new_model_calls=1,usage=cost['usage'])
    try:
        if response['choices'][0]['finish_reason']!='stop':raise ValueError('incomplete proposal')
        content=response['choices'][0]['message']['content']
        if op=='gemm':
            try:proposal=apply_reviewable_patch(content,task['source'].encode())
            except ValueError as error:
                atomic_json(folder/'strict-rejection.json',{'error':str(error)});proposal=apply_reviewable_patch(content,task['source'].encode(),recount=True)
            (folder/'wmma_sm86.cu').write_bytes(proposal.pop('updated_source'));proposal['candidate_id']=hashlib.sha256(canonical({'source_sha256':proposal['source_sha256'],'warps':4})).hexdigest();base['status']='NEEDS_TRUSTED_SOURCE_REVIEW'
        else:
            d=strict_json(content)
            if not isinstance(d,dict) or set(d)!={'schema_version','parent_id','page_size','splits','reason'} or type(d['schema_version']) is not int or d['schema_version']!=1 or d['parent_id']!='current' or type(d['page_size']) is not int or d['page_size'] not in [1,16,64] or type(d['splits']) is not int or d['splits'] not in [1,2,4] or not isinstance(d['reason'],str) or not 1<=len(d['reason'])<=1000:raise ValueError('invalid controlled MLA proposal')
            proposal={'proposal':d,'config':{k:d[k] for k in ['page_size','splits']},'candidate_id':hashlib.sha256(canonical({'source':task['source_sha256'],'page_size':d['page_size'],'splits':d['splits']})).hexdigest()};base['status']='CONTROLLED_PARAMETER_PROPOSAL'
        atomic_json(folder/'proposal.json',proposal);atomic_json(folder/'result.json',{**base,**{k:v for k,v in proposal.items() if k in ['candidate_id','config','source_sha256']}})
    except (ValueError,KeyError,TypeError) as error:
        base.update(status='REJECT',reason=str(error)[:1000]);atomic_json(folder/'result.json',base);cid=sha(folder/'response.json');atomic_json(folder/'decision.json',{'status':'REJECT','reason':'invalid_proposal','execution_feedback':{'candidate_id':cid,'compile_status':'NOT_RUN','compile_error':None,'correctness':'NOT_RUN','latency_ms':[],'evaluation_wall_seconds':0.},'evidence':[str(folder/'result.json')],'production_release':False})
    print(json.dumps(base))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=['prepare','propose']);p.add_argument('--root',type=Path,required=True);p.add_argument('--op',choices=['gemm','mla']);p.add_argument('--method',choices=['A','B','C']);p.add_argument('--step',type=int);a=p.parse_args()
    if a.action=='prepare':prepare(a.root)
    else:
        if a.op is None or a.method is None or a.step is None:p.error('--op --method --step required')
        propose(a.root,a.op,a.method,a.step)
