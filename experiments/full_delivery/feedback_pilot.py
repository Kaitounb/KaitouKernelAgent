"""Two real B/C interface proposals from the same frozen public GEMM parent.

Historical execution/profile data qualify wiring only, not a fresh A/B/C experiment.
The existing provider and patch guard are reused. No generated CUDA is executed here.
"""
import argparse,hashlib,json,logging,os,time
from pathlib import Path
from kernel_agent.optimization.feedback_channels import prepare_channel
from kernel_agent.optimization.identity import identity
from kernel_agent.optimization.patches import apply_reviewable_patch
from kernel_agent.optimization.provider import provider_class
from kernel_agent.optimization.call_budget import CallBudget
ROOT=Path(__file__).resolve().parents[2]


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def prepare(root):
    root.mkdir(exist_ok=False);source=ROOT/'kernels/gemm/wmma_sm86.cu';parent=identity(4)
    task={'source':source.read_text(),'source_sha256':sha(source),'contract':'GEMM FP16 row-major input in [-4,4], FP32 accumulation/output. Current 4 warps. Public anchor M=N=K=512. Host dispatcher and cuBLAS tails unchanged. Historical feedback, no fresh performance claim.',
      'gpu':'RTX 3070 SM86, Torch2.14 CUDA13','allowed_changes':'One small CUDA-only structural change inside the supplied gemm_wmma device template, e.g. bounded K-loop prefetch or compiler unroll strategy. Preserve exact order of FP32 MMA accumulation, warp mapping and all output semantics. Preserve includes/license and complete host launcher. No new dependencies, global state, timing/reference/evaluator edits, inline assembly, printf, device allocation or unsupported instructions. Return one fixed-file unified diff. Do not merely rename variables/comments or repeat manual K2 unroll from an earlier rejected candidate. A hypothesis is not a measured gain. Cite available fact IDs in reason only if present.',
      'model':'deepseek-flash','temperature':.2,'max_tokens':768}
    report_path=ROOT/'evidence/20261122-p1-04-feedback/evaluation/run/session-0.json';report=json.loads(report_path.read_text());measurement=next(x for x in report['measurements'] if x['shape_mnk']==[512,512,512] and x['input_count']==1)
    execution={'candidate_id':parent['id'],'compile_status':'PASS','compile_error':None,'correctness':'PASS','latency_ms':[x['elapsed_ms']/(report['graph_launches']*report['replays']) for x in measurement['segments'] if x['label']=='incumbent'],'evaluation_wall_seconds':0.}
    # Historical phase time is unavailable; 0 here means no new evaluation in this interface pilot.
    facts_path=ROOT/'evidence/20261122-p1-04-feedback/proposal/facts.json'
    def profile_loader():
        data=json.loads(facts_path.read_text());facts=[x for x in data['facts'] if x['shape_mnk']==[512,512,512]];provenance=[{k:x[k] for k in ['csv_sha256','binary_sha256','upstream_profiler_sha256']} for x in data['provenance'] if x['profile']=='warps-4-512']
        return {'facts':[{'id':x['id'],'metric':x['metric'],'value':x['median'],'unit':x['unit'],'status':'observed'} for x in facts],'provenance':provenance,'profile_wall_seconds':None}
    schema={'schema_version':1,'parent_id':'current','patch':'diff --git a/kernels/gemm/wmma_sm86.cu b/kernels/gemm/wmma_sm86.cu\n...','reason':'One falsifiable hypothesis, <=1000 chars'};bindings={}
    for method in ['B','C']:
        channel=prepare_channel(method,task,[execution],profile_loader=profile_loader)
        request={'model':'deepseek-flash','max_tokens':768,'temperature':.2,'timeout_seconds':60,'max_retries':0,'messages':[{'role':'system','content':'Return only JSON with exactly this schema: '+json.dumps(schema)},{'role':'user','content':json.dumps(channel['payload'])}]}
        folder=root/method;folder.mkdir();(folder/'channel.json').write_text(json.dumps(channel,indent=2)+'\n');(folder/'request.json').write_text(json.dumps(request,indent=2)+'\n');bindings[method]={'request_sha256':sha(folder/'request.json'),'channel_sha256':sha(folder/'channel.json'),'common_sha256':channel['common_sha256']}
    assert bindings['B']['common_sha256']==bindings['C']['common_sha256']
    protocol={'kind':'P3-03 historical-data live interface pilot; NOT P3-04 or formal A/B/C','maximum_calls':2,'source_sha256':sha(source),'parent_id':parent['id'],'bindings':bindings,'historical_execution_sha256':sha(report_path),'historical_profile_sha256':sha(facts_path),'historical_profile_wall_seconds':None,'new_gpu_runs':0,'heldout_read':False,'control_files':{str(p.relative_to(ROOT)):sha(p) for p in [Path(__file__),ROOT/'kernel_agent/optimization/feedback_channels.py',ROOT/'kernel_agent/optimization/call_budget.py']}}
    (root/'protocol.json').write_text(json.dumps(protocol,indent=2)+'\n');print(json.dumps({'status':'PREPARED','maximum_calls':2,'new_model_calls':0,'common_sha256':bindings['B']['common_sha256'],'B_profile_reads':0,'C_profile_reads':1}))


def run(root,method):
    protocol=json.loads((root/'protocol.json').read_text());folder=root/method
    if any(sha(ROOT/p)!=digest for p,digest in protocol['control_files'].items()):raise ValueError('pilot control source drift')
    if sha(folder/'request.json')!=protocol['bindings'][method]['request_sha256']:raise ValueError('pilot request drift')
    budget=CallBudget(ROOT/'evidence/20261122-full-project-model-calls',ROOT/'evidence/20261122-full-project-readiness/model-budget.json');reserved=budget.reserve('feedback-pilot-'+method,sha(folder/'request.json'))
    if not reserved['dispatch_authorized']:raise RuntimeError('call already reserved; no retry')
    os.environ['DEEPSEEK_API_KEY']=(ROOT/'.private/deepseek-api-key').read_text().strip();secret=os.environ['DEEPSEEK_API_KEY'];request=json.loads((folder/'request.json').read_text());start=time.monotonic();raw=[]
    try:
        cls=provider_class();provider=cls(api_key_env='DEEPSEEK_API_KEY',base_url='https://api.deepseek.com');provider.client=provider.client.with_options(timeout=60,max_retries=0);logging.getLogger('nka_kernelagent_providers.openai_base').setLevel(logging.WARNING)
        def record(response):response.read();raw.append(response.json())
        provider.client._client.event_hooks['response']=[record];response=provider.get_response('deepseek-flash',request['messages'],max_tokens=768,temperature=.2)
        if not raw:raise ValueError('provider response not captured')
        (folder/'response.json').write_text(json.dumps(raw[-1],indent=2)+'\n');u=raw[-1].get('usage',{});usage={k:u[k] for k in ['prompt_tokens','completion_tokens','total_tokens'] if k in u};budget.finish('feedback-pilot-'+method,status='RECEIVED',usage=usage)
        result={'status':'RECEIVED','method':method,'usage':usage,'wall_seconds':time.monotonic()-start,'new_gpu_runs':0,'candidate_execution':'NOT_EXECUTED_REQUIRES_REVIEW','performance_claim':False}
        try:
            if raw[-1]['choices'][0]['finish_reason']!='stop':raise ValueError('incomplete output')
            content=raw[-1]['choices'][0]['message']['content'];base=(ROOT/'kernels/gemm/wmma_sm86.cu').read_bytes()
            try:proposal=apply_reviewable_patch(content,base)
            except ValueError as error:
                (folder/'strict-rejection.json').write_text(json.dumps({'error':str(error)[:1000]})+'\n');proposal=apply_reviewable_patch(content,base,recount=True)
            (folder/'wmma_sm86.cu').write_bytes(proposal.pop('updated_source'));(folder/'proposal.json').write_text(json.dumps(proposal,indent=2)+'\n');result['patch_status']='APPLIED_REQUIRES_REVIEW';result['source_sha256']=sha(folder/'wmma_sm86.cu')
        except Exception as error:result.update(patch_status='REJECT',patch_error=str(error).replace(secret,'[REDACTED]')[:1000])
        (folder/'result.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result))
    except Exception as error:
        # The reservation remains consumed even if no response was returned.
        result={'status':'CALL_FAILED_OR_UNKNOWN','error_type':type(error).__name__,'error':str(error).replace(secret,'[REDACTED]')[:1000],'wall_seconds':time.monotonic()-start,'automatic_retry':False};(folder/'failure.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result));raise RuntimeError('pilot call failed; reservation retained') from None
    finally:os.environ.pop('DEEPSEEK_API_KEY',None)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('action',choices=['prepare','run']);parser.add_argument('--root',type=Path,required=True);parser.add_argument('--method',choices=['B','C']);args=parser.parse_args()
    if args.action=='prepare':prepare(args.root)
    else:
        if not args.method:parser.error('--method required for run')
        run(args.root,args.method)
