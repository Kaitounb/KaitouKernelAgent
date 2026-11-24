"""Real B/C SDK configuration proposals through the pinned KernelAgent provider.

This is a bounded graph/configuration capability, not arbitrary MLA CUDA rewriting.
No heldout file is read, and no model credential leaves the local provider process.
"""
import argparse,hashlib,json,logging,os,statistics,time
from pathlib import Path
from kernel_agent.optimization.call_budget import CallBudget
from kernel_agent.optimization.feedback_channels import prepare_channel
from kernel_agent.optimization.identity import strict_json
from kernel_agent.optimization.provider import provider_class
ROOT=Path(__file__).resolve().parents[2]


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def prepare(root):
    root.mkdir(exist_ok=False);cal=ROOT/'evidence/20261123-mla-sdk-aa-sm86-exclusive-v2';prof=ROOT/'evidence/20261123-mla-ncu-sm86-pinned-native';validation=json.loads((prof/'remote/run/validation.json').read_text());parsed=json.loads((prof/'remote/run/parsed.json').read_text());execution=json.loads((prof/'remote/run/mla-dev04.execution.json').read_text());sdk=json.loads((cal/'remote/sdk-before.json').read_text());assert validation['status']=='PASS' and execution['exit']==0 and execution['cleanup']['removed'] and len(parsed['launches'])==3;assert validation['native_sha256'] and all(sdk['native'].get(n)==h for n,h in validation['native_sha256'].items())
    native=next(iter(validation['native_sha256'].values()));facts=[]
    for i,metric in enumerate(parsed['launches'][0]['metrics'],1):
        rows=[x['metrics'][metric] for x in parsed['launches']];assert all(x['status']=='observed' and x['unit']==rows[0]['unit'] for x in rows);facts.append({'id':'F'+str(i),'metric':metric,'value':statistics.median(x['value'] for x in rows),'unit':rows[0]['unit'],'status':'observed'})
    profile={'facts':facts,'provenance':[{'csv_sha256':parsed['csv_sha256'],'binary_sha256':native,'upstream_profiler_sha256':parsed['upstream_profiler_sha256']}],'profile_wall_seconds':execution['wall_seconds']};(root/'profile.json').write_text(json.dumps(profile,indent=2)+'\n')
    source=(ROOT/'kernel_agent/operators/flashinfer_backend.py').read_text()+'\n# Page schedule\n'+(ROOT/'kernel_agent/operators/partition.py').read_text();body={'source_sha256':hashlib.sha256(source.encode()).hexdigest(),'native_sha256':sdk['native'],'initial_config':{'page_size':16,'splits':1},'protocol_sha256':sha(ROOT/'workloads/mla_measurement_v2.json')};parent=hashlib.sha256(json.dumps(body,sort_keys=True,separators=(',',':')).encode()).hexdigest();(root/'parent.json').write_text(json.dumps({**body,'id':parent},indent=2)+'\n')
    task={'source':source,'source_sha256':body['source_sha256'],'contract':'MLA core Z and natural-log LSE. FP16 inputs quantized before CPU FP64 reference, FP32 accumulation, R512/Drope64, original scale 1/sqrt192, finite +/-4. Public target B1 H128 L8192 (mla-dev-04), 16-byte alignment, valid random permutation page table. Full 8 public development workloads remain protection checks; no full score materialization. RTX3070 SM86 Torch2.14+cu130, FlashInfer0.2.8. Current baseline page16/split1 numerical QA passed but its three-process AA was REJECTED (one pair 3.184% >2%). This call tests a NEW configuration and rejection workflow, not evidence of stable gain. New configuration needs independent AA and external decision; repeated/failed identity is never resampled.',
      'gpu':'RTX3070 SM86, Torch2.14.0+cu130, CUDA13.0','allowed_changes':'Return only page_size in {1,16,64} and splits in {1,2,4}, and a falsifiable reason <=1000 chars. These are mapped to the supplied fixed native CUDA library and explicit split/combine graph; no code, tolerances, shapes, evaluator or dependencies may change. Both B/C have identical transform space. Do not claim measured gain. All partial/combine kernels are included in timing. Native SDK already has internal split planning; explicit split can add launches and FP16 intermediate rounding. Initial page16/split1 is a duplicate. Cite only available fact IDs; no fact citation required without profile.',
      'model':'deepseek-flash','temperature':.2,'max_tokens':768}
    reports=[json.loads((cal/f'remote/aa-{i}.json').read_text()) for i in range(3)];latencies=[x['elapsed_ms']/1024 for r in reports for row in r['rows'] if row['workload']=='mla-dev-04' and row['input_count']==1 for x in row['segments'] if x['label']=='incumbent'];history=[{'candidate_id':parent,'compile_status':'PASS','compile_error':None,'correctness':'PASS','latency_ms':latencies,'evaluation_wall_seconds':sum(x['wall_seconds'] for x in json.loads((cal/'remote/result.json').read_text())['records'] if x['label'] in ['aa-0','aa-1','aa-2'])}];bindings={}
    schema={'schema_version':1,'parent_id':parent,'page_size':16,'splits':2,'reason':'one testable hypothesis; citations only from supplied facts'}
    for method in ['B','C']:
        channel=prepare_channel(method,task,history,profile_loader=lambda:profile);folder=root/method;folder.mkdir();request={'model':'deepseek-flash','max_tokens':768,'temperature':.2,'messages':[{'role':'system','content':'Return one JSON object with exactly this schema: '+json.dumps(schema)},{'role':'user','content':json.dumps(channel['payload'])}]};(folder/'channel.json').write_text(json.dumps(channel,indent=2)+'\n');(folder/'request.json').write_text(json.dumps(request,indent=2)+'\n');bindings[method]={'request_sha256':sha(folder/'request.json'),'common_sha256':channel['common_sha256'],'profile_reads':channel['profile_loader_calls']}
    assert bindings['B']['common_sha256']==bindings['C']['common_sha256'] and bindings['B']['profile_reads']==0 and bindings['C']['profile_reads']==1
    protocol={'scope':'P4-05 bounded MLA SDK graph/configuration live workflow, not P3-04/formal ABC or free CUDA rewriting','bindings':bindings,'parent_id':parent,'maximum_calls':2,'profile_sha256':sha(root/'profile.json'),'calibration_sha256':sha(cal/'verification.json'),'source_calibration':'SM86 baseline failed AA, no stable performance claim','controls':{str(p.relative_to(ROOT)):sha(p) for p in [Path(__file__),ROOT/'kernel_agent/optimization/feedback_channels.py',ROOT/'kernel_agent/optimization/call_budget.py',ROOT/'kernel_agent/operators/flashinfer_backend.py',ROOT/'kernel_agent/operators/partition.py',ROOT/'workloads/mla_measurement_v2.json']},'heldout_read':False};(root/'protocol.json').write_text(json.dumps(protocol,indent=2)+'\n');print(json.dumps({'status':'PREPARED','maximum_calls':2,'B_profile_reads':0,'C_profile_reads':1,'profile_fields':len(facts),'profile_wall_seconds':profile['profile_wall_seconds'],'new_model_calls':0}))


def run(root,method):
    protocol=json.loads((root/'protocol.json').read_text());folder=root/method;assert all(sha(ROOT/p)==h for p,h in protocol['controls'].items()) and sha(folder/'request.json')==protocol['bindings'][method]['request_sha256'];key='mla-sdk-first-'+method;budget=CallBudget(ROOT/'evidence/20261122-full-project-model-calls',ROOT/'evidence/20261122-full-project-readiness/model-budget.json');reservation=budget.reserve(key,sha(folder/'request.json'))
    if not reservation['dispatch_authorized']:raise RuntimeError('already reserved; never repeat uncertain calls')
    secret=(ROOT/'.private/deepseek-api-key').read_text().strip();os.environ['DEEPSEEK_API_KEY']=secret;request=json.loads((folder/'request.json').read_text());raw=[];start=time.monotonic();received=False
    try:
        cls=provider_class();provider=cls(api_key_env='DEEPSEEK_API_KEY',base_url='https://api.deepseek.com');provider.client=provider.client.with_options(timeout=60,max_retries=0);logging.getLogger('nka_kernelagent_providers.openai_base').setLevel(logging.WARNING)
        def record(response):response.read();raw.append(response.json())
        provider.client._client.event_hooks['response']=[record];provider.get_response(request['model'],request['messages'],max_tokens=request['max_tokens'],temperature=request['temperature'])
        if not raw:raise ValueError('provider response not captured')
        response=raw[-1];(folder/'response.json').write_text(json.dumps(response,indent=2)+'\n');usage={k:response.get('usage',{})[k] for k in ['prompt_tokens','completion_tokens','total_tokens'] if k in response.get('usage',{})};budget.finish(key,status='RECEIVED',usage=usage);received=True;result={'status':'REJECT','method':method,'usage':usage,'model_wall_seconds':time.monotonic()-start,'cost_currency':None,'billed_amount':None,'production_release':False}
        try:
            assert response['choices'][0]['finish_reason']=='stop';proposal=strict_json(response['choices'][0]['message']['content'])
            if set(proposal)!={'schema_version','parent_id','page_size','splits','reason'} or type(proposal['schema_version']) is not int or proposal['schema_version']!=1 or proposal['parent_id']!=protocol['parent_id']:raise ValueError('proposal schema/parent mismatch')
            if type(proposal['page_size']) is not int or proposal['page_size'] not in [1,16,64] or type(proposal['splits']) is not int or proposal['splits'] not in [1,2,4] or not isinstance(proposal['reason'],str) or not 1<=len(proposal['reason'])<=1000:raise ValueError('configuration/reason outside frozen contract')
            config={k:proposal[k] for k in ['page_size','splits']};identity={'parent_id':protocol['parent_id'],'config':config};cid=hashlib.sha256(json.dumps(identity,sort_keys=True,separators=(',',':')).encode()).hexdigest();(folder/'proposal.json').write_text(json.dumps({**proposal,'candidate_id':cid},indent=2)+'\n');result.update(status='DUPLICATE_BASELINE' if config=={'page_size':16,'splits':1} else 'VALID_CONTROLLED_CONFIG',candidate_id=cid,config=config,free_source_executed=False,evaluation='NOT_RUN')
        except (ValueError,KeyError,TypeError,AssertionError) as error:result['reason']=str(error)[:512]
        (folder/'result.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result))
    except Exception as error:
        if not received:budget.finish(key,status='UNKNOWN')
        result={'status':'CALL_FAILED_OR_UNKNOWN','error_type':type(error).__name__,'error':str(error).replace(secret,'[REDACTED]')[:1000],'automatic_retry':False};(folder/'failure.json').write_text(json.dumps(result,indent=2)+'\n');raise RuntimeError('MLA model call failed; consumed reservation retained') from None
    finally:os.environ.pop('DEEPSEEK_API_KEY',None)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=['prepare','run']);p.add_argument('--root',type=Path,required=True);p.add_argument('--method',choices=['B','C']);a=p.parse_args()
    if a.action=='prepare':prepare(a.root)
    else:
        if not a.method:p.error('--method required')
        run(a.root,a.method)
