"""One bounded source proposal via the already validated KernelAgent DeepSeek provider."""
import argparse,json,logging,os,time
from pathlib import Path
from evaluators.gemm_decision import ROOT,identity,sha
from experiments.p1_03.propose import provider_class
from experiments.p3_01.patches import apply_reviewable_patch


def call_once(prepared,output,*,provider_factory=None):
    prepared=Path(prepared);output=Path(output)
    if output.exists():raise FileExistsError('Use a new output; every attempt has a separate budget record')
    if not os.environ.get('DEEPSEEK_API_KEY'):raise ValueError('DEEPSEEK_API_KEY missing')
    raw_request=(prepared/'request.json').read_bytes()
    if len(raw_request)>32768:raise ValueError('prepared request too large')
    request=json.loads(raw_request);binding=json.loads((prepared/'parent-binding.json').read_text());source=ROOT/'kernels/gemm/wmma_sm86.cu'
    if binding['actual_parent']!=identity(4) or binding['source_sha256']!=sha(source):raise ValueError('prepared parent changed')
    if any(request.get(k)!=v for k,v in {'model':'deepseek-flash','max_tokens':768,'maximum_calls':1,'max_retries':0,'timeout_seconds':60,'thinking':'disabled','response_format':'json_object'}.items()):raise ValueError('prepared call budget/config changed')
    if not isinstance(request.get('messages'),list) or len(request['messages'])!=2:raise ValueError('prepared messages invalid')
    output.mkdir(parents=True);(output/'request.json').write_bytes(raw_request);(output/'parent-binding.json').write_bytes((prepared/'parent-binding.json').read_bytes())
    usage={'status':'ATTEMPTED','real_external_model_call':provider_factory is None,'calls_attempted':1,'maximum_calls':1,'max_output_tokens':768,'max_retries':0,
           'provider':'deepseek','model':'deepseek-flash','credentials':'local process environment only; not archived','billed_cost':None,'billed_cost_status':'not returned by API'}
    start=time.monotonic();usage_path=output/'usage.json';usage_path.write_text(json.dumps(usage,indent=2)+'\n')
    try:
        cls=provider_factory or provider_class();provider=cls(api_key_env='DEEPSEEK_API_KEY',base_url='https://api.deepseek.com')
        if not provider.is_available():raise RuntimeError('provider unavailable')
        provider.client=provider.client.with_options(timeout=60,max_retries=0)
        logging.getLogger('nka_kernelagent_providers.openai_base').setLevel(logging.WARNING);raw=[]
        def record(response):response.read();raw.append(response.json())
        provider.client._client.event_hooks['response']=[record]
        response=provider.get_response(request['model'],request['messages'],max_tokens=768,temperature=.2)
        if not raw:raise ValueError('missing raw provider response')
        (output/'response.json').write_text(json.dumps(raw[-1],indent=2)+'\n');usage.update(status='RECEIVED',usage=response.usage)
        if raw[-1]['choices'][0]['finish_reason']!='stop':raise ValueError('model output incomplete')
        if sha(source)!=binding['source_sha256'] or identity(4)!=binding['actual_parent']:raise ValueError('parent changed during proposal')
        result=apply_reviewable_patch(response.content,source.read_bytes());files={name:(ROOT/'kernels/gemm'/name).read_bytes() for name in ['cublas_sm86.cu','wmma_sm86.cu','gemm_contract.cuh']};files['wmma_sm86.cu']=result.pop('updated_source')
        for name,data in files.items():
            target=output/'source/kernels/gemm'/name;target.parent.mkdir(parents=True,exist_ok=True);target.write_bytes(data)
        result['actual_parent_id']=binding['actual_parent']['id'];result['source_hashes']={name:sha(output/'source/kernels/gemm'/name) for name in files};result['execution_status']='NOT_EXECUTED'
        (output/'proposal.json').write_text(json.dumps(result,indent=2)+'\n');usage['proposal_status']='PATCH_APPLIED_REQUIRES_SOURCE_REVIEW'
        return result
    except Exception as error:
        secret=os.environ.get('DEEPSEEK_API_KEY','');usage.update(status='FAILED',error_type=type(error).__name__,error=str(error).replace(secret,'[REDACTED]')[:1000])
        raise RuntimeError('Source proposal failed; sanitized usage retained; no retry') from None
    finally:
        usage['wall_seconds']=time.monotonic()-start;usage_path.write_text(json.dumps(usage,indent=2)+'\n')


def main():
    p=argparse.ArgumentParser();p.add_argument('--prepared',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    result=call_once(a.prepared,a.output);print(json.dumps({'status':result['status'],'actual_parent_id':result['actual_parent_id'],'source_sha256':result['source_sha256'],'execution_status':result['execution_status']}))


if __name__=='__main__':main()
