"""One durable, bounded provider dispatch; API credentials remain local to this process."""
import json,logging,os,time
from pathlib import Path
from .provider import provider_class


def dispatch(request,folder,budget,key,request_sha256,key_file):
    reservation=budget.reserve(key,request_sha256)
    if not reservation['dispatch_authorized']:raise RuntimeError('call already reserved; no automatic retry')
    secret=Path(key_file).read_text().strip();os.environ['DEEPSEEK_API_KEY']=secret;raw=[];start=time.monotonic();received=False;folder=Path(folder)
    try:
        cls=provider_class();provider=cls(api_key_env='DEEPSEEK_API_KEY',base_url='https://api.deepseek.com');provider.client=provider.client.with_options(timeout=60,max_retries=0);logging.getLogger('nka_kernelagent_providers.openai_base').setLevel(logging.WARNING)
        def capture(response):response.read();raw.append(response.json())
        provider.client._client.event_hooks['response']=[capture];provider.get_response(request['model'],request['messages'],max_tokens=request['max_tokens'],temperature=request['temperature'])
        if not raw:raise ValueError('provider response not captured')
        response=raw[-1];(folder/'response.json').write_text(json.dumps(response,indent=2)+'\n');usage={k:response.get('usage',{})[k] for k in ['prompt_tokens','completion_tokens','total_tokens'] if k in response.get('usage',{})};budget.finish(key,status='RECEIVED',usage=usage);received=True;cost={'status':'RECEIVED','usage':usage,'model_wall_seconds':time.monotonic()-start,'cost_currency':None,'billed_amount':None,'automatic_retry':False};(folder/'usage.json').write_text(json.dumps(cost,indent=2)+'\n');return response,cost
    except Exception as error:
        if not received:budget.finish(key,status='UNKNOWN')
        record={'status':'FAILED_OR_UNKNOWN','error_type':type(error).__name__,'error':str(error).replace(secret,'[REDACTED]')[:1000],'automatic_retry':False,'wall_seconds':time.monotonic()-start};(folder/'failure.json').write_text(json.dumps(record,indent=2)+'\n');raise RuntimeError('dispatch failed; consumed reservation retained') from None
    finally:os.environ.pop('DEEPSEEK_API_KEY',None)
