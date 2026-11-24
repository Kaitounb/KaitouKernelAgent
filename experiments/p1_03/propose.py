"""One bounded DeepSeek proposal via pinned KernelAgent's OpenAI-compatible provider."""
import argparse
from dataclasses import asdict
import importlib
import importlib.metadata
import json
import logging
import os
from pathlib import Path
import sys
import time
import types

from evaluators.gemm_decision import ROOT,identity,validate_proposal,sha


def provider_class():
    folder=ROOT/'provenance/upstream-reviews/20261121-p1-03/KernelAgent/utils/providers'
    records=json.loads((folder.parents[1]/'source-record.json').read_text())
    for row in records:
        if sha(folder.parents[1]/row['path'])!=row['sha256']:
            raise RuntimeError('upstream provider file drift')
    package=types.ModuleType('nka_kernelagent_providers');package.__path__=[str(folder)]
    sys.modules[package.__name__]=package
    upstream=importlib.import_module(package.__name__+'.openai_base')
    class DeepSeekProvider(upstream.OpenAICompatibleProvider):
        @property
        def name(self):return 'deepseek'
        def _build_api_params(self,model_name,messages,**kwargs):
            params=super()._build_api_params(model_name,messages,**kwargs)
            params.update(response_format={'type':'json_object'},extra_body={'thinking':{'type':'disabled'}})
            return params
    return DeepSeekProvider


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--model',default='deepseek-flash')
    args=p.parse_args()
    if args.output.exists():p.error('Use a new directory; every attempt consumes the one-call budget')
    if not os.getenv('DEEPSEEK_API_KEY'):p.error('DEEPSEEK_API_KEY is missing')
    args.output.mkdir(parents=True)
    parent=identity(4)
    # Minimal public task description: never export project source/contracts/workload lists.
    task={"operation":"Public CUDA WMMA GEMM example, FP16 input and FP32 accumulation/output",
          "public_reference":"https://github.com/NVIDIA/cuda-samples/blob/3f1c50965017932fc81e6d94a3fc9e04c105b312/Samples/3_CUDA_Features/cudaTensorCoreGemm/cudaTensorCoreGemm.cu",
          "public_hardware":"RTX 3070, SM86",
          "template_description":"A warp computes one 16x16 output tile using WMMA; a block contains 1, 2, 4 or 8 warps mapped along M. Current setting is 4. Row-major A and B. One runtime integer changes warps per block. Edge shapes fall back to cuBLAS.",
          "instructions":"Propose one alternative warp setting (1/2/8) to measure. No source edits or tools. There is no supplied benchmark data, so describe a testable hypothesis and do not claim a measured gain.",
          "parent_alias":"current"}
    messages=[{'role':'system','content':'Return ONLY one JSON object with exactly schema_version (integer 1), parent_id (literal "current"), warps (integer 1, 2, 4, or 8), reason (1 to 1000 characters). No Markdown or extra keys. Example JSON: {"schema_version":1,"parent_id":"current","warps":2,"reason":"Test fewer warps per block."}'},
              {'role':'user','content':json.dumps(task,ensure_ascii=False)}]
    request={'model':args.model,'messages':messages,'max_tokens':768,'temperature':.2,'response_format':{'type':'json_object'},'thinking':{'type':'disabled'},
             'endpoint':'https://api.deepseek.com','max_retries':0,'timeout_seconds':60,'maximum_calls':1}
    (args.output/'request.json').write_text(json.dumps(request,indent=2,ensure_ascii=False)+'\n')
    usage={'status':'ATTEMPTED','provider':'deepseek','model':args.model,'calls_attempted':1,'max_calls':1,'max_output_tokens':768,
           'max_retries':0,'credentials':'process environment only; not archived','real_external_model_call':True,'wall_seconds':None}
    usage_path=args.output/'usage.json';usage_path.write_text(json.dumps(usage,indent=2)+'\n')
    started=time.monotonic()
    try:
        cls=provider_class();provider=cls(api_key_env='DEEPSEEK_API_KEY',base_url='https://api.deepseek.com')
        if not provider.is_available():raise RuntimeError('OpenAI SDK or DeepSeek credential unavailable')
        provider.client=provider.client.with_options(timeout=60,max_retries=0)
        logging.getLogger('nka_kernelagent_providers.openai_base').setLevel(logging.WARNING)
        raw=[]
        def record_response(response):raw.append(response.json())
        # Preserve request id/resolved model/finish_reason without changing upstream get_response.
        provider.client._client.event_hooks['response']=[lambda r:(r.read(),record_response(r))]
        response=provider.get_response(args.model,messages,max_tokens=768,temperature=.2)
        (args.output/'response.json').write_text(json.dumps(raw[-1] if raw else asdict(response),indent=2,ensure_ascii=False)+'\n')
        usage.update(status='RECEIVED',usage=response.usage,sdk_version=importlib.metadata.version('openai'))
        proposal=validate_proposal(response.content,'current')
        proposal['parent_id']=parent['id']
        (args.output/'parent-binding.json').write_text(json.dumps({'request_alias':'current','actual_parent':parent,'scope':'bound locally; actual identity not sent to model'},indent=2)+'\n')
        (args.output/'proposal.json').write_text(json.dumps(proposal,indent=2,ensure_ascii=False)+'\n')
        usage['proposal_status']='VALIDATED'
        print(json.dumps({'provider':'deepseek','model':args.model,'proposal_warps':proposal['warps'],'usage':response.usage},indent=2))
    except Exception as error:
        secret=os.environ.get('DEEPSEEK_API_KEY','')
        usage.update(status='FAILED',error_type=type(error).__name__,error=str(error).replace(secret,'[REDACTED]')[:1200])
        raise RuntimeError('DeepSeek proposal failed; inspect sanitized usage.json; no automatic retry') from None
    finally:
        usage['wall_seconds']=time.monotonic()-started
        usage_path.write_text(json.dumps(usage,indent=2,ensure_ascii=False)+'\n')


if __name__=='__main__':main()
