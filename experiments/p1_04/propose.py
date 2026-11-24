"""Bounded hardware-feedback proposal using the previously validated KernelAgent provider."""
import argparse
import json
import logging
import math
import os
from pathlib import Path
import statistics
import time

from evaluators.gemm_decision import ROOT,identity,strict_json,validate_proposal,sha
from experiments.p1_03.propose import provider_class


def make_facts(profile_run):
    facts=[];provenance=[];unavailable=[]
    for path in sorted(Path(profile_run).glob('*.metrics.json')):
        data=json.loads(path.read_text());tag=path.name.removesuffix('.metrics.json')
        if len(data['launches'])!=3:raise ValueError('three profile launches required')
        config=json.loads(path.with_name(tag+'.config.json').read_text())
        if config['warps']!=4:raise ValueError('feedback must describe current 4-warps incumbent')
        for name,cell in data['launches'][0]['metrics'].items():
            cells=[r['metrics'][name] for r in data['launches']]
            if any(c['status']!='observed' or c['value'] is None for c in cells):
                unavailable.append({'profile':tag,'metric':name,'reason':'missing/nonfinite in profile'});continue
            values=[c['value'] for c in cells]
            facts.append({'id':f'F{len(facts)+1}','profile':tag,'shape_mnk':config['shape_mnk'],'metric':name,
                          'median':statistics.median(values),'min':min(values),'max':max(values),'unit':cell['unit'],'launch_count':3})
        provenance.append({'profile':tag,'metrics_json_sha256':sha(path),'csv_sha256':data['csv_sha256'],
                           'binary_sha256':config['binary_sha256'],'upstream_profiler_sha256':data['upstream_profiler_sha256']})
    if len(provenance)!=2:raise ValueError('two development shapes required')
    selection=json.loads((Path(profile_run)/'metric-selection.json').read_text())
    unavailable.extend({'metric':k,'reason':v} for k,v in selection['unavailable_from_query'].items())
    return {'facts':facts,'unavailable':unavailable,'provenance':provenance,
            'scope':'NCU kernel replay, cache-control all, clock-control none, 3 launches per shape; not formal Event latency or speedup',
            'interpretation_constraints':['Reported active/elapsed percentage denominators differ; do not directly equate them.',
             'launch shared memory allocated includes tool/runtime accounting; source has no explicit __shared__ array. Do not infer user shared tiling from allocated bytes.',
             'Total warp tiles are unchanged by warps-per-block; more blocks alone does not prove more resident warps.',
             'An old 8-warps proposal passed correctness but was rejected for A/A timing instability; no evidence of algorithmic loss from that result.']}


def validate_feedback(text,parent,facts):
    d=strict_json(text)
    if not isinstance(d,dict) or set(d)!={'proposal','fact_ids','hypothesis','competing_explanations','expected_changes'}:
        raise ValueError('unexpected feedback schema')
    proposal=validate_proposal(json.dumps(d['proposal']),'current')
    known={f['id'] for f in facts['facts']}
    if not isinstance(d['fact_ids'],list) or not 1<=len(d['fact_ids'])<=8 or any(x not in known for x in d['fact_ids']):raise ValueError('unknown/missing cited facts')
    if not isinstance(d['hypothesis'],str) or not 1<=len(d['hypothesis'])<=1500:raise ValueError('invalid hypothesis')
    if not isinstance(d['competing_explanations'],list) or not 1<=len(d['competing_explanations'])<=4 or any(not isinstance(x,str) or not 1<=len(x)<=600 for x in d['competing_explanations']):raise ValueError('competing explanation required')
    metrics={f['metric'] for f in facts['facts']}
    if not isinstance(d['expected_changes'],list) or not 1<=len(d['expected_changes'])<=5:raise ValueError('expected changes required')
    for change in d['expected_changes']:
        if not isinstance(change,dict) or set(change)!={'metric','direction'} or change['metric'] not in metrics or change['direction'] not in ['increase','decrease','uncertain']:
            raise ValueError('invalid prediction')
    proposal['parent_id']=parent['id'];d['proposal']=proposal
    return d


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--profile-run',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    args=p.parse_args()
    if args.output.exists():p.error('Use a new output; no automatic retries')
    if not os.environ.get('DEEPSEEK_API_KEY'):p.error('DEEPSEEK_API_KEY missing')
    facts=make_facts(args.profile_run);parent=identity(4);args.output.mkdir(parents=True)
    (args.output/'facts.json').write_text(json.dumps(facts,indent=2)+'\n')
    task={'current_warps':4,'allowed_warps':[1,2,4,8],'parent_alias':'current','gpu':'RTX 3070 / SM86',
          'source':(ROOT/'kernels/gemm/wmma_sm86.cu').read_text(),'profile_facts':facts,
          'constraints':'Only choose a runtime warp parameter; no code execution, source changes, reference/tolerance/benchmark changes. Supported semantics stay FP16 row-major inputs with FP32 accumulation/output and cuBLAS tails. Prefer an unmeasured 1 or 2 alternative if justified. A no-change proposal is legal. Hypotheses and predictions are not measurements.'}
    schema={'proposal':{'schema_version':1,'parent_id':'current','warps':2,'reason':'Testable reason, <=1000 chars'},
            'fact_ids':['F1'],'hypothesis':'A falsifiable interpretation, not a claimed root cause',
            'competing_explanations':['At least one plausible alternative'],
            'expected_changes':[{'metric':'sm__cycles_active.avg','direction':'decrease'}]}
    messages=[{'role':'system','content':'Return only valid JSON with exactly the supplied schema. Cite existing fact IDs; do not invent counter values or claim measured gains. direction must be increase/decrease/uncertain. Keep each explanation concise. Schema: '+json.dumps(schema)},
              {'role':'user','content':json.dumps(task)}]
    request={'model':'deepseek-flash','messages':messages,'max_tokens':768,'temperature':.2,'thinking':'disabled','response_format':'json_object','timeout':60,'max_retries':0,'maximum_calls':1}
    (args.output/'request.json').write_text(json.dumps(request,indent=2)+'\n')
    usage={'status':'ATTEMPTED','real_external_model_call':True,'provider':'deepseek','model':'deepseek-flash','calls_attempted':1,'max_calls':1,
           'max_tokens':768,'max_retries':0,'billed_cost':None,'billed_cost_status':'not returned in API response','credentials':'local process environment only'}
    start=time.monotonic()
    try:
        cls=provider_class();provider=cls(api_key_env='DEEPSEEK_API_KEY',base_url='https://api.deepseek.com')
        if not provider.is_available():raise RuntimeError('model provider unavailable')
        provider.client=provider.client.with_options(timeout=60,max_retries=0)
        logging.getLogger('nka_kernelagent_providers.openai_base').setLevel(logging.WARNING)
        raw=[]
        def record(r):r.read();raw.append(r.json())
        provider.client._client.event_hooks['response']=[record]
        response=provider.get_response('deepseek-flash',messages,max_tokens=768,temperature=.2)
        (args.output/'response.json').write_text(json.dumps(raw[-1],indent=2)+'\n')
        usage.update(status='RECEIVED',usage=response.usage)
        if raw[-1]['choices'][0]['finish_reason']!='stop':raise ValueError('model output incomplete')
        feedback=validate_feedback(response.content,parent,facts)
        (args.output/'feedback.json').write_text(json.dumps(feedback,indent=2)+'\n')
        (args.output/'proposal.json').write_text(json.dumps(feedback['proposal'],indent=2)+'\n')
        (args.output/'parent-binding.json').write_text(json.dumps({'request_alias':'current','actual_parent':parent},indent=2)+'\n')
        usage['feedback_status']='VALIDATED'
        print(json.dumps({'warps':feedback['proposal']['warps'],'fact_ids':feedback['fact_ids'],'hypothesis':feedback['hypothesis'],'usage':response.usage},indent=2))
    except Exception as error:
        usage.update(status='FAILED',error_type=type(error).__name__,error=str(error).replace(os.environ['DEEPSEEK_API_KEY'],'[REDACTED]')[:1000])
        raise RuntimeError('hardware-feedback call failed; sanitized usage retained; no retry') from None
    finally:
        usage['wall_seconds']=time.monotonic()-start
        (args.output/'usage.json').write_text(json.dumps(usage,indent=2)+'\n')


if __name__=='__main__':main()
