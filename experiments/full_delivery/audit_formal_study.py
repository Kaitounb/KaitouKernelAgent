"""Replay frozen study bookkeeping, information boundaries and cost receipts on CPU."""
import argparse,collections,hashlib,json,random,re
from pathlib import Path
from experiments.full_delivery.formal_study import ROOT,verify,branch
from experiments.full_delivery.study_cache import validate_entry,descriptor
from kernel_agent.optimization.study_feedback import prepare
from kernel_agent.optimization.identity import strict_json
from kernel_agent.optimization.patches import apply_reviewable_patch


def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def decode_response(response,op,task,parent):
    assert response['choices'][0]['finish_reason']=='stop';d=strict_json(response['choices'][0]['message']['content']);keys={'schema_version','parent_id','action','reason'}|({'patch'} if op=='gemm' else {'page_size','splits'})
    if not isinstance(d,dict) or set(d)!=keys or type(d['schema_version']) is not int or d['schema_version']!=1 or d['parent_id']!='current' or d['action'] not in ['modify','keep_parent'] or not isinstance(d['reason'],str) or not 1<=len(d['reason'])<=1000:raise ValueError('invalid schema')
    if op=='mla':
        if type(d['page_size']) is not int or d['page_size'] not in [1,16,64] or type(d['splits']) is not int or d['splits'] not in [1,2,4]:raise ValueError('invalid SDK config')
        config={k:d[k] for k in ['page_size','splits']}
        if d['action']=='keep_parent' and config!={'page_size':16,'splits':1}:raise ValueError('changed keep_parent')
        return descriptor(op,config=config),config,None,None
    if d['action']=='keep_parent':
        if d['patch']!='':raise ValueError('changed keep_parent')
        return parent,{'source_sha256':task['source_sha256'],'patch':''},None,None
    legacy={k:d[k] for k in ['schema_version','parent_id','patch','reason']};content=json.dumps(legacy)
    try:parsed=apply_reviewable_patch(content,task['source'].encode())
    except ValueError:parsed=apply_reviewable_patch(content,task['source'].encode(),recount=True)
    normal=dict(response);normal['choices']=[{**response['choices'][0],'message':{**response['choices'][0]['message'],'content':content}}]
    return descriptor(op,source_sha=parsed['source_sha256']),{'source_sha256':parsed['source_sha256'],'patch':d['patch']},parsed['updated_source'],normal


def audit(root,allow_incomplete=False):
    protocol=verify(root);ledger=json.loads((ROOT/'evidence/20261122-full-project-model-calls/ledger.json').read_text())['calls'];cache=json.loads((root/'cache.initial.json').read_text());cache.update({p.stem:json.loads(p.read_text()) for p in (root/'cache-updates').glob('*.json')})
    for entry in cache.values():validate_entry(entry)
    searches=[];rows=[];calls=[];pending=[];global_outcomes=collections.Counter()
    for op,ids in protocol['targets'].items():
        for wid in ids:
            taskdir=root/'tasks'/op/wid;task=json.loads((taskdir/'task.json').read_text());baseline=json.loads((taskdir/'baseline.json').read_text());assert sha(taskdir/'task.json')==protocol['tasks'][op+'/'+wid]['task_sha256'] and sha(taskdir/'baseline.json')==protocol['tasks'][op+'/'+wid]['baseline_sha256']
            for rep in range(3):
                profile_dir=root/'profiles'/op/wid/str(rep);profile=json.loads((profile_dir/'profile.json').read_text());contexts=json.loads((profile_dir/'context.json').read_text());pb=protocol['profiles'][op+'/'+wid+'/'+str(rep)];assert sha(profile_dir/'profile.json')==pb['profile_sha256'] and sha(profile_dir/'context.json')==pb['context_sha256'] and sha(ROOT/pb['origin_profile'])==pb['origin_sha256']
                for method in ['A','B','C']:
                    b=branch(root,op,wid,rep,method);history=[];executions=[];tokens=0;api_wall=0.;fresh_wall=0.;outcomes=collections.Counter();order=list(protocol['A_spaces'][op]);random.Random(protocol['A_order_seed_base']+rep).shuffle(order)
                    folders=sorted((x for x in b.iterdir() if x.is_dir() and x.name.isdecimal()),key=lambda x:int(x.name));assert [int(x.name) for x in folders]==list(range(len(folders)))
                    for step,folder in enumerate(folders):
                        if method!='A':
                            assert (folder/'request.json').exists();request=json.loads((folder/'request.json').read_text());channel=json.loads((folder/'channel.json').read_text());expected=prepare(method,op,task,baseline,executions,history,profile_loader=lambda:profile,profile_context_loader=lambda:contexts);assert channel==expected and json.loads(request['messages'][1]['content'])==channel['payload'];assert request['model']==protocol['model'] and request['temperature']==protocol['temperature'] and request['max_tokens']==protocol['max_tokens']
                            if method=='B':assert channel['profile_loader_calls']==0 and 'profile' not in channel['payload']['feedback'] and 'profile_contexts' not in channel['payload']['feedback']
                            else:assert channel['profile_loader_calls']==1 and channel['payload']['feedback']['profile']==profile
                            key='formal-v1-'+op+'-'+wid+'-'+str(rep)+'-'+method+'-'+str(step);receipt=ledger[key];assert receipt['request_sha256']==sha(folder/'request.json');calls.append(key)
                            if receipt['status']=='RECEIVED':
                                usage=json.loads((folder/'usage.json').read_text());response=json.loads((folder/'response.json').read_text());assert response['model']==request['model'] and receipt['usage']==usage['usage'];assert usage['usage']['prompt_tokens']+usage['usage']['completion_tokens']==usage['usage']['total_tokens'];tokens+=usage['usage']['total_tokens'];api_wall+=usage['model_wall_seconds'];assert usage['billed_amount'] is None and usage['automatic_retry'] is False
                            else:assert receipt['status']=='UNKNOWN' and (folder/'failure.json').exists()
                        else:assert not (folder/'request.json').exists() and not (folder/'usage.json').exists()
                        if not (folder/'decision.json').exists():pending.append(str(folder.relative_to(root)));break
                        decision=json.loads((folder/'decision.json').read_text());d=decision['outcome'];execution=decision['execution'];assert d['step']==step and d['candidate_id']==execution['candidate_id'] and decision['production_release'] is False;assert all(sha(ROOT/n)==h for n,h in decision['proofs'].items())
                        if (folder/'proposal.json').exists():
                            proposal=json.loads((folder/'proposal.json').read_text());assert (proposal['op'],proposal['workload'],proposal['repetition'],proposal['method'],proposal['step'])==(op,wid,rep,method,step);cid=proposal['descriptor']['id'];assert cid==d['candidate_id'] and proposal['source_change']==d['source_change'];entry=cache[cid];assert entry['descriptor']==proposal['descriptor']
                            if method=='A':assert proposal['descriptor']==descriptor(op,config={'warps':order[step]} if op=='gemm' else {'page_size':order[step],'splits':1})
                            else:
                                desc,change,source,normal=decode_response(response,op,task,protocol['tasks'][op+'/'+wid]['parent']);assert desc==proposal['descriptor'] and change==proposal['source_change']
                                if source is not None:
                                    assert (folder/'wmma_sm86.cu').read_bytes()==source==(folder/'eval-input/wmma_sm86.cu').read_bytes() and json.loads((folder/'eval-input/response.json').read_text())==normal;adapt=json.loads((folder/'schema-adaptation.json').read_text());assert adapt['raw_response_sha256']==sha(folder/'response.json') and adapt['derived_evaluation_response_sha256']==sha(folder/'eval-input/response.json')
                            static=folder/'static-review.json'
                            if static.exists():
                                review=json.loads(static.read_text());assert review['source_sha256']==proposal['source_change']['source_sha256']==sha(folder/'wmma_sm86.cu')
                                if review.get('identifier_mapping'):
                                    mapping=review['identifier_mapping'];ref=ROOT/review['reference_source'];assert sha(ref)==review['reference_sha256'] and re.sub(r'\b(?:'+'|'.join(re.escape(k) for k in mapping)+r')\b',lambda m:mapping[m[0]],(folder/'wmma_sm86.cu').read_text())==ref.read_text()
                            repeated=any(x['candidate_id']==cid for x in history);expected_reason='duplicate' if repeated and entry['reason']!='no_change' else entry['reason'];assert d['reason']==expected_reason and d['verdict']==('DUPLICATE' if expected_reason=='duplicate' else entry['verdict'])
                            assert execution['compile_status']==entry['compile_status'] and execution['correctness']==entry['correctness'] and execution['latency_ms']==entry.get('latencies',{}).get(wid,[])
                            if d['evaluation_scope']=='fresh':
                                assert not repeated;assert (root/'evaluation-reservations'/(cid+'.json')).exists();fresh_wall+=execution['evaluation_wall_seconds']
                            else:assert execution['evaluation_wall_seconds']==0.
                        else:
                            assert d['evaluation_scope']=='static' and d['reason'] in ['invalid_patch','invalid_json','model_call_failed'] and execution['compile_status']==execution['correctness']=='NOT_RUN' and not execution['latency_ms']
                            if d['reason']!='model_call_failed':
                                try:decode_response(response,op,task,protocol['tasks'][op+'/'+wid]['parent'])
                                except (AssertionError,ValueError,KeyError,TypeError,IndexError):pass
                                else:raise AssertionError('invalid proposal decoded as valid')
                        history.append(d);executions.append(execution);outcomes[d['verdict']]+=1;global_outcomes[d['verdict']]+=1;rows.append({'op':op,'workload':wid,'rep':rep,'method':method,'step':step,'verdict':d['verdict'],'reason':d['reason'],'candidate_id':d['candidate_id'],'scope':d['evaluation_scope'],'compile_status':execution['compile_status'],'correctness':execution['correctness'],'nominal_fallback_score':1.,'speedup_measured':False})
                    closed=json.loads((b/'closed.json').read_text()) if (b/'closed.json').exists() else None
                    if closed:
                        assert closed['proposals']==len(history) and closed['production_release'] is False
                        reason='screening_pass' if history and history[-1]['verdict']=='SCREENING_PASS' else 'three_consecutive_duplicate_or_invalid' if len(history)>=3 and all(x['reason'] in {'duplicate','no_change','invalid_json','invalid_patch','static_invalid','model_call_failed'} for x in history[-3:]) else 'proposal_cap' if len(history)==12 else None
                        if method=='A' and len(history)>=len(order):reason='finite_parameter_space_exhausted'
                        assert closed['reason']==reason and reason is not None
                    else:pending.append(str(b.relative_to(root)))
                    searches.append({'op':op,'workload':wid,'rep':rep,'method':method,'closed':bool(closed),'stop':closed['reason'] if closed else None,'proposals':len(history),'outcomes':dict(outcomes),'model_calls':0 if method=='A' else len(folders),'tokens':tokens,'api_wall_seconds':api_wall,'fresh_worker_wall_seconds':fresh_wall,'initial_profile_wall_seconds':pb['actual_profile_wall_seconds'] if method=='C' else 0.,'nominal_best_score':1.,'measured_gain':False,'billed_amount':None})
                b0=branch(root,op,wid,rep,'B')/'0/channel.json';c0=branch(root,op,wid,rep,'C')/'0/channel.json'
                if b0.exists() and c0.exists():
                    bvalue=json.loads(b0.read_text());cvalue=json.loads(c0.read_text());assert bvalue['common_sha256']==cvalue['common_sha256'] and bvalue['payload']['objective']==cvalue['payload']['objective']
    assert len(searches)==54 and len(calls)==len(set(calls));assert set(calls)=={k for k in ledger if k.startswith('formal-v1-')};assert len(ledger)<=700
    if not allow_incomplete:assert not pending and all(x['closed'] for x in searches)
    # A negative study must never plot an invented measured 1.0x result.
    assert not any(x['verdict']=='SCREENING_PASS' for x in rows),'positive screening requires a separately audited quantitative curve implementation'
    limitations=['Fixed-parent source/config search; C receives initial parent native-kernel profiles, not updated candidate or complete-wrapper profiles.','The common prompt identifies the selected public target, not all seven other guard shapes; execution latency feedback uses first-session single-input candidate samples. The other input policy and complete guard outcomes remain in external evidence.','Safe feedback collapses insufficient gain and guard regression into one reason enum and omits detailed compiler/static error explanations. B/C share this restriction.','Independent model searches reuse exact physical GPU outcomes; cache hits are not new independent timing samples. Fresh construction cost is charged to its first query, so asynchronous cost allocation is not a causal method-efficiency estimate.','Different source/DSO identities do not establish distinct generated CUDA machine code or distinct optimization algorithms.','No screening pass; fallback score1.0 is nominal, not a measured speedup. Currency and complete end-to-end elapsed are unknown.']
    result={'status':'INCOMPLETE' if pending else 'PASS','scope':'CPU replay of actual 54-branch fixed-parent study; not a new GPU measurement','protocol_sha256':sha(root/'protocol.json'),'searches':searches,'rows':rows,'closed_searches':sum(x['closed'] for x in searches),'attempts':len(rows),'new_model_calls':len(calls),'tokens':sum(x['tokens'] for x in searches),'outcomes':dict(global_outcomes),'pending':pending,'global_new_calls_used':len(ledger),'global_remaining_calls':700-len(ledger),'initial_accepted_profile_wall_seconds':sum(x['initial_profile_wall_seconds'] for x in searches),'cost_scope':'Measured API/profile/worker subphases only. Historical cache construction, rejected diagnostic profiles, CPU orchestration and end-to-end elapsed are separate, not zero. Currency is unknown. No billing estimate.','limitations':limitations,'heldout_read':False,'runtime_activation':False,'production_release':False};(root/('verification.partial.json' if pending else 'verification.json')).write_text(json.dumps(result,indent=2)+'\n');return result


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('root',type=Path);p.add_argument('--allow-incomplete',action='store_true');a=p.parse_args();r=audit(a.root.resolve(),a.allow_incomplete);print(json.dumps({k:r[k] for k in ['status','closed_searches','attempts','new_model_calls','tokens','global_new_calls_used','outcomes']}))
