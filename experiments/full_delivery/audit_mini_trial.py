"""Independently replay the 36-attempt bookkeeping and feedback boundary, CPU only."""
import argparse,collections,hashlib,json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]


def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def audit(root):
    root=Path(root);protocol=json.loads((root/'protocol.json').read_text());ledger=json.loads((ROOT/'evidence/20261122-full-project-model-calls/ledger.json').read_text());rows=[];cells={};call_count=0;token_count=0
    for name,h in protocol['controls'].items():assert sha(root/'sources'/name)==h
    for op in ['gemm','mla']:
        assert sha(root/(op+'-task.json'))==protocol['tasks'][op] and sha(root/(op+'-profile.json'))==protocol['profiles'][op]
        for method in ['A','B','C']:
            branch=root/op/method;previous=[];outcomes=collections.Counter();tokens=0;wall=0.;unique=set()
            for step in range(6):
                folder=branch/str(step);result=json.loads((folder/'result.json').read_text());decision=json.loads((folder/'decision.json').read_text());assert result['op']==op and result['method']==method and result['step']==step and decision['status'] in ['REJECT','DUPLICATE','SCREENING_PASS'] and not decision.get('promotion_performed',False) and not decision['production_release'];
                if 'promotion_performed' not in decision:assert decision['reason']=='invalid_proposal' and result['status']=='REJECT' and decision['execution_feedback']['compile_status']==decision['execution_feedback']['correctness']=='NOT_RUN'
                if isinstance(decision['evidence'],dict):assert all(sha(ROOT/n)==h for n,h in decision['evidence'].items())
                else:
                    assert decision['reason']=='invalid_proposal' and [Path(x).resolve() for x in decision['evidence']]==[(folder/'result.json').resolve()] and decision['execution_feedback']['candidate_id']==sha(folder/'response.json')
                    from kernel_agent.optimization.patches import apply_reviewable_patch
                    content=json.loads((folder/'response.json').read_text())['choices'][0]['message']['content'];source=json.loads((root/(op+'-task.json')).read_text())['source'].encode()
                    try:
                        try:apply_reviewable_patch(content,source)
                        except ValueError:apply_reviewable_patch(content,source,recount=True)
                    except ValueError as error:assert str(error)==result['reason']
                    else:raise AssertionError('invalid proposal no longer rejected')
                outcomes[decision['status']]+=1;unique.add(decision['execution_feedback']['candidate_id'])
                if method=='A':assert result['new_model_calls']==0 and result['config']==({'warps':protocol['A_sequences'][op][step]} if op=='gemm' else {'page_size':protocol['A_sequences'][op][step],'splits':1})
                else:
                    channel=json.loads((folder/'channel.json').read_text());request=json.loads((folder/'request.json').read_text());response=json.loads((folder/'response.json').read_text());usage=json.loads((folder/'usage.json').read_text());reservation=ledger['calls']['mini-v2-'+op+'-'+method+'-'+str(step)];assert reservation['status']=='RECEIVED' and reservation['request_sha256']==sha(folder/'request.json') and reservation['usage']==usage['usage']==result['usage'];assert response['model']==request['model']==protocol['model']=='deepseek-flash' and request['temperature']==.2 and request['max_tokens']==768;assert json.loads(request['messages'][1]['content'])==channel['payload'];assert channel['payload']['feedback']['executions']==previous and channel['payload']['task']==json.loads((root/(op+'-task.json')).read_text())
                    if method=='B':assert channel['profile_loader_calls']==0 and 'profile' not in channel['payload']['feedback']
                    else:assert channel['profile_loader_calls']==1 and channel['payload']['feedback']['profile']==json.loads((root/(op+'-profile.json')).read_text())
                    assert reservation['usage']['prompt_tokens']+reservation['usage']['completion_tokens']==reservation['usage']['total_tokens'];tokens+=usage['usage']['total_tokens'];wall+=usage['model_wall_seconds'];call_count+=1;token_count+=usage['usage']['total_tokens']
                previous.append(decision['execution_feedback']);rows.append({'op':op,'method':method,'step':step,'status':decision['status'],'reason':decision['reason'],'candidate_id':decision['execution_feedback']['candidate_id'],'nominal_fallback_score':1.0,'measured_speedup_claim':False})
            cells[op+'-'+method]={'attempts':6,'outcomes':dict(outcomes),'unique_feedback_identities':len(unique),'model_calls':0 if method=='A' else 6,'tokens':tokens,'api_wall_seconds':wall,'best_after':'unchanged_initial','nominal_fallback_score':1.0}
        b=json.loads((root/op/'B/0/channel.json').read_text());c=json.loads((root/op/'C/0/channel.json').read_text());assert b['common_sha256']==c['common_sha256']
    assert len(rows)==36 and call_count==24 and not any(r['status']=='SCREENING_PASS' for r in rows)
    result={'status':'PASS','scope':'P3-04 single small trial, not formal P6 and not evidence of hardware-feedback benefit','attempts':36,'new_model_calls':call_count,'tokens':token_count,'cells':cells,'rows':rows,'initial_profile_wall_seconds':{op:json.loads((root/(op+'-profile.json')).read_text())['profile_wall_seconds'] for op in ['gemm','mla']},'cost_scope':'Actual API tokens/wall and referenced compile/worker/profile phase costs; exact cache hits have zero marginal GPU work. Cache construction costs are separate; no complete billed currency or total end-to-end wall time claimed.','limitations':['Only one independent search per cell.','Execution feedback schema included compile/correctness/latency but omitted final gate reason and previous source patches; repeated/rename/no-op proposals reveal missing search context. Formal protocol must add explicit safe verdict/duplicate/source history and optimization objective.','SM86 parameter/source AA noise rejected several identities; no resampling and no stable performance-gain claim.','MLA uses fixed CUDA SDK configuration/graph transforms; free MLA native source is outside this study.'],'early_stopped_v1_calls':4,'early_stopped_v1_excluded':True,'heldout_read':False,'production_release':False};(root/'verification.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps({k:result[k] for k in ['status','attempts','new_model_calls','tokens','cells','production_release']}));return result


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('root',type=Path);a=p.parse_args();audit(a.root)
