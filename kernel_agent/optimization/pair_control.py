"""Ordered independent A/A then A/B batches. No retries, activation or private outputs in child."""
import json
from pathlib import Path
from .pair_validation import validate_report,validate_ledgers
from .pair_decision import decide_pair
from .pair_contract import validate_descriptor
from .cases import make_cases
from .identity import ROOT


def collect_batches(runner,work,contract,workloads,ledgers,descriptor,idle,*,evidence_kind='GPU'):
    work=Path(work);aa=[];ab=[];unstable=[]
    state={'aa_status':'NOT_RUN','ab_status':'NOT_RUN','verdict':'REJECT','reason':'not_started','sessions':[]}
    def finish(reason,aa_status=None,ab_status=None,decision=None):
        state.update(reason=reason)
        if aa_status:state['aa_status']=aa_status
        if ab_status:state['ab_status']=ab_status
        if decision is not None:state.update(verdict=decision['verdict'],decision=decision)
        (work/'batch-summary.json').write_text(json.dumps(state,indent=2)+'\n');return state
    try:
        validate_descriptor(descriptor);_,_,cases=make_cases(ROOT);validate_ledgers(ledgers,cases,descriptor)
        preflight=decide_pair([],[],ledgers,descriptor,evidence_kind=evidence_kind)
        if preflight['reason'] in ['known_candidate_aa_unstable','duplicate']:return finish(preflight['reason'],'NOT_RESAMPLED',decision=preflight)
    except (ValueError,KeyError,TypeError) as error:
        state['detail']=str(error);return finish('invalid_pair_prerequisites','NOT_RUN')
    if not idle():return finish('gpu_busy_before_aa','SKIPPED_GPU_BUSY')
    for phase,collection in [('aa',aa),('ab',ab)]:
        if phase=='ab' and not idle():return finish('gpu_busy_before_ab',ab_status='SKIPPED_GPU_BUSY')
        for session in range(3):
            report,record=runner.run({'mode':phase,'session':session,'cases':contract,'workloads':workloads},work/(phase+'-'+str(session)))
            collection.append({'report':report,'record':record});state['sessions'].append({'phase':phase,'session':session,'worker_status':record['status']})
            try:bad=validate_report(report,record,contract,workloads,ledgers,descriptor,phase,session);unstable+=bad
            except (ValueError,KeyError,TypeError,IndexError,OverflowError) as error:
                state['detail']=str(error);return finish('invalid_'+phase+'_witness',**{phase+'_status':'INVALID_WITNESS'})
            (work/'batch-summary.json').write_text(json.dumps(state,indent=2)+'\n')
        if phase=='aa':
            state['aa_status']='FAIL_STABILITY' if unstable else 'PASS'
            if unstable:return finish('candidate_aa_unstable',ab_status='NOT_RUN_AA_FAILED',decision=decide_pair(aa,[],ledgers,descriptor,evidence_kind=evidence_kind))
        else:state['ab_status']='PASS'
    decision=decide_pair(aa,ab,ledgers,descriptor,evidence_kind=evidence_kind)
    return finish(decision['reason'],decision=decision)
