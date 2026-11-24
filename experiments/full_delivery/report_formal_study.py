"""Export audited negative-study tables and standard Matplotlib figures."""
import argparse,collections,csv,hashlib,json,statistics
from pathlib import Path
from experiments.full_delivery.audit_formal_study import audit
from experiments.full_delivery.formal_results import evaluation_evidence
ROOT=Path(__file__).resolve().parents[2]


def csv_file(path,rows):
    if not rows:return
    with path.open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)


def report(root,output):
    result=audit(root);output.mkdir(exist_ok=False);csv_file(output/'proposals.csv',result['rows']);csv_file(output/'searches.csv',[{k:v for k,v in row.items() if k!='outcomes'} for row in result['searches']]);costs=[];latencies=[]
    for p in sorted((root/'cache-updates').glob('*.json')):
        entry=json.loads(p.read_text());cid=entry['descriptor']['id'];reserve=root/'evaluation-reservations'/(cid+'.json')
        if not reserve.exists() or (root/'evaluation-reservations'/(cid+'.cancelled-before-execution.json')).exists():continue
        reservation=json.loads(reserve.read_text());folder=evaluation_evidence(root,cid);vfile=folder/'verification.json';failure=folder/'build-failure-verification.json'
        if not vfile.exists() and not failure.exists():continue
        verification=json.loads((vfile if vfile.exists() else failure).read_text());gpu_ms=0.;segments=0
        for rp in sorted((folder/'remote/run/evaluation').glob('*-*/report.json')):
            if not rp.parent.name.startswith(('aa-','ab-')):continue
            r=json.loads(rp.read_text());phase=rp.parent.name.split('-')[0]
            for row in r['candidate_aa' if phase=='aa' else 'measurements']:
                for label in ['incumbent','candidate']:
                    values=[x['elapsed_ms']/384 for x in row['segments'] if x['label']==label];gpu_ms+=sum(x['elapsed_ms'] for x in row['segments'] if x['label']==label);segments+=len(values);latencies.append({'candidate_id':cid,'phase':phase,'session':r['session'],'workload':row['workload'],'input_count':row['input_count'],'role':label,'median_call_ms':statistics.median(values),'segments':len(values),'timed_calls_per_segment':384,'screening_verdict':entry['verdict'],'screening_reason':entry['reason'],'new_performance_claim':False})
        costs.append({'candidate_id':cid,'compile_status':entry['compile_status'],'correctness':entry['correctness'],'reason':entry['reason'],'build_wall_seconds':entry['build_wall_seconds'],'all_worker_wall_seconds':entry['evaluation_wall_seconds'],'measured_cuda_event_seconds':gpu_ms/1000,'raw_event_segments':segments,'gpu_execution_performed':verification.get('gpu_execution_performed',bool(segments)),'evidence':str(folder.relative_to(ROOT))})
    csv_file(output/'physical-evaluations.csv',costs);csv_file(output/'workload-latencies.csv',latencies)
    curves=[]
    for search in result['searches']:
        b=root/'searches'/search['op']/search['workload']/str(search['rep'])/search['method'];tokens=0;wall=search['initial_profile_wall_seconds'];gpu=0.
        curves.append({**{k:search[k] for k in ['op','workload','rep','method']},'proposal_count':0,'known_model_tokens':0,'measured_subphase_wall_seconds':wall,'fresh_worker_wall_seconds':0.,'retained_fallback_score':1.,'measured_gain':False})
        for step in range(search['proposals']):
            folder=b/str(step);d=json.loads((folder/'decision.json').read_text());usage=folder/'usage.json'
            if usage.exists():u=json.loads(usage.read_text());tokens+=u['usage']['total_tokens'];wall+=u['model_wall_seconds']
            failure=folder/'failure.json'
            if failure.exists():wall+=json.loads(failure.read_text())['wall_seconds']
            wall+=d['execution']['evaluation_wall_seconds'];gpu+=d['execution']['evaluation_wall_seconds'];curves.append({**{k:search[k] for k in ['op','workload','rep','method']},'proposal_count':step+1,'known_model_tokens':tokens,'measured_subphase_wall_seconds':wall,'fresh_worker_wall_seconds':gpu,'retained_fallback_score':1.,'measured_gain':False})
    csv_file(output/'search-curves.csv',curves)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import numpy as np
    colors={'A':'#2563eb','B':'#0f766e','C':'#d97706'};fig,axes=plt.subplots(1,3,figsize=(14,4.4));methods=['A','B','C']
    for method in methods:
        values=[x for x in curves if x['method']==method];groups=collections.defaultdict(list)
        for row in values:groups[(row['op'],row['workload'],row['rep'])].append(row)
        for i,rows in enumerate(groups.values()):axes[0].plot([x['proposal_count'] for x in rows],[x['retained_fallback_score'] for x in rows],color=colors[method],alpha=.4,label=method if i==0 else None,linestyle={'A':':','B':'--','C':'-'}[method])
    axes[0].set(xlabel='Actual proposals before stop',ylabel='Nominal retained fallback score',ylim=(.97,1.03),title='No screening-qualified gain');axes[0].legend(title='Method');bottom=np.zeros(3)
    for verdict,color in [('REJECT','#64748b'),('DUPLICATE','#cbd5e1')]:
        values=[sum(x['verdict']==verdict and x['method']==m for x in result['rows']) for m in methods];axes[1].bar(methods,values,bottom=bottom,color=color,label=verdict);bottom+=values
    axes[1].set(ylabel='Actual attempts',title='All failures and duplicates retained');axes[1].legend(fontsize=8)
    bottom=np.zeros(3)
    for key,label,color in [('api_wall_seconds','API','#2563eb'),('initial_profile_wall_seconds','Initial profile','#d97706'),('fresh_worker_wall_seconds','Fresh build + workers','#64748b')]:
        values=[sum(s[key] for s in result['searches'] if s['method']==m) for m in methods];axes[2].bar(methods,values,bottom=bottom,color=color,label=label);bottom+=values
    axes[2].set(ylabel='Measured subphase wall seconds',title='Historical cache construction separate');axes[2].legend(fontsize=8)
    for ax in axes:ax.spines[['top','right']].set_visible(False)
    fig.suptitle('54 fixed-parent searches · retained fallback scores are nominal',fontsize=13);fig.tight_layout();fig.savefig(output/'study.png',dpi=180);fig.savefig(output/'study.svg');plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(11,4.4))
    for method in methods:
        groups=collections.defaultdict(list)
        for row in curves:
            if row['method']==method:groups[(row['op'],row['workload'],row['rep'])].append(row)
        for i,values in enumerate(groups.values()):
            for ax,key in zip(axes,['known_model_tokens','measured_subphase_wall_seconds']):ax.plot([x[key] for x in values],[x['retained_fallback_score'] for x in values],color=colors[method],alpha=.4,linestyle={'A':':','B':'--','C':'-'}[method],label=method if i==0 else None)
    for ax,label in zip(axes,['Known model tokens (not currency)','Measured subphase wall seconds']):ax.set(xlabel=label,ylabel='Nominal retained fallback score',ylim=(.97,1.03));ax.spines[['top','right']].set_visible(False);ax.legend(title='Method')
    fig.suptitle('Observed search budgets · no qualified improvement',fontsize=13);fig.text(.5,.02,'Currency unknown; wall excludes historical cache construction and unmeasured orchestration.',ha='center',fontsize=9);fig.tight_layout(rect=(0,.05,1,.94));fig.savefig(output/'budget-curves.png',dpi=180);fig.savefig(output/'budget-curves.svg');plt.close(fig)
    totals={'searches':54,'attempts':result['attempts'],'model_calls':result['new_model_calls'],'tokens':result['tokens'],'screening_passes':0,'new_physical_evaluations':len(costs),'fresh_build_wall_seconds':sum(x['build_wall_seconds'] for x in costs),'fresh_all_worker_wall_seconds':sum(x['all_worker_wall_seconds'] for x in costs),'fresh_event_seconds':sum(x['measured_cuda_event_seconds'] for x in costs),'initial_profile_wall_seconds':result['initial_accepted_profile_wall_seconds'],'api_wall_seconds':sum(x['api_wall_seconds'] for x in result['searches']),'billed_amount':None,'targets_1_03_and_1_05':'NOT_REACHED','hardware_feedback_benefit':'NOT_ESTABLISHED','production_release':False};(output/'totals.json').write_text(json.dumps(totals,indent=2));(output/'plot-environment.json').write_text(json.dumps({'matplotlib':matplotlib.__version__,'numpy':np.__version__},indent=2));names=['experiments/full_delivery/report_formal_study.py','experiments/full_delivery/audit_formal_study.py','experiments/full_delivery/formal_results.py'];(output/'code-hashes.json').write_text(json.dumps({n:hashlib.sha256((ROOT/n).read_bytes()).hexdigest() for n in names},indent=2));manifest={str(p.relative_to(output)):hashlib.sha256(p.read_bytes()).hexdigest() for p in output.iterdir() if p.is_file()};(output/'manifest.json').write_text(json.dumps(manifest,indent=2));return totals


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('root',type=Path);p.add_argument('--output',type=Path,required=True);a=p.parse_args();print(json.dumps(report(a.root.resolve(),a.output.resolve())))
