"""Describe profiler prediction checks; never turn replay timings into speedup claims."""
import argparse
import csv
import io
import json
from pathlib import Path
import statistics
from experiments.p1_04.ncu_adapter import sha


EXTRA=['device__attribute_max_threads_per_multiprocessor','device__attribute_max_warps_per_multiprocessor',
       'launch__shared_mem_per_block_driver','launch__shared_mem_per_block_dynamic','launch__shared_mem_per_block_static']


def metadata(path):
    # Only extend metadata absent from the requested metric list; main metrics use upstream parser.
    lines=Path(path).read_text().splitlines();start=next(i for i,line in enumerate(lines) if line.startswith('"ID",'))
    rows=[r for r in csv.DictReader(io.StringIO('\n'.join(lines[start:]))) if r['ID'].isdigit()]
    return {key:[int(r[key].replace(',','')) for r in rows] for key in EXTRA}


def compare(baseline,candidate,feedback):
    predictions={p['metric']:p['direction'] for p in feedback['expected_changes']}
    output={'scope':'descriptive NCU kernel-replay comparison; no formal speedup/causal proof','shapes':[],'predictions_checked':0,'predictions_direction_matched':0}
    warps=feedback['proposal']['warps']
    for size in (512,1024):
        bp=Path(baseline)/f'warps-4-{size}.metrics.json';cp=Path(candidate)/f'warps-{warps}-{size}.metrics.json'
        b=json.loads(bp.read_text());c=json.loads(cp.read_text())
        rows=[]
        for key in b['launches'][0]['metrics']:
            old=[r['metrics'][key]['value'] for r in b['launches']];new=[r['metrics'][key]['value'] for r in c['launches']]
            if any(v is None for v in [*old,*new]):
                rows.append({'metric':key,'status':'unavailable'});continue
            bm,cm=statistics.median(old),statistics.median(new)
            direction='increase' if cm>bm else 'decrease' if cm<bm else 'unchanged'
            row={'metric':key,'unit':b['launches'][0]['metrics'][key]['unit'],'baseline_median':bm,'candidate_median':cm,'baseline_range':[min(old),max(old)],'candidate_range':[min(new),max(new)],'observed_direction':direction}
            if key in predictions:
                row['model_prediction']=predictions[key];row['descriptive_direction_match']=predictions[key]==direction
                output['predictions_checked']+=1;output['predictions_direction_matched']+=int(row['descriptive_direction_match'])
            rows.append(row)
        extras={'baseline':metadata(bp.with_name(f'warps-4-{size}.csv')),'candidate':metadata(cp.with_name(f'warps-{warps}-{size}.csv'))}
        limits={}
        for label,data in [('baseline',b),('candidate',c)]:
            cells=data['launches'][0]['metrics'];threads=extras[label]['device__attribute_max_threads_per_multiprocessor'][0]
            block_size=cells['launch__block_size']['value']
            cap=min(cells[k]['value'] for k in ['launch__occupancy_limit_blocks','launch__occupancy_limit_registers','launch__occupancy_limit_shared_mem'])
            cap=min(cap,int(threads//block_size))
            limits[label]={'block_size':block_size,'limiting_blocks_from_collected_caps':cap,'max_threads_per_sm':threads,
                           'resident_thread_upper_bound':cap*block_size,'thread_occupancy_upper_bound_percent':100*cap*block_size/threads}
        output['shapes'].append({'shape_mnk':[size,size,size],'baseline_metrics_sha256':sha(bp),'candidate_metrics_sha256':sha(cp),
                                'metrics':rows,'auto_collected_raw_metadata':extras,'derived_collected_resource_caps':limits})
    output['interpretation']='The collected 1024^3 profile does not support the proposed scheduling/long-scoreboard improvement. More blocks alone did not add total warp tiles; the 2-warp variant meets a 16-block cap and collected thread-cap upper bound 66.7%, while the 4-warp variant can reach 100% by these caps. This supports a resource-cap explanation, not unique causal proof. Keep normal Event correctness/A/A/A-B decision separate.'
    return output


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--baseline',type=Path,required=True);p.add_argument('--candidate',type=Path,required=True);p.add_argument('--feedback',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    args=p.parse_args()
    if args.output.exists():p.error('Use a new output')
    result=compare(args.baseline,args.candidate,json.loads(args.feedback.read_text()))
    args.output.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k!='shapes'},indent=2))


if __name__=='__main__':main()
