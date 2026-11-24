"""Freeze final-only long-segment counts from the preserved noisy sizing batch."""
import argparse,json,math,statistics
from pathlib import Path
from evaluators.gemm_decision import ROOT,sha


def make_plan(source,target_ms=100):
    source=Path(source);dev=json.loads((ROOT/'workloads/gemm_development_v1.json').read_text())['workloads']
    reports=[json.loads((source/'run'/f'final-{s}.json').read_text()) for s in range(3)]
    rows=[]
    for w in dev:
        for count in (1,16):
            medians=[]
            for report in reports:
                row=next(c for c in report['candidate_aa'] if c['workload']==w['id'] and c['input_count']==count)
                for label in ['incumbent','candidate']:medians.append(statistics.median(s['elapsed_ms'] for s in row['segments'] if s['label']==label))
            estimate=min(medians)
            if not math.isfinite(estimate) or estimate<=0:raise ValueError('invalid calibration estimate')
            replays=max(12,math.ceil((12*target_ms/estimate)/16)*16)
            if replays>4096:raise ValueError('calibration replay budget exceeded')
            rows.append({'workload':w['id'],'shape_mnk':w['shape_mnk'],'input_count':count,'replays':replays,'calls_per_segment':32*replays,'sizing_min_session_label_median_ms':estimate})
    return {'mode':'frozen-final-long-v2','target_segment_ms':target_ms,'graph_launches':32,'pairs':9,'sessions':3,
            'source_final_sha256':{f'final-{s}.json':sha(source/'run'/f'final-{s}.json') for s in range(3)},
            'freeze_sha256':sha(source/'freeze.json'),'core_protocol_unchanged':True,'max_pair_deviation':.02,'max_group_geomean_deviation':.015,'rows':rows,
            'purpose':'final same-identity verification only; one identical-pair collection IS the A/A control; not search promotion'}


def main():
    p=argparse.ArgumentParser();p.add_argument('--source',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--target-segment-ms',type=int,choices=[100,250],default=100);a=p.parse_args()
    if a.output.exists():p.error('Do not overwrite the frozen plan')
    d=make_plan(a.source,a.target_segment_ms);a.output.write_text(json.dumps(d,indent=2)+'\n');print(json.dumps(d,indent=2))


if __name__=='__main__':main()
