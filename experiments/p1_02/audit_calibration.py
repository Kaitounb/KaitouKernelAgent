"""Recompute the P1-02 A/A negative controls directly from archived Event segments."""
import argparse
import hashlib
import json
import math
from pathlib import Path
import statistics


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    if args.output.exists():
        parser.error('Use a new audit output')
    root=Path(__file__).resolve().parents[2]
    protocol_path=root/'workloads/gemm_measurement_v1.json'
    protocol=json.loads(protocol_path.read_text())
    observed=[]
    for folder in ['20261121-p1-02-gemm-aa-anchors','20261121-p1-02-gemm-aa-template']:
        run=root/'evidence'/folder/'run'
        execution=json.loads((run/'execution.json').read_text())
        assert execution['preflight_idle']
        for session in range(3):
            path=run/f'aa-session-{session}.json'
            data=json.loads(path.read_text())
            assert data['status']=='PASS' and data['session']==session and len(data['checks'])==6
            assert data['graph_launches']==32 and data['replays']==12
            for case in data['checks']:
                assert case['cpu_reference_and_inputs_pass']
                assert len(case['segments'])==18
                ratios=[]
                for pair in range(9):
                    rows=[r for r in case['segments'] if r['pair']==pair]
                    assert len(rows)==2 and {r['label'] for r in rows}=={'A','A_prime'}
                    assert {r['order'] for r in rows}=={0,1}
                    values={r['label']:r['elapsed_ms'] for r in rows}
                    assert all(math.isfinite(v) and v>1 for v in values.values())
                    ratios.append(values['A']/values['A_prime'])
                logs=list(map(math.log,ratios));mean=statistics.mean(logs);sem=statistics.stdev(logs)/3
                gm=math.exp(mean);ci=[math.exp(mean-2.306*sem),math.exp(mean+2.306*sem)]
                assert abs(gm-case['geomean'])<1e-12
                assert max(abs(a-b) for a,b in zip(ci,case['exploratory_t8_95_interval']))<1e-12
                maxdev=max(abs(x-1) for x in ratios)
                # Empirical calibration envelope, separate from candidate promotion.
                assert maxdev<.02 and abs(gm-1)<.015
                threshold=protocol['screening']
                false_promotion=gm>=threshold['minimum_anchor_geomean'] and ci[0]>threshold['minimum_exploratory_interval_lower']
                assert not false_promotion
                observed.append({'source':str(path.relative_to(root)), 'source_sha256':hashlib.sha256(path.read_bytes()).hexdigest(),
                                 'shape_mnk':case['shape_mnk'],'input_count':case['input_count'],'geomean':gm,
                                 'max_pair_deviation':maxdev,'exploratory_interval':ci,'would_promote':False})
    assert len(observed)==36
    report={'status':'PASS','kind':'CPU recomputation of actual GPU evidence; not a new GPU run',
            'protocol_sha256':hashlib.sha256(protocol_path.read_bytes()).hexdigest(),
            'groups':36,'pairs':324,'segments':648,'false_promotions':0,
            'max_pair_deviation':max(r['max_pair_deviation'] for r in observed),
            'max_group_geomean_deviation':max(abs(r['geomean']-1) for r in observed),'observed':observed}
    args.output.write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({k:v for k,v in report.items() if k!='observed'},indent=2))


if __name__=='__main__':
    main()
