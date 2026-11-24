"""CPU-only independent audit of public SDK measurement archives."""
import argparse,hashlib,io,json,math,zipfile
from pathlib import Path
from kernel_agent.operators.measurement import assess
from kernel_agent.optimization.identity import group_log_ratio


def audit(root):
    root=Path(root)
    if (root/'scheduling-invalidated.json').exists():raise ValueError('entire batch invalidated before numeric review')
    raw=(root/'raw.zip').read_bytes();wire=json.loads((root/'result.json').read_text());assert hashlib.sha256(raw).hexdigest()==wire['archive_sha256']
    with zipfile.ZipFile(io.BytesIO(raw)) as z:
        manifest=json.loads(z.read('manifest.json'))
        for name,meta in manifest.items():
            p=Path(name);assert not p.is_absolute() and '..' not in p.parts;data=z.read('blobs/'+meta['sha256']);assert len(data)==meta['bytes'] and hashlib.sha256(data).hexdigest()==meta['sha256'] and (root/'remote'/p).read_bytes()==data
    r=root/'remote';freeze=json.loads((r/'freeze.json').read_text());result=json.loads((r/'result.json').read_text());sdk=json.loads((r/'sdk-before.json').read_text());assert sdk==json.loads((r/'sdk-after.json').read_text()) and result['sdk_unchanged'] and result['freeze_id']==freeze['id']
    assert all(hashlib.sha256((r/p).read_bytes()).hexdigest()==h for p,h in freeze['files'].items());assert result['source_hashes_before']==result['source_hashes_after']==freeze['files']
    d=json.loads((r/'workloads/mla_development_v1.json').read_text());workloads=[w for w in d['workloads'] if freeze['phase']!='calibration' or w['id'] in d['anchors']];reports={};cap=[8,6] if freeze['target']=='3070' else [8,9];segments=[];deviations=[];peak=0
    for mode in ['aa','ab']:
        reports[mode]=[]
        for path in sorted(r.glob(mode+'-?.json')):
            report=json.loads(path.read_text());session=report['session'];assert report['mode']==mode and report['config']==freeze['config'] and report['capability']==cap and report['graph_launches']==report['replays']==32
            expected={'incumbent':freeze['config'] if mode=='aa' else {'page_size':16,'splits':1},'candidate':freeze['config']}
            for row in report['rows']:
                assert row['qa_after_pass'] and row['inputs_and_guards_unchanged'] and row['configs']==expected and row['capture_order']==(['incumbent','candidate'] if session%2==0 else ['candidate','incumbent'])
                assert row['native_hashes'] and all(sdk['native'].get(n)==h for n,h in row['native_hashes'].items())
                assert all(len(row['input_hashes'][role])==row['input_count'] and len(row['output_hashes'][role])==32 and len(row['qa_before_errors'][role])==row['input_count'] for role in ['incumbent','candidate'])
                mean,dev=group_log_ratio(row,session);deviations.append(dev) if mode=='aa' else None;segments.extend(x['elapsed_ms'] for x in row['segments']);peak=max(peak,row['peak_allocated_bytes'])
            reports[mode].append(report)
    decision=assess(reports['aa'],reports['ab'],workloads,d['anchors']);assert decision==result['decision'];commands=[x for x in result['records'] if x['label'].startswith(('aa-0','aa-1','aa-2','ab-0','ab-1','ab-2'))];assert all(x['exit']==0 for x in commands)
    monitors=json.loads((r/'gpu-monitor.json').read_text());admission=[x for x in monitors if x['phase']=='idle-admission'];assert len(admission)>=3*len(commands)
    return {'status':'PASS','scope':'CPU raw/hash/recomputed gate audit; numerical and Event data came from GPU','measurement_status':result['status'],'decision':decision,'raw_files_verified':len(manifest),'raw_segments':len(segments),'segment_ms_range':[min(segments),max(segments)],'max_aa_pair_deviation':max(deviations),'peak_allocated_bytes':peak,'capability':cap,'native_sha256':sdk['native'],'new_model_calls':0,'performance_claim':False,'production_release':False}


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('root',type=Path);a=p.parse_args();value=audit(a.root);(a.root/'verification.json').write_text(json.dumps(value,indent=2)+'\n');print(json.dumps(value))
