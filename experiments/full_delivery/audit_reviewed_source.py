"""CPU-only receipts/frames/manifest/gate replay for the installed source evaluator."""
import argparse,hashlib,json,math
from pathlib import Path
from experiments.p3_01.audit import boundary,npy_digest,witness
from kernel_agent.optimization.source_packets import unpack
from kernel_agent.protocol import decode_frame
from kernel_agent.build_artifacts import ARCHIVE_LIMIT,decode_artifacts
from kernel_agent.operators.measurement import assess
from kernel_agent.optimization.identity import group_log_ratio


def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def audit(folder,expected_source):
    folder=Path(folder);execution=json.loads((folder/'execution.json').read_text())
    for n,m in execution['manifest'].items():
        p=Path(n);assert not p.is_absolute() and '..' not in p.parts;assert (folder/'remote'/p).stat().st_size==m['bytes'] and sha(folder/'remote'/p)==m['sha256']
    run=folder/'remote/run';summary=json.loads((run/'run.json').read_text());work=run/'evaluation';result=json.loads((work/'result.json').read_text());assert summary['source_sha256']==expected_source and summary['new_external_model_calls']==0 and summary['production_release'] is False;descriptor=json.loads((work/'pair-descriptor.json').read_text());assert descriptor['candidate_source_sha256']['wmma_sm86.cu']==expected_source and descriptor['binary_sha256']['candidate']==result['binary_sha256'];build=json.loads((work/'build/record.json').read_text());meta,data=decode_frame(work/'build/capture/stdout.bin',ARCHIVE_LIMIT);artifacts=decode_artifacts(data);assert hashlib.sha256(artifacts['nka_gemm_v1.so']).hexdigest()==result['binary_sha256']==sha(work/'build/artifacts/nka_gemm_v1.so');assert meta['gpu_available'] is False and '--gpus' not in build['command'] and build['source_unchanged'] and build['deps_unchanged'] and build['cleanup']['removed']
    ledgers=json.loads((work/'oracle-ledgers.json').read_text());checks=0;sanitizer_checks=0;wall=build['wall_seconds']
    for job in sorted(work.iterdir()):
        if not job.is_dir() or not (job/'record.json').exists() or job.name=='build':continue
        record=json.loads((job/'record.json').read_text());request=json.loads((job/'inputs/request.json').read_text());boundary(record,job);wall+=record['wall_seconds'];report=json.loads((job/'report.json').read_text());assert report['pair_sha256']==result['pair_sha256'] and report['candidate_binary_sha256']==result['binary_sha256'] and report['cpu_reference_available'] is False and report['module_identity_distinct'] and report['function_identity_distinct'];assert all('reference' not in x for x in request)
        if request['mode']=='qa':
            cases=request['cases'];role=request.get('role','candidate');limits={c['id']+'.npy':math.prod(c['shape_mnk'][:2])*4+16384 for c in cases};frame,packet=decode_frame(job/'capture/stdout.bin',sum(limits.values())+2*1024*1024);raw_report,outputs=unpack(packet,limits);assert raw_report==report and frame['pair_sha256']==result['pair_sha256']
            for row,case in zip(report['checks'],cases):
                raw=outputs[case['id']+'.npy'];assert raw==(job/'outputs'/(case['id']+'.npy')).read_bytes() and npy_digest(raw,case['shape_mnk'][:2])==row['output_sha256'];witness(row,case,ledgers[role]);checks+=1
            if job.name.startswith('sanitizer-'):assert 'ERROR SUMMARY: 0 errors' in (job/'capture/stderr.bin').read_text();sanitizer_checks+=len(cases)
    d=json.loads((Path(__file__).resolve().parents[2]/'workloads/gemm_development_v1.json').read_text());aa=[];ab=[];segments=0
    for mode,reports,key in [('aa',aa,'candidate_aa'),('ab',ab,'measurements')]:
        for i in range(3):
            job=work/(mode+'-'+str(i))
            if not job.exists():continue
            report=json.loads((job/'report.json').read_text());assert report['session']==i and report['graph_launches']==32 and report['replays']==12 and report['capability']==[8,6];reports.append({'status':report['status'],'mode':mode,'session':i,'rows':report[key]});segments+=sum(len(x['segments']) for x in report[key])
    decision=assess(aa,ab,d['workloads'],d['anchors']);assert decision['verdict']==result['verdict'] and decision['reason']==result['reason'];assert checks-sanitizer_checks==result['qa_checks']==320 and sanitizer_checks==16 and all(x['status']=='PASS' for x in result['sanitizers']);assert not result['frozen_best_updated'] and not result['runtime_activation'] and not result['heldout_executed'];value={'status':'PASS','scope':'CPU archive/binary/NKA1/NPY/independent-host-FP64-receipt/low-privilege-boundary/raw-gate audit; not a new GPU run','source_sha256':expected_source,'candidate_id':descriptor['candidate']['id'],'binary_sha256':result['binary_sha256'],'raw_files_verified':len(execution['manifest']),'independently_checked_qa_outputs':320,'sanitizer_checks':16,'sanitizer_errors':0,'raw_segments':segments,'decision':decision,'build_wall_seconds':build['wall_seconds'],'all_worker_wall_seconds':wall,'new_model_calls':0,'production_release':False};(folder/'verification.json').write_text(json.dumps(value,indent=2)+'\n');return value


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('folder',type=Path);p.add_argument('--source-sha',required=True);a=p.parse_args();print(json.dumps(audit(a.folder,a.source_sha)))
