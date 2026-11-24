"""CPU-only replay of the deployed-default qualification, including honest AA rejection."""
import argparse,hashlib,json,math,tempfile,zipfile
from pathlib import Path
from experiments.p3_01.audit import boundary,npy_digest,witness
from experiments.p3_01.cases import make_cases,case_key
from experiments.full_delivery.gemm_power_host import support_files
from experiments.full_delivery.gemm_deployment_gate import assess_deployment
from kernel_agent.optimization.identity import group_log_ratio
from kernel_agent.protocol import decode_frame
from kernel_agent.optimization.source_packets import unpack
from kernel_agent.operators.measurement import assess
ROOT=Path(__file__).resolve().parents[2]


def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def audit(folder):
    archive=folder/'segmented-archive';r=archive/'remote';execution=json.loads((archive/'execution.json').read_text());collection=json.loads((archive/'collection.json').read_text());assert sha(archive/'raw.zip')==collection['archive_sha256'];assert all((r/n).stat().st_size==m['bytes'] and sha(r/n)==m['sha256'] for n,m in execution['manifest'].items());protocol=json.loads((r/'protocol.json').read_text());freeze=json.loads((folder/'protocol.freeze.json').read_text());assert sha(r/'protocol.json')==freeze['protocol_sha256'] and sha(r/'host.py')==protocol['evaluator_sha256']==freeze['evaluator_sha256'];wheel=ROOT/'artifacts/wheels/nvidia_kernel_agent-0.1.1-py3-none-any.whl';assert sha(wheel)=='f2ef0ec1f054a9331a4134e6545e6ce2a84b5aae22923f81063cb5c7418a2706'
    with tempfile.TemporaryDirectory() as td,zipfile.ZipFile(wheel) as z:
        temp=Path(td);prefix=temp/'prefix'
        for n,h in protocol['installed_controls'].items():
            raw=z.read(n);assert hashlib.sha256(raw).hexdigest()==h;p=prefix/n;p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes(raw)
        generated=support_files(temp,prefix)
        for p in generated.rglob('*'):
            if p.is_file():assert sha(p)==sha(r/'qualification'/p.relative_to(generated))
    workloads,contract,cases=make_cases(ROOT);ledgers=json.loads((r/'oracle-ledgers.json').read_text());result=json.loads((r/'result.json').read_text());assert result['backend']=='cublas' and result['protocol_sha256']==sha(r/'protocol.json') and result['production_release'] is False and result['runtime_activation'] is False;qa=0;aa=[];ab=[];deviations=[];event_ms=0.;segments=0
    for job in sorted(r.iterdir()):
        if not job.is_dir() or not (job/'record.json').exists():continue
        record=json.loads((job/'record.json').read_text());request=json.loads((job/'inputs/request.json').read_text());report=json.loads((job/'report.json').read_text());boundary(record,job);assert all(sha(job/n)==h for n,h in record['protected_hashes'].items());assert request['diagnostic_backends']==report['backends']=={'incumbent':'cublas','candidate':'cublas'} and request['candidate_sha256']==protocol['default_binary_sha256'] and request['incumbent_sha256']==protocol['fallback_binary_sha256'] and request['pair_sha256']==report['pair_sha256']==sha(r/'protocol.json');assert report['module_identity_distinct'] and report['function_identity_distinct'] and report['cpu_reference_available'] is False
        if request['mode']=='qa':
            role=request['role'];selected=[c for c in cases if c['group']==job.name.split('qa-'+role+'-',1)[1]];assert request['cases']==selected;limits={c['id']+'.npy':math.prod(c['shape_mnk'][:2])*4+16384 for c in selected};meta,payload=decode_frame(job/'capture/stdout.bin',sum(limits.values())+2*1024*1024);raw_report,outputs=unpack(payload,limits);assert raw_report==report
            for row,case in zip(report['checks'],selected):
                raw=outputs[case['id']+'.npy'];assert raw==(job/'outputs'/(case['id']+'.npy')).read_bytes() and npy_digest(raw,case['shape_mnk'][:2])==row['output_sha256'];witness(row,case,ledgers[role]);qa+=1
        else:
            phase=request['mode'];session=request['session'];assert request['cases']==contract and request['workloads']==workloads and report['session']==session and report['graph_launches']==32 and report['replays']=='row_calibrated';rows=report['candidate_aa' if phase=='aa' else 'measurements'];assert len(rows)==16 and {(x['workload'],x['input_count']) for x in rows}=={(w['id'],n) for w in workloads for n in [1,16]}
            for role in ['incumbent','candidate']:
                ledger=ledgers['candidate' if phase=='aa' else role]
                for key in ['checks_before','checks_after']:
                    values=report[key] if phase=='aa' else report[key][role];assert len(values)==len(contract)
                    for row,case in zip(values,contract):witness(row,case,ledger)
            for row in rows:
                shape=next(w['shape_mnk'] for w in workloads if w['id']==row['workload']);assert row['shape_mnk']==shape and row['capture_order']==(['incumbent','candidate'] if session%2==0 else ['candidate','incumbent']);pilot=row['pilot_elapsed_ms'];assert len(pilot)==3 and all(math.isfinite(v) and v>0 for v in pilot) and row['replays']==max(1,min(8192,math.ceil(4000/sorted(pilot)[1]))) and row['timed_calls_per_segment']==32*row['replays']
                for role in ['incumbent','candidate']:
                    ledger=ledgers['candidate' if phase=='aa' else role];values=[ledger[case_key({'shape_mnk':shape,'seed':701+i,'distribution':'random','offset_guard':False})] for i in range(row['input_count'])];assert row['input_hashes']==[v['input_hashes'] for v in values] and row['output_hashes'][role]==[values[i%row['input_count']]['output_sha256'] for i in range(32)]
                event_ms+=sum(s['elapsed_ms'] for s in row['segments']);segments+=len(row['segments'])
            (aa if phase=='aa' else ab).append({'mode':phase,'session':session,'status':report['status'],'rows':rows})
    decision=assess(aa,ab,workloads,json.loads((ROOT/'workloads/gemm_development_v1.json').read_text())['anchors']);assert decision==result['optimization_screening_decision'] and qa==result['qa_checks']==320 and segments==result['raw_segments']
    policy=json.loads((r/'policy.json').read_text());assert sha(r/'policy.json')==protocol['policy_sha256'] and sha(r/'deployment_gate.py')==protocol['deployment_gate_sha256']
    research=decision;decision=assess_deployment(aa,ab,workloads,policy)
    if result['deployment_decision'] is not None:assert decision==result['deployment_decision']
    assert not (r/'clock-control.json').exists()
    telemetry=json.loads((r/'ordinary-telemetry.json').read_text());assert len(telemetry)>=30 and all(x['exit']==0 for x in telemetry);loaded=[]
    for sample in telemetry:
        f=[x.strip() for x in sample['stdout'].split(',')]
        if float(f[1])>=90:loaded.append([float(f[2]),float(f[3]),float(f[4])])
    power=json.loads((r/'power-control.json').read_text());assert power['requested_watts']==100 and power['original_watts']==240 and power['restored'] and power['thermal_counter_before_us']==power['thermal_counter_after_us'];assert protocol['power_limit_watts']==100 and protocol['restore_power_limit_watts']==240
    caps=[float(x['stdout'].split(',')[7].strip()) for x in telemetry if float(x['stdout'].split(',')[1].strip())>=90];physical=bool(caps) and all(x==100 for x in caps) and all(x[2]<83 for x in loaded);assert result['operating_profile']['thermal_valid']==physical and result['operating_profile']['restored'];assert result['operating_profile']['loaded_samples']==len(loaded)
    clock_proof={'control':'100W ceiling, dynamic clocks; restored240W','loaded_samples':len(loaded),'sm_mhz_range':[min(x[0] for x in loaded),max(x[0] for x in loaded)] if loaded else None,'maximum_temperature_c':max((x[2] for x in loaded),default=None)}
    qualified=physical and decision.get('aa_status')=='PASS' and len(ab)==3 and len(decision['policies'])==2 and all(min(p['workload_speedups'].values())>=.97 for p in decision['policies'])
    assert result['status']==('PASS_BASELINE_LATENCY_GUARDS' if qualified else 'AA_REJECTED') or (not qualified and result['status'] in ['GPU_BUSY_UNSAMPLED_REMAINDER','OPERATING_PROFILE_REJECTED'])
    if not qualified and len(aa)==3:assert decision['reason']=='aa_unstable_v2' and not ab
    maximum=max((group_log_ratio(row,report['session'])[1] for report in aa for row in report['rows']),default=None)
    group_max=max((abs(math.exp(group_log_ratio(row,report['session'])[0])-1) for report in aa for row in report['rows']),default=None)
    minimum=min((min(p['workload_speedups'].values()) for p in decision['policies']),default=None)
    value={'status':'AUDIT_PASS_QUALIFICATION_PASS' if qualified else 'AUDIT_PASS_QUALIFICATION_REJECT','scope':'CPU archive/whole-file SHA, generated-support delta, NKA1/NPY/host-FP64 receipt/Graph witnesses/pilot calculation and original gate replay; no new GPU run','qualification_status':result['status'],'qa_checks':qa,'raw_segments':segments,'raw_files_verified':len(execution['manifest']),'max_aa_pair_deviation':maximum,'max_aa_group_deviation':group_max,'minimum_nonregression_geomean':minimum,'aa_sessions':len(aa),'ab_sessions':len(ab),'decision':decision,'research_diagnostic':research,'policy_sha256':protocol['policy_sha256'],'protocol_sha256':sha(r/'protocol.json'),'measured_event_seconds':event_ms/1000,'clock_environment_verification':clock_proof,'new_model_calls':0,'certificate_generated':False,'production_release':False};(folder/'verification.json').write_text(json.dumps(value,indent=2));return value


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('folder',type=Path);a=p.parse_args();print(json.dumps(audit(a.folder.resolve())))
