"""CPU-only raw evidence replay for trusted-native SM89 stability; not an isolation certificate."""
import argparse,hashlib,json,math,tempfile,zipfile
from pathlib import Path
from experiments.p3_01.audit import npy_digest,witness
from experiments.p3_01.cases import make_cases,case_key
from kernel_agent.protocol import decode_frame
from kernel_agent.optimization.source_packets import unpack
from kernel_agent.optimization.identity import group_log_ratio
from kernel_agent.operators.measurement import assess as research_assess
from experiments.full_delivery.sm89_cublas_host import assess_deployment
from experiments.full_delivery.sm89_cublas_support import support_files
ROOT=Path(__file__).resolve().parents[2]


def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def audit(folder):
    archive=folder/'segmented-archive';raw=archive/'remote';execution=json.loads((archive/'execution.json').read_text());collection=json.loads((archive/'collection.json').read_text())
    assert sha(archive/'raw.zip')==collection['archive_sha256']
    for n,m in execution['manifest'].items():assert (raw/n).stat().st_size==m['bytes'] and sha(raw/n)==m['sha256']
    protocol=json.loads((raw/'protocol.json').read_text());freeze=json.loads((folder/'protocol.freeze.json').read_text());pair_sha=sha(raw/'protocol.json')
    assert pair_sha==freeze['protocol_sha256'] and sha(raw/'experiments/full_delivery/sm89_cublas_host.py')==protocol['host_sha256']
    assert all(sha(raw/n)==h for n,h in protocol['controlled_files'].items())
    assert sha(raw/'precheck-result.json')==protocol['precheck_result_sha256'] and sha(raw/'workloads/gemm_deployment_stability_v2.json')==protocol['policy_sha256']
    policy=json.loads((raw/'workloads/gemm_deployment_stability_v2.json').read_text());assert policy['aa_pair_deviation_limit_exclusive']==.03 and policy['aa_group_geomean_deviation_limit_exclusive']==.015
    build=json.loads((raw/'build-result.json').read_text());digests={role:build['builds'][role]['binary_sha256'] for role in ['incumbent','candidate']};remote_root=protocol['remote_root'];assert len(set(digests.values()))==2
    for role in digests:
        info=build['builds'][role];binary=Path(info['binary']).relative_to(remote_root);assert sha(raw/binary)==digests[role]
        assert info['cuda_source_sha256']==sha(raw/'kernels/gemm/cublas_sm86.cu')==sha(ROOT/'kernels/gemm/cublas_sm86.cu')
        assert info['contract_sha256']==sha(raw/'kernels/gemm/gemm_contract.cuh')
        ninja=raw/binary.parent/'build.ninja';assert 'compute_89' in ninja.read_text() and 'sm_89' in ninja.read_text()
    with tempfile.TemporaryDirectory() as td:
        temp=Path(td);prefix=temp/'prefix';prefix.mkdir()
        with zipfile.ZipFile(raw/'dev6.whl') as z:z.extractall(prefix)
        p=temp/'experiments/full_delivery/cublas_sustained_qualification_host.py';p.parent.mkdir(parents=True);p.write_bytes((raw/'experiments/full_delivery/cublas_sustained_qualification_host.py').read_bytes())
        support=support_files(temp,prefix)
        for p in support.rglob('*'):
            if p.is_file() and '__pycache__' not in p.parts:assert sha(p)==sha(raw/'qualification'/p.relative_to(support))
    pre=json.loads((raw/'precheck-result.json').read_text());result=json.loads((raw/'qualification-result.json').read_text());assert pre['status']=='PRECHECK_PASS' and pre['qa_checks']==320
    assert pre['environment']==result['environment']==protocol['environment'] and result['protocol_sha256']==pair_sha
    assert result['container_isolation_verified'] is False and result['production_release'] is False and result['heldout_executed'] is False
    workloads,contract,cases=make_cases(ROOT);ledgers=json.loads((raw/'oracle-ledgers.json').read_text());aa=[];ab=[];qa=0;sanity=0;recovery=0;faults=[];segments=0;event_ms=0.;maximum=0.
    for job in sorted(raw.iterdir()):
        if not job.is_dir() or not (job/'record.json').exists():continue
        record=json.loads((job/'record.json').read_text());req=json.loads((job/'inputs/request.json').read_text());phase=req['mode'];role=req.get('role','candidate');is_fault=job.name.startswith('fault-')
        assert record['execution_boundary']=='trusted-native-fixed-cublas-only' and record['container_isolation_verified'] is False
        assert record['protected_unchanged'] and record['cleanup']['removed'] and not record['cleanup']['remaining_owned_pids']
        assert all(sha(job/n)==h for n,h in record['protected_hashes'].items())
        assert req['candidate_sha256']==digests['candidate'] and req['incumbent_sha256']==digests['incumbent']
        expected_pair=sha(raw/'precheck-protocol.json') if phase=='qa' else pair_sha;assert req['pair_sha256']==expected_pair
        for stream in ['stdout','stderr']:assert (job/'capture'/(stream+'.bin')).stat().st_size==record['stored_bytes'][stream]<=record['limits'][stream]
        if is_fault:
            assert record['status']=='REJECT'
            if job.name=='fault-timeout':assert record['reason']=='timeout'
            else:assert record['exit']!=0
            faults.append(job.name);continue
        assert record['exit']==0 and record['reason'] is None and record['status']=='WITNESSES_READY'
        report=json.loads((job/'report.json').read_text());assert report['capability']==[8,9] and report['torch']=='2.7.1+cu126' and report['pair_sha256']==expected_pair and report['cpu_reference_available'] is False
        assert report['module_identity_distinct'] and report['function_identity_distinct'] and report['backends']=={'incumbent':'cublas','candidate':'cublas'}
        if phase=='qa':
            selected=req['cases'];limits={c['id']+'.npy':math.prod(c['shape_mnk'][:2])*4+16384 for c in selected};meta,payload=decode_frame(job/'capture/stdout.bin',sum(limits.values())+2*1024*1024);decoded,files=unpack(payload,limits);assert decoded==report and len(report['checks'])==len(selected)
            assert meta['uid']==0 and meta['pair_sha256']==expected_pair
            for row,case in zip(report['checks'],selected):
                data=files[case['id']+'.npy'];assert data==(job/'outputs'/(case['id']+'.npy')).read_bytes()
                assert npy_digest(data,case['shape_mnk'][:2])==row['output_sha256'];witness(row,case,ledgers[role])
                item=ledgers[role][case_key(case)];assert item['binary_sha256']==digests[role] and math.isfinite(item['max_abs_error'])
            if job.name.startswith('qa-'):qa+=len(selected)
            elif job.name.startswith(('memcheck-','synccheck-')):
                assert 'ERROR SUMMARY: 0 errors' in (job/'capture/stderr.bin').read_text();sanity+=len(selected)
            elif job.name.startswith('recovery-'):recovery+=len(selected)
        else:
            session=req['session'];assert req['cases']==contract and req['workloads']==workloads and report['session']==session
            assert report['graph_launches']==32 and report['replays']=='row_calibrated'
            meta,payload=decode_frame(job/'capture/stdout.bin',4*1024*1024);assert json.loads(payload)==report and meta['pair_sha256']==pair_sha
            for r in ['incumbent','candidate']:
                ledger=ledgers['candidate' if phase=='aa' else r]
                for key in ['checks_before','checks_after']:
                    values=report[key] if phase=='aa' else report[key][r];assert len(values)==len(contract)
                    for row,case in zip(values,contract):witness(row,case,ledger)
            rows=report['candidate_aa' if phase=='aa' else 'measurements'];assert len(rows)==16 and {(r['workload'],r['input_count']) for r in rows}=={(w['id'],n) for w in workloads for n in [1,16]}
            for row in rows:
                shape=next(w['shape_mnk'] for w in workloads if w['id']==row['workload']);assert row['shape_mnk']==shape and row['capture_order']==(['incumbent','candidate'] if session%2==0 else ['candidate','incumbent'])
                pilot=row['pilot_elapsed_ms'];assert len(pilot)==3 and all(math.isfinite(x) and x>0 for x in pilot) and row['replays']==max(1,min(8192,math.ceil(4000/sorted(pilot)[1]))) and row['timed_calls_per_segment']==32*row['replays']
                mean,deviation=group_log_ratio(row,session)
                if phase=='aa':maximum=max(maximum,deviation)
                for r in ['incumbent','candidate']:
                    ledger=ledgers['candidate' if phase=='aa' else r];values=[ledger[case_key({'shape_mnk':shape,'seed':701+i,'distribution':'random','offset_guard':False})] for i in range(row['input_count'])]
                    assert row['input_hashes']==[v['input_hashes'] for v in values] and row['output_hashes'][r]==[values[i%row['input_count']]['output_sha256'] for i in range(32)]
                segments+=len(row['segments']);event_ms+=sum(s['elapsed_ms'] for s in row['segments'])
            (aa if phase=='aa' else ab).append({'status':report['status'],'mode':phase,'session':session,'rows':rows})
    assert qa==320 and sanity==32 and recovery==2 and set(faults)=={'fault-crash','fault-timeout'}
    assert len(aa)==result['aa_sessions'] and len(ab)==result['ab_sessions'] and segments==result['raw_segments']
    decision=assess_deployment(aa,ab,workloads,policy);assert decision==result['deployment_decision']
    assert research_assess(aa,ab,workloads,json.loads((ROOT/'workloads/gemm_development_v1.json').read_text())['anchors'])==result['original_research_decision']
    passed=len(aa)==len(ab)==3 and decision['verdict']=='PASS'
    assert (result['status']=='PASS_DEPLOYMENT_STABILITY_AND_GUARDS')==passed
    value={'status':'AUDIT_PASS_STABILITY_PASS' if passed else 'AUDIT_PASS_STABILITY_NOT_PASSED','scope':'Trusted SM89 fixed cuBLAS, public numerical FP64 acceptance attributed to remote host; raw frame/NPY/witness/Graph/Event/policy replay; not worker isolation or production certificate','raw_files_verified':len(execution['manifest']),'qa_checks':qa,'sanitizer_checks':sanity,'recovery_checks':recovery,'expected_faults_rejected':len(faults),'aa_sessions':len(aa),'ab_sessions':len(ab),'raw_segments':segments,'measured_event_seconds':event_ms/1000,'max_aa_pair_deviation':maximum,'deployment_decision':decision,'new_model_calls':0,'heldout_executed':False,'container_isolation_verified':False,'production_release':False}
    (folder/'verification.json').write_text(json.dumps(value,indent=2));return value


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('folder',type=Path);a=p.parse_args();print(json.dumps(audit(a.folder.resolve())))
