"""CPU replay of the RTX3060 public fixed-cuBLAS counterpart; never touches sealed data."""
import argparse,hashlib,json,math,tempfile,zipfile
from pathlib import Path
from experiments.p3_01.audit import npy_digest,witness
from experiments.p3_01.cases import make_cases,case_key
from kernel_agent.protocol import decode_frame
from kernel_agent.optimization.source_packets import unpack
from kernel_agent.optimization.identity import group_log_ratio
from kernel_agent.operators.measurement import assess as research_assess
from experiments.full_delivery.gemm_deployment_gate import assess_deployment
from experiments.p3_01.audit import boundary
import sys,types,importlib.util
from experiments.full_delivery.gemm_ordinary_host import support_files as ordinary_support
ROOT=Path(__file__).resolve().parents[2]


def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def audit(folder):
    archive=folder/'segmented-archive';raw=archive/'remote';execution=json.loads((archive/'execution.json').read_text());collection=json.loads((archive/'collection.json').read_text())
    assert sha(archive/'raw.zip')==collection['archive_sha256']
    for n,m in execution['manifest'].items():assert (raw/n).stat().st_size==m['bytes'] and sha(raw/n)==m['sha256']
    protocol=json.loads((raw/'protocol.json').read_text());freeze=json.loads((folder/'qualification-freeze.json').read_text());pair_sha=sha(raw/'protocol.json')
    assert pair_sha==freeze['protocol_sha256'] and sha(raw/'host.py')==protocol['host_sha256']
    assert all(sha(raw/n)==h for n,h in protocol['controlled_files'].items())
    assert sha(raw/'precheck-result.json')==protocol['precheck_result_sha256'] and sha(raw/'workloads/gemm_deployment_stability_v2.json')==protocol['policy_sha256']
    policy=json.loads((raw/'workloads/gemm_deployment_stability_v2.json').read_text());assert policy['aa_pair_deviation_limit_exclusive']==.03 and policy['aa_group_geomean_deviation_limit_exclusive']==.015
    build=json.loads((raw/'artifact-binding.json').read_text());digests={role:build['builds'][role]['binary_sha256'] for role in ['incumbent','candidate']};remote_root=execution['remote_root'];assert len(set(digests.values()))==2
    assert build['status']=='REUSED_VERIFIED_SM86_ARTIFACTS' and build['new_compilation'] is False
    assert digests=={'candidate':'66fcc80f9122c919f92c72e37a01794a792c91a58b71d5705b160e0bc979dfc3','incumbent':'93fb2f4344521b7e9b246b54d1c30d4b1a8facf731e6385e18f6119e9c49c72c'}
    for role,info in build['builds'].items():assert sha(raw/Path(info['binary']).relative_to(remote_root))==digests[role]
    wheel=raw/'runtime.whl';assert sha(wheel)=='f2ef0ec1f054a9331a4134e6545e6ce2a84b5aae22923f81063cb5c7418a2706'
    with zipfile.ZipFile(wheel) as z:
        for n in z.namelist():
            if not n.endswith('/'):assert hashlib.sha256(z.read(n)).hexdigest()==sha(raw/'prefix'/n)
    # Reproduce generated reviewed support with original absolute root/prefix strings.
    # Adapter substitution only changes where generated text is written for CPU replay.
    with tempfile.TemporaryDirectory() as td:
        temp=Path(td);prefix=temp/'prefix';prefix.mkdir()
        with zipfile.ZipFile(wheel) as z:z.extractall(prefix)
        (temp/'runtime-binding.json').write_bytes((raw/'runtime-binding.json').read_bytes())
        stub=types.ModuleType('ordinary_host');stub.support_files=ordinary_support;prior=sys.modules.get('ordinary_host');sys.modules['ordinary_host']=stub
        try:
            spec=importlib.util.spec_from_file_location('counterpart_replay',raw/'counterpart_support.py');module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module);support=module.support_files(temp,prefix)
        finally:
            if prior is None:del sys.modules['ordinary_host']
            else:sys.modules['ordinary_host']=prior
        for p in support.rglob('*'):
            if p.is_file() and '__pycache__' not in p.parts:
                generated=p.read_bytes().replace(('nka.counterpart='+temp.name).encode(),('nka.counterpart='+Path(remote_root).name).encode());assert hashlib.sha256(generated).hexdigest()==sha(raw/'qualification'/p.relative_to(support)),str(p)
    pre_protocol=json.loads((raw/'precheck-protocol.json').read_text());assert sha(raw/'precheck-protocol.json')==protocol['precheck_protocol_sha256']
    assert all(sha(raw/n)==h for n,h in pre_protocol['controlled_files'].items())
    assert sha(raw/'precheck-protocol.json')==json.loads((folder/'precheck-freeze.json').read_text())['protocol_sha256']
    proof=json.loads((raw/'gpu-probe-result.json').read_text());assert proof['status']=='GPU_ABI_BOUNDARY_PASS' and proof['container_isolation_verified']
    import numpy as np
    probe=raw/'gpu-probe-cdi';record=json.loads((probe/'record.json').read_text());meta,payload=decode_frame(probe/'capture/stdout.bin',17*19*4+16384)
    assert meta==record['metadata']==proof['metadata'] and payload==(probe/'result.npy').read_bytes()
    assert record['cleanup']['removed'] and record['protected_files_unchanged'] and record['exit']==0 and record['reason'] is None
    for key in ['capabilities_zero','no_new_privs','artifact_write_denied','inputs_write_denied','external_connect_denied']:assert meta[key] is True
    for key in ['docker_socket_visible','control_canary_in_env','host_private_visible','host_pid_visible']:assert meta[key] is False
    assert meta['uid']==65534 and meta['interfaces']==['lo']
    actual=np.load(probe/'result.npy',allow_pickle=False);rng=np.random.default_rng(3060);x=(rng.standard_normal((17,23))*.25).astype(np.float16);y=(rng.standard_normal((23,19))*.25).astype(np.float16);reference=x.astype(np.float64)@y.astype(np.float64);assert np.all(np.abs(actual-reference)<=3e-5+1e-4*np.abs(reference))
    assert all(json.loads((raw/'runtime-binding.json').read_text())[k]==v for k,v in [('target_uuid','GPU-0140b895-e4e9-7c67-aba1-c7a953df1847'),('target_driver','580.95.05'),('capability',[8,6])])
    pre=json.loads((raw/'precheck-result.json').read_text());result=json.loads((raw/'qualification-result.json').read_text());assert pre['status']=='PRECHECK_PASS' and pre['qa_checks']==320
    assert pre['environment']==result['environment']==protocol['environment'] and result['protocol_sha256']==pair_sha
    assert result['container_isolation_verified'] is True and result['production_release'] is False and result['heldout_executed'] is False
    workloads,contract,cases=make_cases(ROOT);ledgers=json.loads((raw/'oracle-ledgers.json').read_text());aa=[];ab=[];qa=0;sanity=0;recovery=0;faults=[];segments=0;event_ms=0.;maximum=0.
    for job in sorted(raw.iterdir()):
        if not job.is_dir() or not job.name.startswith(('qa-','memcheck-','synccheck-','fault-','recovery-','aa-','ab-')) or not (job/'record.json').exists():continue
        record=json.loads((job/'record.json').read_text());req=json.loads((job/'inputs/request.json').read_text());phase=req['mode'];role=req.get('role','candidate');is_fault=job.name.startswith('fault-')
        cmd=record['command']
        for flag in ['--network=none','--sig-proxy=false','--read-only','--log-driver=none','--cap-drop=ALL','--security-opt=no-new-privileges','--pids-limit=32','--memory=2g','--memory-swap=2g','--cpus=2','--runtime=runc']:assert cmd.count(flag)==1,flag
        assert cmd[cmd.index('--user')+1]=='65534:65534' and cmd[cmd.index('--device')+1]=='nvidia.com/gpu=0' and 'NVIDIA_VISIBLE_DEVICES=void' in cmd and '--gpus' not in cmd
        assert protocol['image_id'] in cmd and cmd[cmd.index('--label')+1]=='nka.counterpart='+Path(remote_root).name
        assert cmd.count('--mount')==2 and all('readonly' in cmd[i+1] and 'docker.sock' not in cmd[i+1] for i,x in enumerate(cmd) if x=='--mount')
        from kernel_agent.bounded_io import confirmed_removed
        assert record['protected_unchanged'] and record['cleanup']['removed'] and confirmed_removed(record['cleanup']['after'])
        assert all(sha(job/n)==h for n,h in record['protected_hashes'].items())
        assert req['candidate_sha256']==digests['candidate'] and req['incumbent_sha256']==digests['incumbent']
        expected_pair=sha(raw/'precheck-protocol.json') if phase=='qa' else pair_sha;assert req['pair_sha256']==expected_pair
        for stream in ['stdout','stderr']:assert (job/'capture'/(stream+'.bin')).stat().st_size==record['stored_bytes'][stream]<=record['limits'][stream]
        if is_fault:
            assert record['status']=='REJECT'
            if job.name=='fault-timeout':assert record['reason']=='timeout' and 'NKA_FINITE_CUDA_READY' in (job/'capture/stderr.bin').read_text()
            else:assert record['exit']!=0
            faults.append(job.name);continue
        assert record['exit']==0 and record['reason'] is None and record['status']=='WITNESSES_READY'
        report=json.loads((job/'report.json').read_text());assert report['capability']==[8,6] and report['torch']=='2.14.0+cu130' and report['pair_sha256']==expected_pair and report['cpu_reference_available'] is False
        assert report['module_identity_distinct'] and report['function_identity_distinct'] and report['backends']=={'incumbent':'cublas','candidate':'cublas'}
        if phase=='qa':
            selected=req['cases'];limits={c['id']+'.npy':math.prod(c['shape_mnk'][:2])*4+16384 for c in selected};meta,payload=decode_frame(job/'capture/stdout.bin',sum(limits.values())+2*1024*1024);decoded,files=unpack(payload,limits);assert decoded==report and len(report['checks'])==len(selected)
            assert meta['uid']==65534 and meta['pair_sha256']==expected_pair
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
    telemetry=json.loads((raw/'qualification-telemetry.json').read_text());assert len(telemetry)>=30 and all(x['exit']==0 for x in telemetry)
    fields=[[float(x.strip()) for x in t['stdout'].split(',')[1:]] for t in telemetry];loaded=[v for v in fields if v[0]>=90];assert loaded
    thermal={'loaded_samples':len(loaded),'maximum_temperature_c':max(v[3] for v in loaded),'loaded_sm_mhz_range':[min(v[1] for v in loaded),max(v[1] for v in loaded)]}
    group_max=max(abs(math.exp(group_log_ratio(row,rep['session'])[0])-1) for rep in aa for row in rep['rows'])
    minimum=min((min(p['workload_speedups'].values()) for p in decision['policies']),default=None)
    assert not (raw/'clock-control.json').exists()
    passed=len(aa)==len(ab)==3 and decision['verdict']=='PASS'
    assert (result['status']=='PASS_DEPLOYMENT_STABILITY_AND_GUARDS')==passed
    post=json.loads((raw/'postflight.json').read_text());assert post['status']=='POSTFLIGHT_IDLE_NO_OWNED_CONTAINERS' and post['power_limit_unchanged_watts']==170 and post['gpu_lock_acquired']
    value={'status':'AUDIT_PASS_STABILITY_PASS' if passed else 'AUDIT_PASS_STABILITY_NOT_PASSED','scope':'RTX3060 nativeLinux boundedDocker fixed cuBLAS; numericalFP64 receipt attributed to trustedparent; wholefile/frame/NPY/input-output witness/pilot/Event/policy replay, no productioncertificate or3070inheritance','raw_files_verified':len(execution['manifest']),'qa_checks':qa,'sanitizer_checks':sanity,'recovery_checks':recovery,'expected_faults_rejected':len(faults),'aa_sessions':len(aa),'ab_sessions':len(ab),'raw_segments':segments,'measured_event_seconds':event_ms/1000,'max_aa_pair_deviation':maximum,'max_aa_group_deviation':group_max,'minimum_nonregression_geomean':minimum,'thermal_observation':thermal,'protocol_sha256':pair_sha,'environment':protocol['environment'],'image_id':protocol['image_id'],'binary_sha256':digests,'deployment_decision':decision,'new_model_calls':0,'heldout_executed':False,'container_isolation_verified':True,'production_release':False}
    (folder/'verification.json').write_text(json.dumps(value,indent=2));return value


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('folder',type=Path);a=p.parse_args();print(json.dumps(audit(a.folder.resolve())))
