"""Independent CPU replay of installed wheels, fixed outputs, real boundaries and rollback."""
import argparse,hashlib,io,json,math,zipfile
from pathlib import Path
import numpy as np
from kernel_agent.protocol import decode_frame
from kernel_agent.sm89.runner import validate_boundary
ROOT=Path(__file__).resolve().parents[2]


def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def audit(folder):
    archive=folder/'segmented-archive';raw=archive/'remote';manifest=json.loads((archive/'execution.json').read_text());receipt=json.loads((archive/'collection.json').read_text());assert sha(archive/'raw.zip')==receipt['archive_sha256']
    for n,m in manifest['manifest'].items():assert (raw/n).stat().st_size==m['bytes'] and sha(raw/n)==m['sha256']
    result=json.loads((raw/'install-result.json').read_text());assert result['status']=='PASS';remote=Path(result['remote_root']);work=raw/'installed-validation';validation=json.loads((work/'result.json').read_text());assert validation==result['validation'] and validation['status']=='PASS' and validation['cpu_tests']==27
    assert validation['baseline_created_for_first_sm89_rollback'] is True and validation['new_event_samples']==0 and validation['production_release'] is False
    wheels={};members={}
    for tag in ['dev7','rc2']:
        meta=result['wheels'][tag];wheel=raw/Path(meta['path']).relative_to(remote);assert sha(wheel)==meta['sha256'];wheels[tag]=wheel
        with zipfile.ZipFile(wheel) as z:
            members[tag]={n:z.read(n) for n in z.namelist() if n.startswith('kernel_agent/')}
            assert not any(n.endswith(('qualified_releases.json','gemm_qualification.json')) for n in z.namelist())
            for name,data in members[tag].items():assert (raw/('prefix-'+tag)/name).read_bytes()==data
            entry=next(n for n in z.namelist() if n.endswith('entry_points.txt'));assert 'kernel-agent-sm89 = kernel_agent.sm89.cli:main' in z.read(entry).decode()
    assert set(members['dev7'])==set(members['rc2'])
    differences=[n for n in members['dev7'] if members['dev7'][n]!=members['rc2'][n]];assert differences==['kernel_agent/__init__.py']
    digests=set(json.loads(members['rc2']['kernel_agent/sm89/data/releases.json'])['versions'][k]['binary_sha256'] for k in ['sm89-cublas-current','sm89-cublas-baseline'])
    expected={'4572c346c20a70630a2521eac8c4a83163f26d931a7d07bbcd549e28220a43de','72d466d30e6b777b06d8e4216bdd3f4aef1306127926ac6f0976923d3406009d'};assert digests==expected
    for digest in digests:assert hashlib.sha256(members['rc2']['kernel_agent/sm89/data/artifacts/'+digest+'.so']).hexdigest()==digest
    outputs=0;attempts=0;boundary_records=0
    for check in validation['checks']:
        job=raw/Path(check['run_directory']).relative_to(remote);is_direct=check['label']=='boundary'
        if is_direct:records=[json.loads((job/'record.json').read_text())];attempt_jobs=[job];actual=np.load(job/'result.npy',allow_pickle=False)
        else:
            execution=json.loads((job/'execution.json').read_text());assert execution['status']=='PASS' and execution['fallback_used']==check['fallback'] and execution['production_release'] is False;records=execution['attempts'];attempt_jobs=[job/('attempt-'+str(i)) for i in range(len(records))];actual=np.load(work/(check['label']+'.npy'),allow_pickle=False)
        assert len(records)==check['attempts'];attempts+=len(records)
        for record,path in zip(records,attempt_jobs):
            assert record['cleanup']['removed'] and not record['cleanup']['remaining_owned_pids'] and record['cleanup']['exposed_files_unchanged'] and record['protected_files_unchanged']
            assert all(sha(path/n)==h for n,h in record['protected_hashes'].items())
            for stream in ['stdout','stderr']:assert (path/'capture'/(stream+'.bin')).stat().st_size==record['stored_bytes'][stream]<=record['limits'][stream]
            command=record['command'];assert command[:2]==['/usr/bin/env','-i']
            for flag in ['--reuid=65534','--regid=65534','--clear-groups','--no-new-privs','--bounding-set=-all','--inh-caps=-all','--ambient-caps=-all','-I','-B']:assert command.count(flag)==1
            for name,target in [('pipe_worker.py','kernel_agent/sm89/pipe_worker.py'),('restrictions.py','kernel_agent/sm89/restrictions.py'),('protocol.py','kernel_agent/protocol.py')]:assert (path/'artifact'/name).read_bytes()==members['rc2'][target]
            if record['status']=='REJECT':
                assert record['case'] in ['crash','timeout']
                if record['case']=='timeout':assert record['reason']=='timeout' and 'NKA_SM89_FINITE_CUDA_READY' in (path/'capture/stderr.bin').read_text()
                continue
            assert record['status']=='OUTPUT_VALID' and record['exit']==0 and record['reason'] is None
            a=np.load(path/'inputs/a.npy',allow_pickle=False);b=np.load(path/'inputs/b.npy',allow_pickle=False);meta,payload=decode_frame(path/'capture/stdout.bin',a.shape[0]*b.shape[1]*4+16384);validate_boundary(meta);assert meta==record['metadata'] and meta['binary_sha256'] in expected
            value=np.load(io.BytesIO(payload),allow_pickle=False);assert payload==(path/'result.npy').read_bytes();reference=a.astype(np.float64)@b.astype(np.float64);assert value.dtype==np.float32 and np.all(np.abs(value.astype(np.float64)-reference)<=3e-5+1e-4*np.abs(reference));assert np.array_equal(actual,value)
            boundary_records+=1
        outputs+=1
    assert outputs==validation['gpu_outputs']==9 and attempts==validation['gpu_attempts']==11 and boundary_records==9
    for case in ['stdout_flood','stderr_flood']:
        job=work/case;record=json.loads((job/'record.json').read_text());assert record['status']=='REJECT' and record['reason']==case.split('_')[0]+'_limit' and record['cleanup']['removed']
    observed=[json.loads((work/(label+'.stdout')).read_text())['version'] for label in ['installed-version','package-rollback-version','package-restore-version']];assert observed==['0.1.0rc2','0.1.0.dev7','0.1.0rc2']
    assert 'Ran 27 tests' in (work/'installed-cpu-tests.stderr').read_text() and (work/'installed-cpu-tests.stderr').read_text().rstrip().endswith('OK')
    commands={r['label']:r for r in validation['commands']}
    for name in ['qualified-rejected','invalid-input-rejected','wrong-runtime-rejected']:assert commands[name]['exit']==1
    assert commands['source-command-rejected']['exit']==2
    # Preserve the exact immutable artifacts locally; never rebuild the same version.
    for tag,wheel in wheels.items():
        target=ROOT/'artifacts/wheels'/wheel.name
        if target.exists():assert sha(target)==sha(wheel)
        else:target.write_bytes(wheel.read_bytes())
    value={'status':'AUDIT_PASS','scope':'CPU whole-file SHA/wheel/source/installed CLI/FP64 NPY/frame/real worker boundary and real version/package rollback replay','raw_files_verified':len(manifest['manifest']),'wheel_sha256':{k:sha(v) for k,v in wheels.items()},'installed_cpu_tests':27,'gpu_outputs_verified':outputs,'gpu_attempts':attempts,'successful_boundary_records':boundary_records,'host_flood_fixtures_rejected':2,'package_versions_observed':observed,'baseline_created_for_first_sm89_rollback':True,'max_abs_error':max(c['max_abs_error'] for c in validation['checks']),'target_qualified_channel_rejected':True,'arbitrary_source_command_rejected':True,'new_model_calls':0,'new_event_samples':0,'heldout_executed':False,'container_isolation_verified':False,'production_release':False};(folder/'verification.json').write_text(json.dumps(value,indent=2));return value


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('folder',type=Path);a=p.parse_args();print(json.dumps(audit(a.folder.resolve())))
