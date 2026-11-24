"""CPU replay of the final scoped qualified package and its installed operational checks."""
import argparse,hashlib,io,json,zipfile
from pathlib import Path
import numpy as np
from kernel_agent.protocol import decode_frame
from kernel_agent.sm89.runner import validate_boundary
ROOT=Path(__file__).resolve().parents[2]


def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def audit(folder):
    archive=folder/'segmented-archive';raw=archive/'remote';execution=json.loads((archive/'execution.json').read_text());receipt=json.loads((archive/'collection.json').read_text());assert sha(archive/'raw.zip')==receipt['archive_sha256']
    for n,m in execution['manifest'].items():assert (raw/n).stat().st_size==m['bytes'] and sha(raw/n)==m['sha256']
    result=json.loads((raw/'result.json').read_text());validation=json.loads((raw/'validation/result.json').read_text());assert result['status']==validation['status']=='PASS' and validation==result['validation'];remote=Path(result['location']['remote_root']);wheel=raw/Path(result['wheel_path']).relative_to(remote);assert sha(wheel)==result['wheel_sha256'];mapping=json.loads((raw/'rollback-reference.json').read_text())
    with zipfile.ZipFile(wheel) as z:
        members={n:z.read(n) for n in z.namelist() if n.startswith('kernel_agent/')}
        for n,data in members.items():assert (raw/'prefix'/n).read_bytes()==data
        cert_raw=members['kernel_agent/sm89/data/gemm_qualification.json'];cert=json.loads(cert_raw);catalog=json.loads(members['kernel_agent/sm89/data/qualified_releases.json']);assert catalog['qualification_sha256']==hashlib.sha256(cert_raw).hexdigest()
        assert cert['status']=='QUALIFIED' and cert['scope']=='SM89_REVIEWED_FIXED_CUBLAS' and cert['source_execution_enabled'] is False
        for n,h in cert['evidence'].items():assert sha(ROOT/n)==h
        with zipfile.ZipFile(ROOT/'artifacts/wheels/nvidia_kernel_agent-0.1.0rc2-py3-none-any.whl') as parent:
            for n,h in cert['runtime_files'].items():assert hashlib.sha256(members[n]).hexdigest()==h and members[n]==parent.read(n)
    cfg=json.loads((raw/'worker-runtime.json').read_text());assert sha(raw/'software-manifest.json')==cfg['manifest_sha256']==cert['environment']['public_runtime_manifest_sha256']
    count=0;attempts=0;qualified=0
    for check in validation['checks']:
        origin=check['run_directory'];job=raw/Path(origin).relative_to(remote) if Path(origin).is_relative_to(remote) else raw/mapping[origin]
        run=json.loads((job/'execution.json').read_text());assert run['status']=='PASS' and run['production_release'] is check['production_release'] and len(run['attempts'])==check['attempts'];actual=np.load(raw/'validation'/(check['label']+'.npy'),allow_pickle=False)
        for i,record in enumerate(run['attempts']):
            path=job/('attempt-'+str(i));assert record['cleanup']['removed'] and record['protected_files_unchanged'] and record['cleanup']['exposed_files_unchanged'] and not record['cleanup']['remaining_owned_pids'];assert all(sha(path/n)==h for n,h in record['protected_hashes'].items())
            for name,target in [('pipe_worker.py','kernel_agent/sm89/pipe_worker.py'),('restrictions.py','kernel_agent/sm89/restrictions.py'),('protocol.py','kernel_agent/protocol.py')]:assert (path/'artifact'/name).read_bytes()==members[target]
            for stream in ['stdout','stderr']:assert (path/'capture'/(stream+'.bin')).stat().st_size==record['stored_bytes'][stream]<=record['limits'][stream]
            if record['status']=='REJECT':
                assert record['case'] in ['crash','timeout']
                if record['case']=='timeout':assert record['reason']=='timeout' and 'NKA_SM89_FINITE_CUDA_READY' in (path/'capture/stderr.bin').read_text()
                continue
            assert record['status']=='OUTPUT_VALID' and record['reason'] is None and record['exit']==0
            a=np.load(path/'inputs/a.npy',allow_pickle=False);b=np.load(path/'inputs/b.npy',allow_pickle=False);meta,payload=decode_frame(path/'capture/stdout.bin',a.shape[0]*b.shape[1]*4+16384);validate_boundary(meta);assert meta==record['metadata'] and meta['binary_sha256'] in cert['binary_sha256'];value=np.load(io.BytesIO(payload),allow_pickle=False);reference=a.astype(np.float64)@b.astype(np.float64);assert payload==(path/'result.npy').read_bytes() and np.array_equal(actual,value) and np.all(np.abs(value.astype(np.float64)-reference)<=3e-5+1e-4*np.abs(reference))
        count+=1;attempts+=len(run['attempts']);qualified+=check['production_release']
    assert count==validation['gpu_outputs']==6 and attempts==validation['gpu_attempts']==8 and qualified==validation['qualified_outputs']==5
    work=raw/'validation';assert 'Ran 30 tests' in (work/'cpu-tests.stderr').read_text() and (work/'cpu-tests.stderr').read_text().rstrip().endswith('OK')
    assert (work/'version.stdout').read_text().strip()=='0.1.0' and (work/'rollback-package-version.stdout').read_text().strip()=='0.1.0rc2' and (work/'restore-package-version.stdout').read_text().strip()=='0.1.0'
    commands={c['label']:c for c in validation['commands']};assert commands['qualified-wrong-runtime']['exit']==1
    post=json.loads((raw/'postflight.json').read_text());assert post['gpu_exit']==post['process_query_exit']==0 and post['compute_processes'].strip()=='' and not post['remaining_exposed_jobs']
    target=ROOT/'artifacts/wheels'/wheel.name
    if target.exists():assert sha(target)==sha(wheel)
    else:target.write_bytes(wheel.read_bytes())
    value={'status':'AUDIT_PASS','scope':'SM89_REVIEWED_FIXED_CUBLAS','wheel_sha256':sha(wheel),'qualification_sha256':hashlib.sha256(cert_raw).hexdigest(),'raw_files_verified':len(execution['manifest']),'accepted_runtime_files_identical_to_rc2':len(cert['runtime_files']),'installed_cpu_tests':30,'gpu_outputs_verified':count,'gpu_attempts':attempts,'qualified_outputs':qualified,'package_versions':['0.1.0','0.1.0rc2','0.1.0'],'new_event_samples':0,'heldout_reexecuted':False,'new_model_calls':0,'source_execution_enabled':False,'container_isolation_verified':False,'production_release':True};(folder/'verification.json').write_text(json.dumps(value,indent=2));return value


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('folder',type=Path);a=p.parse_args();print(json.dumps(audit(a.folder.resolve())))
