"""CPU-only installed-wheel, output, recovery and package-rollback audit."""
import argparse,hashlib,json,zipfile
from pathlib import Path
import numpy as np
from kernel_agent.protocol import decode_frame
from kernel_agent.bounded_io import confirmed_removed
ROOT=Path(__file__).resolve().parents[2]


def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def audit(folder):
    a=folder/'archive';r=a/'remote';collection=json.loads((a/'collection.json').read_text());assert sha(a/'raw.zip')==collection['archive_sha256']
    with zipfile.ZipFile(a/'raw.zip') as z:manifest=json.loads(z.read('manifest.json'))
    assert all((r/n).stat().st_size==m['bytes'] and sha(r/n)==m['sha256'] for n,m in manifest.items());install=json.loads((r/'result.json').read_text());assert install['status']=='PASS' and all(x['exit']==0 for x in install['records']);assert '--no-build-isolation' in install['records'][0]['command'] and '--no-index' in install['records'][1]['command'];wheel=next((r/'dist').glob('*.whl'));assert sha(wheel)==install['wheel_sha256']=='d5b9ae9e1141b8b91bd937807918b11b5d46e467764c031115700e85bef6f7c4';sources=json.loads((folder/'source-hashes.json').read_text())
    with zipfile.ZipFile(wheel) as z:
        assert not any(n.endswith(('qualified_releases.json','gemm_qualification.json')) for n in z.namelist())
        for n,h in sources.items():
            assert sha(r/'source'/n)==h
            if n.startswith('kernel_agent/'):assert hashlib.sha256(z.read(n)).hexdigest()==h
    work=r/'validation';v=json.loads((work/'result.json').read_text());assert v['status']=='PASS' and v['cpu_tests']==19 and v['gpu_outputs']==5 and v['gpu_attempts']==7 and v['qualified_gpu_attempts']==0 and v['production_release'] is False and not v['performance_sampling'];assert 'Ran 19 tests' in (work/'cpu-tests.stderr').read_text();assert json.loads((work/'installed-version.stdout').read_text())['version']=='0.1.0rc1' and (work/'old-version.stdout').read_text().strip()=='0.1.0.dev6';assert not (work/'not-qualified').exists() and 'qualified GEMM release is not packaged' in (work/'qualified-rejected.stderr').read_text();expected=np.load(work/'a.npy',allow_pickle=False).astype(np.float64)@np.load(work/'b.npy',allow_pickle=False).astype(np.float64);attempts=0
    for row in v['checks']:
        actual=np.load(work/(row['label']+'.npy'),allow_pickle=False);assert np.all(np.abs(actual.astype(np.float64)-expected)<=3e-5+1e-4*np.abs(expected));response=json.loads((work/(row['label']+'.stdout')).read_text());execution=r/Path(response['run_directory']).relative_to(Path(collection['remote_root']))/'execution.json';recorded=json.loads(execution.read_text());assert recorded=={k:value for k,value in response.items() if k!='output'};assert response['status']=='PASS' and response['production_release'] is False and not response['promotion_performed'] and response['binary_sha256']==row['binary_sha256'] and len(response['attempts'])==row['attempts'];assert response['fallback_used']==row['fallback'] and all(x['cleanup']['removed'] and x['protected_files_unchanged'] for x in response['attempts']);assert row['fallback']==(row['label'] in ['crash-recovery','timeout-recovery'])
        for i,record in enumerate(response['attempts']):
            attempts+=1;job=execution.parent/('attempt-'+str(i));command=record['command'];assert command[command.index('--user')+1]=='65534:65534' and all(x in command for x in ['--network=none','--read-only','--cap-drop=ALL','--security-opt=no-new-privileges','--pids-limit=32']);assert command.count('--mount')==2 and all(command[k+1].endswith(',readonly') for k,x in enumerate(command) if x=='--mount');assert confirmed_removed(record['cleanup']['inspect_after_cleanup']) and all(sha(job/n)==h for n,h in record['protected_sha256'].items());assert sha(job/'artifact/nka_gemm_v1.so')==record['binary_sha256']
            if record['status']=='OUTPUT_VALID':
                meta,payload=decode_frame(job/'capture/stdout.bin',17*19*4+16384);assert meta==record['metadata'] and meta['uid']==65534 and meta['case']=='cublas' and payload==(job/'result.npy').read_bytes();value=np.load(job/'result.npy',allow_pickle=False);assert np.array_equal(value,actual)
            else:
                assert row['fallback'] and i==0 and record['case'] in ['crash','timeout']
                if record['case']=='timeout':assert record['cleanup']['host_pids_absent_after_cleanup']
                else:
                    state=json.loads(record['cleanup']['inspect']['stdout']);assert state['Running'] is False and state['Pid']==0 and state['Status']=='exited'
    assert attempts==7;result={'status':'PASS','scope':'CPU archive/wheel/source/installed CLI/NPY/low-privilege lifecycle and real package rollback replay; not new GPU work','raw_files_verified':len(manifest),'wheel_sha256':sha(wheel),'installed_cpu_tests':19,'gpu_outputs_verified':5,'gpu_attempts':7,'max_abs_error':max(x['max_abs_error'] for x in v['checks']),'qualified_channel_rejected_before_gpu':True,'real_qualification_certificate_present':False,'package_rollback':'0.1.0rc1 -> 0.1.0.dev6','new_model_calls':0,'performance_sampling':False,'production_release':False};(folder/'verification.json').write_text(json.dumps(result,indent=2));return result


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('folder',type=Path);a=p.parse_args();print(json.dumps(audit(a.folder.resolve())))
