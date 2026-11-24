"""Real installed CLI/registry/package rollback and fixed low-privilege GPU checks; no Event trial."""
import argparse,fcntl,json,os,subprocess,sys,time
from pathlib import Path
import numpy as np


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);a=p.parse_args();root=a.root;work=root/'installed-validation';work.mkdir(exist_ok=False);empty=work/'empty';empty.mkdir();current=root/'prefix-rc2';baseline=root/'prefix-dev7';live=work/'live';live.symlink_to(current);env={k:v for k,v in os.environ.items() if k in ['HOME','LANG','LD_LIBRARY_PATH']};env.update(PATH='/usr/local/cuda-12.6/bin:/usr/local/bin:/usr/bin:/bin',PYTHONPATH=str(live),OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',NKA_CONTROL_CANARY='host-only-install-validation');records=[];checks=[]
    def command(label,args,console=False):
        cmd=[str(live/'bin/kernel-agent-sm89'),*args] if console else [sys.executable,*args];start=time.monotonic();r=subprocess.run(cmd,cwd=empty,env=env,capture_output=True,text=True,timeout=180);(work/(label+'.stdout')).write_text(r.stdout);(work/(label+'.stderr')).write_text(r.stderr);records.append({'label':label,'exit':r.returncode,'wall_seconds':time.monotonic()-start,'command':cmd});(work/'commands.json').write_text(json.dumps(records,indent=2));return r
    version="import json,kernel_agent;print(json.dumps({'version':kernel_agent.__version__,'path':kernel_agent.__file__}))"
    r=command('installed-version',['-c',version]);assert r.returncode==0;info=json.loads(r.stdout);assert info['version']=='0.1.0rc2' and Path(info['path']).resolve().is_relative_to(current)
    r=command('installed-cpu-tests',['-m','unittest','discover','-s',str(root/'source/tests'),'-v']);assert r.returncode==0 and 'Ran 27 tests' in r.stderr
    r=command('qualified-rejected',['init','--root',str(work/'qualified-rejected'),'--channel','qualified'],True);assert r.returncode==1 and not (work/'qualified-rejected').exists()
    r=command('source-command-rejected',['optimize'],True);assert r.returncode==2
    store=work/'store';assert command('initialize',['init','--root',str(store)],True).returncode==0
    x=(np.arange(17*32).reshape(17,32)%13/16).astype(np.float16);y=(np.arange(32*19).reshape(32,19)%11/16).astype(np.float16);np.save(work/'a.npy',x);np.save(work/'b.npy',y);expected=x.astype(np.float64)@y.astype(np.float64)
    def run(label,*,fault=None,fallback=False,output_expected=True,runtime=None,input_path=None):
        output=work/(label+'.npy');args=['run','--root',str(store),'--a',str(input_path or work/'a.npy'),'--b',str(work/'b.npy'),'--output',str(output),'--worker-runtime',str(runtime or root/'worker-runtime.json')]
        if fault:args+=['--inject-failure',fault]
        r=command(label,args,True)
        if not output_expected:assert r.returncode==1 and not output.exists();return
        assert r.returncode==0;result=json.loads(r.stdout);actual=np.load(output,allow_pickle=False);assert np.all(np.abs(actual.astype(np.float64)-expected)<=3e-5+1e-4*np.abs(expected));assert result['status']=='PASS' and result['production_release'] is False and result['fallback_used']==fallback
        assert all(a['cleanup']['removed'] for a in result['attempts'])
        checks.append({'label':label,'max_abs_error':float(np.abs(actual.astype(np.float64)-expected).max()),'attempts':len(result['attempts']),'fallback':result['fallback_used'],'executed_version':result['executed_version'],'binary_sha256':result['binary_sha256'],'run_directory':result['run_directory']});return result
    lock=open('/tmp/nka-reviewed-gpu-execution.lock','a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    admission=[]
    for _ in range(3):
        r=subprocess.run(['nvidia-smi','--query-gpu=utilization.gpu','--format=csv,noheader,nounits'],capture_output=True,text=True,timeout=15);admission.append(r.stdout);assert r.returncode==0 and int(r.stdout.strip())==0;time.sleep(1)
    run('default');run('crash-recovery',fault='crash',fallback=True);run('timeout-recovery',fault='timeout',fallback=True)
    active=store/'versions/sm89-cublas-current/nka_gemm_v1.so';original=active.read_bytes();active.chmod(0o644);active.write_bytes(b'installed corrupt active fixture');run('corrupt-active-fallback',fallback=True);active.write_bytes(original);active.chmod(0o444)
    assert command('version-rollback',['rollback','--root',str(store)],True).returncode==0;value=run('after-version-rollback');assert value['executed_version']=='sm89-cublas-baseline'
    assert command('activate-current',['activate','--root',str(store),'--version','sm89-cublas-current'],True).returncode==0
    next_link=work/'next';next_link.symlink_to(baseline);os.replace(next_link,live);r=command('package-rollback-version',['-c',version]);assert r.returncode==0 and json.loads(r.stdout)['version']=='0.1.0.dev7';run('after-package-rollback')
    next_link.symlink_to(current);os.replace(next_link,live);r=command('package-restore-version',['-c',version]);assert r.returncode==0 and json.loads(r.stdout)['version']=='0.1.0rc2';run('after-package-restore')
    bad=x.copy();bad[0,0]=np.nan;np.save(work/'invalid.npy',bad);before=set((store/'runs').iterdir());run('invalid-input-rejected',input_path=work/'invalid.npy',output_expected=False);assert set((store/'runs').iterdir())==before
    cfg=json.loads((root/'worker-runtime.json').read_text());cfg['python_sha256']='0'*64;wrong=work/'wrong-runtime.json';wrong.write_text(json.dumps(cfg));run('wrong-runtime-rejected',runtime=wrong,output_expected=False)
    # Exercise the fixed runner fixtures from the installed package, never the source cwd.
    sys.path.insert(0,str(current));from kernel_agent.sm89.catalog import registry
    from kernel_agent.sm89.runner import NativeRunner
    selected=registry(store);binary,meta,_=selected.select();runner=NativeRunner(binary,meta['binary_sha256'],runtime=root/'worker-runtime.json')
    actual,record=runner.run(x,y,work/'boundary',case='boundaries');assert actual is not None and np.all(np.abs(actual.astype(np.float64)-expected)<=3e-5+1e-4*np.abs(expected));checks.append({'label':'boundary','max_abs_error':float(np.abs(actual.astype(np.float64)-expected).max()),'attempts':1,'fallback':False,'binary_sha256':meta['binary_sha256'],'run_directory':str(work/'boundary')})
    rejected=[]
    for case in ['stdout_flood','stderr_flood']:
        actual,record=runner.run(x,y,work/case,case=case);assert actual is None and record['reason']==case.split('_')[0]+'_limit' and record['cleanup']['removed'];rejected.append(case)
    run('after-flood-recovery')
    value={'status':'PASS','scope':'Installed SM89 preview, real UID/cap/NoNewPrivs/IP denial/private-read/readonly/byte-time boundaries, faults/fallback and app/package rollback; no namespace isolation claim','package':'0.1.0rc2','baseline_package':'0.1.0.dev7','baseline_created_for_first_sm89_rollback':True,'cpu_tests':27,'commands':records,'checks':checks,'gpu_outputs':len(checks),'gpu_attempts':sum(x['attempts'] for x in checks),'host_flood_fixtures_rejected':rejected,'admission':admission,'package_pointer':str(live.resolve()),'container_isolation_verified':False,'new_model_calls':0,'new_event_samples':0,'heldout_executed':False,'production_release':False};(work/'result.json').write_text(json.dumps(value,indent=2));print(json.dumps({k:value[k] for k in ['status','gpu_outputs','gpu_attempts','cpu_tests','baseline_package','package','production_release']}))


if __name__=='__main__':main()
