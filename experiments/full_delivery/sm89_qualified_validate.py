"""Installed qualified-channel GPU/guard/fallback/rollback checks; no heldout rerun or timing."""
import argparse,fcntl,json,os,re,subprocess,sys,time
from pathlib import Path
import numpy as np


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);a=p.parse_args();root=a.root;work=root/'validation';work.mkdir(exist_ok=False);empty=work/'empty';empty.mkdir();live=work/'live';live.symlink_to(root/'prefix');env={k:v for k,v in os.environ.items() if k in ['HOME','LANG','LD_LIBRARY_PATH']};env.update(PATH='/usr/bin:/bin',PYTHONPATH=str(live),OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',PYTHONDONTWRITEBYTECODE='1',NKA_CONTROL_CANARY='qualified-host-only');records=[];checks=[]
    def command(label,args):
        start=time.monotonic();cmd=[sys.executable,'-B',*args];r=subprocess.run(cmd,cwd=empty,env=env,capture_output=True,text=True,timeout=180);(work/(label+'.stdout')).write_text(r.stdout);(work/(label+'.stderr')).write_text(r.stderr);records.append({'label':label,'exit':r.returncode,'wall_seconds':time.monotonic()-start,'command':cmd});(work/'commands.json').write_text(json.dumps(records,indent=2));return r
    r=command('version',['-c',"import kernel_agent;print(kernel_agent.__version__)"]);assert r.returncode==0 and r.stdout.strip()=='0.1.0'
    r=command('cpu-tests',['-m','unittest','discover','-s',str(root/'source/tests'),'-v']);assert r.returncode==0 and 'Ran 30 tests' in r.stderr
    store=work/'store-qualified';assert command('init',['-m','kernel_agent.sm89.cli','init','--root',str(store),'--channel','qualified']).returncode==0
    x=(np.arange(17*32).reshape(17,32)%13/16).astype(np.float16);y=(np.arange(32*19).reshape(32,19)%11/16).astype(np.float16);np.save(work/'a.npy',x);np.save(work/'b.npy',y);reference=x.astype(np.float64)@y.astype(np.float64)
    def run(label,*,fault=None,production=True,selected_store=None,runtime=None,success=True):
        out=work/(label+'.npy');args=['-m','kernel_agent.sm89.cli','run','--root',str(selected_store or store),'--channel','qualified' if production else 'development','--a',str(work/'a.npy'),'--b',str(work/'b.npy'),'--output',str(out),'--worker-runtime',str(runtime or root/'worker-runtime.json')]
        if fault:args+=['--inject-failure',fault]
        r=command(label,args)
        if not success:assert r.returncode==1 and not out.exists();return
        assert r.returncode==0;value=json.loads(r.stdout);actual=np.load(out,allow_pickle=False);assert value['production_release'] is production and value['status']=='PASS' and np.all(np.abs(actual.astype(np.float64)-reference)<=3e-5+1e-4*np.abs(reference));assert all(rec['cleanup']['removed'] for rec in value['attempts']);checks.append({'label':label,'production_release':production,'attempts':len(value['attempts']),'fallback':value['fallback_used'],'version':value['executed_version'],'binary_sha256':value['binary_sha256'],'run_directory':value['run_directory'],'max_abs_error':float(np.abs(actual.astype(np.float64)-reference).max())});return value
    lock=open('/tmp/nka-reviewed-gpu-execution.lock','a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    for _ in range(3):
        r=subprocess.run(['nvidia-smi','--query-gpu=utilization.gpu','--format=csv,noheader,nounits'],capture_output=True,text=True,timeout=15);assert r.returncode==0 and int(r.stdout.strip())==0;time.sleep(1)
    run('qualified-default');assert run('qualified-crash',fault='crash')['fallback_used'];assert run('qualified-timeout',fault='timeout')['fallback_used']
    assert command('qualified-rollback',['-m','kernel_agent.sm89.cli','rollback','--root',str(store),'--channel','qualified']).returncode==0;assert run('qualified-after-rollback')['executed_version']=='sm89-cublas-baseline'
    assert command('qualified-activate',['-m','kernel_agent.sm89.cli','activate','--root',str(store),'--channel','qualified','--version','sm89-cublas-current']).returncode==0
    cfg=json.loads((root/'worker-runtime.json').read_text());cfg['manifest_sha256']='0'*64;bad=work/'wrong-runtime.json';bad.write_text(json.dumps(cfg));before=set((store/'runs').iterdir());run('qualified-wrong-runtime',runtime=bad,success=False);assert set((store/'runs').iterdir())==before
    preview=json.loads((root/'preview-location.json').read_text());next_link=work/'next';next_link.symlink_to(preview['prefix']);os.replace(next_link,live);r=command('rollback-package-version',['-c',"import kernel_agent;print(kernel_agent.__version__)"]);assert r.returncode==0 and r.stdout.strip()=='0.1.0rc2';run('package-rollback-preview',production=False,selected_store=Path(preview['store']))
    next_link.symlink_to(root/'prefix');os.replace(next_link,live);r=command('restore-package-version',['-c',"import kernel_agent;print(kernel_agent.__version__)"]);assert r.returncode==0 and r.stdout.strip()=='0.1.0';run('qualified-after-package-restore')
    result={'status':'PASS','scope':'Qualified fixed SM89 cuBLAS runtime only; existing accepted GPU binary/worker lineage unchanged','version':'0.1.0','cpu_tests':30,'commands':records,'checks':checks,'gpu_outputs':len(checks),'gpu_attempts':sum(c['attempts'] for c in checks),'qualified_outputs':sum(c['production_release'] for c in checks),'new_event_samples':0,'heldout_executed':False,'new_model_calls':0,'container_isolation_verified':False,'qualified_scope':'SM89_REVIEWED_FIXED_CUBLAS','production_release':True};(work/'result.json').write_text(json.dumps(result,indent=2));print(json.dumps({k:result[k] for k in ['status','version','cpu_tests','gpu_outputs','gpu_attempts','qualified_outputs','qualified_scope','production_release']}))


if __name__=='__main__':main()
