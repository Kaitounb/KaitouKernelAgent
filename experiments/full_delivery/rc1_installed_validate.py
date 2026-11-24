"""Unreleased rc1 install/runtime/recovery/rollback checks, never Event sampling."""
import argparse,fcntl,json,os,subprocess,sys,time
from pathlib import Path
import numpy as np


def main():
    p=argparse.ArgumentParser();p.add_argument('--prefix',type=Path,required=True);p.add_argument('--work',type=Path,required=True);a=p.parse_args();a.work.mkdir();empty=a.work/'empty';empty.mkdir();old=Path('/tmp/nka-operators-dev6-izkosqud/prefix');current=a.work/'current';current.symlink_to(a.prefix);env=dict(os.environ);env['PYTHONPATH']=str(current);records=[];checks=[]
    def command(label,args):
        start=time.monotonic();r=subprocess.run([sys.executable,*args],cwd=empty,env=env,capture_output=True,text=True,timeout=180);(a.work/(label+'.stdout')).write_text(r.stdout);(a.work/(label+'.stderr')).write_text(r.stderr);records.append({'label':label,'exit':r.returncode,'wall_seconds':time.monotonic()-start,'arguments':args});return r
    r=command('installed-version',['-c','import kernel_agent,json;print(json.dumps({"version":kernel_agent.__version__,"path":kernel_agent.__file__}))']);assert r.returncode==0;info=json.loads(r.stdout);assert info['version']=='0.1.0rc1' and Path(info['path']).resolve().is_relative_to(a.prefix.resolve())
    r=command('cpu-tests',['-m','unittest','discover','-s',str(a.work.parent/'installed-tests'),'-v']);assert r.returncode==0 and 'Ran 19 tests' in r.stderr
    r=command('qualified-rejected',['-m','kernel_agent.cli','init','--root',str(a.work/'not-qualified'),'--channel','qualified']);assert r.returncode==1 and 'qualified GEMM release is not packaged' in r.stderr and not (a.work/'not-qualified').exists()
    store=a.work/'store';assert command('initialize',['-m','kernel_agent.cli','init','--root',str(store)]).returncode==0
    x=(np.arange(17*32).reshape(17,32)%13/16).astype(np.float16);y=(np.arange(32*19).reshape(32,19)%11/16).astype(np.float16);np.save(a.work/'a.npy',x);np.save(a.work/'b.npy',y);expected=x.astype(np.float64)@y.astype(np.float64);lock=open('/tmp/nka-reviewed-gpu-execution.lock','a');fcntl.flock(lock,fcntl.LOCK_EX)
    def run(label,fault=None):
        out=a.work/(label+'.npy');args=['-m','kernel_agent.cli','run','--root',str(store),'--a',str(a.work/'a.npy'),'--b',str(a.work/'b.npy'),'--output',str(out)]
        if fault:args+=['--inject-failure',fault]
        r=command(label,args);assert r.returncode==0;d=json.loads(r.stdout);actual=np.load(out,allow_pickle=False);assert d['status']=='PASS' and d['production_release'] is False and d['fallback_used']==bool(fault) and np.all(np.abs(actual.astype(np.float64)-expected)<=3e-5+1e-4*np.abs(expected)) and all(x['cleanup']['removed'] for x in d['attempts']);checks.append({'label':label,'max_abs_error':float(np.abs(actual.astype(np.float64)-expected).max()),'attempts':len(d['attempts']),'fallback':d['fallback_used'],'binary_sha256':d['binary_sha256']})
    run('default');run('crash-recovery','crash');run('timeout-recovery','timeout');assert command('registry-rollback',['-m','kernel_agent.cli','rollback','--root',str(store)]).returncode==0;run('after-registry-rollback');staged=a.work/'current-next';staged.symlink_to(old);os.replace(staged,current);r=command('old-version',['-c','import kernel_agent;print(kernel_agent.__version__)']);assert r.returncode==0 and r.stdout.strip()=='0.1.0.dev6';run('after-package-rollback')
    result={'status':'PASS','scope':'Unreleased rc1 installed runtime delta; real certificate absent and qualified channel rejected before GPU','new_version':'0.1.0rc1','old_version':'0.1.0.dev6','cpu_tests':19,'commands':records,'checks':checks,'gpu_outputs':len(checks),'gpu_attempts':sum(x['attempts'] for x in checks),'qualified_gpu_attempts':0,'new_model_calls':0,'performance_sampling':False,'production_release':False};(a.work/'result.json').write_text(json.dumps(result,indent=2));print(json.dumps({k:result[k] for k in ['status','gpu_outputs','gpu_attempts','qualified_gpu_attempts','production_release']}))


if __name__=='__main__':main()
