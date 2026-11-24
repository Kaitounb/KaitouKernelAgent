"""Switch an application prefix pointer dev6 -> archived dev5 and run real cuBLAS.

This is application-package rollback, separate from the already tested GEMM artifact
pointer rollback. The operator extension becomes unavailable on the archived version.
"""
import argparse,fcntl,json,os,subprocess,sys,tempfile,time
from pathlib import Path
import numpy as np


def main():
    p=argparse.ArgumentParser();p.add_argument('--new-prefix',type=Path,required=True);p.add_argument('--old-prefix',type=Path,required=True);p.add_argument('--work',type=Path,required=True);a=p.parse_args();a.work.mkdir(exist_ok=False);empty=a.work/'empty-cwd';empty.mkdir();lock=open('/tmp/nka-reviewed-gpu-execution.lock','a');fcntl.flock(lock,fcntl.LOCK_EX);records=[];link=a.work/'current';store=a.work/'gemm-store';a0=(np.arange(17*32).reshape(17,32)%13/16).astype(np.float16);b0=(np.arange(32*19).reshape(32,19)%11/16).astype(np.float16);np.save(a.work/'a.npy',a0);np.save(a.work/'b.npy',b0);expected=a0.astype(np.float64)@b0.astype(np.float64)
    for prefix,version in [(a.new_prefix,'0.1.0.dev6'),(a.old_prefix,'0.1.0.dev5')]:
        temp=a.work/'next';temp.symlink_to(prefix,target_is_directory=True);os.replace(temp,link);env={k:v for k,v in os.environ.items() if k in ['HOME','LANG','PATH','LD_LIBRARY_PATH']};env.update(PYTHONPATH=str(link),OMP_NUM_THREADS='1');tag=version.split('.')[-1]
        def run(label,args):
            start=time.monotonic();r=subprocess.run([sys.executable,*args],cwd=empty,env=env,capture_output=True,text=True,timeout=150);(a.work/(tag+'-'+label+'.stdout')).write_text(r.stdout);(a.work/(tag+'-'+label+'.stderr')).write_text(r.stderr);records.append({'version':version,'label':label,'exit':r.returncode,'wall_seconds':time.monotonic()-start,'prefix':str(prefix)});assert r.returncode==0;return r
        r=run('version',['-c','import kernel_agent,json;print(json.dumps({"version":kernel_agent.__version__,"module":kernel_agent.__file__}))']);assert json.loads(r.stdout)['version']==version
        if not store.exists():run('init',['-m','kernel_agent.cli','init','--root',str(store)])
        run('status',['-m','kernel_agent.cli','status','--root',str(store)]);out=a.work/(tag+'-output.npy');r=run('run',['-m','kernel_agent.cli','run','--root',str(store),'--a',str(a.work/'a.npy'),'--b',str(a.work/'b.npy'),'--output',str(out)]);record=json.loads(r.stdout);actual=np.load(out,allow_pickle=False);assert record['status']=='PASS' and record['backend']=='cublas' and np.all(np.abs(actual.astype(np.float64)-expected)<=1e-4+1e-4*np.abs(expected));records[-1]['max_abs_error']=float(np.abs(actual-expected).max())
    result={'status':'PASS','kind':'actual application prefix rollback dev6 to archived dev5, each with real cuBLAS output','checks':records,'selected_prefix_after':str(link.resolve()),'new_model_calls':0,'performance_sampling':False,'production_release':False};(a.work/'result.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps({'status':'PASS','versions':['0.1.0.dev6','0.1.0.dev5'],'gpu_outputs':2,'production_release':False}))


if __name__=='__main__':main()
