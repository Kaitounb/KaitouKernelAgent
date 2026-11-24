"""Exercise the installed wheel from an empty cwd; independent references stay outside workers."""
import argparse,hashlib,json,os,subprocess,sys
from pathlib import Path
import numpy as np
import torch


def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    p=argparse.ArgumentParser();p.add_argument('--prefix',type=Path,required=True);p.add_argument('--work',type=Path,required=True);p.add_argument('--development',type=Path,required=True);a=p.parse_args()
    work=a.work;work.mkdir();cwd=work/'empty-cwd';cwd.mkdir();root=work/'deployment';records=[];failures=[];torch.set_num_threads(1)
    env={k:v for k,v in os.environ.items() if k in ['HOME','PATH','LANG','LD_LIBRARY_PATH']};env.update(PYTHONPATH=str(a.prefix),OPENBLAS_NUM_THREADS='1')
    def command(args,tag):
        cmd=[sys.executable,'-m','kernel_agent.cli',*args];p=subprocess.run(cmd,cwd=cwd,env=env,capture_output=True,text=True,timeout=120)
        (work/(tag+'.stdout.log')).write_text(p.stdout);(work/(tag+'.stderr.log')).write_text(p.stderr)
        rec={'tag':tag,'command':cmd,'exit':p.returncode}
        if p.returncode==0:rec['response']=json.loads(p.stdout)
        else:rec['error']=p.stderr[:2048]
        records.append(rec);(work/'commands.partial.json').write_text(json.dumps(records,indent=2)+'\n');return rec
    def invoke(command_name,tag,*extra):return command([command_name,'--root',str(root),*extra],tag)
    if invoke('init','init')['exit']:raise RuntimeError('installed init failed')
    pointer_before=sha(root/'current.json')
    original='sm86-cublas-original';container='sm86-cublas-container';active=root/'versions'/container;active_binary=active/'nka_gemm_v1.so';manifest=active/'manifest.json'
    artifact_data=active_binary.read_bytes();manifest_data=manifest.read_bytes();fallback_data=(root/'fallback.so').read_bytes()
    checked=[]
    def run_case(tag,shape,expected_version=container,fallback=False,extra=()):
        m,n,k=shape;g=torch.Generator().manual_seed(171)
        x=torch.randn(m,k,generator=g).mul_(.25).half();y=torch.randn(k,n,generator=g).mul_(.25).half();reference=(x.double()@y.double()).float().numpy()
        folder=work/tag;folder.mkdir();np.save(folder/'a.npy',x.numpy(),allow_pickle=False);np.save(folder/'b.npy',y.numpy(),allow_pickle=False)
        output=folder/'result.npy'
        row=invoke('run',tag,'--a',str(folder/'a.npy'),'--b',str(folder/'b.npy'),'--output',str(output),*extra)
        ok=row['exit']==0 and output.exists()
        if ok:
            result=np.load(output,allow_pickle=False);delta=np.abs(result.astype(np.float64)-reference.astype(np.float64))
            ok=result.shape==(m,n) and result.dtype==np.float32 and bool(np.isfinite(result).all()) and bool(np.all(delta<=3e-5+1e-4*np.abs(reference)))
            row['max_abs_error']=float(delta.max());r=row['response'];ok=ok and r['executed_version']==expected_version and r['fallback_used']==fallback and not r['promotion_performed'] and not r['production_release']
        row['independent_correctness']=bool(ok);checked.append(tag)
        if not ok:failures.append(tag)
        return row
    development=json.loads(a.development.read_text())['workloads']
    for row in development:run_case('normal-'+row['id'],row['shape_mnk'])
    shape=(17,19,23)
    try:
        active_binary.rename(active/'binary.disabled');run_case('missing-active',shape,original,True)
    finally:
        if (active/'binary.disabled').exists():(active/'binary.disabled').rename(active_binary)
    try:
        active_binary.chmod(0o644);active_binary.write_bytes(b'corrupt');run_case('corrupt-active',shape,original,True)
    finally:active_binary.write_bytes(artifact_data);active_binary.chmod(0o444)
    try:
        changed=json.loads(manifest_data);changed['torch']='incompatible';manifest.chmod(0o644);manifest.write_text(json.dumps(changed));run_case('incompatible-manifest',shape,original,True)
    finally:manifest.write_bytes(manifest_data);manifest.chmod(0o444)
    for fault in ['crash','timeout']:run_case('execution-'+fault,shape,original,True,('--inject-failure',fault))
    if sha(root/'current.json')!=pointer_before:failures.append('fallback_modified_active_pointer')
    if invoke('activate','activate-original','--version',original)['exit']:failures.append('activate-original')
    run_case('activated-original',shape,original,False)
    if invoke('activate','activate-container','--version',container)['exit']:failures.append('activate-container')
    if invoke('rollback','rollback-to-original')['exit']:failures.append('rollback')
    run_case('rolled-back-original',shape,original,False)
    if invoke('activate','restore-container','--version',container)['exit']:failures.append('restore-container')
    # Independent fallback corruption must fail closed; it must not start an unverified .so.
    try:
        active_binary.rename(active/'binary.disabled');fallback_path=root/'fallback.so';fallback_path.chmod(0o644);fallback_path.write_bytes(b'bad')
        folder=work/'negative-fallback';folder.mkdir();np.save(folder/'a.npy',np.ones((1,1),np.float16));np.save(folder/'b.npy',np.ones((1,1),np.float16))
        row=invoke('run','corrupt-fallback-fails-closed','--a',str(folder/'a.npy'),'--b',str(folder/'b.npy'),'--output',str(folder/'result.npy'))
        if row['exit']==0 or (folder/'result.npy').exists():failures.append('corrupt-fallback-fails-closed')
    finally:
        if (active/'binary.disabled').exists():(active/'binary.disabled').rename(active_binary)
        (root/'fallback.so').write_bytes(fallback_data);(root/'fallback.so').chmod(0o444)
    # Invalid request values must be rejected before creating a GPU job or trying fallback.
    runs_before={p.name for p in (root/'runs').iterdir()}
    for name,value in [('nonfinite',float('nan')),('out-of-domain',5.0)]:
        folder=work/('negative-'+name);folder.mkdir();np.save(folder/'a.npy',np.array([[value]],np.float16));np.save(folder/'b.npy',np.ones((1,1),np.float16))
        row=invoke('run','reject-'+name,'--a',str(folder/'a.npy'),'--b',str(folder/'b.npy'),'--output',str(folder/'result.npy'))
        if row['exit']==0 or (folder/'result.npy').exists() or {p.name for p in (root/'runs').iterdir()}!=runs_before:failures.append('reject-'+name)
    info=subprocess.run([sys.executable,'-c','import json, pathlib, kernel_agent, numpy; print(json.dumps({"module":str(pathlib.Path(kernel_agent.__file__).resolve()),"numpy":numpy.__version__}))'],cwd=cwd,env=env,capture_output=True,text=True,timeout=30)
    module=json.loads(info.stdout)
    if not Path(module['module']).is_relative_to(a.prefix):failures.append('imported-source-tree-instead-of-wheel')
    if sha(root/'current.json')!=pointer_before:failures.append('final-pointer-differs')
    if sha(active_binary)!=hashlib.sha256(artifact_data).hexdigest() or sha(manifest)!=hashlib.sha256(manifest_data).hexdigest():failures.append('artifact-not-restored')
    result={'status':'PASS' if not failures else 'FAIL','installed_module':module,'cases':records,'independent_correctness_calls':len(checked),'failures':failures,'pointer_unchanged_after_failures':sha(root/'current.json')==pointer_before,
            'development_only':True,'production_release':False,'heldout_executed':False,'model_calls':0,'performance_claim':False}
    (work/'result.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps({k:result[k] for k in ['status','independent_correctness_calls','failures','installed_module']}))
    if failures:raise SystemExit(1)


if __name__=='__main__':main()
