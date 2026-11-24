"""Run in the authorized SM86 host; every installed CLI invocation uses an empty cwd."""
import argparse,hashlib,json,os,shutil,subprocess,sys,zipfile
from pathlib import Path


def main():
    p=argparse.ArgumentParser();p.add_argument('--project',type=Path,required=True);p.add_argument('--work',type=Path,required=True);a=p.parse_args()
    a.work.mkdir();prefix=a.work/'prefix';dist=a.work/'dist';empty=a.work/'empty';empty.mkdir();commands=[]
    env={k:v for k,v in os.environ.items() if k in ['HOME','PATH','LANG','LD_LIBRARY_PATH']};env.update(OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1')
    def call(label,cmd,*,cwd=empty,installed=False,expected=0,timeout=120):
        log=a.work/label;log.mkdir();e=env.copy()
        if installed:e['PYTHONPATH']=str(prefix)
        r=subprocess.run(cmd,cwd=cwd,env=e,capture_output=True,text=True,timeout=timeout)
        (log/'stdout.log').write_text(r.stdout);(log/'stderr.log').write_text(r.stderr)
        row={'label':label,'command':cmd,'cwd':str(cwd),'installed':installed,'exit':r.returncode,'expected_exit':expected};commands.append(row);(a.work/'commands.json').write_text(json.dumps(commands,indent=2)+'\n')
        if r.returncode!=expected:raise RuntimeError(label+' unexpected exit; logs retained')
        return r
    call('wheel',[sys.executable,'-m','pip','wheel','--no-deps','--no-build-isolation','--wheel-dir',str(dist),str(a.project)],timeout=180)
    wheel=next(dist.glob('*.whl'))
    with zipfile.ZipFile(wheel) as archive:
        names=archive.namelist()
        if any('.private' in x or '__pycache__' in x or x.endswith('.pyc') for x in names):raise ValueError('private/cache files in wheel')
        for name in ['kernel_agent/optimization/assets/kernels/gemm/wmma_sm86.cu','kernel_agent/optimization/assets/provenance/upstream-reviews/20261121-p1-03/KernelAgent/LICENSE','kernel_agent/optimization/bundle.json']:
            if name not in names:raise ValueError('required installed resource missing '+name)
    call('install',[sys.executable,'-m','pip','install','--no-index','--no-deps','--target',str(prefix),str(wheel)])
    executable=str(prefix/'bin/kernel-agent');response=a.project/'evidence/20261122-p3-01-deepseek-source/response.json'
    installed_info=call('package-info',[sys.executable,'-c',"import kernel_agent,json;from kernel_agent.optimization.resources import verify_bundle;print(json.dumps({'version':kernel_agent.__version__,'module':kernel_agent.__file__,'bundle_sha256':verify_bundle()}))"],installed=True)
    info=json.loads(installed_info.stdout)
    if info['version']!='0.1.0.dev3' or not Path(info['module']).is_relative_to(prefix):raise ValueError('repository shadowed installed package')
    strict=call('strict-reject',[executable,'optimize','--run-dir',str(a.work/'strict-run'),'--response',str(response)],installed=True)
    if json.loads(strict.stdout)['status']!='REJECT':raise ValueError('malformed hunk not rejected')
    knowncmd=[executable,'optimize','--run-dir',str(a.work/'known-run'),'--response',str(response),'--recount']
    known=json.loads(call('known-reject',knowncmd,installed=True).stdout)
    if known['reason']!='known_candidate_aa_unstable; no resampling' or known['new_external_model_calls']!=0:raise ValueError('known failure unexpectedly retried')
    before=(a.work/'known-run/trajectory.jsonl').read_bytes()
    repeated=json.loads(call('known-idempotent',knowncmd,installed=True).stdout)
    if repeated!=known or (a.work/'known-run/trajectory.jsonl').read_bytes()!=before:raise ValueError('terminal run repeated work')
    call('status',[executable,'optimize-status','--run-dir',str(a.work/'known-run')],installed=True)
    call('invalid-budget',[executable,'optimize','--run-dir',str(a.work/'invalid-budget-run'),'--response',str(response),'--max-calls','1'],installed=True,expected=1)
    # Public dependencies already installed on this host, copied read-only for the builder.
    deps=a.work/'deps';shutil.copytree('/tmp/nka-p1-02-pydeps-hw9k37vu',deps,ignore=shutil.ignore_patterns('__pycache__'));deps.chmod(0o755)
    for path in deps.rglob('*'):path.chmod(0o755 if path.is_dir() else 0o644)
    import ninja
    (deps/'bin').mkdir();shutil.copyfile(Path(ninja.BIN_DIR)/'ninja',deps/'bin/ninja');(deps/'bin/ninja').chmod(0o755)
    (deps/'readonly-guard').write_text('unchanged');(deps/'readonly-guard').chmod(0o666)
    checkcmd=[executable,'optimize','--run-dir',str(a.work/'check-run'),'--response',str(response),'--recount','--checks-only','--review-sha','2ec37a08a96b374eac5038af49e81c987ad0dd74f4822b264f5503f9a74773d2','--deps',str(deps)]
    checked=json.loads(call('installed-build-qa-sanitizer',checkcmd,installed=True,timeout=1800).stdout)
    if checked['status']!='CHECKS_COMPLETE' or checked['aa_status']!='NOT_RUN_CHECKS_ONLY' or checked['qa_checks']!=160:raise ValueError('installed checks-only pipeline not accepted')
    evaluation=json.loads((a.work/'check-run/evaluation/result.json').read_text())
    if len(evaluation['sanitizers'])!=2 or any(x['status']!='PASS' for x in evaluation['sanitizers']) or evaluation['aa_sessions']:raise ValueError('sanitizer/timing maintenance scope changed')
    # One run validates CLI integration with the unchanged cuBLAS deployment and rollback manifests.
    store=a.work/'runtime-store';call('runtime-init',[executable,'init','--root',str(store)],installed=True)
    import numpy as np
    generator=np.random.default_rng(771);x=(generator.standard_normal((17,23))*.25).astype(np.float16);y=(generator.standard_normal((23,19))*.25).astype(np.float16)
    np.save(a.work/'a.npy',x);np.save(a.work/'b.npy',y)
    deployment=json.loads(call('runtime-cublas',[executable,'run','--root',str(store),'--a',str(a.work/'a.npy'),'--b',str(a.work/'b.npy'),'--output',str(a.work/'out.npy')],installed=True).stdout)
    actual=np.load(a.work/'out.npy',allow_pickle=False);expected=(x.astype(np.float64)@y.astype(np.float64)).astype(np.float32)
    if not np.all(np.abs(actual-expected)<=3e-5+1e-4*np.abs(expected)):raise ValueError('installed default cuBLAS output mismatch')
    result={'status':'PASS','version':info['version'],'wheel':str(wheel),'wheel_sha256':hashlib.sha256(wheel.read_bytes()).hexdigest(),'package':info,'commands':commands,'installed_qa_outputs':160,'sanitizers':[{'tool':r['tool'],'status':r['status'],'checks':len(r['checks'])} for r in evaluation['sanitizers']],
      'candidate_binary_sha256':evaluation['binary_sha256'],'known_rejection_reused':True,'idempotent_no_retry':True,'new_model_calls':0,'aa_segments':0,'runtime_default_cublas_correct':True,'runtime_deployment':deployment,'release_activated':False,'heldout_executed':False,'performance_claim':False,'production_release':False}
    (a.work/'result.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps({k:result[k] for k in ['status','version','installed_qa_outputs','new_model_calls','aa_segments']}))


if __name__=='__main__':main()
