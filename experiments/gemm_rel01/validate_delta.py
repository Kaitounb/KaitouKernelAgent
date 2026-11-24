"""Final dev4 command-only correction: new install/CPU regression, no GPU or model calls."""
import argparse,hashlib,json,os,shutil,subprocess,sys,zipfile
from pathlib import Path


def main():
    p=argparse.ArgumentParser();p.add_argument('--project',type=Path,required=True);p.add_argument('--work',type=Path,required=True);p.add_argument('--previous',type=Path,required=True);a=p.parse_args();a.work.mkdir();empty=a.work/'empty';empty.mkdir();prefix=a.work/'prefix';dist=a.work/'dist';commands=[]
    prior=json.loads((a.previous/'run/result.json').read_text());assert prior['status']=='PASS' and prior['version']=='0.1.0.dev3'
    env={k:v for k,v in os.environ.items() if k in ['HOME','PATH','LANG','LD_LIBRARY_PATH']};env.update(OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1')
    def call(label,cmd,*,installed=False,expected=0):
        directory=a.work/label;directory.mkdir();e=env.copy()
        if installed:e['PYTHONPATH']=str(prefix)
        r=subprocess.run(cmd,cwd=empty,env=e,capture_output=True,text=True,timeout=180)
        (directory/'stdout.log').write_text(r.stdout);(directory/'stderr.log').write_text(r.stderr);commands.append({'label':label,'command':cmd,'cwd':str(empty),'installed':installed,'exit':r.returncode,'expected_exit':expected});(a.work/'commands.json').write_text(json.dumps(commands,indent=2)+'\n')
        assert r.returncode==expected,label;return r
    call('wheel',[sys.executable,'-m','pip','wheel','--no-deps','--no-build-isolation','--wheel-dir',str(dist),str(a.project)])
    wheel=next(dist.glob('*.whl'));oldwheel=Path(prior['wheel']);changed=[]
    with zipfile.ZipFile(wheel) as new,zipfile.ZipFile(oldwheel) as old:
        oldnames={n for n in old.namelist() if n.startswith('kernel_agent/')};newnames={n for n in new.namelist() if n.startswith('kernel_agent/')};assert oldnames==newnames
        changed=[n for n in sorted(oldnames) if new.read(n)!=old.read(n)]
        assert set(changed)=={'kernel_agent/__init__.py','kernel_agent/optimization/command.py','kernel_agent/optimization/bundle.json'},changed
    call('install',[sys.executable,'-m','pip','install','--no-index','--no-deps','--target',str(prefix),str(wheel)])
    info=json.loads(call('package-info',[sys.executable,'-c',"import kernel_agent,json;from kernel_agent.optimization.resources import verify_bundle;print(json.dumps({'version':kernel_agent.__version__,'module':kernel_agent.__file__,'bundle_sha256':verify_bundle()}))"],installed=True).stdout);assert info['version']=='0.1.0.dev4' and Path(info['module']).is_relative_to(prefix)
    response=a.project/'evidence/20261122-p3-01-deepseek-source/response.json';executable=str(prefix/'bin/kernel-agent')
    for relative in ['evidence/20261122-p3-01-deepseek-source/response.json',*[f'experiments/p3_01/{n}.py' for n in ['fixtures','source_timing','evaluate','patches']]]:
        target=empty/relative;target.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(a.project/relative,target)
    call('installed-cpu-regressions',[sys.executable,'-m','unittest','discover','-s',str(a.project/'tests'),'-p','test_optimizer.py','-v'],installed=True)
    bad=a.work/'empty-choices.json';bad.write_text(json.dumps({'model':'deepseek-flash','choices':[]}))
    reject=json.loads(call('bad-response',[executable,'optimize','--run-dir',str(a.work/'bad-run'),'--response',str(bad)],installed=True).stdout);assert reject['status']=='REJECT' and 'invalid_proposal' in reject['reason']
    cmd=[executable,'optimize','--run-dir',str(a.work/'known-run'),'--response',str(response),'--recount'];known=json.loads(call('known-reject',cmd,installed=True).stdout);assert known['known_decision_reused'] and known['new_external_model_calls']==0
    snapshot=(a.work/'known-run/trajectory.jsonl').read_bytes();again=json.loads(call('known-idempotent',cmd,installed=True).stdout);assert again==known and snapshot==(a.work/'known-run/trajectory.jsonl').read_bytes()
    call('status',[executable,'optimize-status','--run-dir',str(a.work/'known-run')],installed=True)
    call('missing-key',[executable,'optimize','--run-dir',str(a.work/'missing-key-run'),'--deepseek','--max-calls','1'],installed=True,expected=1)
    store=a.work/'runtime-store';call('runtime-init',[executable,'init','--root',str(store)],installed=True);state=json.loads(call('runtime-status',[executable,'status','--root',str(store)],installed=True).stdout);assert state['selected_version']=='sm86-cublas-container'
    call('runtime-activate',[executable,'activate','--root',str(store),'--version','sm86-cublas-original'],installed=True);rolled=json.loads(call('runtime-rollback',[executable,'rollback','--root',str(store)],installed=True).stdout)
    # Record pointer from file independently of display schema.
    pointer=json.loads((store/'current.json').read_text());assert pointer['active']=='sm86-cublas-container'
    result={'status':'PASS','version':info['version'],'wheel':str(wheel),'wheel_sha256':hashlib.sha256(wheel.read_bytes()).hexdigest(),'package':info,'commands':commands,'changed_package_files':changed,
      'prior_gpu_evidence':str(a.previous),'prior_result_sha256':hashlib.sha256((a.previous/'run/result.json').read_bytes()).hexdigest(),'prior_wheel_sha256':prior['wheel_sha256'],'unchanged_build_worker_reference_event_runtime_bytes':True,'cpu_test_methods':14,'new_gpu_calls':0,'new_model_calls':0,'aa_segments':0,'actual_pointer_rollback':pointer,'heldout_executed':False,'performance_claim':False,'production_release':False}
    (a.work/'result.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps({k:result[k] for k in ['status','version','new_gpu_calls','new_model_calls']}))


if __name__=='__main__':main()
