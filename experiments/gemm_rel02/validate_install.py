"""Installed dual-extension fault regression and preregistered non-screening WMMA/cuBLAS diagnostic."""
import argparse,hashlib,json,os,shutil,subprocess,sys,threading,time,zipfile
from pathlib import Path


def main():
    p=argparse.ArgumentParser();p.add_argument('--project',type=Path,required=True);p.add_argument('--work',type=Path,required=True);a=p.parse_args();a.work.mkdir();empty=a.work/'empty';empty.mkdir();prefix=a.work/'prefix';dist=a.work/'dist';commands=[]
    env={k:v for k,v in os.environ.items() if k in ['HOME','PATH','LANG','LD_LIBRARY_PATH']};env.update(OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1')
    def call(label,cmd,installed=False,expected=0):
        log=a.work/label;log.mkdir();e=env.copy()
        if installed:e['PYTHONPATH']=str(prefix)
        r=subprocess.run(cmd,cwd=empty,env=e,capture_output=True,text=True,timeout=180);(log/'stdout.log').write_text(r.stdout);(log/'stderr.log').write_text(r.stderr)
        commands.append({'label':label,'command':cmd,'cwd':str(empty),'installed':installed,'exit':r.returncode,'expected_exit':expected});(a.work/'commands.json').write_text(json.dumps(commands,indent=2)+'\n')
        if r.returncode!=expected:raise RuntimeError(label+' unexpected exit; raw logs retained')
        return r
    call('wheel',[sys.executable,'-m','pip','wheel','--no-deps','--no-build-isolation','--wheel-dir',str(dist),str(a.project)])
    wheel=next(dist.glob('*.whl'))
    with zipfile.ZipFile(wheel) as archive:
        if any('.private' in n or '__pycache__' in n or n.endswith('.pyc') or 'wrong-output' in n for n in archive.namelist()):raise ValueError('unintended file in wheel')
    call('install',[sys.executable,'-m','pip','install','--no-index','--no-deps','--target',str(prefix),str(wheel)])
    info=json.loads(call('package-info',[sys.executable,'-c',"import kernel_agent,json;from kernel_agent.optimization.resources import verify_bundle;print(json.dumps({'version':kernel_agent.__version__,'module':kernel_agent.__file__,'bundle_sha256':verify_bundle()}))"],True).stdout)
    if info['version']!='0.1.0.dev5' or not Path(info['module']).is_relative_to(prefix):raise ValueError('installed package shadowed')
    for relative in ['evidence/20261122-p3-01-deepseek-source/response.json',*[f'experiments/p3_01/{n}.py' for n in ['fixtures','source_timing','evaluate','patches']]]:
        target=empty/relative;target.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(a.project/relative,target)
    for pattern in ['test_optimizer.py','test_pair_decision.py']:
        call('cpu-'+pattern.removesuffix('.py'),[sys.executable,'-m','unittest','discover','-s',str(a.project/'tests'),'-p',pattern,'-v'],True)
    executable=str(prefix/'bin/kernel-agent');response=a.project/'evidence/20261122-p3-01-deepseek-source/response.json';knowncmd=[executable,'optimize','--run-dir',str(a.work/'known-run'),'--response',str(response),'--recount']
    known=json.loads(call('known-reject',knowncmd,True).stdout);assert known['known_decision_reused'] and known['new_external_model_calls']==0
    before=(a.work/'known-run/trajectory.jsonl').read_bytes();again=json.loads(call('known-idempotent',knowncmd,True).stdout);assert known==again and before==(a.work/'known-run/trajectory.jsonl').read_bytes()
    # Direct diagnostics use the installed controller/worker, never import native code in host.
    sys.path.insert(0,str(prefix))
    import kernel_agent,torch
    from kernel_agent.optimization.source_runner import SourceRunner
    from kernel_agent.optimization.evaluate import expected_before,validate_qa
    from kernel_agent.optimization.pair_validation import witness
    from kernel_agent.optimization.cases import case_key
    from kernel_agent.optimization.identity import canonical,group_log_ratio
    assert Path(kernel_agent.__file__).is_relative_to(prefix);torch.set_num_threads(1)
    artifacts=prefix/'kernel_agent/data/artifacts';inc=artifacts/'93fb2f4344521b7e9b246b54d1c30d4b1a8facf731e6385e18f6119e9c49c72c.so';correct=artifacts/'66fcc80f9122c919f92c72e37a01794a792c91a58b71d5705b160e0bc979dfc3.so';wrong=a.project/'diagnostics/wrong-output.so'
    digest=lambda p:hashlib.sha256(p.read_bytes()).hexdigest();incsha=digest(inc)
    fixture_cases=[{'id':'d'+str(i),'group':'diagnostic','kind':'contract','shape_mnk':list(shape),'seed':171,'distribution':'random','offset_guard':offset} for i,(shape,offset) in enumerate([((32,48,64),False),((17,19,23),True),((16,32,16),False),((32,48,48),False)])]
    records=[];ledgers={};runners={};bindings={}
    for label,native in [('correct',correct),('wrong',wrong)]:
        descriptor={'kind':'FIXED_DUAL_IMPORT_DIAGNOSTIC','source':'reviewed artifacts; not an optimization candidate','candidate_sha256':digest(native),'incumbent_sha256':incsha,'bundle_sha256':info['bundle_sha256']};pid=hashlib.sha256(canonical(descriptor)).hexdigest();bindings[label]=descriptor
        runner=SourceRunner(native,digest(native),incumbent=inc,incumbent_sha256=incsha,pair_sha256=pid);runners[label]=runner;ledgers[label]={'incumbent':{},'candidate':{}}
        for role in ['incumbent','candidate']:
            job=a.work/(label+'-'+role);expected=expected_before(fixture_cases);report,record=runner.run({'mode':'qa','role':role,'cases':fixture_cases},job)
            assert report is not None and report['module_identity_distinct'] and report['function_identity_distinct'],record
            for row,case in zip(report['checks'],fixture_cases):
                ok=True;error=None
                try:validate_qa({**report,'checks':[row]},record,[case],{case['id']:expected[case['id']]},job,ledgers[label][role])
                except ValueError as e:ok=False;error=str(e)
                expected_ok=not(label=='wrong' and role=='candidate')
                records.append({'pair':label,'role':role,'case':case['id'],'correct':ok,'expected_correct':expected_ok,'error':error})
                assert ok==expected_ok,(label,role,case,error)
    (a.work/'diagnostic-bindings.json').write_text(json.dumps(bindings,indent=2)+'\n');(a.work/'qa-verdicts.json').write_text(json.dumps(records,indent=2)+'\n')
    sanitizer=[]
    for tool in ['memcheck','synccheck']:
        job=a.work/('sanitizer-'+tool);expected=expected_before(fixture_cases);report,record=runners['correct'].run({'mode':'qa','role':'candidate','cases':fixture_cases},job,sanitizer=tool)
        if report is None:raise RuntimeError('dual '+tool+' failed; record retained')
        rows=validate_qa(report,record,fixture_cases,expected,job,{});sanitizer.append({'tool':tool,'checks':len(rows),'status':'PASS'})
    # Verify artifact mismatch is rejected before even creating a GPU job.
    hash_rejects=0
    for role in ['incumbent','candidate']:
        try:SourceRunner(correct,'0'*64 if role=='candidate' else digest(correct),incumbent=inc,incumbent_sha256='0'*64 if role=='incumbent' else incsha,pair_sha256='1'*64)
        except ValueError:hash_rejects+=1
    assert hash_rejects==2
    diag_backends={'incumbent':'template','candidate':'cublas'};oracle={'incumbent':dict(ledgers['correct']['incumbent']),'candidate':{}};runner=runners['correct']
    extra=a.work/'diagnostic-contract';expected=expected_before(fixture_cases);report,record=runner.run({'mode':'qa','role':'candidate','cases':fixture_cases,'diagnostic_backends':diag_backends},extra);validate_qa(report,record,fixture_cases,expected,extra,oracle['candidate'])
    workloads=[{'id':'diagnostic-1','shape_mnk':[128,128,128]},{'id':'diagnostic-2','shape_mnk':[32,48,48]}];graph_checks=0
    for wi,w in enumerate(workloads):
        cases=[{'id':'g'+str(wi)+str(i).zfill(2),'group':w['id'],'kind':'graph','shape_mnk':w['shape_mnk'],'seed':701+i,'distribution':'random','offset_guard':False} for i in range(16)];expected=expected_before(cases)
        for role in ['incumbent','candidate']:
            job=a.work/('diagnostic-'+w['id']+'-'+role);report,record=runner.run({'mode':'qa','role':role,'cases':cases,'diagnostic_backends':diag_backends},job);graph_checks+=len(validate_qa(report,record,cases,expected,job,oracle[role]))
    (a.work/'diagnostic-oracles.json').write_text(json.dumps(oracle,indent=2)+'\n')
    monitors=[]
    def sample(phase):
        r=subprocess.run(['nvidia-smi','--query-gpu=timestamp,utilization.gpu,memory.used,temperature.gpu,clocks.sm,power.draw','--format=csv,noheader'],capture_output=True,text=True,timeout=10);row={'phase':phase,'time':time.time(),'exit':r.returncode,'stdout':r.stdout,'stderr':r.stderr};monitors.append(row);return row
    consecutive=0;idle=False
    for _ in range(10):
        row=sample('idle-before-diagnostic')
        try:ok=row['exit']==0 and int(row['stdout'].split(',')[1].strip().split()[0])==0
        except (ValueError,IndexError):ok=False
        consecutive=consecutive+1 if ok else 0
        if consecutive==3:idle=True;break
        time.sleep(1)
    diagnostic_status='SKIPPED_GPU_BUSY';raw_segments=0
    if idle:
        stop=threading.Event()
        def monitor():
            while not stop.is_set():sample('diagnostic');stop.wait(.5)
        thread=threading.Thread(target=monitor);thread.start()
        try:report,record=runner.run({'mode':'ab','session':0,'cases':fixture_cases,'workloads':workloads,'diagnostic_backends':diag_backends},a.work/'diagnostic-ab')
        finally:stop.set();thread.join()
        assert report is not None and report['backends']==diag_backends,record
        for phase in ['checks_before','checks_after']:
            for role in ['incumbent','candidate']:
                assert len(report[phase][role])==4
                for row,case in zip(report[phase][role],fixture_cases):witness(row,case,oracle[role])
        assert len(report['measurements'])==4
        for row in report['measurements']:
            shape=next(w['shape_mnk'] for w in workloads if w['id']==row['workload']);count=row['input_count'];assert row['shape_mnk']==shape and row['capture_order']==['incumbent','candidate'];group_log_ratio(row,0)
            for role in ['incumbent','candidate']:
                values=[oracle[role][case_key({'shape_mnk':shape,'seed':701+i,'distribution':'random','offset_guard':False})] for i in range(count)]
                assert row['input_hashes']==[v['input_hashes'] for v in values] and row['output_hashes'][role]==[values[i%count]['output_sha256'] for i in range(32)]
            raw_segments+=len(row['segments'])
        assert raw_segments==72;diagnostic_status='GPU_DIAGNOSTIC_PASS_NOT_SCREENING_ELIGIBLE'
    sample('after');(a.work/'gpu-monitor.json').write_text(json.dumps(monitors,indent=2)+'\n')
    result={'status':'PASS','version':info['version'],'wheel':str(wheel),'wheel_sha256':digest(wheel),'package':info,'commands':commands,'dual_import_qa_outcomes':records,'sanitizers':sanitizer,'host_artifact_hash_rejections':hash_rejects,'diagnostic_graph_qa_checks':graph_checks,'diagnostic_status':diagnostic_status,'diagnostic_raw_segments':raw_segments,
            'diagnostic_screening_eligible':False,'real_candidate_aa_sessions':0,'new_model_calls':0,'frozen_best_updated':False,'runtime_activation':False,'heldout_executed':False,'formal_performance_claim':False,'production_release':False}
    (a.work/'result.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps({k:result[k] for k in ['status','version','diagnostic_status','diagnostic_raw_segments','new_model_calls']}))


if __name__=='__main__':main()
