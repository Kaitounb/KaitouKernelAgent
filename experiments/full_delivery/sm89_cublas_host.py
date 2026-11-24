"""Trusted fixed cuBLAS QA/recovery and frozen-v2 paired timing on SM89; no model source."""
import argparse,fcntl,hashlib,json,math,os,statistics,subprocess,sys,threading,time,traceback
from pathlib import Path


def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def assess_deployment(aa,ab,workloads,policy):
    from kernel_agent.optimization.identity import group_log_ratio
    ids={w['id'] for w in workloads};grid={(w,n) for w in ids for n in [1,16]}
    value={'verdict':'REJECT','aa_status':'INCOMPLETE','policies':[],'production_release':False}
    if len(aa)!=3 or {r['session'] for r in aa}!={0,1,2}:return {**value,'reason':'three_aa_required'}
    violations=[]
    for report in aa:
        assert report['mode']=='aa' and report['status']=='PASS' and len(report['rows'])==16
        assert {(r['workload'],r['input_count']) for r in report['rows']}==grid
        for row in report['rows']:
            mean,deviation=group_log_ratio(row,report['session'])
            if deviation>=policy['aa_pair_deviation_limit_exclusive'] or abs(math.exp(mean)-1)>=policy['aa_group_geomean_deviation_limit_exclusive']:
                violations.append({'session':report['session'],'workload':row['workload'],'input_count':row['input_count'],'max_pair_deviation':deviation,'geomean':math.exp(mean)})
    if violations:return {**value,'aa_status':'REJECT','reason':'aa_unstable_v2','unstable_groups':violations}
    value['aa_status']='PASS'
    if len(ab)!=3 or {r['session'] for r in ab}!={0,1,2}:return {**value,'reason':'AA_PASS_AB_NOT_RUN'}
    logs={key:[] for key in grid}
    for report in ab:
        assert report['mode']=='ab' and report['status']=='PASS' and len(report['rows'])==16
        assert {(r['workload'],r['input_count']) for r in report['rows']}==grid
        for row in report['rows']:logs[(row['workload'],row['input_count'])].append(group_log_ratio(row,report['session'])[0])
    for count in [1,16]:
        guards={w:math.exp(statistics.mean(logs[(w,count)])) for w in sorted(ids)}
        value['policies'].append({'input_count':count,'workload_speedups':guards,'pass':min(guards.values())>=policy['deployment_nonregression_guard']})
    passed=all(p['pass'] for p in value['policies'])
    return {**value,'verdict':'PASS' if passed else 'REJECT','reason':'deployment_guards_pass' if passed else 'deployment_guard_regression'}


def environment():
    import torch
    r=subprocess.run(['nvidia-smi','--query-gpu=uuid,name,driver_version','--format=csv,noheader'],capture_output=True,text=True,timeout=15)
    assert r.returncode==0
    return {'gpu_identity':r.stdout.strip(),'boot_id':Path('/proc/sys/kernel/random/boot_id').read_text().strip(),'torch':str(torch.__version__),'cuda':torch.version.cuda,'capability':list(torch.cuda.get_device_capability(0)),'python':sys.version.split()[0]}


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--phase',choices=['precheck','qualification'],required=True);a=p.parse_args();root=a.root;prefix=root/'prefix';sys.path.insert(0,str(prefix));sys.path.insert(0,str(root/'experiments/full_delivery'));sys.path.insert(0,str(root))
    from sm89_cublas_support import support_files
    if a.phase=='precheck':support_files(root,prefix)
    from qualification.source_runner import SourceRunner
    from kernel_agent.optimization.cases import make_cases,case_key
    from kernel_agent.optimization.evaluate import expected_before,validate_qa,witness
    from kernel_agent.operators.measurement import assess as research_assess
    import torch
    torch.set_num_threads(1);torch.set_num_interop_threads(1)
    assert str(torch.__version__)=='2.7.1+cu126' and torch.cuda.get_device_capability(0)==(8,9)
    env=environment();build=json.loads((root/'build-result.json').read_text());assert build['status']=='BUILT'
    binaries={r:Path(build['builds'][r]['binary']) for r in ['incumbent','candidate']}
    assert all(sha(binaries[r])==build['builds'][r]['binary_sha256'] for r in binaries)
    workloads,contract,cases=make_cases(prefix/'kernel_agent/optimization/assets');policy=json.loads((root/'workloads/gemm_deployment_stability_v2.json').read_text());assert policy['aa_pair_deviation_limit_exclusive']==.03 and policy['aa_group_geomean_deviation_limit_exclusive']==.015
    lock=open('/tmp/nka-reviewed-gpu-execution.lock','a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    telemetry=[];admission=[];records=[];stop=threading.Event()
    def smi():
        r=subprocess.run(['nvidia-smi','--query-gpu=timestamp,utilization.gpu,clocks.sm,clocks.mem,temperature.gpu,power.draw,memory.used','--format=csv,noheader,nounits'],capture_output=True,text=True,timeout=10)
        return {'wall':time.time(),'exit':r.returncode,'stdout':r.stdout,'stderr':r.stderr}
    def idle():
        count=0
        for _ in range(10):
            sample=smi();admission.append(sample);(root/(a.phase+'-admission.json')).write_text(json.dumps(admission,indent=2))
            good=sample['exit']==0 and int(sample['stdout'].split(',')[1].strip())==0
            count=count+1 if good else 0
            if count==3:return True
            time.sleep(1)
        return False
    def monitor():
        while not stop.is_set():
            try:telemetry.append(smi())
            except Exception as e:telemetry.append({'wall':time.time(),'exit':-1,'error':repr(e)})
            (root/(a.phase+'-telemetry.json')).write_text(json.dumps(telemetry,indent=2));stop.wait(.5)
    thread=threading.Thread(target=monitor,daemon=True);thread.start()
    pair_path=root/('precheck-protocol.json' if a.phase=='precheck' else 'protocol.json')
    binding=json.loads(pair_path.read_text());pair_sha=sha(pair_path)
    if a.phase=='qualification':
        assert env==binding['environment'] and sha(Path(__file__))==binding['host_sha256']
        assert all(sha(root/n)==h for n,h in binding['controlled_files'].items())
        assert sha(root/'precheck-result.json')==binding['precheck_result_sha256']
        assert sha(root/'workloads/gemm_deployment_stability_v2.json')==binding['policy_sha256']
    runner=SourceRunner(binaries['candidate'],sha(binaries['candidate']),incumbent=binaries['incumbent'],incumbent_sha256=sha(binaries['incumbent']),pair_sha256=pair_sha)
    def run(request,name,sanitizer=None):
        start=time.time();report,record=runner.run({**request,'diagnostic_backends':{'incumbent':'cublas','candidate':'cublas'}},root/name,sanitizer=sanitizer)
        records.append({'job':name,'phase':request['mode'],'started_wall':start,'ended_wall':time.time(),'wall_seconds':record['wall_seconds'],'status':record['status']})
        (root/(a.phase+'-progress.json')).write_text(json.dumps(records,indent=2));return report,record
    ledgers={'incumbent':{},'candidate':{}};qa=[];sanity=[];recovery=[];aa=[];ab=[];decision=None;status='FAIL'
    try:
        if a.phase=='precheck':
            if not idle():raise RuntimeError('GPU busy before precheck; no GPU QA started')
            for group in [w['id'] for w in workloads]+['boundaries']:
                selected=[c for c in cases if c['group']==group];expected=expected_before(selected)
                for role in ['incumbent','candidate']:
                    name='qa-'+role+'-'+group;r,record=run({'mode':'qa','role':role,'cases':selected},name);qa.extend(validate_qa(r,record,selected,expected,root/name,ledgers[role]))
                (root/'oracle-ledgers.json').write_text(json.dumps(ledgers,indent=2))
            selected=contract[:4]+[c for c in contract if c['group']=='boundaries'][:4];expected=expected_before(selected)
            for tool in ['memcheck','synccheck']:
                for role in ['incumbent','candidate']:
                    name=tool+'-'+role;r,record=run({'mode':'qa','role':role,'cases':selected},name,sanitizer=tool);rows=validate_qa(r,record,selected,expected,root/name,ledgers[role]);sanity.append({'tool':tool,'role':role,'checks':len(rows),'errors':0})
            selected=contract[:1];expected=expected_before(selected)
            for fault in ['crash','timeout']:
                name='fault-'+fault;r,record=run({'mode':'qa','role':'candidate','cases':selected,'reviewed_failure':fault},name);assert r is None and record['cleanup']['removed'] and record['status']=='REJECT'
                if fault=='timeout':assert record['reason']=='timeout'
                name='recovery-'+fault;r,record=run({'mode':'qa','role':'candidate','cases':selected},name);rows=validate_qa(r,record,selected,expected,root/name,ledgers['candidate']);recovery.append({'fault':fault,'rejected':True,'recovery_checks':len(rows),'fresh_process':True})
            (root/'oracle-ledgers.json').write_text(json.dumps(ledgers,indent=2));status='PRECHECK_PASS'
        else:
            ledgers=json.loads((root/'oracle-ledgers.json').read_text())
            for phase,collection in [('aa',aa),('ab',ab)]:
                for session in range(3):
                    if not idle():status='GPU_BUSY_UNSAMPLED_REMAINDER';break
                    name=phase+'-'+str(session);r,record=run({'mode':phase,'session':session,'cases':contract,'workloads':workloads},name)
                    assert r and r['mode']==phase and r['session']==session and r['pair_sha256']==pair_sha and r['graph_launches']==32 and r['replays']=='row_calibrated'
                    for role in ['incumbent','candidate']:
                        ledger=ledgers['candidate' if phase=='aa' else role]
                        for key in ['checks_before','checks_after']:
                            values=r[key] if phase=='aa' else r[key][role];assert len(values)==len(contract)
                            for row,case in zip(values,contract):witness(row,case,ledger)
                    rows=r['candidate_aa' if phase=='aa' else 'measurements'];assert len(rows)==16
                    for row in rows:
                        shape=next(w['shape_mnk'] for w in workloads if w['id']==row['workload']);assert row['shape_mnk']==shape and row['capture_order']==(['incumbent','candidate'] if session%2==0 else ['candidate','incumbent'])
                        pilot=row['pilot_elapsed_ms'];assert len(pilot)==3 and all(math.isfinite(x) and x>0 for x in pilot) and row['replays']==max(1,min(8192,math.ceil(4000/sorted(pilot)[1]))) and row['timed_calls_per_segment']==32*row['replays']
                        for role in ['incumbent','candidate']:
                            ledger=ledgers['candidate' if phase=='aa' else role];values=[ledger[case_key({'shape_mnk':shape,'seed':701+i,'distribution':'random','offset_guard':False})] for i in range(row['input_count'])]
                            assert row['input_hashes']==[v['input_hashes'] for v in values] and row['output_hashes'][role]==[values[i%row['input_count']]['output_sha256'] for i in range(32)]
                    collection.append({'status':r['status'],'mode':phase,'session':session,'rows':rows})
                if status=='GPU_BUSY_UNSAMPLED_REMAINDER':break
                decision=assess_deployment(aa,ab,workloads,policy)
                (root/'decision-progress.json').write_text(json.dumps(decision,indent=2))
                if phase=='aa' and decision['aa_status']!='PASS':status='AA_REJECTED';break
                if phase=='ab':status='PASS_DEPLOYMENT_STABILITY_AND_GUARDS' if decision['verdict']=='PASS' else 'AB_GUARD_REJECTED'
    except Exception:
        (root/(a.phase+'-failure.log')).write_text(traceback.format_exc());status='FAIL'
    finally:
        stop.set();thread.join(15)
    result={'status':status,'phase':a.phase,'environment':env,'protocol_sha256':pair_sha,'qa_checks':len(qa),'sanitizers':sanity,'recovery':recovery,'records':records,'aa_sessions':len(aa),'ab_sessions':len(ab),'raw_segments':sum(len(row['segments']) for rep in aa+ab for row in rep['rows']),'deployment_decision':decision,'original_research_decision':research_assess(aa,ab,workloads,json.loads((prefix/'kernel_agent/optimization/assets/workloads/gemm_development_v1.json').read_text())['anchors']) if aa else None,'execution_scope':'trusted-native-fixed-cublas-only','container_isolation_verified':False,'nc_u_profile_executed':False,'new_model_calls':0,'heldout_executed':False,'runtime_activation':False,'production_release':False}
    (root/(a.phase+'-result.json')).write_text(json.dumps(result,indent=2));print(json.dumps(result))
    if status not in ['PRECHECK_PASS','PASS_DEPLOYMENT_STABILITY_AND_GUARDS','AA_REJECTED','AB_GUARD_REJECTED','GPU_BUSY_UNSAMPLED_REMAINDER']:raise SystemExit(1)


if __name__=='__main__':main()
