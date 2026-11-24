"""Final deployed-default cuBLAS qualification; unrelated WMMA rejections stay frozen."""
import argparse,fcntl,hashlib,json,math,os,shutil,statistics,sys,time,traceback
from pathlib import Path


def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def support_files(root,prefix):
    source=prefix/'kernel_agent/optimization';support=root/'qualification';support.mkdir();(support/'__init__.py').write_text('')
    for n in ['source_runner.py','source_worker.py','fixtures.py','source_packets.py','pair_worker.py','source_timing.py']:shutil.copyfile(source/n,support/n)
    shutil.copyfile(prefix/'kernel_agent/protocol.py',root/'protocol.py');original=support/'original';original.mkdir()
    for n in ['pair_worker.py','source_timing.py']:shutil.copyfile(support/n,original/n)
    worker=(support/'pair_worker.py').read_text();old="if backends not in [{'incumbent':'template','candidate':'template'},{'incumbent':'template','candidate':'cublas'}]";assert worker.count(old)==1;worker=worker.replace(old,"if backends != {'incumbent':'cublas','candidate':'cublas'}");old="report.update(session=req['session'],graph_launches=32,replays=12)";assert worker.count(old)==1;worker=worker.replace(old,"report.update(session=req['session'],graph_launches=32,replays='row_calibrated',target_segment_ms=250)");(support/'pair_worker.py').write_text(worker)
    timing=(support/'source_timing.py').read_text();old='        segments=[]\n';assert timing.count(old)==1;timing=timing.replace(old,"""        pilot_elapsed_ms=[]
        for _ in range(3):
            start.record(stream)
            for _ in range(4):graphs[capture_order[0]].replay()
            end.record(stream);end.synchronize();pilot_elapsed_ms.append(start.elapsed_time(end))
        if any(not math.isfinite(x) or x<=0 for x in pilot_elapsed_ms):raise ValueError('invalid pilot')
        replays=max(1,min(2048,math.ceil(1000/sorted(pilot_elapsed_ms)[1])))
        segments=[]
""");assert timing.count('for _ in range(12)')==1;timing=timing.replace('for _ in range(12)','for _ in range(replays)');assert timing.count("'witnesses_ready':True}")==1;timing=timing.replace("'witnesses_ready':True}","'witnesses_ready':True,'replays':replays,'timed_calls_per_segment':32*replays,'pilot_elapsed_ms':pilot_elapsed_ms,'pilot_replays':4,'target_segment_ms':250}");(support/'source_timing.py').write_text(timing);return support


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);a=p.parse_args();root=a.root;protocol=json.loads((root/'protocol.json').read_text());prefix=Path(protocol['installed_prefix']);assert sha(Path(__file__))==protocol['evaluator_sha256'] and all(sha(prefix/n)==h for n,h in protocol['installed_controls'].items());sys.path.insert(0,str(prefix));sys.path.insert(0,str(root));support_files(root,prefix)
    import torch
    from qualification.source_runner import SourceRunner
    from kernel_agent.optimization.cases import make_cases,case_key
    from kernel_agent.optimization.evaluate import expected_before,validate_qa,witness
    from kernel_agent.optimization.fixtures import make_inputs
    from kernel_agent.operators.measurement import assess
    import numpy as np
    torch.set_num_threads(1);default=prefix/'kernel_agent/data/artifacts'/(protocol['default_binary_sha256']+'.so');fallback=prefix/'kernel_agent/data/artifacts'/(protocol['fallback_binary_sha256']+'.so');assert sha(default)==protocol['default_binary_sha256'] and sha(fallback)==protocol['fallback_binary_sha256'];lock=open('/tmp/nka-reviewed-gpu-execution.lock','a');fcntl.flock(lock,fcntl.LOCK_EX);pair_sha=sha(root/'protocol.json');runner=SourceRunner(default,sha(default),incumbent=fallback,incumbent_sha256=sha(fallback),pair_sha256=pair_sha);workloads,contract,cases=make_cases(prefix/'kernel_agent/optimization/assets');ledgers={'incumbent':{},'candidate':{}};qa=[];records=[];aa=[];ab=[];samples=[];decision=None;status='FAIL';backends={'incumbent':'cublas','candidate':'cublas'}
    def run(request,path):
        report,record=runner.run({**request,'diagnostic_backends':backends},path);records.append({'job':path.name,'wall_seconds':record['wall_seconds'],'status':record['status']});return report,record
    def idle():
        import subprocess
        consecutive=0
        for _ in range(10):
            r=subprocess.run(['/usr/lib/wsl/lib/nvidia-smi','--query-gpu=timestamp,utilization.gpu,memory.used,temperature.gpu,clocks.sm,power.draw','--format=csv,noheader'],capture_output=True,text=True,timeout=10);samples.append({'time':time.time(),'stdout':r.stdout,'stderr':r.stderr,'exit':r.returncode})
            try:good=r.returncode==0 and int(r.stdout.split(',')[1].strip().split()[0])==0
            except (ValueError,IndexError):good=False
            consecutive=consecutive+1 if good else 0
            if consecutive==3:return True
            time.sleep(1)
        return False
    def validate(report,mode,session):
        assert report and report['mode']==mode and report['session']==session and report['backends']==backends and report['graph_launches']==32 and report['replays']=='row_calibrated' and report['pair_sha256']==pair_sha
        for role in ['incumbent','candidate']:
            before=report['checks_before'] if mode=='aa' else report['checks_before'][role];after=report['checks_after'] if mode=='aa' else report['checks_after'][role];ledger=ledgers['candidate' if mode=='aa' else role]
            assert len(before)==len(after)==len(contract)
            for values in [before,after]:
                for value,case in zip(values,contract):witness(value,case,ledger)
        rows=report['candidate_aa' if mode=='aa' else 'measurements'];assert len(rows)==16 and {(x['workload'],x['input_count']) for x in rows}=={(w['id'],c) for w in workloads for c in [1,16]}
        for row in rows:
            shape=next(w['shape_mnk'] for w in workloads if w['id']==row['workload']);assert row['shape_mnk']==shape and row['capture_order']==(['incumbent','candidate'] if session%2==0 else ['candidate','incumbent']);pilot=row['pilot_elapsed_ms'];assert len(pilot)==3 and all(math.isfinite(x) and x>0 for x in pilot) and row['replays']==max(1,min(2048,math.ceil(1000/sorted(pilot)[1]))) and row['timed_calls_per_segment']==32*row['replays']
            for role in ['incumbent','candidate']:
                ledger=ledgers['candidate' if mode=='aa' else role];expected=[ledger[case_key({'shape_mnk':shape,'seed':701+i,'distribution':'random','offset_guard':False})] for i in range(row['input_count'])];assert row['input_hashes']==[x['input_hashes'] for x in expected] and row['output_hashes'][role]==[expected[i%row['input_count']]['output_sha256'] for i in range(32)]
        return {'status':report['status'],'mode':mode,'session':session,'rows':rows}
    try:
        for group in [w['id'] for w in workloads]+['boundaries']:
            selected=[c for c in cases if c['group']==group];expected=expected_before(selected)
            for role in ['incumbent','candidate']:
                job=root/('qa-'+role+'-'+group);r,record=run({'mode':'qa','role':role,'cases':selected},job);qa.extend(validate_qa(r,record,selected,expected,job,ledgers[role]))
            (root/'oracle-ledgers.json').write_text(json.dumps(ledgers,indent=2))
        for phase,collection in [('aa',aa),('ab',ab)]:
            for session in range(3):
                if not idle():status='GPU_BUSY_UNSAMPLED_REMAINDER';break
                r,record=run({'mode':phase,'session':session,'cases':contract,'workloads':workloads},root/(phase+'-'+str(session)));collection.append(validate(r,phase,session))
            if status=='GPU_BUSY_UNSAMPLED_REMAINDER':break
            decision=assess(aa,ab,workloads,json.loads((prefix/'kernel_agent/optimization/assets/workloads/gemm_development_v1.json').read_text())['anchors'])
            if phase=='aa' and decision['reason']!='AA_PASS_AB_NOT_RUN':status='AA_REJECTED';break
            if phase=='ab':status='PASS_BASELINE_LATENCY_GUARDS' if decision.get('aa_status')=='PASS' and len(decision['policies'])==2 and all(min(p['workload_speedups'].values())>=.97 for p in decision['policies']) else 'FAIL_BASELINE_LATENCY_GUARDS'
    except Exception:
        (root/'failure.log').write_text(traceback.format_exc());status='FAIL'
    result={'status':status,'scope':'Existing default/fallback cuBLAS deployment qualification, not WMMA search resampling or optimization promotion','protocol_sha256':pair_sha,'backend':'cublas','default_binary_sha256':sha(default),'fallback_binary_sha256':sha(fallback),'qa_checks':len(qa),'records':records,'optimization_screening_decision':decision,'raw_segments':sum(len(x['segments']) for r in aa+ab for x in r['rows']),'aa_sessions':len(aa),'ab_sessions':len(ab),'new_model_calls':0,'heldout_executed':False,'runtime_activation':False,'production_release':False};(root/'gpu-admission.json').write_text(json.dumps(samples,indent=2));(root/'result.json').write_text(json.dumps(result,indent=2));print(json.dumps(result))


if __name__=='__main__':main()
