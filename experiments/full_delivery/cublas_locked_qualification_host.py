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
    torch.set_num_threads(1);default=prefix/'kernel_agent/data/artifacts'/(protocol['default_binary_sha256']+'.so');fallback=prefix/'kernel_agent/data/artifacts'/(protocol['fallback_binary_sha256']+'.so');assert sha(default)==protocol['default_binary_sha256'] and sha(fallback)==protocol['fallback_binary_sha256'];lock=open('/tmp/nka-reviewed-gpu-execution.lock','a');fcntl.flock(lock,fcntl.LOCK_EX);global _CLOCK_CONTROL;_CLOCK_CONTROL=ClockControl(root);_CLOCK_CONTROL.start();pair_sha=sha(root/'protocol.json');runner=SourceRunner(default,sha(default),incumbent=fallback,incumbent_sha256=sha(fallback),pair_sha256=pair_sha);workloads,contract,cases=make_cases(prefix/'kernel_agent/optimization/assets');ledgers={'incumbent':{},'candidate':{}};qa=[];records=[];aa=[];ab=[];samples=[];decision=None;status='FAIL';backends={'incumbent':'cublas','candidate':'cublas'}
    def run(request,path):
        started_wall=time.time();report,record=runner.run({**request,'diagnostic_backends':backends},path);records.append({'job':path.name,'phase':request['mode'],'started_wall':started_wall,'ended_wall':time.time(),'wall_seconds':record['wall_seconds'],'status':record['status']});return report,record
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


import threading,subprocess
_CLOCK_CONTROL=None

def validate_clock_evidence(samples,records):
    spans=[r for r in records if r.get('phase') in ['aa','ab']]
    selected=[]
    for sample in samples:
        if sample.get('exit')!=0:raise ValueError('telemetry command failed')
        if not any(r['started_wall']+5<=sample['wall']<=r['ended_wall']-2 for r in spans):continue
        fields=[v.strip() for v in sample['stdout'].strip().split(',')]
        if len(fields)!=6:raise ValueError('invalid telemetry fields')
        utilization,sm,memory,temperature,power=map(float,fields[1:])
        if utilization<90:continue
        selected.append((sm,memory,temperature,power))
    if len(selected)<30:raise ValueError('insufficient loaded telemetry')
    if any(not 1185<=sm<=1215 or not 6798<=memory<=6802 or temperature>=80 for sm,memory,temperature,power in selected):raise ValueError('locked environment deviated')
    return {'status':'PASS','loaded_samples':len(selected),'sm_mhz_range':[min(v[0] for v in selected),max(v[0] for v in selected)],'memory_mhz_range':[min(v[1] for v in selected),max(v[1] for v in selected)],'maximum_temperature_c':max(v[2] for v in selected),'maximum_power_w':max(v[3] for v in selected)}

class ClockControl:
    def __init__(self,root):
        self.root=root;self.records=[];self.samples=[];self.stop_event=threading.Event();self.thread=None
    def command(self,args):
        r=subprocess.run(['/mnt/c/Windows/System32/nvidia-smi.exe','-i','0',*args],capture_output=True,text=True,timeout=30)
        self.records.append({'wall':time.time(),'args':args,'exit':r.returncode,'stdout':r.stdout,'stderr':r.stderr});(self.root/'clock-control.json').write_text(json.dumps(self.records,indent=2));return r
    def start(self):
        assert self.command(['--lock-gpu-clocks=1200,1200']).returncode==0
        assert self.command(['--lock-memory-clocks=7001,7001']).returncode==0
        def monitor():
            while not self.stop_event.is_set():
                try:
                    r=subprocess.run(['/usr/lib/wsl/lib/nvidia-smi','--query-gpu=timestamp,utilization.gpu,clocks.sm,clocks.mem,temperature.gpu,power.draw','--format=csv,noheader,nounits'],capture_output=True,text=True,timeout=10)
                    self.samples.append({'wall':time.time(),'exit':r.returncode,'stdout':r.stdout,'stderr':r.stderr})
                except Exception as e:self.samples.append({'wall':time.time(),'exit':-1,'error':type(e).__name__})
                (self.root/'clock-telemetry.json').write_text(json.dumps(self.samples,indent=2));self.stop_event.wait(.5)
        self.thread=threading.Thread(target=monitor,daemon=True);self.thread.start()
    def close(self):
        self.stop_event.set()
        if self.thread is not None:self.thread.join(15)
        reset=[self.command(['--reset-gpu-clocks']),self.command(['--reset-memory-clocks'])]
        result_path=self.root/'result.json'
        result=json.loads(result_path.read_text()) if result_path.exists() else {'status':'FAIL_BEFORE_QUALIFICATION_RESULT','production_release':False}
        try:
            proof=validate_clock_evidence(self.samples,result.get('records',[]))
            if self.thread is not None and self.thread.is_alive():raise ValueError('telemetry thread not stopped')
            if any(r.returncode!=0 for r in reset):raise ValueError('clock restoration failed')
        except Exception as e:proof={'status':'FAIL','reason':str(e)}
        result['clock_environment_verification']=proof;result['clock_restoration_verified']=all(r.returncode==0 for r in reset)
        if proof['status']!='PASS':result['unqualified_original_status']=result['status'];result['status']='CLOCK_ENVIRONMENT_REJECTED'
        result_path.write_text(json.dumps(result,indent=2));(self.root/'clock-environment-verification.json').write_text(json.dumps(proof,indent=2))


if __name__=='__main__':
    try:main()
    finally:
        if _CLOCK_CONTROL is not None:_CLOCK_CONTROL.close()

