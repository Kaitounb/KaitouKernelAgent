"""One reviewed real-model CUDA source: bounded build, independent oracle, sanitizer and one AA batch."""
import argparse,hashlib,json,os,subprocess,threading,time
from pathlib import Path
import numpy as np
import torch
from kernel_agent.builder import DockerBuilder,cuda_identity
from kernel_agent.runner import IMAGE,digest
from experiments.p3_01.cases import make_cases
from experiments.p3_01.evaluate import expected_before,validate_qa,validate_aa
from experiments.p3_01.source_runner import SourceRunner


def main():
    p=argparse.ArgumentParser();p.add_argument('--project',type=Path,required=True);p.add_argument('--candidate',type=Path,required=True);p.add_argument('--deps',type=Path,required=True);p.add_argument('--work',type=Path,required=True);a=p.parse_args()
    work=a.work;work.mkdir();torch.set_num_threads(1);public=json.loads((a.project/'source-binding.json').read_text());records=[];ledger={};checks=[];monitors=[];unstable=[];sanitizers=[]
    files={name:(a.candidate/name).read_bytes() for name in ['cublas_sm86.cu','wmma_sm86.cu','gemm_contract.cuh']}
    binary,build=DockerBuilder(a.project,a.deps).build(files,work/'build',reviewed_sha256=cuda_identity(files))
    if binary is None:raise RuntimeError('real source build failed; record retained')
    runner=SourceRunner(binary,build['binary_sha256']);workloads,contract,cases=make_cases(a.project)
    def progress():
        (work/'progress.json').write_text(json.dumps({'build':build,'qa_groups':records,'checks':checks,'sanitizers':sanitizers,'unstable':unstable},indent=2)+'\n')
    for group in [w['id'] for w in workloads]+['boundaries']:
        selected=[case for case in cases if case['group']==group]
        expected=expected_before(selected)  # Trusted host FP64 reference exists BEFORE subject invocation.
        job=work/('qa-'+group);report,record=runner.run({'mode':'qa','cases':selected},job)
        (work/'qa-records').mkdir(exist_ok=True);(work/'qa-records'/(group+'.json')).write_text(json.dumps(record,indent=2)+'\n')
        rows=validate_qa(report,record,selected,expected,job,ledger);checks+=rows;records.append({'group':group,'count':len(rows),'worker_status':record['status']});progress()
    (work/'oracle-ledger.json').write_text(json.dumps(ledger,indent=2)+'\n')
    finish(work,public,build,runner,workloads,contract,checks,records,ledger,sanitizers,monitors,unstable,progress)


def finish(work,public,build,runner,workloads,contract,checks,records,ledger,sanitizers,monitors,unstable,progress):
    # Probe the already present image tool; no installation or weaker container permissions.
    probe_cmd=['docker','run','--rm','--pull=never','--network=none','--read-only','--cap-drop=ALL','--security-opt=no-new-privileges','--user','65534:65534','--env','NVIDIA_VISIBLE_DEVICES=void','--entrypoint','/bin/sh',IMAGE,'-c','test -x /usr/local/cuda/bin/compute-sanitizer']
    probe=subprocess.run(probe_cmd,capture_output=True,text=True,timeout=30)
    (work/'sanitizer-probe.json').write_text(json.dumps({'command':probe_cmd,'exit':probe.returncode,'stdout':probe.stdout,'stderr':probe.stderr},indent=2)+'\n')
    if probe.returncode==0:
        selected=[case for case in contract if case['group']=='boundaries' and case['seed']==171]
        for tool in ['memcheck','synccheck']:
            expected=expected_before(selected);job=work/('sanitizer-'+tool);report,record=runner.run({'mode':'qa','cases':selected},job,sanitizer=tool)
            row={'tool':tool,'status':'UNAVAILABLE_OR_FAILED','exit':record['exit'],'rejection':record.get('rejection')}
            if report is not None:
                try:row.update(status='PASS',checks=validate_qa(report,record,selected,expected,job,{}))
                except ValueError as error:row['error']=str(error)
            sanitizers.append(row);progress()
    else:sanitizers=[{'status':'UNAVAILABLE','reason':'fixed image has no executable compute-sanitizer; no tool installed'}]
    def sample(phase):
        cmd=['nvidia-smi','--query-gpu=timestamp,utilization.gpu,memory.used,temperature.gpu,clocks.sm,power.draw','--format=csv,noheader'];r=subprocess.run(cmd,capture_output=True,text=True,timeout=10)
        row={'phase':phase,'time':time.time(),'exit':r.returncode,'stdout':r.stdout,'stderr':r.stderr};monitors.append(row);return row
    def idle():
        consecutive=0
        for _ in range(10):
            r=sample('idle-before-aa')
            try:is_idle=r['exit']==0 and int(r['stdout'].split(',')[1].strip().split()[0])==0
            except (ValueError,IndexError):is_idle=False
            consecutive=consecutive+1 if is_idle else 0
            if consecutive==3:return True
            time.sleep(1)
        return False
    aa_reports=[];aa_status='SKIPPED_GPU_BUSY';stop=threading.Event()
    def monitor():
        while not stop.is_set():sample('aa');stop.wait(.5)
    if any(row['status'] not in ['PASS'] for row in sanitizers):
        aa_status='NOT_RUN_SANITIZER_NOT_ACCEPTED'
    elif idle():
        thread=threading.Thread(target=monitor);thread.start()
        try:
            for session in range(3):
                report,record=runner.run({'mode':'aa','session':session,'cases':contract,'workloads':workloads},work/('aa-'+str(session)))
                unknown=None
                try:bad=validate_aa(report,record,contract,workloads,ledger,session);unstable+=bad
                except (ValueError,KeyError,TypeError) as error:unknown=str(error)
                aa_reports.append({'session':session,'worker_status':record['status'],'witness_error':unknown,'report_available':report is not None})
                progress()
                if unknown:aa_status='INVALID_WITNESS';break
            else:aa_status='FAIL_STABILITY' if unstable else 'PASS'
        finally:stop.set();thread.join()
    sample('after');(work/'gpu-monitor.json').write_text(json.dumps(monitors,indent=2)+'\n')
    result={'status':'SOURCE_CORRECTNESS_PASS','candidate':public,'source_identity':build['source_identity'],'binary_sha256':build['binary_sha256'],'build_status':build['status'],
         'qa_checks':len(checks),'qa_groups':records,'sanitizers':sanitizers,'aa_status':aa_status,'aa_sessions':aa_reports,'unstable_groups':unstable,
         'ab_status':'NOT_RUN_AA_NOT_ACCEPTED' if aa_status!='PASS' else 'PENDING_DUAL_BINARY_ADAPTER','verdict':'REJECT' if aa_status!='PASS' else 'NO_PROMOTION_PENDING_AB',
         'reason':aa_status,'incumbent_after':public['parent_id'],'model_calls_this_gpu_run':0,'heldout_executed':False,'formal_performance_claim':False,'production_release':False}
    (work/'result.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps({k:result[k] for k in ['status','qa_checks','aa_status','verdict','reason']}))


if __name__=='__main__':main()
