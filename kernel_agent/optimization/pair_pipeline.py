"""Reviewed source build, both private oracles, dual-context sanitizer/A-A/A-B, external decision."""
import json,subprocess,threading,time
from pathlib import Path
import torch
from kernel_agent.builder import DockerBuilder,cuda_identity
from kernel_agent.runner import IMAGE,digest
from .cases import make_cases
from .evaluate import expected_before,validate_qa
from .source_runner import SourceRunner
from .pair_contract import make_descriptor,pair_id
from .pair_control import collect_batches


def run_pair(project,candidate,deps,work,public):
    project=Path(project);candidate=Path(candidate);work=Path(work);work.mkdir();torch.set_num_threads(1)
    known=json.loads((project/'known-rejections.json').read_text()).get(public['candidate']['id'])
    if known:
        result={'verdict':'REJECT','reason':'known_candidate_aa_unstable','aa_status':'NOT_RESAMPLED_KNOWN_FAILURE','ab_status':'NOT_RUN','qa_checks':0,'production_release':False}
        (work/'result.json').write_text(json.dumps(result,indent=2)+'\n');return result
    files={name:(candidate/name).read_bytes() for name in ['cublas_sm86.cu','wmma_sm86.cu','gemm_contract.cuh']}
    binary,build=DockerBuilder(project,deps).build(files,work/'build',reviewed_sha256=cuda_identity(files))
    if binary is None:raise RuntimeError('pair candidate build failed; raw record retained')
    descriptor=make_descriptor(public,build);pid=pair_id(descriptor);baseline=Path(__file__).resolve().parents[1]/'data/artifacts'/(descriptor['binary_sha256']['incumbent']+'.so')
    if digest(baseline)!=descriptor['binary_sha256']['incumbent']:raise ValueError('packaged incumbent artifact changed')
    (work/'pair-descriptor.json').write_text(json.dumps(descriptor,indent=2)+'\n')
    runner=SourceRunner(binary,build['binary_sha256'],incumbent=baseline,incumbent_sha256=descriptor['binary_sha256']['incumbent'],pair_sha256=pid)
    workloads,contract,cases=make_cases(project);ledgers={'incumbent':{},'candidate':{}};qa=[];sanitizers=[];monitors=[]
    for group in [w['id'] for w in workloads]+['boundaries']:
        selected=[c for c in cases if c['group']==group];expected=expected_before(selected)
        for role in ['incumbent','candidate']:
            job=work/('qa-'+role+'-'+group);report,record=runner.run({'mode':'qa','role':role,'cases':selected},job)
            checks=validate_qa(report,record,selected,expected,job,ledgers[role]);qa.append({'role':role,'group':group,'checks':checks})
        (work/'qa-progress.json').write_text(json.dumps(qa,indent=2)+'\n')
    (work/'oracle-ledgers.json').write_text(json.dumps(ledgers,indent=2)+'\n')
    selected=[c for c in contract if c['group']=='boundaries' and c['seed']==171]
    for tool in ['memcheck','synccheck']:
        expected=expected_before(selected);job=work/('sanitizer-'+tool);report,record=runner.run({'mode':'qa','role':'candidate','cases':selected},job,sanitizer=tool)
        entry={'tool':tool,'status':'FAILED','exit':record['exit'],'rejection':record.get('rejection')}
        if report is not None:
            checks=validate_qa(report,record,selected,expected,job,{})
            entry.update(status='PASS',checks=checks)
        sanitizers.append(entry)
        (work/'sanitizers.json').write_text(json.dumps(sanitizers,indent=2)+'\n')
        if entry['status']!='PASS':break
    def sample(phase):
        cmd=['nvidia-smi','--query-gpu=timestamp,utilization.gpu,memory.used,temperature.gpu,clocks.sm,power.draw','--format=csv,noheader'];r=subprocess.run(cmd,capture_output=True,text=True,timeout=10)
        row={'phase':phase,'time':time.time(),'exit':r.returncode,'stdout':r.stdout,'stderr':r.stderr};monitors.append(row);return row
    def idle():
        consecutive=0
        for _ in range(10):
            row=sample('idle-before-batch')
            try:ok=row['exit']==0 and int(row['stdout'].split(',')[1].strip().split()[0])==0
            except (ValueError,IndexError):ok=False
            consecutive=consecutive+1 if ok else 0
            if consecutive==3:return True
            time.sleep(1)
        return False
    state={'verdict':'REJECT','reason':'sanitizer_not_accepted','aa_status':'NOT_RUN_SANITIZER_FAILED','ab_status':'NOT_RUN'};stop=threading.Event()
    def monitor():
        while not stop.is_set():sample('pair-batch');stop.wait(.5)
    if len(sanitizers)==2 and all(s['status']=='PASS' for s in sanitizers):
        thread=threading.Thread(target=monitor);thread.start()
        try:state=collect_batches(runner,work,contract,workloads,ledgers,descriptor,idle)
        finally:stop.set();thread.join()
    sample('after');(work/'gpu-monitor.json').write_text(json.dumps(monitors,indent=2)+'\n')
    result={**state,'status':'PAIR_EVALUATION_COMPLETE','pair_sha256':pid,'candidate':public,'binary_sha256':build['binary_sha256'],'source_identity':build['source_identity'],
            'qa_checks':sum(len(r['checks']) for r in qa),'sanitizers':sanitizers,'frozen_best_updated':False,'runtime_activation':False,'model_calls_this_gpu_run':0,'heldout_executed':False,'formal_performance_claim':False,'production_release':False}
    (work/'result.json').write_text(json.dumps(result,indent=2)+'\n');return result
