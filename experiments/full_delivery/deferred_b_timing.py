"""One deferred timing batch for the new B source whose prior run sampled zero segments.

Reuse the installed, frozen dev5 pair controller. The original busy rejection is
immutable; this separate record never activates a runtime release or overwrites best.
"""
import argparse,hashlib,json,subprocess,threading,time
from pathlib import Path
from kernel_agent.optimization.cases import make_cases
from kernel_agent.optimization.identity import ROOT
from kernel_agent.optimization.pair_contract import pair_id,validate_descriptor
from kernel_agent.optimization.pair_control import collect_batches
from kernel_agent.optimization.source_runner import SourceRunner
from kernel_agent.runner import digest


def main():
    p=argparse.ArgumentParser();p.add_argument('--prior',type=Path,required=True);p.add_argument('--work',type=Path,required=True);a=p.parse_args();prior=a.prior/'run/evaluation';old=json.loads((prior/'result.json').read_text())
    assert old['reason']=='gpu_busy_before_aa' and old['aa_status']=='SKIPPED_GPU_BUSY' and not old['sessions'] and old['qa_checks']==320
    assert len(old['sanitizers'])==2 and all(s['status']=='PASS' and len(s['checks'])==8 for s in old['sanitizers'])
    assert not list(prior.glob('aa-*')) and not list(prior.glob('ab-*'))
    a.work.mkdir(exist_ok=False);descriptor=json.loads((prior/'pair-descriptor.json').read_text());validate_descriptor(descriptor);assert descriptor['candidate']['id']=='0b7283a329ed98288c28eb87a323944fab34fe8fae4ff61836f2a24e7f64957b'
    binary=prior/'build/artifacts/nka_gemm_v1.so';assert digest(binary)==descriptor['binary_sha256']['candidate']=='690c6c2f5312e1074b206f3cc20fc70d20ec615ce91ff10956e850ae37bba138';baseline=Path(__import__('kernel_agent').__file__).parent/'data/artifacts'/(descriptor['binary_sha256']['incumbent']+'.so')
    ledgers=json.loads((prior/'oracle-ledgers.json').read_text());workloads,contract,_=make_cases(ROOT);runner=SourceRunner(binary,digest(binary),incumbent=baseline,incumbent_sha256=digest(baseline),pair_sha256=pair_id(descriptor));samples=[];stop=threading.Event()
    def sample(phase):
        r=subprocess.run(['/usr/lib/wsl/lib/nvidia-smi','--query-gpu=timestamp,utilization.gpu,memory.used,temperature.gpu,clocks.sm,power.draw','--format=csv,noheader'],capture_output=True,text=True,timeout=10);row={'phase':phase,'time':time.time(),'exit':r.returncode,'stdout':r.stdout,'stderr':r.stderr};samples.append(row);return row
    def idle():
        count=0
        for _ in range(10):
            r=sample('idle-before-batch')
            try:good=r['exit']==0 and int(r['stdout'].split(',')[1].strip().split()[0])==0
            except (ValueError,IndexError):good=False
            count=count+1 if good else 0
            if count==3:return True
            time.sleep(1)
        return False
    def monitor():
        while not stop.is_set():sample('deferred-batch');stop.wait(.5)
    t=threading.Thread(target=monitor);t.start()
    try:state=collect_batches(runner,a.work,contract,workloads,ledgers,descriptor,idle)
    finally:stop.set();t.join()
    (a.work/'gpu-monitor.json').write_text(json.dumps(samples,indent=2)+'\n');result={**state,'prior_qa_sha256':hashlib.sha256((prior/'result.json').read_bytes()).hexdigest(),'prior_qa_outputs':320,'prior_sanitizer_outputs':16,'prior_timing_segments':0,'candidate_id':descriptor['candidate']['id'],'binary_sha256':digest(binary),'new_model_calls':0,'frozen_best_updated':False,'runtime_activation':False,'production_release':False,'performance_claim':False};(a.work/'result.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result))


if __name__=='__main__':main()
