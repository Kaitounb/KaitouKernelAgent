"""One preregistered longer-segment final retest, preserving the failed short batch."""
import argparse,base64,json
from pathlib import Path
from evaluators.gemm_decision import ROOT,identity,sha,protected_hashes
from tools.ssh_3070 import execute_python


def main():
    p=argparse.ArgumentParser();p.add_argument('--source',type=Path,required=True);p.add_argument('--plan',type=Path,required=True);p.add_argument('--evidence',type=Path,required=True)
    a=p.parse_args();dest=a.evidence.resolve()
    if dest.exists():p.error('Use a new evidence directory')
    plan=json.loads(a.plan.read_text());binding=json.loads((a.source/'bindings.json').read_text())
    if binding['harness']!=protected_hashes() or binding['best']!=identity(4) or binding['initial']!=identity(4):raise ValueError('freeze/harness changed')
    if sha(a.source/'freeze.json')!=plan['freeze_sha256']:raise ValueError('wrong frozen state')
    for name,value in plan['source_final_sha256'].items():
        if sha(a.source/'run'/name)!=value:raise ValueError('sizing evidence changed')
    dest.mkdir(parents=True);(dest/'plan.json').write_bytes(a.plan.read_bytes());(dest/'freeze.json').write_bytes((a.source/'freeze.json').read_bytes())
    (dest/'bindings.json').write_bytes((a.source/'bindings.json').read_bytes())
    paths=[*binding['harness'],'experiments/p1_05/pairing_long.py','experiments/p1_05/long_worker.py']
    files={name:base64.b64encode((ROOT/name).read_bytes()).decode() for name in paths}
    upstream=ROOT/'provenance/upstream-reviews/20261121-p1-02/ScalingIntelligence__KernelBench/src/kernelbench'
    for path in upstream.glob('*.py'):files['upstream/kernelbench/'+path.name]=base64.b64encode(path.read_bytes()).decode()
    files['plan.json']=base64.b64encode(a.plan.read_bytes()).decode()
    for name,data in files.items():
        path=dest/'sources'/name;path.parent.mkdir(parents=True,exist_ok=True);path.write_bytes(base64.b64decode(data))
    for name in ['experiments/p1_05/run_long_3070.py','tools/ssh_3070.py']:
        path=dest/'sources'/name;path.parent.mkdir(parents=True,exist_ok=True);path.write_bytes((ROOT/name).read_bytes())
    payload=r'''
import base64,json,os,pathlib,signal,subprocess,sys,tempfile,threading,time
work=pathlib.Path(tempfile.mkdtemp(prefix='nka-p1-05-long-'))
for name,data in FILES.items():
 p=work/name;p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes(base64.b64decode(data))
env={k:v for k,v in os.environ.items() if k in ['HOME','LANG','PATH','LD_LIBRARY_PATH']}
env.update(PYTHONPATH=str(work/'upstream')+':/tmp/nka-p1-02-pydeps-hw9k37vu',PYTHONUNBUFFERED='1')
samples=[];records=[];consecutive=0
for _ in range(10):
 p=subprocess.run(['nvidia-smi','--query-gpu=timestamp,utilization.gpu,memory.used,temperature.gpu,clocks.sm,power.draw','--format=csv,noheader'],text=True,capture_output=True)
 row={'time':time.time(),'stdout':p.stdout,'stderr':p.stderr,'exit':p.returncode};samples.append(row)
 consecutive=consecutive+1 if p.returncode==0 and ' 0 %,' in p.stdout else 0
 if consecutive==3:break
 time.sleep(1)
idle=consecutive==3
if idle:
 stop=threading.Event()
 def monitor():
  while not stop.is_set():
   p=subprocess.run(['nvidia-smi','--query-gpu=timestamp,utilization.gpu,memory.used,temperature.gpu,clocks.sm,power.draw','--format=csv,noheader'],text=True,capture_output=True)
   samples.append({'time':time.time(),'stdout':p.stdout,'stderr':p.stderr,'exit':p.returncode});stop.wait(.5)
 thread=threading.Thread(target=monitor);thread.start()
 try:
  for session in range(3):
   cmd=[sys.executable,'-m','experiments.p1_05.long_worker','--binary','/tmp/nka-p1-02-contract-eeyx9qg9/extensions/nka_gemm_v1/nka_gemm_v1.so','--plan',str(work/'plan.json'),'--session',str(session),'--output',str(work/f'final-{session}.json')]
   start=time.monotonic();timed=False
   with (work/f'final-{session}.stdout.log').open('w') as out,(work/f'final-{session}.stderr.log').open('w') as err:
    proc=subprocess.Popen(cmd,cwd=work,env=env,stdout=out,stderr=err,start_new_session=True)
    try:proc.wait(timeout=360)
    except subprocess.TimeoutExpired:
     timed=True;os.killpg(proc.pid,signal.SIGTERM)
     try:proc.wait(timeout=5)
     except subprocess.TimeoutExpired:os.killpg(proc.pid,signal.SIGKILL);proc.wait(timeout=5)
   records.append({'command':cmd,'exit':proc.returncode,'timed_out':timed,'session':session,'wall_seconds':time.monotonic()-start})
   (work/'execution.partial.json').write_text(json.dumps(records,indent=2))
 finally:stop.set();thread.join()
(work/'gpu-monitor.json').write_text(json.dumps(samples,indent=2))
(work/'execution.json').write_text(json.dumps({'workdir':str(work),'commands':records,'preflight_idle':idle,'mode':'frozen-final-long-v2','new_model_calls':0,'profile_runs':0,'heldout_executed':False},indent=2))
print('P1_05_LONG='+json.dumps({'artifacts':{p.name:base64.b64encode(p.read_bytes()).decode() for p in work.iterdir() if p.is_file()}}))
'''.replace('FILES',repr(files))
    (dest/'remote-driver.py').write_text(payload)
    result=execute_python(payload)
    (dest/'transport.log').write_text(result.stdout+result.stderr)
    envelopes=[s.split('=',1)[1] for s in result.stdout.splitlines() if s.startswith('P1_05_LONG=')]
    if not envelopes:raise RuntimeError('No final envelope; inspect retained transport log')
    for name,data in json.loads(envelopes[-1])['artifacts'].items():
        target=(dest/'run'/name).resolve()
        if not target.is_relative_to((dest/'run').resolve()):raise ValueError('unsafe artifact path')
        target.parent.mkdir(parents=True,exist_ok=True);target.write_bytes(base64.b64decode(data))
    print((dest/'run/execution.json').read_text())


if __name__=='__main__':main()
