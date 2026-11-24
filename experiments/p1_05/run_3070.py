"""Fresh frozen V0 final retest; no model, profiler, new kernel, or heldout execution."""
import argparse,base64,json,os,re,subprocess
from pathlib import Path
from evaluators.gemm_decision import ROOT,identity,protected_hashes,sha


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--freeze',type=Path,required=True);p.add_argument('--evidence',type=Path,required=True)
    args=p.parse_args();dest=args.evidence.resolve()
    if dest.exists():p.error('Use a new evidence directory')
    frozen=json.loads(args.freeze.read_text());protocol=json.loads((ROOT/'workloads/gemm_measurement_v1.json').read_text())
    if frozen['candidate']!=identity(4) or frozen['binary_sha256']!=protocol['initial_template']['binary_sha256']:
        raise ValueError('freeze source/artifact mismatch; do not refreeze silently')
    guard=protected_hashes();dest.mkdir(parents=True)
    (dest/'freeze.json').write_bytes(args.freeze.read_bytes())
    (dest/'bindings.json').write_text(json.dumps({'initial':identity(4),'best':frozen['candidate'],'harness':guard,'freeze_sha256':sha(args.freeze)},indent=2)+'\n')
    files={rel:base64.b64encode((ROOT/rel).read_bytes()).decode() for rel in [*guard,'experiments/p1_05/expert_worker.py']}
    upstream=ROOT/'provenance/upstream-reviews/20261121-p1-02/ScalingIntelligence__KernelBench/src/kernelbench'
    for path in upstream.glob('*.py'):files['upstream/kernelbench/'+path.name]=base64.b64encode(path.read_bytes()).decode()
    for rel,data in files.items():
        path=dest/'sources'/rel;path.parent.mkdir(parents=True,exist_ok=True);path.write_bytes(base64.b64decode(data))
    (dest/'sources/run_3070.py').write_bytes(Path(__file__).read_bytes())
    payload=r'''
import base64,hashlib,json,os,pathlib,signal,subprocess,sys,tempfile,threading,time
work=pathlib.Path(tempfile.mkdtemp(prefix='nka-p1-05-final-'))
for name,data in FILES.items():
 p=work/name;p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes(base64.b64decode(data))
env={k:v for k,v in os.environ.items() if k in ['HOME','LANG','PATH','LD_LIBRARY_PATH']}
env.update(PYTHONPATH=str(work/'upstream')+':/tmp/nka-p1-02-pydeps-hw9k37vu',PYTHONUNBUFFERED='1',CUDA_HOME='/usr/local/cuda-13.0',TORCH_CUDA_ARCH_LIST='8.6',MAX_JOBS='1')
binary=pathlib.Path('/tmp/nka-p1-02-contract-eeyx9qg9/extensions/nka_gemm_v1/nka_gemm_v1.so')
expected='93fb2f4344521b7e9b246b54d1c30d4b1a8facf731e6385e18f6119e9c49c72c'
assert hashlib.sha256(binary.read_bytes()).hexdigest()==expected
records=[];samples=[];phase='preflight';stop=threading.Event()
def sample():
 p=subprocess.run(['nvidia-smi','--query-gpu=timestamp,utilization.gpu,memory.used,temperature.gpu,clocks.sm,power.draw','--format=csv,noheader'],capture_output=True,text=True,timeout=10)
 row={'time':time.time(),'phase':phase,'exit':p.returncode,'stdout':p.stdout,'stderr':p.stderr};samples.append(row);return row
def idle_window():
 consecutive=0
 for _ in range(10):
  row=sample()
  try:idle=row['exit']==0 and int(row['stdout'].split(',')[1].strip().split()[0])==0
  except (ValueError,IndexError):idle=False
  consecutive=consecutive+1 if idle else 0
  if consecutive==3:return True
  time.sleep(1)
 return False
def monitor():
 while not stop.is_set():sample();stop.wait(.5)
def run(cmd,tag):
 start=time.monotonic();timed_out=False
 with (work/(tag+'.stdout.log')).open('w') as out,(work/(tag+'.stderr.log')).open('w') as err:
  p=subprocess.Popen(cmd,cwd=work,env=env,stdout=out,stderr=err,start_new_session=True)
  try:p.wait(timeout=360)
  except subprocess.TimeoutExpired:
   timed_out=True;os.killpg(p.pid,signal.SIGTERM)
   try:p.wait(timeout=5)
   except subprocess.TimeoutExpired:os.killpg(p.pid,signal.SIGKILL);p.wait(timeout=5)
 records.append({'tag':tag,'command':cmd,'exit':p.returncode,'timed_out':timed_out,'wall_seconds':time.monotonic()-start})
 (work/'execution.partial.json').write_text(json.dumps(records,indent=2))
idles={}
try:
 for group in ['final','expert']:
  phase='idle-before-'+group;idles[group]=idle_window()
  if not idles[group]:continue
  phase=group;thread=threading.Thread(target=monitor);thread.start()
  try:
   for session in range(3):
    tag=group+'-'+str(session)
    module='evaluators.gemm_candidate' if group=='final' else 'experiments.p1_05.expert_worker'
    cmd=[sys.executable,'-m',module,'--binary',str(binary),'--sha256',expected,'--session',str(session),'--output',str(work/(tag+'.json'))]
    if group=='final':cmd+=['--warps','4']
    run(cmd,tag)
  finally:stop.set();thread.join();stop.clear()
finally:
 phase='after';sample()
 (work/'gpu-monitor.json').write_text(json.dumps(samples,indent=2))
 (work/'execution.json').write_text(json.dumps({'workdir':str(work),'commands':records,'idle_before':idles,'worker_env_keys':sorted(env),
 'scope':'fresh frozen best/initial final retest plus separate cuBLAS comparison','model_calls':0,'profile_runs':0,'heldout_executed':False},indent=2))
 print('P1_05_RESULT='+json.dumps({'artifacts':{p.name:base64.b64encode(p.read_bytes()).decode() for p in work.iterdir() if p.is_file()}}))
'''.replace('FILES',repr(files))
    line=next(x for x in (ROOT/'.private/3070-connection.txt').read_text().splitlines() if 'Windows' in x and '密码' in x)
    env=os.environ.copy();env['SSHPASS']=re.search(r'密码[：:]\s*(\S+)',line).group(1)
    for name in list(env):
        if 'API_KEY' in name.upper() or 'AUTH_TOKEN' in name.upper():env.pop(name)
    user=re.search(r'账号\s*([^，,\s]+)',line).group(1)
    command=['/opt/homebrew/bin/sshpass','-e','ssh','-p','21022','-o','StrictHostKeyChecking=yes','-o','ConnectTimeout=10',user+'@127.0.0.1','wsl -d Ubuntu -u root -- /home/dministrator/cuda-lab/.venv/bin/python -']
    result=subprocess.run(command,input=payload,env=env,capture_output=True,text=True,timeout=2400)
    (dest/'transport.log').write_text(result.stdout+result.stderr)
    envelopes=[s.split('=',1)[1] for s in result.stdout.splitlines() if s.startswith('P1_05_RESULT=')]
    if not envelopes:raise RuntimeError('No remote envelope; inspect preserved transport.log')
    record=json.loads(envelopes[-1])
    for name,data in record['artifacts'].items():
        target=(dest/'run'/name).resolve()
        if not target.is_relative_to((dest/'run').resolve()):raise ValueError('unsafe artifact path')
        target.parent.mkdir(parents=True,exist_ok=True);target.write_bytes(base64.b64decode(data))
    if protected_hashes()!=guard:raise ValueError('protected source changed while measuring')
    execution=json.loads((dest/'run/execution.json').read_text());print(json.dumps(execution,indent=2))
    if len(execution['commands'])!=6 or any(r['exit']!=0 for r in execution['commands']):raise RuntimeError('Incomplete retest; partial evidence retained')


if __name__=='__main__':main()
