"""Single-round KDA task transport: trusted warp parameters, external evaluator, replayable verdict."""
import argparse
import base64
import json
import os
from pathlib import Path
import re
import subprocess
import sys

from evaluators.gemm_decision import ROOT,identity,validate_proposal,protected_hashes,decide,sha


def atomic_json(path,value):
    temp=path.with_suffix('.tmp');temp.write_text(json.dumps(value,indent=2,ensure_ascii=False)+'\n');temp.replace(path)


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--proposal',type=Path,required=True);ap.add_argument('--evidence',type=Path,required=True)
    args=ap.parse_args();dest=args.evidence.resolve()
    if dest.exists():ap.error('Use a new evidence directory')
    parent=identity(4);proposal=validate_proposal(args.proposal.read_text(),parent['id']);candidate=identity(proposal['warps'])
    guard=protected_hashes();dest.mkdir(parents=True)
    atomic_json(dest/'initial-incumbent.json',parent);atomic_json(dest/'incumbent.json',parent)
    atomic_json(dest/'proposal.json',proposal);atomic_json(dest/'bindings.json',{'parent':parent,'candidate':candidate,'harness':guard})
    files={}
    for rel in [*guard,'experiments/p1_02/timeout_recovery.py']:
        files[rel]=base64.b64encode((ROOT/rel).read_bytes()).decode()
    upstream=ROOT/'provenance/upstream-reviews/20261121-p1-02/ScalingIntelligence__KernelBench/src/kernelbench'
    for p in upstream.glob('*.py'):files['upstream/kernelbench/'+p.name]=base64.b64encode(p.read_bytes()).decode()
    for name,data in files.items():
        p=dest/'sources'/name;p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes(base64.b64decode(data))
    (dest/'sources/run_3070.py').write_bytes(Path(__file__).read_bytes())
    payload=r'''
import base64,hashlib,json,os,pathlib,signal,subprocess,sys,tempfile,threading,time
work=pathlib.Path(tempfile.mkdtemp(prefix='nka-p1-03-param-'))
for name,data in FILES.items():
 p=work/name;p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes(base64.b64decode(data))
env={k:v for k,v in os.environ.items() if k in ['HOME','LANG','PATH','LD_LIBRARY_PATH']}
env.update(CUDA_HOME='/usr/local/cuda-13.0',TORCH_CUDA_ARCH_LIST='8.6',MAX_JOBS='1',PYTHONUNBUFFERED='1',
 PYTHONPATH=str(work/'upstream')+':/tmp/nka-p1-02-pydeps-hw9k37vu')
binary=pathlib.Path('/tmp/nka-p1-02-contract-eeyx9qg9/extensions/nka_gemm_v1/nka_gemm_v1.so')
expected=json.loads((work/'workloads/gemm_measurement_v1.json').read_text())['initial_template']['binary_sha256']
assert hashlib.sha256(binary.read_bytes()).hexdigest()==expected
records=[]
def group_pids(pgid):
 found=[]
 for p in pathlib.Path('/proc').iterdir():
  if p.name.isdigit():
   try:
    if os.getpgid(int(p.name))==pgid:found.append(int(p.name))
   except ProcessLookupError:pass
 return found
def run(cmd,tag,timeout=360,wait_ready=False):
 start=time.monotonic()
 with (work/(tag+'.stdout.log')).open('w') as out,(work/(tag+'.stderr.log')).open('w') as err:
  p=subprocess.Popen(cmd,cwd=work,env=env,stdout=out,stderr=err,start_new_session=True)
  timed_out=False;ready=False
  try:
   if wait_ready:
    deadline=time.monotonic()+30
    while p.poll() is None and time.monotonic()<deadline:
     if (work/(tag+'.ready.json')).exists():ready=True;break
     time.sleep(.05)
    if not ready:raise RuntimeError('timeout fixture did not complete finite CUDA work')
   p.wait(timeout=.3 if wait_ready else timeout)
  except subprocess.TimeoutExpired:
   timed_out=True;os.killpg(p.pid,signal.SIGTERM)
   try:p.wait(timeout=5)
   except subprocess.TimeoutExpired:os.killpg(p.pid,signal.SIGKILL);p.wait(timeout=5)
  finally:
   if p.poll() is None:os.killpg(p.pid,signal.SIGKILL);p.wait(timeout=5)
 record={'tag':tag,'command':cmd,'exit':p.returncode,'timed_out':timed_out,'ready':ready,'remaining_group_pids':group_pids(p.pid),'wall_seconds':time.monotonic()-start}
 records.append(record)
 (work/'execution.partial.json').write_text(json.dumps(records,indent=2))
 return record
base=[sys.executable,'-m','evaluators.gemm_candidate','--binary',str(binary),'--sha256',expected,'--warps',str(WARPS),'--session','0']
for fault in ['constant','pollution','host-error','host-timeout']:
 tag='fault-'+fault
 run([*base,'--fault',fault,'--output',str(work/(tag+'.json'))],tag,wait_ready=fault=='host-timeout')
run([sys.executable,'-c','import json; from pathlib import Path; from experiments.p1_02.timeout_recovery import gpu_check; print(json.dumps(gpu_check(Path('+repr(str(binary))+'))))'],'recovery',45)
samples=[]
def sample():
 p=subprocess.run(['nvidia-smi','--query-gpu=timestamp,utilization.gpu,memory.used,temperature.gpu,clocks.sm,power.draw','--format=csv,noheader'],text=True,capture_output=True,timeout=10)
 return {'time':time.time(),'exit':p.returncode,'stdout':p.stdout,'stderr':p.stderr}
for _ in range(3):samples.append(sample());time.sleep(1)
idle=all(' 0 %,' in s['stdout'] and s['exit']==0 for s in samples)
if idle:
 stop=threading.Event()
 def monitor():
  while not stop.is_set():samples.append(sample());stop.wait(.5)
 t=threading.Thread(target=monitor);t.start()
 try:
  for session in range(3):
   tag=f'session-{session}';cmd=base.copy();cmd[cmd.index('--session')+1]=str(session)
   run([*cmd,'--output',str(work/(tag+'.json'))],tag)
 finally:stop.set();t.join()
(work/'gpu-monitor.json').write_text(json.dumps(samples,indent=2))
(work/'execution.json').write_text(json.dumps({'workdir':str(work),'commands':records,'preflight_idle':idle,'worker_env_keys':sorted(env),'no_model_credentials_in_worker':not any('KEY' in k or 'TOKEN' in k or 'PASS' in k for k in env)},indent=2))
print('P1_03_RESULT='+json.dumps({'workdir':str(work),'artifacts':{p.name:base64.b64encode(p.read_bytes()).decode() for p in work.iterdir() if p.is_file()}}))
'''.replace('FILES',repr(files)).replace('WARPS',str(proposal['warps']))
    # Transport only public evaluator/code + development manifest. No heldout or model key.
    line=next(x for x in (ROOT/'.private/3070-connection.txt').read_text().splitlines() if 'Windows' in x and '密码' in x)
    env=os.environ.copy();env['SSHPASS']=re.search(r'密码[：:]\s*(\S+)',line).group(1)
    # Do not pass an API key to SSH even if this CLI is launched from a model process.
    for name in list(env):
        if any(s in name.upper() for s in ['API_KEY','AUTH_TOKEN']):env.pop(name)
    user=re.search(r'账号\s*([^，,\s]+)',line).group(1)
    command=['/opt/homebrew/bin/sshpass','-e','ssh','-p','21022','-o','StrictHostKeyChecking=yes','-o','ConnectTimeout=10',user+'@127.0.0.1','wsl -d Ubuntu -u root -- /home/dministrator/cuda-lab/.venv/bin/python -']
    p=subprocess.run(command,input=payload,env=env,text=True,capture_output=True,timeout=1200)
    (dest/'transport.log').write_text(p.stdout+p.stderr)
    envelopes=[line.split('=',1)[1] for line in p.stdout.splitlines() if line.startswith('P1_03_RESULT=')]
    if not envelopes:raise RuntimeError('No remote envelope; inspect preserved transport.log')
    data=json.loads(envelopes[-1])
    for name,encoded in data['artifacts'].items():
        target=(dest/'run'/name).resolve()
        if not target.is_relative_to((dest/'run').resolve()):raise ValueError('unsafe artifact path')
        target.parent.mkdir(parents=True,exist_ok=True);target.write_bytes(base64.b64decode(encoded))
    execution=json.loads((dest/'run/execution.json').read_text());events=[{'event':'initialize','incumbent':parent}]
    fixtures_ok=True
    for row in execution['commands']:
        if row['tag'].startswith('fault-'):
            expected_timeout=row['tag']=='fault-host-timeout'
            rejected=row['timed_out'] if expected_timeout else row['exit']!=0 and not row['timed_out']
            fixtures_ok=fixtures_ok and rejected and not row['remaining_group_pids']
            events.append({'event':'reject_fault','case':row['tag'],'rejected':rejected,'remaining_group_pids':row['remaining_group_pids'],'incumbent_after':parent['id']})
    recovery=next(r for r in execution['commands'] if r['tag']=='recovery')
    fixtures_ok=fixtures_ok and recovery['exit']==0
    events.append({'event':'recovery','pass':recovery['exit']==0,'incumbent_after':parent['id']})
    reports=[json.loads((dest/'run'/f'session-{i}.json').read_text()) for i in range(3) if (dest/'run'/f'session-{i}.json').exists()]
    verdict=decide(reports,parent,candidate,guard)
    if not fixtures_ok or not execution['preflight_idle'] or protected_hashes()!=guard:
        verdict.update(verdict='REJECT',reason='fault_recovery_idle_or_harness_check_failed',incumbent_after=parent['id'])
    final=candidate if verdict['verdict']=='KEEP' else parent
    events.append({'event':'candidate','proposal_sha256':sha(args.proposal),'decision':verdict})
    (dest/'candidates.jsonl').write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in events))
    atomic_json(dest/'decision.json',verdict);atomic_json(dest/'incumbent.json',final)
    atomic_json(dest/'summary.json',{'status':'PASS' if fixtures_ok and len(reports)==3 and all(r['status']=='PASS' for r in reports) else 'INCOMPLETE',
                                    'scope':'trusted parameter proposal, GPU correctness/measurement, deterministic first-round decision','fixture_rejections_pass':fixtures_ok,
                                    'sessions':len(reports),'verdict':verdict,'incumbent':final,'real_model_proposal_file':str(args.proposal)})
    print(json.dumps({'verdict':verdict,'fixture_rejections_pass':fixtures_ok,'sessions':len(reports)},indent=2))


if __name__=='__main__':main()
