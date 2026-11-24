"""Trusted P1-04 CUDA profiling transport; API credentials and heldout never enter worker."""
import argparse
import base64
import json
import os
from pathlib import Path
import re
import subprocess

ROOT=Path(__file__).resolve().parents[2]


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--evidence',type=Path,required=True);p.add_argument('--warps',type=int,choices=[1,2,4,8],default=4)
    args=p.parse_args();dest=args.evidence.resolve()
    if dest.exists():p.error('Use a new evidence directory')
    paths=['experiments/p1_04/ncu_adapter.py','experiments/p1_04/profile_entry.py','tests/test_ncu_adapter.py',
       'provenance/upstream-reviews/20261118/meta-pytorch__KernelAgent/files/kernel_perf_agent/kernel_opt/profiler/ncu_profiler.py',
       'provenance/upstream-reviews/20261118/meta-pytorch__KernelAgent/files/kernel_perf_agent/kernel_opt/diagnose_prompt/metric_schema.py',
       'provenance/upstream-reviews/20261118/meta-pytorch__KernelAgent/files/LICENSE']
    files={path:base64.b64encode((ROOT/path).read_bytes()).decode() for path in paths}
    dest.mkdir(parents=True)
    for name,data in files.items():
        file=dest/'sources'/name;file.parent.mkdir(parents=True,exist_ok=True);file.write_bytes(base64.b64decode(data))
    (dest/'sources/run_3070.py').write_bytes(Path(__file__).read_bytes())
    payload=r'''
import base64,gzip,hashlib,importlib.metadata,json,os,pathlib,subprocess,sys,tempfile,time
work=pathlib.Path(tempfile.mkdtemp(prefix='nka-p1-04-ncu-'))
for name,data in FILES.items():
 p=work/name;p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes(base64.b64decode(data))
sys.path[:0]=[str(work),'/tmp/nka-p1-04-parser-deps']
os.environ['PYTHONPATH']=':'.join(sys.path)
from experiments.p1_04.ncu_adapter import select_metrics,profile_cuda,parse_profile
query=pathlib.Path('/tmp/nka-p1-04-metrics.txt')
metrics,missing=select_metrics(query.read_text())
(work/'metric-selection.json').write_text(json.dumps({'selected':metrics,'unavailable_from_query':missing,'query_sha256':hashlib.sha256(query.read_bytes()).hexdigest(),'query_bytes':query.stat().st_size,'launch_statistics':'not enumerated by hardware query; validated from actual profile'},indent=2))
with gzip.open(work/'query-metrics.txt.gz','wb') as out:out.write(query.read_bytes())
(work/'ncu-help.txt').write_bytes(pathlib.Path('/tmp/nka-p1-04-help.txt').read_bytes())
p=subprocess.run([sys.executable,'-m','unittest','discover','-s','tests','-p','test_ncu_adapter.py'],cwd=work,text=True,capture_output=True,timeout=60)
(work/'cpu-tests.stdout.log').write_text(p.stdout);(work/'cpu-tests.stderr.log').write_text(p.stderr)
meta={'workdir':str(work),'parser_tests_exit':p.returncode,'warps':WARPS,'dependencies':{n:importlib.metadata.version(n) for n in ['torch','numpy','pandas','python-dateutil','pytz','tzdata','six']},'profiles':[]}
try:
 if p.returncode:raise RuntimeError('parser CPU fixtures failed before profiling')
 for shape in [(512,512,512),(1024,1024,1024)]:
  tag='warps-WARPS-'+str(shape[0]);config={'binary':'/tmp/nka-p1-02-contract-eeyx9qg9/extensions/nka_gemm_v1/nka_gemm_v1.so','binary_sha256':'93fb2f4344521b7e9b246b54d1c30d4b1a8facf731e6385e18f6119e9c49c72c','warps':WARPS,'shape_mnk':shape,'output':str(work/(tag+'.checks.json'))}
  config_path=work/(tag+'.config.json');config_path.write_text(json.dumps(config,indent=2))
  script=work/(tag+'.py');script.write_text('from experiments.p1_04.profile_entry import main\nmain('+repr(str(config_path))+')\n')
  started=time.monotonic()
  raw=profile_cuda(script,work,tag,metrics,pathlib.Path('/opt/nvidia/nsight-compute/2026.3.1/ncu'))
  parsed=parse_profile(raw,metrics,WARPS)
  (work/(tag+'.metrics.json')).write_text(json.dumps(parsed,indent=2,allow_nan=False)+'\n')
  meta['profiles'].append({'tag':tag,'wall_seconds':time.monotonic()-started,'status':'PASS'})
except Exception as error:
 import traceback
 meta['error']=traceback.format_exc()
finally:
 meta['status']='PASS' if len(meta['profiles'])==2 and 'error' not in meta else 'FAIL'
 (work/'execution.json').write_text(json.dumps(meta,indent=2))
 gpu=subprocess.run(['nvidia-smi','--query-gpu=timestamp,name,utilization.gpu,memory.used,temperature.gpu,clocks.sm,power.draw','--format=csv,noheader'],text=True,capture_output=True)
 (work/'gpu-after.csv').write_text(gpu.stdout+gpu.stderr)
 print('P1_04_RESULT='+json.dumps({'artifacts':{str(p.relative_to(work)):base64.b64encode(p.read_bytes()).decode() for p in work.iterdir() if p.is_file()},'status':meta['status']}))
'''.replace('WARPS',str(args.warps)).replace('FILES',repr(files))
    line=next(x for x in (ROOT/'.private/3070-connection.txt').read_text().splitlines() if 'Windows' in x and '密码' in x)
    env=os.environ.copy();env['SSHPASS']=re.search(r'密码[：:]\s*(\S+)',line).group(1)
    user=re.search(r'账号\s*([^，,\s]+)',line).group(1)
    command=['/opt/homebrew/bin/sshpass','-e','ssh','-p','21022','-o','StrictHostKeyChecking=yes','-o','ConnectTimeout=10',user+'@127.0.0.1','wsl -d Ubuntu -u root -- /home/dministrator/cuda-lab/.venv/bin/python -']
    result=subprocess.run(command,input=payload,env=env,capture_output=True,text=True,timeout=600)
    (dest/'transport.log').write_text(result.stdout+result.stderr)
    envelope=[s.split('=',1)[1] for s in result.stdout.splitlines() if s.startswith('P1_04_RESULT=')]
    if not envelope:raise RuntimeError('No profile envelope; inspect preserved transport.log')
    record=json.loads(envelope[-1])
    for name,data in record['artifacts'].items():
        target=(dest/'run'/name).resolve()
        if not target.is_relative_to((dest/'run').resolve()):raise ValueError('unsafe artifact path')
        target.parent.mkdir(parents=True,exist_ok=True);target.write_bytes(base64.b64decode(data))
    print(json.dumps(json.loads((dest/'run/execution.json').read_text()),indent=2))
    if record['status']!='PASS':raise RuntimeError('Profile run failed; all partial evidence preserved')


if __name__=='__main__':main()
