"""Trusted GEMM regression transport, adapted from P1-01 run_3070.py.

Only fixed repository files; credentials supplied exclusively via child environment.
Requires the already-authorized local SSH tunnel on port 21022.
"""
import argparse
import base64
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import sys


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--evidence',type=Path,required=True)
    args = ap.parse_args()
    root = Path(__file__).resolve().parents[2]
    dest = args.evidence.resolve()
    if dest.exists():
        ap.error('Use a new evidence directory')
    paths = [root/x for x in ['evaluators/gemm_regression.py','evaluators/gemm_kernelbench.py',
             'kernels/gemm/cublas_sm86.cu','kernels/gemm/wmma_sm86.cu','kernels/gemm/gemm_contract.cuh',
             'workloads/gemm_contract_v1.json','workloads/gemm_development_v1.json']]
    files = {str(p.relative_to(root)):base64.b64encode(p.read_bytes()).decode() for p in paths}
    upstream = root/'provenance/upstream-reviews/20261121-p1-02/ScalingIntelligence__KernelBench/src/kernelbench'
    for p in upstream.glob('*.py'):
        files['upstream/kernelbench/'+p.name] = base64.b64encode(p.read_bytes()).decode()
    dest.mkdir(parents=True)
    for name,data in files.items():
        target = dest/'sources'/name
        target.parent.mkdir(parents=True,exist_ok=True)
        target.write_bytes(base64.b64decode(data))
    (dest/'sources/run_3070.py').write_bytes(Path(__file__).read_bytes())
    payload = r'''
import base64,hashlib,json,os,pathlib,signal,subprocess,sys,tempfile,time
work=pathlib.Path(tempfile.mkdtemp(prefix='nka-p1-02-contract-'))
for name,data in FILES.items():
 p=work/name;p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes(base64.b64decode(data))
deps=pathlib.Path('/tmp/nka-p1-02-pydeps-hw9k37vu')
assert (deps/'pydantic').is_dir(), 'Prior isolated KernelBench dependencies absent; inspect before installing'
env={k:v for k,v in os.environ.items() if k in ['HOME','LANG','PATH','LD_LIBRARY_PATH']}
env.update(CUDA_HOME='/usr/local/cuda-13.0',MAX_JOBS='1',TORCH_CUDA_ARCH_LIST='8.6',
 TORCH_EXTENSIONS_DIR=str(work/'extensions'),PYTHONPATH=str(work/'upstream')+':'+str(deps),PYTHONUNBUFFERED='1')
env['PATH']=env['CUDA_HOME']+'/bin:'+env.get('PATH','')
records=[]
def execute(cmd,tag,timeout=240):
 try:
  p=subprocess.run(cmd,cwd=work,env=env,text=True,capture_output=True,timeout=timeout)
  (work/(tag+'.stdout.log')).write_text(p.stdout);(work/(tag+'.stderr.log')).write_text(p.stderr)
  records.append({'tag':tag,'command':cmd,'exit':p.returncode})
  return p.returncode
 except subprocess.TimeoutExpired as e:
  (work/(tag+'.stdout.log')).write_bytes(e.stdout or b'');(work/(tag+'.stderr.log')).write_bytes(e.stderr or b'')
  records.append({'tag':tag,'command':cmd,'timeout':True});return 124
for warps in [4,1,2,8]:
 cmd=[sys.executable,'-m','evaluators.gemm_regression','--warps',str(warps),'--output',str(work/f'warps-{warps}.json')]
 if execute(cmd,f'warps-{warps}') != 0: break
else:
 for tool in ['memcheck','synccheck']:
  cmd=['/usr/local/cuda-13.0/bin/compute-sanitizer','--tool',tool,'--error-exitcode','97',sys.executable,'-m','evaluators.gemm_regression','--sanitizer-smoke','--output',str(work/(tool+'.json'))]
  execute(cmd,tool,180)
 execute([sys.executable,'-m','evaluators.gemm_kernelbench','--output',str(work/'original-entry.json')],'original-entry')
# Preserve exact build commands, generated sources, all partial results and logs on failure.
artifacts=[]
for p in (work/'extensions').rglob('*'):
 if p.is_file(): artifacts.append({'path':str(p.relative_to(work)),'bytes':p.stat().st_size,'sha256':hashlib.sha256(p.read_bytes()).hexdigest()})
(work/'execution.json').write_text(json.dumps({'workdir':str(work),'commands':records,'env_overrides':env,'build_artifacts':artifacts},indent=2))
subprocess.run(['nvidia-smi','--query-gpu=timestamp,name,utilization.gpu,memory.used,temperature.gpu,clocks.sm,power.draw','--format=csv,noheader'],stdout=(work/'gpu-after.csv').open('w'),text=True)
outputs=[p for p in work.glob('*') if p.is_file()]
outputs += [p for p in (work/'extensions').rglob('*') if p.is_file() and p.suffix in ['.ninja','.cpp','.cu']]
print('P1_02_RESULT='+json.dumps({'artifacts':{str(p.relative_to(work)):base64.b64encode(p.read_bytes()).decode() for p in outputs},'commands':records,'workdir':str(work)}))
'''.replace('FILES',repr(files))
    line = next(x for x in (root/'.private/3070-connection.txt').read_text().splitlines() if 'Windows' in x and '密码' in x)
    env = os.environ.copy()
    env['SSHPASS'] = re.search(r'密码[：:]\s*(\S+)',line).group(1)
    user = re.search(r'账号\s*([^，,\s]+)',line).group(1)
    command = ['/opt/homebrew/bin/sshpass','-e','ssh','-p','21022','-o','StrictHostKeyChecking=yes','-o','ConnectTimeout=10',
               user+'@127.0.0.1','wsl -d Ubuntu -u root -- /home/dministrator/cuda-lab/.venv/bin/python -']
    p = subprocess.run(command,input=payload,env=env,text=True,capture_output=True,timeout=1200)
    (dest/'transport.log').write_text(p.stdout+p.stderr)
    envelopes = [line.split('=',1)[1] for line in p.stdout.splitlines() if line.startswith('P1_02_RESULT=')]
    if not envelopes:
        raise RuntimeError('No result envelope; inspect preserved transport.log')
    data = json.loads(envelopes[-1])
    for name,blob in data['artifacts'].items():
        target = (dest/'run'/name).resolve()
        if not target.is_relative_to((dest/'run').resolve()):
            raise ValueError('unsafe artifact path')
        target.parent.mkdir(parents=True,exist_ok=True)
        target.write_bytes(base64.b64decode(blob))
    print(json.dumps({k:data[k] for k in ['commands','workdir']},indent=2))
    return p.returncode or int(any(x.get('exit',124) != 0 for x in data['commands']))


if __name__ == '__main__':
    sys.exit(main())
