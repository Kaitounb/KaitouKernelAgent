"""GPU-free admission negatives, persistent RC staging and read-only occupancy diagnosis."""
import json
from tools.ssh_3070 import ROOT,execute_python

def main():
 out=ROOT/'evidence/20261123-sm86-operators-rc1';root=json.loads((out/'location.json').read_text())['remote_root']
 payload='ROOT='+repr(root)+'\n'+r'''
import hashlib,json,os,shutil,subprocess,time
from pathlib import Path
import numpy as np
root=Path(ROOT);assert json.loads((root/'installed/result.json').read_text())['status']=='PASS';negative=root/'admission-negatives';negative.mkdir(exist_ok=False);py='/home/dministrator/cuda-lab/.venv/bin/python';env={k:v for k,v in os.environ.items() if k in ['HOME','PATH','LANG','LD_LIBRARY_PATH']};env.update(PYTHONPATH=str(root/'prefix'),PYTHONDONTWRITEBYTECODE='1',OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1');checks=[]
for label in ['nan','wrong-dtype','invalid-pages','request-fault','existing-output']:
 base=root/'installed'/('mla-ragged' if label=='invalid-pages' else 'attention-mask');dest=negative/label;dest.mkdir();shutil.copytree(base/'inputs',dest/'inputs');req=json.loads((base/'request.json').read_text())
 if label=='nan':
  a=np.load(dest/'inputs/q.npy',allow_pickle=False);a[0,0,0,0]=np.nan;np.save(dest/'inputs/q.npy',a,allow_pickle=False)
 if label=='wrong-dtype':
  a=np.load(dest/'inputs/q.npy',allow_pickle=False);np.save(dest/'inputs/q.npy',a.astype(np.float32),allow_pickle=False)
 if label=='invalid-pages':
  a=np.load(dest/'inputs/indices.npy',allow_pickle=False);a[:]=0;np.save(dest/'inputs/indices.npy',a,allow_pickle=False)
 if label=='request-fault':req['fault']='constant'
 if label=='existing-output':(dest/'outputs').mkdir()
 (dest/'request.json').write_text(json.dumps(req));cmd=[py,'-B','-m','kernel_agent.operators.cli','run','--bundle',str(root/'sdk'),'--request',str(dest/'request.json'),'--inputs',str(dest/'inputs'),'--output',str(dest/'outputs'),'--work',str(dest/'run')];p=subprocess.run(cmd,env=env,capture_output=True,text=True,timeout=30);(dest/'stdout').write_text(p.stdout);(dest/'stderr').write_text(p.stderr);assert p.returncode!=0 and not (dest/'run').exists();checks.append({'label':label,'exit':p.returncode,'no_gpu_work_created':True,'error':p.stderr.strip()})
(negative/'result.json').write_text(json.dumps({'status':'PASS','checks':checks,'gpu_calls':0},indent=2))
wheel=next((root/'dist').glob('*.whl'));sha=hashlib.sha256(wheel.read_bytes()).hexdigest();persistent=Path('/home/dministrator/cuda-lab/nka-persistent')/('operators-0.1.1rc1-'+sha[:12]);persistent.mkdir(mode=0o700,exist_ok=False)
shutil.copytree(root/'prefix',persistent/'prefix');shutil.copytree(root/'sdk',persistent/'sdk');shutil.copyfile(wheel,persistent/wheel.name)
launcher=persistent/'kernel-agent-operators';launcher.write_text('#!/bin/sh\nexport PYTHONPATH='+str(persistent/'prefix')+'\nexec '+py+' -B -m kernel_agent.operators.cli "$@"\n');launcher.chmod(0o700)
p=subprocess.run([str(launcher),'inspect','--bundle',str(persistent/'sdk')],env=env,capture_output=True,text=True,timeout=30);assert p.returncode==0
manifest=json.loads((persistent/'sdk/aot-manifest.json').read_text());assert all(hashlib.sha256((persistent/'sdk'/n).read_bytes()).hexdigest()==h for n,h in manifest['files'].items())
files={str(p.relative_to(persistent)):hashlib.sha256(p.read_bytes()).hexdigest() for p in persistent.rglob('*') if p.is_file() and '__pycache__' not in p.parts};deployment={'status':'DEVELOPMENT_RC_INSTALLED','root':str(persistent),'launcher':str(launcher),'prefix':str(persistent/'prefix'),'bundle':str(persistent/'sdk'),'wheel_sha256':sha,'files':files,'production_release':False,'source_execution_enabled':False,'restored_archived_dev6_untouched':True,'inspect':json.loads(p.stdout)};(persistent/'deployment.json').write_text(json.dumps(deployment,indent=2));(root/'persistent-deployment.json').write_text(json.dumps(deployment,indent=2));shutil.copyfile(launcher,root/'persistent-launcher')
occupancy=[]
for _ in range(4):
 p=subprocess.run(['nvidia-smi','--query-gpu=utilization.gpu,memory.used,temperature.gpu','--format=csv,noheader,nounits'],capture_output=True,text=True,timeout=15);occupancy.append({'exit':p.returncode,'stdout':p.stdout});time.sleep(1)
ps=r"""$rows=Get-CimInstance Win32_PerfFormattedData_GPUPerformanceCounters_GPUEngine | Where-Object { $_.UtilizationPercentage -gt 0 }; $items=foreach($row in $rows){ if($row.Name -match 'pid_(\d+)_'){ $taskId=[int]$Matches[1]; $proc=Get-Process -Id $taskId -ErrorAction SilentlyContinue; [pscustomobject]@{id=$taskId;process=$proc.ProcessName;utilization=[int]$row.UtilizationPercentage;engine=$row.Name} } }; @($items | Sort-Object utilization -Descending | Select-Object -First 8) | ConvertTo-Json -Compress"""
try:
 p=subprocess.run(['/mnt/c/Windows/System32/WindowsPowerShell/v1.0/powershell.exe','-NoProfile','-NonInteractive','-Command',ps],capture_output=True,text=True,timeout=30);win={'exit':p.returncode,'stdout':p.stdout,'stderr':p.stderr}
except Exception as e:win={'error':type(e).__name__}
post={'samples':occupancy,'windows_gpu_engines':win,'docker_containers':subprocess.run(['docker','ps','--filter','name=nka-native-','--format','{{.Names}}'],capture_output=True,text=True,timeout=15).stdout,'gpu_jobs_started':0};(root/'postflight.json').write_text(json.dumps(post,indent=2));print(json.dumps({'admission_negative_checks':len(checks),'persistent_root':str(persistent),'persistent_files_verified':len(files),'launcher':str(launcher),'production_release':False,'postflight':post}))
'''
 (out/'finish-driver.py').write_text(payload);r=execute_python(payload,180);(out/'finish-transport.stderr').write_text(r.stderr);(out/'finish-transport.stdout').write_text(r.stdout);assert r.returncode==0;v=json.loads(r.stdout);(out/'finish-result.json').write_text(json.dumps(v,indent=2));print(json.dumps(v))
if __name__=='__main__':main()
