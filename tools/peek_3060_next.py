"""Read-only progress of the single frozen3060 counterpart; no extra NVML polling."""
import json
from .ssh_3060 import ROOT,execute_python

def main():
 out=ROOT/'evidence/20261124-3060-next-qualification';loc=json.loads((out/'location.json').read_text());payload='ROOT='+repr(loc['root'])+'\n'+r'''
import json,sys,time
from pathlib import Path
root=Path(ROOT);sys.dont_write_bytecode=True;sys.path.insert(0,str(root/'prefix'));from kernel_agent.optimization.identity import group_log_ratio
v={'completed':[],'active':[],'aa_summaries':[]}
for p in sorted(root.iterdir()):
 if p.is_dir() and p.name.startswith(('aa-','ab-')):
  if (p/'report.json').exists():
   try:r=json.loads((p/'report.json').read_text())
   except json.JSONDecodeError:v['active'].append(p.name+'writing');continue
   v['completed'].append(p.name)
   if p.name.startswith('aa-'):v['aa_summaries'].append({'session':r['session'],'max_pair_deviation':max(group_log_ratio(x,r['session'])[1] for x in r['candidate_aa'])})
  else:v['active'].append(p.name)
for n in ['qualification-result.json','qualification-failure.log','qualification-parent.stderr']:
 p=root/n
 if p.exists():
  if n.endswith('.json'):
   r=json.loads(p.read_text());v[n]={k:r[k] for k in r if k not in ['records','deployment_decision','original_research_decision']}
  else:v[n]=p.read_text()[-1800:]
p=root/'qualification-telemetry.json'
if p.exists():
 for _ in range(3):
  try:r=json.loads(p.read_text());v['telemetry_samples']=len(r);v['latest_existing_gpu_sample']=r[-1] if r else None;break
  except json.JSONDecodeError:time.sleep(.1)
print(json.dumps(v))
'''
 r=execute_python(payload,45);assert r.returncode==0;v=json.loads(r.stdout);(out/'latest-progress.json').write_text(json.dumps(v,indent=2));print(json.dumps(v))
if __name__=='__main__':main()
