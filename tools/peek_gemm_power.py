"""Read-only state of the single frozen 100W-capped GEMM qualification."""
import json,re
from tools.ssh_3070 import ROOT,execute_python

def main():
 out=ROOT/'evidence/20261124-sm86-gemm-power-qualification';root=json.loads((out/'location.json').read_text())['remote_root'];assert re.fullmatch('/tmp/nka-cublas-final-power-[A-Za-z0-9_-]+',root)
 payload='ROOT='+repr(root)+'\n'+r'''
import json
from pathlib import Path
root=Path(ROOT);qa=0;done=[];active=[]
for p in sorted(root.iterdir()):
 if p.is_dir() and p.name.startswith(('qa-','aa-','ab-')):
  if (p/'report.json').exists():
   v=json.loads((p/'report.json').read_text());qa+=len(v.get('checks',[])) if p.name.startswith('qa-') else 0;done.append(p.name)
  else:active.append(p.name)
v={'qa_output_checks_completed':qa,'completed_jobs':done,'active_job_directories':active}
# Read-only partial receipt; the frozen evaluator still completes all3AA before any gate.
import sys
protocol=json.loads((root/'protocol.json').read_text());sys.path.insert(0,protocol['installed_prefix'])
from kernel_agent.optimization.identity import group_log_ratio
v['completed_aa_summaries']=[]
for name in done:
 if name.startswith('aa-'):
  report=json.loads((root/name/'report.json').read_text());rows=report['candidate_aa'];v['completed_aa_summaries'].append({'session':report['session'],'rows':len(rows),'max_pair_deviation':max(group_log_ratio(x,report['session'])[1] for x in rows)})
for name in ['result.json','failure.log','host.stderr']:
 p=root/name
 if p.exists():
  if name.endswith('.json'):
   result=json.loads(p.read_text());v[name]={k:x for k,x in result.items() if k not in ['records','deployment_decision','optimization_screening_decision']}
  else:v[name]=p.read_text()[-1800:]
import subprocess
smi=subprocess.run(['nvidia-smi','--query-gpu=timestamp,utilization.gpu,temperature.gpu,power.limit,clocks.sm','--format=csv,noheader,nounits'],capture_output=True,text=True,timeout=15);v['gpu_snapshot']={'exit':smi.returncode,'stdout':smi.stdout,'stderr':smi.stderr}
print(json.dumps(v))
'''
 r=execute_python(payload,45);assert r.returncode==0;v=json.loads(r.stdout);(out/'latest-progress.json').write_text(json.dumps(v,indent=2));print(json.dumps(v))
if __name__=='__main__':main()
