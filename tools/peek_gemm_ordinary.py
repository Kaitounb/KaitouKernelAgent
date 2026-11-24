"""Read-only state of the single frozen ordinary-clock GEMM qualification."""
import json,re
from tools.ssh_3070 import ROOT,execute_python

def main():
 out=ROOT/'evidence/20261123-sm86-gemm-ordinary-qualification';root=json.loads((out/'location.json').read_text())['remote_root'];assert re.fullmatch('/tmp/nka-cublas-final-ordinary-[A-Za-z0-9_-]+',root)
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
for name in ['result.json','failure.log','host.stderr']:
 p=root/name
 if p.exists():
  if name.endswith('.json'):
   result=json.loads(p.read_text());v[name]={k:x for k,x in result.items() if k not in ['records','deployment_decision','optimization_screening_decision']}
  else:v[name]=p.read_text()[-1800:]
print(json.dumps(v))
'''
 r=execute_python(payload,45);assert r.returncode==0;v=json.loads(r.stdout);(out/'latest-progress.json').write_text(json.dumps(v,indent=2));print(json.dumps(v))
if __name__=='__main__':main()
