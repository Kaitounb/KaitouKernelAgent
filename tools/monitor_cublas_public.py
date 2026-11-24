"""Read-only progress of an exact public qualification root; never schedules work."""
import argparse,json,re,time
from pathlib import Path
from tools.ssh_3070 import execute_python

def main():
 p=argparse.ArgumentParser();p.add_argument('--remote-root');p.add_argument('--output',type=Path,required=True);a=p.parse_args()
 if a.remote_root is not None and not re.fullmatch('/tmp/nka-cublas-final-[A-Za-z0-9_-]+',a.remote_root):raise ValueError('exact public qualification root required')
 payload='REMOTE='+repr(a.remote_root)+'\n'+r'''
import json,math,statistics,time
from pathlib import Path
roots=[Path(REMOTE)] if REMOTE is not None else sorted(Path('/tmp').glob('nka-cublas-final-sustained-*'))
values=[]
for root in roots:
 row={'remote_root':str(root),'qa':len(list(root.glob('qa-*/report.json'))),'aa':len(list(root.glob('aa-*/report.json'))),'ab':len(list(root.glob('ab-*/report.json'))),'completed_sessions':[],'active_jobs':[]}
 for job in sorted(root.iterdir()):
  if job.is_dir() and job.name.startswith(('aa-','ab-')):
   if not (job/'report.json').exists():row['active_jobs'].append({'name':job.name,'elapsed_seconds':round(time.time()-job.stat().st_mtime)})
   else:
    try:report=json.loads((job/'report.json').read_text())
    except ValueError:continue
    devs=[];means=[];durations=[]
    for r in report['candidate_aa' if report['mode']=='aa' else 'measurements']:
     pairs={}
     for s in r['segments']:pairs.setdefault(s['pair'],{})[s['label']]=s['elapsed_ms'];durations.append(s['elapsed_ms'])
     ratios=[v['incumbent']/v['candidate'] for v in pairs.values()];devs.extend(abs(x-1) for x in ratios);means.append(abs(math.exp(statistics.mean(math.log(x) for x in ratios))-1))
    row['completed_sessions'].append({'name':job.name,'max_pair_deviation':max(devs),'max_group_deviation':max(means),'segment_ms_range':[min(durations),max(durations)]})
 p=root/'clock-telemetry.json'
 if p.exists():
  try:
   samples=json.loads(p.read_text());row['telemetry_samples']=len(samples);row['latest_telemetry']=samples[-1].get('stdout') if samples else None
  except ValueError:row['telemetry_read']='concurrent_write_unavailable'
 if (root/'result.json').exists():
  r=json.loads((root/'result.json').read_text());row['status']=r['status'];row['clock_environment_verification']=r.get('clock_environment_verification');row['clock_restoration_verified']=r.get('clock_restoration_verified')
 values.append(row)
print(json.dumps(values))
'''
 r=execute_python(payload,90);a.output.mkdir(parents=True,exist_ok=True)
 if r.returncode:raise RuntimeError('Read-only progress unavailable; do not relaunch workload')
 v=json.loads(r.stdout);(a.output/(str(time.time_ns())+'.json')).write_text(json.dumps(v,indent=2)+'\n');print(json.dumps(v))
if __name__=='__main__':main()
