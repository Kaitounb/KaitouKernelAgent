"""Read-only progress for the reserved SM89 build/QA/qualification; never starts GPU work."""
import json,time
from tools.ssh_server_gpu import ROOT,execute_python


def main():
    output=ROOT/'evidence/20261123-sm89-cublas-v2';root=json.loads((output/'location.json').read_text())['remote_root']
    payload='ROOT='+repr(root)+'\n'+r'''
import json,time,sys,math,hashlib
from pathlib import Path
root=Path(ROOT)
sys.path.insert(0,str(root/'prefix'))
from kernel_agent.optimization.identity import group_log_ratio
def read(name):return json.loads((root/name).read_text()) if (root/name).exists() else None
v={'observed_wall':time.time(),'build_progress':read('build-progress.json'),'precheck_progress':read('precheck-progress.json'),'qualification_progress':read('qualification-progress.json'),'precheck_result':read('precheck-result.json'),'qualification_result':read('qualification-result.json'),'decision':read('decision-progress.json')}
v['errors']={n:(root/n).read_text()[-3000:] for n in ['build.stderr','precheck.stderr','precheck-failure.log','qualification.stderr','qualification-failure.log'] if (root/n).exists()}
v['active_jobs']=[p.name for p in root.iterdir() if p.is_dir() and p.name.startswith(('qa-','aa-','ab-','memcheck-','synccheck-','fault-','recovery-')) and not (p/'record.json').exists()]
v['timing_summaries']=[]
for f in sorted(root.glob('aa-*/report.json')):
 if not (f.parent/'record.json').exists():continue
 report=json.loads(f.read_text());values=[group_log_ratio(row,report['session']) for row in report['candidate_aa']]
 v['timing_summaries'].append({'session':report['session'],'report_sha256':hashlib.sha256(f.read_bytes()).hexdigest(),'max_pair_deviation':max(x[1] for x in values),'max_group_geomean_deviation':max(abs(math.exp(x[0])-1) for x in values)})
print(json.dumps(v))
'''
    r=execute_python(payload,60,connection_path=ROOT/'.private/server-gpu-next-connection.json')
    if r.returncode:
        (output/('observe-error-'+str(int(time.time()))+'.txt')).write_text(r.stderr)
        raise RuntimeError('Read-only observation failed; GPU controller left untouched')
    v=json.loads(r.stdout);progress=output/'progress';progress.mkdir(exist_ok=True);(progress/(str(int(v['observed_wall']))+'.json')).write_text(json.dumps(v,indent=2))
    concise={k:v[k] for k in ['observed_wall','active_jobs','errors','decision','timing_summaries']}
    for phase in ['precheck','qualification']:
        records=v[phase+'_progress'] or []
        concise[phase+'_completed_jobs']=[{'job':r['job'],'status':r['status']} for r in records]
        result=v[phase+'_result'];concise[phase+'_result']={k:result.get(k) for k in ['status','qa_checks','aa_sessions','ab_sessions','raw_segments','deployment_decision']} if result else None
    print(json.dumps(concise))


if __name__=='__main__':main()
