"""Real fixed-image profiler preflight/rejection/recovery; no new API or Event scores."""
import argparse,base64,hashlib,json
from pathlib import Path
from tools.ssh_3070 import execute_python
ROOT=Path(__file__).resolve().parents[2]


def main():
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);a=p.parse_args();a.output.mkdir(exist_ok=False)
    names=['kernel_agent/__init__.py','kernel_agent/bounded_io.py','kernel_agent/protocol.py','kernel_agent/runner.py','kernel_agent/output.py','kernel_agent/optimization/__init__.py','kernel_agent/optimization/profile_runner.py','kernel_agent/optimization/profile_worker.py','kernel_agent/optimization/profile_target.py','kernel_agent/optimization/source_worker.py','kernel_agent/optimization/source_packets.py','kernel_agent/optimization/fixtures.py','experiments/p1_04/ncu_adapter.py','provenance/upstream-reviews/20261118/meta-pytorch__KernelAgent/files/kernel_perf_agent/kernel_opt/profiler/ncu_profiler.py','provenance/upstream-reviews/20261118/meta-pytorch__KernelAgent/files/kernel_perf_agent/kernel_opt/diagnose_prompt/metric_schema.py'];files={n:(ROOT/n).read_bytes() for n in names};(a.output/'sources.json').write_text(json.dumps({n:hashlib.sha256(v).hexdigest() for n,v in files.items()},indent=2)+'\n')
    payload='FILES='+repr({n:base64.b64encode(v).decode() for n,v in files.items()})+'\n'+r'''
import base64,fcntl,hashlib,json,os,subprocess,tempfile
from pathlib import Path
root=Path(tempfile.mkdtemp(prefix='nka-bounded-ncu-'));lock=open('/tmp/nka-reviewed-gpu-execution.lock','a');fcntl.flock(lock,fcntl.LOCK_EX)
for n,v in FILES.items():
 p=root/n;p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes(base64.b64decode(v))
worker=r"""
import hashlib,json
from pathlib import Path
from kernel_agent.optimization.profile_runner import run
from experiments.p1_04.ncu_adapter import HARDWARE_METRICS,LAUNCH_METRICS,parse_profile
root=Path.cwd();base=Path('/tmp/nka-gemm-rel01-500z3apq/run/prefix/kernel_agent/data/artifacts/93fb2f4344521b7e9b246b54d1c30d4b1a8facf731e6385e18f6119e9c49c72c.so');candidate=Path('/tmp/nka-feedback-B-90ea4d43117c/run/evaluation/build/artifacts/nka_gemm_v1.so');assert hashlib.sha256(candidate.read_bytes()).hexdigest()=='3982266ff83f179693c12a1d3ed41b0f69ba3068aff5d27992300a5bd56ee6e4';oracles=json.loads(Path('/tmp/nka-feedback-B-90ea4d43117c/run/evaluation/oracle-ledgers.json').read_text());rows=[]
for label,binary,role,regex in [('baseline',base,'incumbent','.*gemm_wmma.*'),('no-target-rejected',base,'incumbent','__fixed_no_target__'),('recovery-reviewed-source',candidate,'candidate','.*gemm_wmma.*')]:
 work=root/label;files,record=run(binary,hashlib.sha256(binary.read_bytes()).hexdigest(),[512,512,512],HARDWARE_METRICS+LAUNCH_METRICS,work,root/'experiments/p1_04/ncu_adapter.py',kernel_regex=regex);entry={'label':label,'worker_status':record['status'],'wall_seconds':record['wall_seconds'],'cleanup':record['cleanup']['removed'],'protected_unchanged':record['protected_unchanged']};rows.append(entry)
 if not entry['cleanup']:raise RuntimeError('profile cleanup unconfirmed')
 if files is None:
  entry['reason']=record.get('metadata',{}).get('failure') or record.get('rejection') or record.get('reason');(root/'progress.json').write_text(json.dumps(rows,indent=2))
  if label=='no-target-rejected':entry['expected_rejection']=True;continue
  break
 try:
  parsed=parse_profile(work/'metrics.csv',HARDWARE_METRICS+LAUNCH_METRICS,4);(work/'parsed.json').write_text(json.dumps(parsed,indent=2));validation=json.loads((work/'validation.json').read_text());matches=[x for x in oracles[role].values() if x['case']['shape_mnk']==[512,512,512] and x['case']['seed']==171 and x['case']['distribution']=='random' and not x['case']['offset_guard']];assert len(matches)==1;oracle=matches[0];assert validation['input_hashes']==oracle['input_hashes'] and validation['output_hashes']==[oracle['output_sha256']]*3 and validation['cpu_reference_available'] is False;entry.update(parsed=True,launches=len(parsed['launches']),metric_fields=len(parsed['launches'][0]['metrics']),profile_wall_seconds=json.loads((work/'execution.json').read_text())['wall_seconds'],host_oracle_hashes_pass=True)
 except (ValueError,AssertionError,KeyError) as error:
  entry.update(parsed=False,parse_error=str(error)[:512]);assert label=='no-target-rejected';entry['expected_rejection']=True
 if label=='no-target-rejected':assert entry.get('expected_rejection')
 (root/'progress.json').write_text(json.dumps(rows,indent=2))
result={'status':'PASS' if len(rows)==3 and rows[0].get('parsed') and rows[1].get('expected_rejection') and rows[2].get('parsed') else 'UNAVAILABLE_OR_FAILED','rows':rows,'new_model_calls':0,'event_sampling':False,'production_release':False};(root/'result.json').write_text(json.dumps(result,indent=2));print(json.dumps(result))
"""
(root/'probe.py').write_text(worker);env={k:v for k,v in os.environ.items() if k in ['HOME','LANG','LD_LIBRARY_PATH']};env.update(PYTHONPATH=str(root)+':/tmp/nka-p1-04-parser-deps',PATH='/usr/local/cuda-13.0/bin:/usr/lib/wsl/lib:/usr/local/bin:/usr/bin:/bin',OMP_NUM_THREADS='1')
with (root/'controller.stdout').open('w') as out,(root/'controller.stderr').open('w') as err:r=subprocess.run(['/home/dministrator/cuda-lab/.venv/bin/python',str(root/'probe.py')],cwd=root,env=env,stdout=out,stderr=err,timeout=850)
print(json.dumps({'remote_root':str(root),'exit':r.returncode,'result':json.loads((root/'result.json').read_text()) if (root/'result.json').exists() else None}))
'''
    (a.output/'driver.py').write_text(payload);r=execute_python(payload,950);(a.output/'transport.stderr').write_text(r.stderr);(a.output/'transport.stdout').write_text(r.stdout)
    if r.returncode:raise RuntimeError('probe transport failed; inspect existing job only')
    result=json.loads(r.stdout);(a.output/'result.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result))


if __name__=='__main__':main()
