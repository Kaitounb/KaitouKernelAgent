"""Fresh, costed original GEMM profile for the small trial; never an Event resample."""
import argparse,base64,json
from pathlib import Path
from tools.ssh_3070 import execute_python
ROOT=Path(__file__).resolve().parents[2]


def main():
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);a=p.parse_args();a.output.mkdir(exist_ok=False)
    names=['experiments/p1_04/ncu_adapter.py','experiments/p1_04/profile_entry.py','provenance/upstream-reviews/20261118/meta-pytorch__KernelAgent/files/kernel_perf_agent/kernel_opt/profiler/ncu_profiler.py','provenance/upstream-reviews/20261118/meta-pytorch__KernelAgent/files/kernel_perf_agent/kernel_opt/diagnose_prompt/metric_schema.py'];files={n:(ROOT/n).read_bytes() for n in names}
    payload='FILES='+repr({n:base64.b64encode(v).decode() for n,v in files.items()})+'\n'+r'''
import base64,fcntl,gzip,hashlib,json,os,subprocess,tempfile,time
from pathlib import Path
root=Path(tempfile.mkdtemp(prefix='nka-gemm-costed-profile-'));py='/home/dministrator/cuda-lab/.venv/bin/python';lock=open('/tmp/nka-reviewed-gpu-execution.lock','a');fcntl.flock(lock,fcntl.LOCK_EX)
for n,v in FILES.items():
 p=root/n;p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes(base64.b64decode(v))
binary=Path('/tmp/nka-gemm-rel01-500z3apq/run/prefix/kernel_agent/data/artifacts/93fb2f4344521b7e9b246b54d1c30d4b1a8facf731e6385e18f6119e9c49c72c.so');assert hashlib.sha256(binary.read_bytes()).hexdigest()==binary.stem
cfg={'binary':str(binary),'binary_sha256':binary.stem,'shape_mnk':[512,512,512],'warps':4,'output':str(root/'validation.json')};(root/'config.json').write_text(json.dumps(cfg));(root/'target.py').write_text("from experiments.p1_04.profile_entry import main\nmain("+repr(str(root/'config.json'))+")\n")
collector=r"""
import gzip,json,statistics,subprocess,time
from pathlib import Path
from experiments.p1_04.ncu_adapter import select_metrics,profile_cuda,parse_profile
r=Path.cwd();ncu='/opt/nvidia/nsight-compute/2026.3.1/ncu';start=time.monotonic();q=subprocess.run([ncu,'--query-metrics','--query-metrics-mode','all'],capture_output=True,text=True,timeout=90);(r/'query.stdout.gz').write_bytes(gzip.compress(q.stdout.encode()));(r/'query.stderr').write_text(q.stderr);assert q.returncode==0;metrics,unavailable=select_metrics(q.stdout);(r/'selection.json').write_text(json.dumps({'metrics':metrics,'unavailable':unavailable}));profile_cuda(r/'target.py',r,'baseline512',metrics,ncu,timeout=180);parsed=parse_profile(r/'baseline512.csv',metrics,4);(r/'parsed.json').write_text(json.dumps(parsed,indent=2));execution=json.loads((r/'baseline512.execution.json').read_text());config=json.loads((r/'config.json').read_text());assert len(parsed['launches'])==3
facts=[]
for i,m in enumerate(parsed['launches'][0]['metrics'],1):
 rows=[x['metrics'][m] for x in parsed['launches']];assert all(x['status']=='observed' and x['unit']==rows[0]['unit'] for x in rows);facts.append({'id':'F'+str(i),'metric':m,'value':statistics.median(x['value'] for x in rows),'unit':rows[0]['unit'],'status':'observed'})
profile={'facts':facts,'provenance':[{'csv_sha256':parsed['csv_sha256'],'binary_sha256':config['binary_sha256'],'upstream_profiler_sha256':parsed['upstream_profiler_sha256']}],'profile_wall_seconds':execution['wall_seconds']};(r/'profile.json').write_text(json.dumps(profile,indent=2));(r/'result.json').write_text(json.dumps({'status':'PASS','collector_wall_seconds':time.monotonic()-start,'profile_wall_seconds':execution['wall_seconds'],'launches':3,'metric_fields':len(facts),'binary_sha256':config['binary_sha256'],'new_model_calls':0,'event_resampling':False,'performance_claim':False,'production_release':False},indent=2))
"""
(root/'collect.py').write_text(collector);env={k:v for k,v in os.environ.items() if k in ['HOME','LANG','LD_LIBRARY_PATH']};env.update(PYTHONPATH=str(root)+':/tmp/nka-p1-04-parser-deps',PATH='/usr/local/cuda-13.0/bin:/usr/lib/wsl/lib:/usr/local/bin:/usr/bin:/bin',CUDA_HOME='/usr/local/cuda-13.0',OMP_NUM_THREADS='1')
with (root/'collector.stdout').open('w') as out,(root/'collector.stderr').open('w') as err:r=subprocess.run([py,str(root/'collect.py')],cwd=root,env=env,stdout=out,stderr=err,timeout=350)
print(json.dumps({'remote_root':str(root),'exit':r.returncode,'result':json.loads((root/'result.json').read_text()) if (root/'result.json').exists() else None}))
'''
    (a.output/'driver.py').write_text(payload);r=execute_python(payload,500);(a.output/'transport.stderr').write_text(r.stderr);(a.output/'transport.stdout').write_text(r.stdout)
    if r.returncode:raise RuntimeError('profile transport failed; do not relaunch')
    wire=json.loads(r.stdout);(a.output/'result.json').write_text(json.dumps(wire,indent=2)+'\n');print(json.dumps(wire))


if __name__=='__main__':main()
