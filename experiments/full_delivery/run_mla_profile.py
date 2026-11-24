"""Use the existing pinned KernelAgent CUDA collector/parser for one native MLA profile."""
import argparse,base64,hashlib,io,json,zipfile
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]


def main():
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,default=ROOT/'evidence/20261122-mla-ncu-sm89');p.add_argument('--target',choices=['server','3070'],default='server');a=p.parse_args();output=a.output;output.mkdir(exist_ok=False)
    names=['kernel_agent/operators/__init__.py','kernel_agent/operators/references.py','kernel_agent/operators/flashinfer_backend.py','experiments/full_delivery/mla_profile_entry.py','experiments/p1_04/ncu_adapter.py','tests/test_ncu_adapter.py','provenance/upstream-reviews/20261118/meta-pytorch__KernelAgent/files/kernel_perf_agent/kernel_opt/profiler/ncu_profiler.py','provenance/upstream-reviews/20261118/meta-pytorch__KernelAgent/files/kernel_perf_agent/kernel_opt/diagnose_prompt/metric_schema.py'];files={n:(ROOT/n).read_bytes() for n in names};files['kernel_agent/__init__.py']=b''
    payload='FILES='+repr({n:base64.b64encode(v).decode() for n,v in files.items()})+'\nTARGET='+repr(a.target)+'\n'+r'''
import base64,fcntl,gzip,hashlib,io,json,os,subprocess,sys,tempfile,time,traceback,zipfile
from pathlib import Path
root=Path(tempfile.mkdtemp(prefix='nka-mla-ncu-'));work=root/'run';work.mkdir();prefix=Path('/tmp/nka-full-server-dqnrbkzs');py=str(prefix/'venv/bin/python') if TARGET=='server' else '/home/dministrator/cuda-lab/.venv/bin/python';cuda='/usr/local/cuda-12.6' if TARGET=='server' else '/usr/local/cuda-13.0';ncu=cuda+'/bin/ncu' if TARGET=='server' else '/opt/nvidia/nsight-compute/2026.3.1/ncu';jit='/tmp/nka-operators-q31aq63u/sdk-jit' if TARGET=='server' else '/tmp/nka-operators-b0x2q9hl/sdk-jit'
gpu_lock=open('/tmp/nka-reviewed-gpu-execution.lock','a');fcntl.flock(gpu_lock,fcntl.LOCK_EX)
if TARGET=='3070':jit='/tmp/nka-mla-measurement-v1-3070-jit'
for name,data in FILES.items():
 p=root/name;p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes(base64.b64decode(data))
pythonpath=str(root)
if TARGET=='3070':pythonpath+=':/tmp/nka-flashinfer-sdk-local-m0583epr/prefix-cxx20:/tmp/nka-p1-04-parser-deps'
env={k:v for k,v in os.environ.items() if k in ['HOME','LANG','LD_LIBRARY_PATH']};env.update(PYTHONPATH=pythonpath,PATH=str(Path(py).parent)+':'+cuda+'/bin:/usr/lib/wsl/lib:/usr/local/bin:/usr/bin:/bin',CUDA_HOME=cuda,TORCH_CUDA_ARCH_LIST='8.9' if TARGET=='server' else '8.6',FLASHINFER_WORKSPACE_BASE=jit,MAX_JOBS='2',OMP_NUM_THREADS='1',UV_CACHE_DIR=str(prefix/'uv-cache'),UV_NATIVE_TLS='true',NKA_PROFILE_NCU=ncu)
worker=r"""
import gzip,json,os,subprocess,time,traceback
from pathlib import Path
from experiments.p1_04.ncu_adapter import select_metrics,profile_cuda,parse_profile
root=Path.cwd();work=root/'run';ncu=os.environ['NKA_PROFILE_NCU'];records=[]
activity=subprocess.run(['nvidia-smi','--query-gpu=utilization.gpu,clocks.sm,memory.used','--format=csv,noheader'],capture_output=True,text=True,timeout=10);(work/'pre-profile-activity.json').write_text(json.dumps({'exit':activity.returncode,'stdout':activity.stdout,'scope':'profile-only diagnostic, no formal Event sampling; background activity retained'}))
for label,cmd in [('version',[ncu,'--version']),('query',[ncu,'--query-metrics','--query-metrics-mode','all'])]:
 start=time.monotonic();r=subprocess.run(cmd,capture_output=True,text=True,timeout=120);(work/(label+'.stdout.gz')).write_bytes(gzip.compress(r.stdout.encode()));(work/(label+'.stderr')).write_text(r.stderr);records.append({'label':label,'command':cmd,'exit':r.returncode,'wall_seconds':time.monotonic()-start})
 if r.returncode:raise RuntimeError('NCU '+label+' unavailable')
query=gzip.decompress((work/'query.stdout.gz').read_bytes()).decode();metrics,unavailable=select_metrics(query);(work/'selection.json').write_text(json.dumps({'metrics':metrics,'unavailable':unavailable},indent=2))
launcher=root/'profile_target.py';launcher.write_text("import sys\nfrom pathlib import Path\nfrom experiments.full_delivery.mla_profile_entry import main\nsys.argv=[__file__,'--output',str(Path(__file__).parent/'run/validation.json')]\nmain()\n")
profile_cuda(launcher,work,'mla-dev04',metrics,ncu,timeout=300,kernel_regex='.*BatchMLAPagedAttentionKernel.*');record=json.loads((work/'mla-dev04.execution.json').read_text());records.append(record);result={'status':'PROFILE_COLLECTED' if record.get('exit')==0 else 'UNAVAILABLE_OR_FAILED','records':records,'new_model_calls':0,'performance_claim':False}
(work/'result.json').write_text(json.dumps(result,indent=2))
"""
script=root/'collect.py';script.write_text(worker)
with (root/'collector.stdout').open('w') as out,(root/'collector.stderr').open('w') as err:r=subprocess.run([py,str(script)],cwd=root,env=env,stdout=out,stderr=err,timeout=650)
if r.returncode==0 and (work/'result.json').exists() and json.loads((work/'result.json').read_text())['status']=='PROFILE_COLLECTED':
 cmd=[str(prefix/'tools-v2/bin/uv'),'pip','install','--python',py,'--index-url','https://mirrors.aliyun.com/pypi/simple','pandas==2.3.3','numpy==2.2.6']
 dep_exit=0
 if TARGET=='server':
  with (root/'parser-install.stdout').open('w') as out,(root/'parser-install.stderr').open('w') as err:dep=subprocess.run(cmd,env=env,stdout=out,stderr=err,timeout=300)
  dep_exit=dep.returncode
 if dep_exit==0:
  code="from pathlib import Path;import json;from experiments.p1_04.ncu_adapter import parse_profile;r=Path('run');s=json.loads((r/'selection.json').read_text());d=parse_profile(r/'mla-dev04.csv',s['metrics'],None,expected_kernel='BatchMLAPagedAttentionKernel');(r/'parsed.json').write_text(json.dumps(d,indent=2))"
  with (root/'parser.stdout').open('w') as out,(root/'parser.stderr').open('w') as err:parsed=subprocess.run([py,'-c',code],cwd=root,env=env,stdout=out,stderr=err,timeout=120)
  with (root/'parser-tests.stdout').open('w') as out,(root/'parser-tests.stderr').open('w') as err:tests=subprocess.run([py,'-m','unittest','discover','-s','tests','-p','test_ncu_adapter.py','-v'],cwd=root,env=env,stdout=out,stderr=err,timeout=120)
  (work/'parser-status.json').write_text(json.dumps({'parse_exit':parsed.returncode,'cpu_parser_tests_exit':tests.returncode,'test_methods':5}))
manifest={};blobs={}
for path in root.rglob('*'):
 if path.is_file() and not path.is_symlink() and '__pycache__' not in path.parts:
  value=path.read_bytes();digest=hashlib.sha256(value).hexdigest();manifest[str(path.relative_to(root))]={'sha256':digest,'bytes':len(value)};blobs.setdefault(digest,value)
raw=io.BytesIO()
with zipfile.ZipFile(raw,'w',compression=zipfile.ZIP_DEFLATED) as z:
 z.writestr('manifest.json',json.dumps(manifest))
 for digest,value in blobs.items():z.writestr('blobs/'+digest,value)
result=json.loads((work/'result.json').read_text()) if (work/'result.json').exists() else {'status':'FAIL','collector_exit':r.returncode}
print(json.dumps({'remote_work':str(root),'result':result,'archive_sha256':hashlib.sha256(raw.getvalue()).hexdigest(),'archive':base64.b64encode(raw.getvalue()).decode()}))
'''
    (output/'driver.py').write_text(payload)
    if a.target=='server':from tools.ssh_server_gpu import execute_python
    else:from tools.ssh_3070 import execute_python
    r=execute_python(payload,1800);(output/'transport.stderr').write_text(r.stderr)
    if r.returncode:(output/'transport.stdout').write_text(r.stdout);raise RuntimeError('profile transport failed; inspect existing job only')
    wire=json.loads(r.stdout);raw=base64.b64decode(wire.pop('archive'));assert hashlib.sha256(raw).hexdigest()==wire['archive_sha256'];(output/'raw.zip').write_bytes(raw)
    with zipfile.ZipFile(io.BytesIO(raw)) as z:
        manifest=json.loads(z.read('manifest.json'))
        for rel,meta in manifest.items():
            name=Path(rel);assert not name.is_absolute() and '..' not in name.parts;value=z.read('blobs/'+meta['sha256']);assert len(value)==meta['bytes'] and hashlib.sha256(value).hexdigest()==meta['sha256'];target=output/'remote'/name;target.parent.mkdir(parents=True,exist_ok=True);target.write_bytes(value)
    (output/'result.json').write_text(json.dumps(wire,indent=2)+'\n');print(json.dumps({'status':wire['result']['status'],'remote_work':wire['remote_work']}))


if __name__=='__main__':main()
