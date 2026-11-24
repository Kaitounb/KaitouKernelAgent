"""Stage only reviewed public sources, run on authorized GPU, collect raw evidence losslessly."""
import argparse,base64,hashlib,io,json,zipfile
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]


def main():
    p=argparse.ArgumentParser();p.add_argument('--target',choices=['3070','server'],required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--only-softmax',action='store_true');p.add_argument('--sdk-prefix');p.add_argument('--phase',choices=['validate','mla-workloads','mla-configs','attention-bench'],default='validate');p.add_argument('--jit-base');a=p.parse_args();a.output.mkdir(exist_ok=False)
    sources=['kernel_agent/operators/__init__.py','kernel_agent/operators/references.py','kernel_agent/operators/flashinfer_backend.py','kernel_agent/operators/partition.py','kernels/softmax/torch_persistent.cu','experiments/full_delivery/operator_validate.py']
    files={name:(ROOT/name).read_bytes() for name in sources};files['kernel_agent/__init__.py']=b''
    if a.phase in ['mla-workloads','mla-configs']:
        for name in ['experiments/full_delivery/mla_workloads.py','workloads/mla_development_v1.json','workloads/mla_contract_v1.json']:files[name]=(ROOT/name).read_bytes()
    if a.phase=='attention-bench':
        files['experiments/full_delivery/attention_benchmark.py']=(ROOT/'experiments/full_delivery/attention_benchmark.py').read_bytes()
        timing=ROOT/'kernel_agent/optimization/assets/provenance/upstream-reviews/20261121-p1-02/ScalingIntelligence__KernelBench/src/kernelbench/timing.py';files['upstream/kernelbench_timing.py']=timing.read_bytes()
    upstream=ROOT/'provenance/operator-upstreams/pytorch__pytorch-e2d141dbde55'
    files['upstream/PersistentSoftmax.cuh']=(upstream/'aten/src/ATen/native/cuda/PersistentSoftmax.cuh').read_bytes();files['upstream/PYTORCH_LICENSE']=(upstream/'LICENSE').read_bytes()
    record={name:hashlib.sha256(value).hexdigest() for name,value in files.items()};(a.output/'sources.json').write_text(json.dumps(record,indent=2)+'\n')
    py='/home/dministrator/cuda-lab/.venv/bin/python' if a.target=='3070' else '/tmp/nka-full-server-dqnrbkzs/venv/bin/python'
    cuda='/usr/local/cuda-13.0' if a.target=='3070' else '/usr/local/cuda'
    payload='FILES='+repr({name:base64.b64encode(value).decode() for name,value in files.items()})+'\nPY='+repr(py)+'\nCUDA='+repr(cuda)+'\nONLY_SOFTMAX='+repr(a.only_softmax)+'\nSDK_PREFIX='+repr(a.sdk_prefix)+'\nPHASE='+repr(a.phase)+'\nJIT_BASE='+repr(a.jit_base)+'\n'+r'''
import base64,hashlib,io,json,os,subprocess,tempfile,zipfile
from pathlib import Path
root=Path(tempfile.mkdtemp(prefix='nka-operators-'))
for name,data in FILES.items():
 p=root/name;p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes(base64.b64decode(data))
env={k:v for k,v in os.environ.items() if k in ['HOME','LANG','LD_LIBRARY_PATH']};env.update(PYTHONPATH=str(root),CUDA_HOME=CUDA,PATH=CUDA+'/bin:/usr/lib/wsl/lib:/usr/local/bin:/usr/bin:/bin',MAX_JOBS='2',TORCH_CUDA_ARCH_LIST='8.6' if 'cuda-13' in CUDA else '8.9',OMP_NUM_THREADS='2',FLASHINFER_CUDA_ARCH_LIST='8.6' if 'cuda-13' in CUDA else '8.9')
env['FLASHINFER_WORKSPACE_BASE']=JIT_BASE or str(root/'sdk-jit')
if JIT_BASE:
 (root/'reused-jit-manifest.json').write_text(json.dumps({str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in Path(JIT_BASE).rglob('*.so')},indent=2)+'\n')
if SDK_PREFIX:
 env['PYTHONPATH']+=os.pathsep+SDK_PREFIX
if not ONLY_SOFTMAX:
 if SDK_PREFIX:sdk=Path(SDK_PREFIX)/'flashinfer'
 else:
  probe=subprocess.run([PY,'-c','import flashinfer;from pathlib import Path;print(Path(flashinfer.__file__).parent)'],env=env,capture_output=True,text=True,timeout=60)
  if probe.returncode:raise RuntimeError('installed SDK import failed')
  sdk=Path(probe.stdout.strip())
 for name,digest in {'mla.py':'b78276b95b35f6a0a309554df52820032146b24015d5c30349c92cfcd671ae41','prefill.py':'cdb48bfb3ed4b8299d8ccc5fda4a0c5a2e8295164dffefdeb76595088dab81fa','data/include/flashinfer/attention/mla.cuh':'8bcfdfc5d030f6c8ee60909981ececec227798b002143b23886379a3b6fd77cf'}.items():
  if hashlib.sha256((sdk/name).read_bytes()).hexdigest()!=digest:raise ValueError('Installed SDK differs from pinned upstream: '+name)
 sdk_files={str(p.relative_to(sdk)):hashlib.sha256(p.read_bytes()).hexdigest() for p in sdk.rglob('*') if p.is_file() and '__pycache__' not in p.parts}
 (root/'sdk-source-manifest.json').write_text(json.dumps(sdk_files,indent=2)+'\n')
 (root/'sdk-jit-core.py').write_bytes((sdk/'jit/core.py').read_bytes())
probe=subprocess.run([PY,'-c','import ninja;print(ninja.BIN_DIR)'],capture_output=True,text=True,timeout=30)
if probe.returncode==0:env['PATH']=probe.stdout.strip()+':'+env['PATH']
script={'mla-workloads':'mla_workloads.py','mla-configs':'mla_workloads.py','attention-bench':'attention_benchmark.py','validate':'operator_validate.py'}[PHASE]
cmd=[PY,str(root/'experiments/full_delivery'/script),'--root',str(root)]
if PHASE=='mla-configs':cmd.append('--config-grid')
if ONLY_SOFTMAX:cmd.append('--only-softmax')
with (root/'controller.stdout').open('w') as out,(root/'controller.stderr').open('w') as err:r=subprocess.run(cmd,cwd=root,env=env,stdout=out,stderr=err,timeout=3300)
manifest={};blobs={}
for path in root.rglob('*'):
 if path.is_file() and not path.is_symlink() and '__pycache__' not in path.parts and path.suffix!='.o':
  value=path.read_bytes();digest=hashlib.sha256(value).hexdigest();manifest[str(path.relative_to(root))]={'sha256':digest,'bytes':len(value)};blobs.setdefault(digest,value)
data=io.BytesIO()
with zipfile.ZipFile(data,'w',compression=zipfile.ZIP_DEFLATED) as archive:
 archive.writestr('manifest.json',json.dumps(manifest))
 for digest,value in blobs.items():archive.writestr('blobs/'+digest,value)
result=json.loads((root/'result.json').read_text()) if (root/'result.json').exists() else {'status':'FAIL','error':'controller failed before result; raw logs retained'}
print(json.dumps({'root':str(root),'exit':r.returncode,'result':result,'archive_sha256':hashlib.sha256(data.getvalue()).hexdigest(),'archive':base64.b64encode(data.getvalue()).decode()}))
'''
    (a.output/'driver.py').write_text(payload)
    if a.target=='3070':from tools.ssh_3070 import execute_python
    else:from tools.ssh_server_gpu import execute_python
    r=execute_python(payload,3600);(a.output/'transport.stderr').write_text(r.stderr)
    if r.returncode:(a.output/'transport.stdout').write_text(r.stdout);raise RuntimeError('operator transport failed; no automatic retry')
    wire=json.loads(r.stdout);raw=base64.b64decode(wire.pop('archive'));assert hashlib.sha256(raw).hexdigest()==wire['archive_sha256'];(a.output/'raw.zip').write_bytes(raw)
    with zipfile.ZipFile(io.BytesIO(raw)) as archive:
        manifest=json.loads(archive.read('manifest.json'))
        for name,meta in manifest.items():
            relative=Path(name);assert not relative.is_absolute() and '..' not in relative.parts;value=archive.read('blobs/'+meta['sha256']);assert len(value)==meta['bytes'] and hashlib.sha256(value).hexdigest()==meta['sha256'];target=a.output/'remote'/relative;target.parent.mkdir(parents=True,exist_ok=True);target.write_bytes(value)
    (a.output/'result.json').write_text(json.dumps(wire,indent=2)+'\n');result=wire['result'];print(json.dumps({'status':result['status'],'checks':result.get('checks'),'passed':result.get('passed'),'remote_root':wire['root'],'failed':[x['label'] for x in result.get('records',[]) if x['status']!='PASS']}))


if __name__=='__main__':main()
