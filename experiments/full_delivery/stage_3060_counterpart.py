"""Stage the accepted immutable wheel and trusted adapters; no GPU execution."""
import base64,hashlib,json
from tools.ssh_3060 import ROOT,execute_python

def main():
 build=json.loads((ROOT/'evidence/20261124-sm86-3060-runtime-build-mirror/result.json').read_text());assert build['status']=='BUILT';out=ROOT/'evidence/20261124-sm86-3060-counterpart';out.mkdir(exist_ok=False);wheel=ROOT/'artifacts/wheels/nvidia_kernel_agent-0.1.1-py3-none-any.whl';assert hashlib.sha256(wheel.read_bytes()).hexdigest()=='f2ef0ec1f054a9331a4134e6545e6ce2a84b5aae22923f81063cb5c7418a2706'
 files={'runtime.whl':wheel.read_bytes(),'host.py':(ROOT/'experiments/full_delivery/counterpart_cublas_host.py').read_bytes(),'counterpart_support.py':(ROOT/'experiments/full_delivery/counterpart_cublas_support.py').read_bytes(),'ordinary_host.py':(ROOT/'experiments/full_delivery/gemm_ordinary_host.py').read_bytes(),'gpu_probe.py':(ROOT/'experiments/full_delivery/counterpart_gpu_probe.py').read_bytes(),'workloads/gemm_deployment_stability_v2.json':(ROOT/'workloads/gemm_deployment_stability_v2.json').read_bytes()};(out/'source-hashes.json').write_text(json.dumps({n:hashlib.sha256(b).hexdigest() for n,b in files.items()},indent=2))
 payload='FILES='+repr({n:base64.b64encode(b).decode() for n,b in files.items()})+'\nIMAGE='+repr(build['image_id'])+'\n'+r'''
import base64,hashlib,json,os,subprocess,tempfile,zipfile
from pathlib import Path
parent=Path.home()/'nka-persistent';root=Path(tempfile.mkdtemp(prefix='gemm-counterpart-',dir=parent));root.chmod(0o700)
for n,b in FILES.items():p=root/n;p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes(base64.b64decode(b))
prefix=root/'prefix';prefix.mkdir()
with zipfile.ZipFile(root/'runtime.whl') as z:
 for n in z.namelist():
  p=Path(n);assert not p.is_absolute() and '..' not in p.parts
  if n.endswith('/'):continue
  f=prefix/p;f.parent.mkdir(parents=True,exist_ok=True);f.write_bytes(z.read(n))

# Separate diagnostic adapter; accepted package/worker files remain untouched.
adapter=root/'probe_runtime';adapter.mkdir();(adapter/'__init__.py').write_text('')
for n in ['runner.py','pipe_worker.py','protocol.py','bounded_io.py','output.py']:(adapter/n).write_bytes((prefix/'kernel_agent'/n).read_bytes())
worker=(adapter/'pipe_worker.py').read_text();anchor='    emit(meta,raw.getvalue())';assert worker.count(anchor)==1
instrument="""    # Actual loaded numerical libraries, diagnostic metadata only after CUDA completes.
    candidates={}
    for line in Path('/proc/self/maps').read_text().splitlines():
        fields=line.split()
        if not fields or not fields[-1].startswith('/'):continue
        path=Path(fields[-1]);name=path.name
        for key,needle in [('cudart','libcudart.so'),('cublas','libcublas.so'),('cublaslt','libcublasLt.so'),('torchcuda','libtorch_cuda.so'),('c10cuda','libc10_cuda.so')]:
            if name.startswith(needle):candidates[key]=path
    if set(candidates)!={'cudart','cublas','cublaslt','torchcuda','c10cuda'}:raise RuntimeError('loaded numerical library witness incomplete')
    for key,path in candidates.items():
        with path.open('rb') as file:meta['lib_'+key+'_sha256']=hashlib.file_digest(file,'sha256').hexdigest()
        meta['lib_'+key+'_path']=str(path)
    emit(meta,raw.getvalue())"""
(adapter/'pipe_worker.py').write_text(worker.replace(anchor,instrument))
r=subprocess.run(['docker','image','inspect',IMAGE,'--format','{{.Id}}'],capture_output=True,text=True,timeout=30);assert r.returncode==0 and r.stdout.strip()==IMAGE
binding={'image_id':IMAGE,'torch':'2.14.0+cu130','cuda':'13.0','numpy':'2.5.3','target_uuid':'GPU-d464a726-39c2-9ad5-d856-f5d00c1fa870','target_driver':'580.95.05','capability':[8,6],'same_binary_same_architecture_reuse':True};(root/'runtime-binding.json').write_text(json.dumps(binding,indent=2));c=json.loads((prefix/'kernel_agent/data/releases.json').read_text());builds={role:{'binary':str(prefix/'kernel_agent/data/artifacts'/(c['versions'][c[key]]['binary_sha256']+'.so')),'binary_sha256':c['versions'][c[key]]['binary_sha256']} for role,key in [('candidate','default'),('incumbent','fallback')]};(root/'artifact-binding.json').write_text(json.dumps({'status':'REUSED_VERIFIED_SM86_ARTIFACTS','new_compilation':False,'builds':builds},indent=2));lock=Path('/tmp/nka-reviewed-gpu-execution.lock');fd=os.open(lock,os.O_CREAT|os.O_RDWR|os.O_NOFOLLOW,0o600);os.close(fd)
location={'root':str(root),'prefix':str(prefix),'image_id':IMAGE,'uid':os.getuid(),'gid':os.getgid(),'docker_gid':__import__('grp').getgrnam('docker').gr_gid,'gpu_runs_started':0};(root/'location.json').write_text(json.dumps(location,indent=2));print(json.dumps(location))
'''
 (out/'stage-driver.py').write_text(payload);r=execute_python(payload,120);(out/'stage.stderr').write_text(r.stderr);assert r.returncode==0;v=json.loads(r.stdout);(out/'location.json').write_text(json.dumps(v,indent=2));print(json.dumps(v))
if __name__=='__main__':main()
