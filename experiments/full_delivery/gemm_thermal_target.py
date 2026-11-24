"""Bounded trusted cuBLAS thermal diagnostic; no AA or qualification result."""
import hashlib,json,sys,time
from pathlib import Path
sys.path[:0]=['/package','/package/kernel_agent','/package/kernel_agent/optimization']
import torch
from source_worker import load
from fixtures import make_inputs
torch.set_num_threads(1)
digest='66fcc80f9122c919f92c72e37a01794a792c91a58b71d5705b160e0bc979dfc3'
model=load(Path('/package/kernel_agent/data/artifacts')/(digest+'.so'),digest).gemm_cublas
assert torch.__version__=='2.14.0+cu130' and torch.cuda.get_device_capability()==(8,6)
a,b=[x.cuda() for x in make_inputs([1024,1024,1024],701)]
stream=torch.cuda.Stream()
with torch.no_grad(),torch.cuda.stream(stream):
 for _ in range(8):output=model(a,b)
 stream.synchronize();graph=torch.cuda.CUDAGraph();outputs=[]
 with torch.cuda.graph(graph,stream=stream):
  for _ in range(32):outputs.append(model(a,b))
 started=time.monotonic();deadline=started+90;replays=0
 while time.monotonic()<deadline:
  for _ in range(16):graph.replay();replays+=1
  stream.synchronize()
 actual=outputs[-1].cpu();reference=a.cpu().double()@b.cpu().double()
 error=(actual.double()-reference).abs();allowed=3e-5+1e-4*reference.abs()
 assert bool((error<=allowed).all())
 print(json.dumps({'status':'DIAGNOSTIC_COMPLETE','seconds':time.monotonic()-started,'graph_replays':replays,'calls_per_graph':32,'public_shape':[1024]*3,'output_sha256':hashlib.sha256(actual.numpy().tobytes()).hexdigest(),'max_abs_error':float(error.max()),'qualification':False,'production_release':False}))
