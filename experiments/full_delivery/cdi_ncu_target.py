"""Public fixed-cuBLAS profile target; same reviewed DSO loader as the existing pipe_worker."""
import hashlib,importlib.util,json
from pathlib import Path
import numpy as np
import torch

def main():
 root=Path('/work');req=json.loads((root/'request.json').read_text());binary=Path('/artifact/nka_gemm_v1.so');assert hashlib.sha256(binary.read_bytes()).hexdigest()==req['binary_sha256'];assert str(torch.__version__)=='2.14.0+cu130' and torch.cuda.get_device_capability()==(8,6);torch.set_num_threads(1);spec=importlib.util.spec_from_file_location('nka_gemm_v1',binary);module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module);rng=np.random.default_rng(req['seed']);x=(rng.standard_normal((512,512))*.25).astype(np.float16);y=(rng.standard_normal((512,512))*.25).astype(np.float16)
 for name,v in [('a',x),('b',y)]:np.save(root/(name+'.npy'),v,allow_pickle=False)
 a=torch.from_numpy(x.copy()).cuda();b=torch.from_numpy(y.copy()).cuda();outputs=[]
 with torch.no_grad():
  for _ in range(5):module.gemm_cublas(a,b)
  torch.cuda.synchronize();torch.cuda.profiler.start()
  try:
   for _ in range(3):outputs.append(module.gemm_cublas(a,b))
   torch.cuda.synchronize()
  finally:torch.cuda.profiler.stop()
  for i,v in enumerate(outputs):
   out=v.cpu().numpy();assert out.dtype==np.float32 and out.shape==(512,512) and np.isfinite(out).all();np.save(root/('output-'+str(i)+'.npy'),out,allow_pickle=False)
  assert np.array_equal(a.cpu().numpy(),x) and np.array_equal(b.cpu().numpy(),y)
 result={'status':'PUBLIC_CUBLAS_PROFILE_TARGET_PASS','binary_sha256':req['binary_sha256'],'target_launches':3,'shape_mnk':[512,512,512],'seed':req['seed'],'torch':str(torch.__version__),'capability':list(torch.cuda.get_device_capability()),'reference_available_to_target':False};(root/'validation.json').write_text(json.dumps(result,indent=2));print(json.dumps(result))
if __name__=='__main__':main()
