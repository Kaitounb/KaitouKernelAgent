"""Generate a small public Attention/MLA request and run the installed qualified CLI."""
import argparse,json,subprocess
from pathlib import Path
import numpy as np
BUNDLE='cfae38ce5c0fac56545207fe93588c1193646886fc5952f113fab487ef0be6eb'


def main():
 p=argparse.ArgumentParser();p.add_argument('--operation',choices=['attention','mla'],default='attention');p.add_argument('--runtime-dir',type=Path);p.add_argument('--work',type=Path,required=True);p.add_argument('--generate-only',action='store_true');a=p.parse_args()
 if not a.generate_only and a.runtime_dir is None:p.error('--runtime-dir is required to run the GPU CLI')
 work=a.work.resolve();work.mkdir(parents=True,mode=0o700,exist_ok=False);inputs=work/'inputs';inputs.mkdir(mode=0o700);rng=np.random.default_rng(9001)
 def values(shape):return np.clip(rng.standard_normal(shape)*.25,-4,4).astype(np.float16)
 if a.operation=='attention':
  request={'operation':'attention','bundle_id':BUNDLE,'fault':None,'batch':1,'sq':3,'sk':17,'heads':2,'dim':64,'causal':True,'mask':False};arrays={name:values(shape) for name,shape in [('q.npy',(1,3,2,64)),('k.npy',(1,17,2,64)),('v.npy',(1,17,2,64))]}
 else:
  request={'operation':'mla','bundle_id':BUNDLE,'fault':None,'lengths':[0,17],'heads':16,'page_size':16,'splits':1,'num_pages':2};arrays={name:values(shape) for name,shape in [('q_abs.npy',(2,16,512)),('q_rope.npy',(2,16,64)),('cache.npy',(2,16,512)),('rope_cache.npy',(2,16,64))]};arrays.update({'indptr.npy':np.array([0,0,2],np.int32),'indices.npy':np.array([1,0],np.int32)})
 for name,value in arrays.items():np.save(inputs/name,value,allow_pickle=False)
 (work/'request.json').write_text(json.dumps(request,indent=2)+'\n')
 if a.generate_only:print(json.dumps({'status':'INPUTS_GENERATED','operation':a.operation,'work':str(work),'gpu_executed':False}));return
 runtime=a.runtime_dir.resolve();subprocess.run([str(runtime/'kernel-agent-operators'),'run','--channel','qualified','--bundle',str(runtime/'sdk'),'--request',str(work/'request.json'),'--inputs',str(inputs),'--output',str(work/'outputs'),'--work',str(work/'execution')],check=True)

if __name__=='__main__':main()
