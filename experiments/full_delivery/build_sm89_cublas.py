"""Fresh trusted SM89 cuBLAS builds through the exact pinned KernelBench loader function."""
import argparse,ast,hashlib,json,os,subprocess,sys,time
from pathlib import Path


def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def build_role(root,role):
    import torch
    import torch.nn as nn
    path=root/'upstream/kernelbench/eval.py'
    text=path.read_text();tree=ast.parse(text)
    fn=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='load_custom_model')
    function_source=ast.get_source_segment(text,fn)
    namespace={'nn':nn}
    exec(compile(function_source,str(path),'exec'),namespace)
    (root/'builds'/role/'kernelbench_loader_function.py').write_text(function_source)
    cuda=(root/'kernels/gemm/cublas_sm86.cu').read_text()
    wrapper=f'''
from torch import nn
from torch.utils.cpp_extension import load_inline
extension = load_inline(name="nka_gemm_v1",
 cpp_sources="torch::Tensor gemm_cublas(const torch::Tensor& a, const torch::Tensor& b);",
 cuda_sources={cuda!r}, functions={{"gemm_cublas":{('Trusted cuBLAS '+role+' binding')!r}}},
 extra_ldflags=["-lcublas"], extra_include_paths=[{str(root/'kernels/gemm')!r}],
 extra_cuda_cflags=["-lineinfo"], verbose=True)
class ModelNew(nn.Module):
 def forward(self,a,b):return extension.gemm_cublas(a,b)
'''
    (root/'builds'/role/'wrapper.py').write_text(wrapper)
    context={};cls=namespace['load_custom_model'](wrapper,context)
    assert cls is not None
    binary=Path(context['extension'].__file__)
    assert binary.name=='nka_gemm_v1.so'
    record={'role':role,'binary':str(binary),'binary_sha256':sha(binary),
            'torch':str(torch.__version__),'cuda':torch.version.cuda,'architecture':'sm_89',
            'kernelbench_source_sha256':sha(path),'kernelbench_function_sha256':hashlib.sha256(function_source.encode()).hexdigest(),
            'cuda_source_sha256':sha(root/'kernels/gemm/cublas_sm86.cu'),
            'contract_sha256':sha(root/'kernels/gemm/gemm_contract.cuh'),
            'wrapper_sha256':hashlib.sha256(wrapper.encode()).hexdigest(),
            'gpu_execution':False,'source_scope':'fixed reviewed cuBLAS only; role differs in binding docstring'}
    (root/'builds'/role/'result.json').write_text(json.dumps(record,indent=2));print(json.dumps(record))


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--role',choices=['incumbent','candidate']);a=p.parse_args();root=a.root
    if a.role:return build_role(root,a.role)
    records=[]
    for role in ['incumbent','candidate']:
        folder=root/'builds'/role;folder.mkdir(parents=True,exist_ok=False)
        env={k:v for k,v in os.environ.items() if k in ['HOME','LANG','LD_LIBRARY_PATH']}
        env.update(PATH='/usr/local/cuda-12.6/bin:'+str(Path(sys.executable).parent)+':/usr/local/bin:/usr/bin:/bin',CUDA_HOME='/usr/local/cuda-12.6',TORCH_CUDA_ARCH_LIST='8.9',TORCH_EXTENSIONS_DIR=str(folder/'extensions'),MAX_JOBS='1',OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',CUDA_VISIBLE_DEVICES='')
        start=time.monotonic()
        with (folder/'stdout').open('w') as out,(folder/'stderr').open('w') as err:
            r=subprocess.run([sys.executable,__file__,'--root',str(root),'--role',role],env=env,cwd=root,stdout=out,stderr=err,timeout=600)
        records.append({'role':role,'exit':r.returncode,'wall_seconds':time.monotonic()-start})
        (root/'build-progress.json').write_text(json.dumps(records,indent=2))
        if r.returncode:raise RuntimeError('Fresh build failed; retain existing logs, do not rebuild blindly')
    values={r:json.loads((root/'builds'/r/'result.json').read_text()) for r in ['incumbent','candidate']}
    assert values['incumbent']['binary_sha256']!=values['candidate']['binary_sha256']
    result={'status':'BUILT','builds':values,'records':records,'gpu_evaluations':0,'new_model_calls':0,'production_release':False}
    (root/'build-result.json').write_text(json.dumps(result,indent=2));print(json.dumps(result))


if __name__=='__main__':main()
