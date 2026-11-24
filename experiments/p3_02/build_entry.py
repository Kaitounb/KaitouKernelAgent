"""Trusted pinned CUDA builder only; no inputs, references, model keys or generated source."""
import hashlib,json,os,platform
from pathlib import Path
from evaluators.gemm_regression import load_trusted
import torch

module,model=load_trusted(4)
binary=Path(module.__file__)
Path('/build/build.json').write_text(json.dumps({'status':'PASS','binary':str(binary),'binary_sha256':hashlib.sha256(binary.read_bytes()).hexdigest(),
 'uid':os.getuid(),'torch':torch.__version__,'cuda':torch.version.cuda,'python':platform.python_version(),
 'arch_list':os.getenv('TORCH_CUDA_ARCH_LIST'),'source_scope':'unchanged trusted CUDA and pinned KernelBench loader; no generated source'},indent=2)+'\n')
print(Path('/build/build.json').read_text())
