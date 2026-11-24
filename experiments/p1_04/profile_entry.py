"""Trusted profile-only driver: no CUDA Events; gates profiling with start/stop APIs."""
import hashlib
import importlib.util
import json
from pathlib import Path
import torch


def main(config_path):
    cfg=json.loads(Path(config_path).read_text());binary=Path(cfg['binary'])
    if hashlib.sha256(binary.read_bytes()).hexdigest()!=cfg['binary_sha256']:raise ValueError('trusted binary changed')
    spec=importlib.util.spec_from_file_location('nka_gemm_v1',binary)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    torch.set_num_threads(1)
    m,n,k=cfg['shape_mnk'];g=torch.Generator().manual_seed(171)
    a_cpu=torch.randn(m,k,generator=g).mul_(.25).clamp_(-4,4).half()
    b_cpu=torch.randn(k,n,generator=g).mul_(.25).clamp_(-4,4).half()
    expected=(a_cpu.double()@b_cpu.double()).float()
    stream=torch.cuda.Stream();outputs=[]
    with torch.no_grad(),torch.cuda.stream(stream):
        a,b=a_cpu.cuda(),b_cpu.cuda()
        for _ in range(10):module.gemm_template(a,b,cfg['warps'])
        stream.synchronize()
        torch.cuda.profiler.start()
        try:
            for _ in range(3):outputs.append(module.gemm_template(a,b,cfg['warps']))
            stream.synchronize()
        finally:torch.cuda.profiler.stop()
        for out in outputs:torch.testing.assert_close(out.cpu(),expected,atol=3e-5,rtol=1e-4)
        torch.testing.assert_close(a.cpu(),a_cpu,atol=0,rtol=0);torch.testing.assert_close(b.cpu(),b_cpu,atol=0,rtol=0)
    Path(cfg['output']).write_text(json.dumps({'status':'PASS','shape_mnk':[m,n,k],'warps':cfg['warps'],'profiled_target_launches':3,
      'binary_sha256':cfg['binary_sha256'],'cpu_fp64_reference':True,'inputs_immutable':True,'scope':'profile-only; no Event timing'})+'\n')


if __name__=='__main__':
    import sys
    main(sys.argv[1])
