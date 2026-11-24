"""Reference-free profile target for an already reviewed and numerically validated DSO."""
import json
from pathlib import Path
import torch
from fixtures import make_inputs
from source_worker import load,tensor_hash


def main():
    req=json.loads(Path('/inputs/request.json').read_text());m,n,k=req['shape_mnk']
    if torch.__version__!='2.14.0+cu130' or torch.cuda.get_device_capability()!=(8,6):raise ValueError('unsupported profile device/software')
    if any(type(x) is not int or not 16<=x<=4096 or x%16 for x in [m,n,k]):raise ValueError('profile target requires supported WMMA dimensions')
    torch.set_num_threads(1);module=load('/artifact/candidate/nka_gemm_v1.so',req['binary_sha256']);cpu=make_inputs([m,n,k],171,'random');stream=torch.cuda.Stream();outputs=[]
    with torch.no_grad(),torch.cuda.stream(stream):
        a,b=[x.cuda() for x in cpu]
        for _ in range(10):module.gemm_template(a,b,4)
        stream.synchronize();torch.cuda.profiler.start()
        try:
            for _ in range(3):outputs.append(module.gemm_template(a,b,4))
            stream.synchronize()
        finally:torch.cuda.profiler.stop()
        inputs=[tensor_hash(a),tensor_hash(b)];hashes=[tensor_hash(x) for x in outputs]
    assert inputs==[tensor_hash(x) for x in cpu] and all(x.shape==(m,n) and x.dtype==torch.float32 and bool(torch.isfinite(x).all()) for x in outputs)
    Path('/tmp/profile-validation.json').write_text(json.dumps({'status':'PASS','binary_sha256':req['binary_sha256'],'input_hashes':inputs,'output_hashes':hashes,'cpu_reference_available':False,'target_launches':3})+'\n')


if __name__=='__main__':main()
