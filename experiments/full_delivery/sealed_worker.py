"""Independent observer for frozen cuBLAS artifacts: NPY inputs only, no hidden manifest/reference."""
import argparse,hashlib,importlib.util,io,json,os,sys
from pathlib import Path
import numpy as np
import torch
from protocol import encode_frame


def main():
    p=argparse.ArgumentParser();p.add_argument('--request',type=Path,required=True);args=p.parse_args();req=json.loads(args.request.read_text());torch.set_num_threads(1)
    if torch.__version__!='2.14.0+cu130' or torch.cuda.get_device_capability(0)!=(8,6):raise RuntimeError('unsupported frozen environment')
    path=Path('/artifact/nka_gemm_v1.so')
    if hashlib.sha256(path.read_bytes()).hexdigest()!=req['binary_sha256']:raise ValueError('frozen binary mismatch')
    spec=importlib.util.spec_from_file_location('nka_gemm_v1',path);module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    xs=[np.load('/inputs/'+name+'.npy',allow_pickle=False) for name in ['a','b']];stream=torch.cuda.Stream()
    with torch.no_grad(),torch.cuda.stream(stream):
        torch.cuda._sleep(200000);raws=[torch.full((x.size+34,),7,dtype=torch.float16,device='cuda') for x in xs];gpu=[]
        for raw,x in zip(raws,xs):
            view=raw[17:-17].view(x.shape);view.copy_(torch.from_numpy(x));gpu.append(view)
        actual=module.gemm_cublas(*gpu)
        if actual.dtype!=torch.float32 or not actual.is_contiguous() or actual.device!=gpu[0].device or list(actual.shape)!=[xs[0].shape[0],xs[1].shape[1]] or any(torch._C._is_alias_of(actual,x) for x in gpu):raise ValueError('frozen output contract')
        output=actual.clone().cpu().numpy();hashes=[hashlib.sha256(x.cpu().contiguous().numpy().tobytes()).hexdigest() for x in gpu];guards=all(bool((x[:17]==7).all() and (x[-17:]==7).all()) for x in raws)
    stream.synchronize();out=io.BytesIO();np.save(out,output,allow_pickle=False)
    meta={'uid':os.getuid(),'case':'cublas','torch':str(torch.__version__),'device':'sm86','binary_sha256':req['binary_sha256'],'input_hashes':hashes,'guards_ok':guards,'producer_consumer_stream':True,'output_contract':True}
    sys.stdout.buffer.write(encode_frame(meta,out.getvalue()));sys.stdout.buffer.flush()


if __name__=='__main__':main()
