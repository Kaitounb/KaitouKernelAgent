"""Public offset/stream/contract checks for one already-built reviewed softmax DSO."""
import argparse,hashlib,importlib.util,json
from pathlib import Path
import torch


def main():
    p=argparse.ArgumentParser();p.add_argument('--binary',type=Path,required=True);a=p.parse_args();spec=importlib.util.spec_from_file_location('nka_torch_persistent_v271',a.binary);module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module);torch.set_num_threads(1);checks=[]
    for seed in [771,772]:
        torch.manual_seed(seed)
        for dtype in [torch.float16,torch.float32]:
            for width in [7,129,2048,2049]:
                rows=5;raw=torch.full((rows*width+34,),-3.25,dtype=dtype,device='cuda');x=raw[17:-17].view(rows,width);cpu=(torch.randn(rows,width)*20).to(dtype);expected=torch.softmax(cpu.double(),-1);stream=torch.cuda.Stream()
                with torch.cuda.stream(stream):
                    torch.cuda._sleep(50000);x.copy_(cpu);actual=module.softmax(x);consumer=actual.clone().cpu()
                stream.synchronize();error=(consumer.double()-expected).abs();assert bool((error<=2e-6+2e-5*expected.abs()).all());assert bool((raw[:17]==-3.25).all()) and bool((raw[-17:]==-3.25).all());assert torch.equal(x.cpu(),cpu) and actual.dtype==torch.float32 and actual.is_contiguous()
                checks.append({'seed':seed,'width':width,'dtype':str(dtype),'offset_elements':17,'max_abs_error':float(error.max()),'input_guards':True,'producer_consumer_stream':True})
    invalid=[torch.zeros(3,4,device='cuda',dtype=torch.int32),torch.zeros(3,4,device='cuda').T,torch.zeros(3,device='cuda'),torch.zeros(3,0,device='cuda')]
    for value in invalid:
        try:module.softmax(value)
        except RuntimeError:continue
        raise AssertionError('invalid contract accepted')
    print(json.dumps({'status':'PASS','checks':checks,'contract_rejections':len(invalid),'binary_sha256':hashlib.sha256(a.binary.read_bytes()).hexdigest(),'performance_sampling':False}))


if __name__=='__main__':main()
