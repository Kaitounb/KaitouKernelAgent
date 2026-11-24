"""Reviewed SM89 cuBLAS callable only; privilege/network reduction precedes CUDA import."""
import argparse,errno,hashlib,importlib.util,io,json,os,resource,sys,time
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parent))
from restrictions import restrict
from protocol import encode_frame


def denied_read(path):
    try:
        with open(path,'rb') as f:f.read(1)
        return False
    except PermissionError:return True


def denied_write(path):
    try:
        with open(path,'ab') as f:f.write(b'changed')
        return False
    except PermissionError:return True


def main():
    p=argparse.ArgumentParser();p.add_argument('--request',type=Path,required=True);a=p.parse_args();req=json.loads(a.request.read_text());case=req['case'];meta=restrict()
    if case in ['stdout_flood','stderr_flood']:
        fd=1 if case=='stdout_flood' else 2
        for _ in range(256):os.write(fd,b'X'*65536)
        return
    artifact=a.request.parent.parent/'artifact'
    meta.update(case=case,private_read_denied=denied_read(req['host_private_path']),artifact_write_denied=denied_write(artifact/'readonly-guard'),input_write_denied=denied_write(a.request),control_canary_in_env=bool(os.getenv('NKA_CONTROL_CANARY')),credentials_in_env=any('API_KEY' in k.upper() or 'AUTH_TOKEN' in k.upper() or k=='SSHPASS' for k in os.environ))
    try:os.kill(req['host_pid'],0);meta['controller_signal_denied']=False
    except PermissionError:meta['controller_signal_denied']=True
    import numpy as np
    import torch
    torch.set_num_threads(1);torch.set_num_interop_threads(1)
    if str(torch.__version__)!='2.7.1+cu126' or torch.version.cuda!='12.6' or torch.cuda.get_device_capability(0)!=(8,9) or torch.cuda.get_device_name(0)!='NVIDIA GeForce RTX 4090':raise RuntimeError('unsupported SM89 runtime/device')
    meta.update(torch=str(torch.__version__),cuda=torch.version.cuda,device='sm89',execution='reviewed-native')
    binary=artifact/'nka_gemm_v1.so'
    if hashlib.sha256(binary.read_bytes()).hexdigest()!=req['binary_sha256']:raise RuntimeError('SM89 binary hash mismatch')
    spec=importlib.util.spec_from_file_location('nka_gemm_v1',binary);module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    arrays=[np.load(a.request.parent/n,allow_pickle=False) for n in ['a.npy','b.npy']]
    if any(x.dtype!=np.float16 or x.ndim!=2 or not x.flags.c_contiguous or not np.isfinite(x).all() or np.any(np.abs(x)>4) for x in arrays):raise RuntimeError('SM89 input contract invalid')
    x,y=arrays
    if x.shape[1]!=y.shape[0] or not all(1<=d<=4096 for d in [*x.shape,*y.shape]):raise RuntimeError('SM89 dimensions invalid')
    before=[hashlib.sha256(x.tobytes()).hexdigest() for x in arrays];left=torch.from_numpy(x.copy()).cuda();right=torch.from_numpy(y.copy()).cuda()
    if case=='crash':raise RuntimeError('reviewed SM89 crash')
    stream=torch.cuda.Stream()
    with torch.no_grad(),torch.cuda.stream(stream):
        torch.cuda._sleep(200000);out=module.gemm_cublas(left,right);consumed=out.clone()
    stream.synchronize()
    if case=='timeout':sys.stderr.write('NKA_SM89_FINITE_CUDA_READY\n');sys.stderr.flush();time.sleep(120)
    if out.dtype!=torch.float32 or list(out.shape)!=[x.shape[0],y.shape[1]] or not out.is_contiguous() or torch._C._is_alias_of(out,left) or torch._C._is_alias_of(out,right):raise RuntimeError('SM89 output contract invalid')
    if [hashlib.sha256(t.cpu().numpy().tobytes()).hexdigest() for t in [left,right]]!=before:raise RuntimeError('SM89 inputs changed')
    meta.update(input_immutable=True,producer_consumer_stream=True,fresh_output=True,binary_sha256=req['binary_sha256'])
    raw=io.BytesIO();np.save(raw,consumed.cpu().numpy(),allow_pickle=False);sys.stdout.buffer.write(encode_frame(meta,raw.getvalue()));sys.stdout.buffer.flush()


if __name__=='__main__':main()
