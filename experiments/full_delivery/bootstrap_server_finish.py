"""Use the already pinned project environment to build FlashInfer, preserving v1/v2 logs."""
import argparse,json,os,subprocess,time
from pathlib import Path


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);a=p.parse_args();root=a.root;py=str(root/'venv/bin/python');uv=str(root/'tools-v2/bin/uv');records=[]
    env={k:v for k,v in os.environ.items() if k in ['HOME','PATH','LANG','LD_LIBRARY_PATH']};env.update(UV_CACHE_DIR=str(root/'uv-cache'),UV_NATIVE_TLS='true',CUDA_HOME='/usr/local/cuda',PATH='/usr/local/cuda/bin:'+env.get('PATH',''),MAX_JOBS='2')
    def run(label,cmd,timeout):
        start=time.monotonic()
        with (root/(label+'.stdout')).open('w') as out,(root/(label+'.stderr')).open('w') as err:r=subprocess.run(cmd,env=env,stdout=out,stderr=err,timeout=timeout)
        records.append({'label':label,'command':cmd,'exit':r.returncode,'wall_seconds':time.monotonic()-start});(root/'progress-v3.json').write_text(json.dumps(records,indent=2)+'\n')
        if r.returncode:raise RuntimeError(label+' failed; raw evidence retained')
    run('flashinfer-v3',[uv,'pip','install','--no-build-isolation','--python',py,'--index-url','https://mirrors.aliyun.com/pypi/simple','flashinfer-python==0.2.8','torch==2.7.1'],1200)
    run('freeze-v3',[uv,'pip','freeze','--python',py],60)
    run('probe-v3',[py,'-c',"import torch,flashinfer,json,platform;print(json.dumps({'python':platform.python_version(),'torch':torch.__version__,'cuda':torch.version.cuda,'capability':torch.cuda.get_device_capability(),'flashinfer':flashinfer.__version__,'smoke':(torch.ones(2,device='cuda')+1).cpu().tolist()}))"],120)
    result={'status':'PASS','root':str(root),'python':py,'commands':records,'new_model_calls':0};(root/'result-v3.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result))


if __name__=='__main__':main()
