"""Isolated public dependency bootstrap for the user's existing SM89 server; no credentials."""
import json,os,subprocess,sys,tempfile,time
from pathlib import Path


def main():
    root=Path(tempfile.mkdtemp(prefix='nka-full-server-'));commands=[];env=os.environ.copy();env.update(UV_CACHE_DIR=str(root/'uv-cache'),UV_PYTHON_INSTALL_DIR=str(root/'python'),UV_NATIVE_TLS='true',CUDA_HOME='/usr/local/cuda',PATH='/usr/local/cuda/bin:'+os.environ.get('PATH',''),MAX_JOBS='4')
    def run(label,cmd,timeout=1800):
        start=time.monotonic()
        with (root/(label+'.stdout')).open('w') as out,(root/(label+'.stderr')).open('w') as err:r=subprocess.run(cmd,env=env,stdout=out,stderr=err,timeout=timeout)
        commands.append({'label':label,'command':cmd,'exit':r.returncode,'wall_seconds':time.monotonic()-start});(root/'progress.json').write_text(json.dumps(commands,indent=2)+'\n')
        if r.returncode:raise RuntimeError(label+' failed; logs retained')
    (root/'bootstrap.json').write_text(json.dumps({'root':str(root),'started':time.time(),'purpose':'trusted operator libraries and cross-card validation; no free model source on this server'},indent=2)+'\n')
    run('uv-install',[sys.executable,'-m','pip','install','--disable-pip-version-check','--no-deps','--target',str(root/'tools'),'uv==0.9.10'],300)
    uv=str(root/'tools/bin/uv')
    run('python',[uv,'python','install','3.12'],600)
    run('venv',[uv,'venv','--python','3.12',str(root/'venv')],60)
    py=str(root/'venv/bin/python')
    run('torch',[uv,'pip','install','--python',py,'torch==2.7.1','--index-url','https://download.pytorch.org/whl/cu126'],1800)
    run('dependencies',[uv,'pip','install','--python',py,'numpy==2.2.6','ninja==1.11.1.4','packaging==25.0','setuptools==80.9.0','wheel==0.45.1','pytest==8.4.2'],600)
    run('flashinfer',[uv,'pip','install','--python',py,'flashinfer-python==0.2.8','torch==2.7.1'],900)
    run('freeze',[uv,'pip','freeze','--python',py],60)
    run('probe',[py,'-c',"import torch,flashinfer,json,platform;print(json.dumps({'python':platform.python_version(),'torch':torch.__version__,'cuda':torch.version.cuda,'capability':torch.cuda.get_device_capability(),'flashinfer':flashinfer.__version__}))"],120)
    (root/'result.json').write_text(json.dumps({'status':'PASS','root':str(root),'python':py,'commands':commands,'new_model_calls':0},indent=2)+'\n')
    print(json.dumps({'status':'PASS','root':str(root),'python':py}))


if __name__=='__main__':main()
