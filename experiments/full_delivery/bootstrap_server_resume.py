"""Continue isolated dependency setup after a slow official wheel transfer; retain first logs."""
import argparse,hashlib,json,os,subprocess,sys,time,urllib.request
from pathlib import Path


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);a=p.parse_args();root=a.root;mirror='https://mirrors.aliyun.com/pypi/simple';commands=[]
    env={k:v for k,v in os.environ.items() if not any(s in k.upper() for s in ['API_KEY','AUTH_TOKEN','PASSWORD','SECRET'])};env.update(UV_CACHE_DIR=str(root/'uv-cache'),UV_PYTHON_INSTALL_DIR=str(root/'python'),UV_NATIVE_TLS='true',UV_INDEX_URL=mirror,PIP_CONFIG_FILE='/dev/null',CUDA_HOME='/usr/local/cuda',PATH='/usr/local/cuda/bin:'+os.environ.get('PATH',''),MAX_JOBS='4')
    def run(label,cmd,timeout=1800):
        start=time.monotonic()
        try:
            with (root/(label+'.stdout')).open('w') as out,(root/(label+'.stderr')).open('w') as err:r=subprocess.run(cmd,env=env,stdout=out,stderr=err,timeout=timeout)
            row={'label':label,'command':cmd,'exit':r.returncode,'wall_seconds':time.monotonic()-start}
        except subprocess.TimeoutExpired:
            row={'label':label,'command':cmd,'exit':None,'timeout':True,'wall_seconds':time.monotonic()-start};commands.append(row);(root/'progress-v2.json').write_text(json.dumps(commands,indent=2)+'\n');raise
        commands.append(row);(root/'progress-v2.json').write_text(json.dumps(commands,indent=2)+'\n')
        if r.returncode:raise RuntimeError(label+' failed; retained')
    run('uv-mirror',[sys.executable,'-m','pip','download','--index-url',mirror,'--no-deps','--dest',str(root/'uv-wheel'),'uv==0.9.10'],600)
    wheel=next((root/'uv-wheel').glob('*.whl'));official=json.load(urllib.request.urlopen('https://pypi.org/pypi/uv/0.9.10/json',timeout=30));expected=next(x['digests']['sha256'] for x in official['urls'] if x['filename']==wheel.name)
    if hashlib.sha256(wheel.read_bytes()).hexdigest()!=expected:raise ValueError('mirrored uv differs from official SHA')
    (root/'uv-official-sha.json').write_text(json.dumps({'filename':wheel.name,'sha256':expected},indent=2)+'\n')
    run('uv-local-install',[sys.executable,'-m','pip','install','--no-index','--no-deps','--target',str(root/'tools-v2'),str(wheel)],120);uv=str(root/'tools-v2/bin/uv')
    run('python-v2',[uv,'python','install','3.12.12'],900);run('venv-v2',[uv,'venv','--python','3.12.12',str(root/'venv')],120);py=str(root/'venv/bin/python')
    run('torch-v2',[uv,'pip','install','--python',py,'--index-url',mirror,'torch==2.7.1'],2400)
    run('dependencies-v2',[uv,'pip','install','--python',py,'--index-url',mirror,'numpy==2.2.6','ninja==1.11.1.4','packaging==25.0','setuptools==80.9.0','wheel==0.45.1','pytest==8.4.2'],900)
    run('flashinfer-v2',[uv,'pip','install','--python',py,'--index-url',mirror,'flashinfer-python==0.2.8','torch==2.7.1'],1200)
    run('freeze-v2',[uv,'pip','freeze','--python',py],60)
    run('probe-v2',[py,'-c',"import torch,flashinfer,json,platform;print(json.dumps({'python':platform.python_version(),'torch':torch.__version__,'cuda':torch.version.cuda,'capability':torch.cuda.get_device_capability(),'flashinfer':flashinfer.__version__}))"],120)
    result={'status':'PASS','root':str(root),'python':py,'commands':commands,'new_model_calls':0};(root/'result.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps({'status':'PASS','root':str(root),'python':py}))


if __name__=='__main__':main()
