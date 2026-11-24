"""Clean non-root container build through the pinned KernelBench path, then runtime faults."""
import argparse,base64,hashlib,json
from pathlib import Path
from tools.ssh_3070 import execute_python


def main():
    p=argparse.ArgumentParser();p.add_argument('--evidence',type=Path,required=True);a=p.parse_args()
    if a.evidence.exists():p.error('Use a new evidence directory')
    a.evidence.mkdir(parents=True);root=Path(__file__).resolve().parents[2]
    paths=['evaluators/gemm_regression.py','kernels/gemm/cublas_sm86.cu','kernels/gemm/wmma_sm86.cu','kernels/gemm/gemm_contract.cuh',
      'experiments/p3_02/build_entry.py','experiments/p3_02/test_output.py','experiments/p3_02/worker.py','experiments/p3_02/supervisor.py']
    sources={rel:(root/rel).read_bytes() for rel in paths}
    upstream=root/'provenance/upstream-reviews/20261121-p1-02/ScalingIntelligence__KernelBench/src/kernelbench'
    for path in upstream.glob('*.py'):sources['upstream/kernelbench/'+path.name]=path.read_bytes()
    for rel,data in sources.items():
        target=a.evidence/'sources'/rel;target.parent.mkdir(parents=True,exist_ok=True);target.write_bytes(data)
    encoded={rel:base64.b64encode(data).decode() for rel,data in sources.items()}
    payload='SOURCES='+repr(encoded)+'\n'+r'''
import base64,hashlib,json,os,shutil,stat,subprocess,tempfile,time,uuid
from pathlib import Path
root=Path(tempfile.mkdtemp(prefix='nka-p3-02-build-'));root.chmod(0o755);source=root/'source';source.mkdir();source.chmod(0o755)
for name,data in SOURCES.items():
 target=source/name;target.parent.mkdir(parents=True,exist_ok=True);target.write_bytes(base64.b64decode(data))
# Original dependency directory is mode0700; copy its public packages for this UID only.
deps=root/'deps';shutil.copytree('/tmp/nka-p1-02-pydeps-hw9k37vu',deps,ignore=shutil.ignore_patterns('__pycache__'))
deps.chmod(0o755)
dep_hashes={}
for item in deps.rglob('*'):
 if item.is_dir():item.chmod(0o755)
 elif item.is_file():
  item.chmod(0o644);dep_hashes[str(item.relative_to(deps))]=hashlib.sha256(item.read_bytes()).hexdigest()
import ninja
ninja_source=Path(ninja.BIN_DIR)/'ninja'
if not ninja_source.is_file():raise RuntimeError('existing host Ninja binary unavailable')
(deps/'bin').mkdir();shutil.copyfile(ninja_source,deps/'bin/ninja');(deps/'bin/ninja').chmod(0o755)
dep_hashes['bin/ninja']=hashlib.sha256(ninja_source.read_bytes()).hexdigest()
(root/'dependency-manifest.json').write_text(json.dumps(dep_hashes,indent=2)+'\n')
build=root/'build';build.mkdir();build.chmod(0o777)
image='sha256:ce00f6dc2eea43dc508b90b548f624646d643a92d363aa715fdb5a3b4f49290a';name='nka-builder-'+uuid.uuid4().hex[:12]
cmd=['docker','run','--name',name,'--rm','--pull=never','--network=none','--read-only','--cap-drop=ALL','--security-opt=no-new-privileges','--user','65534:65534','--init',
 '--pids-limit=128','--memory=8g','--cpus=4','--env','HOME=/tmp','--env','PYTHONDONTWRITEBYTECODE=1','--env','OMP_NUM_THREADS=1','--env','MKL_NUM_THREADS=1',
 '--env','PATH=/deps/bin:/usr/local/cuda/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin','--env','MAX_JOBS=1','--env','TORCH_CUDA_ARCH_LIST=8.6','--env','CUDA_HOME=/usr/local/cuda','--env','TORCH_EXTENSIONS_DIR=/build/extensions',
 '--env','PYTHONPATH=/source:/source/upstream:/deps','--tmpfs','/tmp:rw,nosuid,nodev,size=512m',
 '--mount',f'type=bind,src={source},dst=/source,readonly','--mount',f'type=bind,src={deps},dst=/deps,readonly',
 '--mount',f'type=bind,src={build},dst=/build','--entrypoint','/opt/venv/bin/python',image,'/source/experiments/p3_02/build_entry.py']
started=time.monotonic();timed_out=False
with (root/'build.stdout.log').open('w') as out,(root/'build.stderr.log').open('w') as err:
 proc=subprocess.Popen(cmd,stdout=out,stderr=err)
 try:proc.wait(timeout=1200)
 except subprocess.TimeoutExpired:
  timed_out=True;subprocess.run(['docker','rm','-f',name],capture_output=True);proc.wait(timeout=15)
record={'command':cmd,'exit':proc.returncode,'timed_out':timed_out,'wall_seconds':time.monotonic()-started,'container_removed':subprocess.run(['docker','inspect',name],capture_output=True).returncode!=0,
 'source_sha256':{rel:hashlib.sha256((source/rel).read_bytes()).hexdigest() for rel in SOURCES},'network':'none','gpu_exposed':False}
(root/'build-execution.json').write_text(json.dumps(record,indent=2)+'\n')
if proc.returncode==0:
 meta=json.loads((build/'build.json').read_text());relative=Path(meta['binary']).relative_to('/build');binary=build/relative
 if hashlib.sha256(binary.read_bytes()).hexdigest()!=meta['binary_sha256']:raise ValueError('build output hash mismatch')
 code=root/'runtime-code';code.mkdir();shutil.copyfile(source/'experiments/p3_02/supervisor.py',code/'supervisor.py');shutil.copyfile(source/'experiments/p3_02/worker.py',code/'worker.py')
 script='import sys;sys.path.insert(0,'+repr(str(code))+');import supervisor;supervisor.BINARY='+repr(str(binary))+';supervisor.EXPECTED='+repr(meta['binary_sha256'])+';sys.argv=["supervisor","--work",'+repr(str(root/'run'))+'];supervisor.main()'
 with (root/'runtime.stdout.log').open('w') as out,(root/'runtime.stderr.log').open('w') as err:
  runtime=subprocess.run(['/home/dministrator/cuda-lab/.venv/bin/python','-c',script],stdout=out,stderr=err,timeout=900)
  shutil.copyfile(source/'experiments/p3_02/test_output.py',code/'test_output.py')
 with (root/'decoder-tests.stdout.log').open('w') as out,(root/'decoder-tests.stderr.log').open('w') as err:
  tested=subprocess.run(['/home/dministrator/cuda-lab/.venv/bin/python',str(code/'test_output.py'),'-v'],stdout=out,stderr=err,timeout=60)
 (root/'decoder-tests.json').write_text(json.dumps({'exit':tested.returncode,'scope':'CPU-only hostile output-file decoder tests; no CUDA work'},indent=2))
 (root/'runtime-execution.json').write_text(json.dumps({'exit':runtime.returncode,'script':script},indent=2))
files={};entries=[]
for item in root.rglob('*'):
 rel=item.relative_to(root)
 if any(part in ['control','artifact','source','runtime-code'] for part in rel.parts):continue
 info=item.lstat()
 if stat.S_ISLNK(info.st_mode):entries.append({'path':str(rel),'type':'symlink','target':str(item.readlink())});continue
 if not stat.S_ISREG(info.st_mode):continue
 if item.name not in ['decoder-tests.json','decoder-tests.stdout.log','decoder-tests.stderr.log','dependency-manifest.json','build.json','build-execution.json','runtime-execution.json','build.stdout.log','build.stderr.log','runtime.stdout.log','runtime.stderr.log','result.json','result.partial.json','stdout.log','stderr.log','worker.json','ready.json','request.json','result.npy','build.ninja','main.cpp','cuda.cu']:continue
 if info.st_size>8*1024*1024:raise ValueError('evidence too large')
 data=item.read_bytes();files[str(rel)]=base64.b64encode(data).decode();entries.append({'path':str(rel),'sha256':hashlib.sha256(data).hexdigest(),'bytes':len(data)})
print(json.dumps({'remote_work':str(root),'build_exit':proc.returncode,'files':files,'entries':entries}))
'''
    (a.evidence/'remote-driver.py').write_text(payload)
    r=execute_python(payload,timeout=2200);(a.evidence/'transport.log').write_text(r.stdout+r.stderr)
    if r.returncode:raise RuntimeError('transport failed; evidence retained')
    data=json.loads(r.stdout)
    for name,encoded in data.pop('files').items():
        rel=Path(name)
        if rel.is_absolute() or '..' in rel.parts:raise ValueError('invalid evidence path')
        target=a.evidence/'remote'/rel;target.parent.mkdir(parents=True,exist_ok=True);target.write_bytes(base64.b64decode(encoded))
    (a.evidence/'execution.json').write_text(json.dumps(data,indent=2)+'\n')
    print(json.dumps({k:data[k] for k in ['remote_work','build_exit']}))
    if data['build_exit']:raise RuntimeError('builder failed; inspect preserved build logs')
    runtime=json.loads((a.evidence/'remote/run/result.json').read_text());print(json.dumps({'status':runtime['status'],'cases':len(runtime['cases']),'failures':runtime['failures']}))


if __name__=='__main__':main()
