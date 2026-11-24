"""Copy public Python dependencies into a separate root-owned worker-readable tree."""
import json
from tools.ssh_server_gpu import ROOT,execute_python


def main():
    output=ROOT/'evidence/20261123-sm89-worker-runtime';output.mkdir(exist_ok=False)
    payload=r'''
import hashlib,json,os,shutil,stat,subprocess,tempfile,time
from pathlib import Path
source=Path('/tmp/nka-full-server-dqnrbkzs');base=Path(tempfile.mkdtemp(prefix='nka-sm89-worker-',dir='/data'));started=time.monotonic()
original={str(p):oct(stat.S_IMODE(p.stat().st_mode)) for p in [source,source/'venv',source/'python']}
for name in ['venv','python']:shutil.copytree(source/name,base/name,symlinks=True)
for p in base.rglob('*'):
 if p.is_symlink():
  target=os.readlink(p)
  if target.startswith(str(source)+'/'):p.unlink();p.symlink_to(str(base)+target[len(str(source)):])
cfg=base/'venv/pyvenv.cfg';cfg.write_text(cfg.read_text().replace(str(source),str(base)))
for p in [base,*base.rglob('*')]:
 if p.is_dir() and not p.is_symlink():os.chown(p,0,65534);p.chmod(0o750)
assert original=={str(p):oct(stat.S_IMODE(p.stat().st_mode)) for p in [source,source/'venv',source/'python']}
files={};links={};total=0
for p in base.rglob('*'):
 if p.is_symlink():
  links[str(p.relative_to(base))]=os.readlink(p);assert p.resolve().is_relative_to(base)
 elif p.is_file():
  h=hashlib.sha256()
  with p.open('rb') as f:
   for chunk in iter(lambda:f.read(1024*1024),b''):h.update(chunk)
  files[str(p.relative_to(base))]={'sha256':h.hexdigest(),'bytes':p.stat().st_size};total+=p.stat().st_size
manifest=base/'software-manifest.json';manifest.write_text(json.dumps({'files':files,'links':links,'scope':'public runtime software only; cloned before package installation'},indent=2));os.chown(manifest,0,65534);manifest.chmod(0o640)
py=base/'venv/bin/python';cmd=['/usr/bin/setpriv','--reuid=65534','--regid=65534','--clear-groups','--no-new-privs','--bounding-set=-all','--inh-caps=-all','--ambient-caps=-all',str(py),'-I','-B','-c',"import json,os,torch;print(json.dumps({'uid':os.getuid(),'torch':torch.__version__,'cuda':torch.version.cuda}))"]
r=subprocess.run(cmd,capture_output=True,text=True,timeout=60,env={'PATH':'/usr/bin:/bin','HOME':'/nonexistent','OMP_NUM_THREADS':'1','OPENBLAS_NUM_THREADS':'1'})
result={'status':'PUBLIC_RUNTIME_READY' if r.returncode==0 else 'IMPORT_FAILED','worker_root':str(base),'python':str(py),'python_sha256':files[str(py.resolve().relative_to(base))]['sha256'],'manifest_sha256':hashlib.sha256(manifest.read_bytes()).hexdigest(),'files':len(files),'links':len(links),'software_bytes':total,'copy_and_verify_wall_seconds':time.monotonic()-started,'old_directory_modes':original,'old_permissions_unchanged':True,'uid_probe_exit':r.returncode,'uid_probe_stdout':r.stdout,'uid_probe_stderr':r.stderr,'new_model_calls':0,'gpu_execution':False,'production_release':False}
(base/'setup-receipt.json').write_text(json.dumps(result,indent=2));print(json.dumps(result))
'''
    (output/'driver.py').write_text(payload);r=execute_python(payload,600,connection_path=ROOT/'.private/server-gpu-next-connection.json');(output/'transport.stderr').write_text(r.stderr)
    if r.returncode:
        (output/'transport.stdout').write_text(r.stdout);raise RuntimeError('Inspect existing copied runtime; do not repeat blindly')
    value=json.loads(r.stdout);(output/'result.json').write_text(json.dumps(value,indent=2));print(json.dumps(value))


if __name__=='__main__':main()
