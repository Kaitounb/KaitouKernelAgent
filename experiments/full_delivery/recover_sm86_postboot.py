"""Restore exact archived runtime bytes to reboot-persistent project paths; no new build."""
import base64,hashlib,json
from pathlib import Path
from tools.ssh_3070 import ROOT,execute_python

def main():
 output=ROOT/'evidence/20261123-hags-postboot-recovery';output.mkdir(exist_ok=False);wheel=ROOT/'artifacts/wheels/nvidia_kernel_agent-0.1.0.dev6-py3-none-any.whl';aot=next((ROOT/'artifacts/operator-runtime').glob('sm86-cfae*.zip'));assert hashlib.sha256(wheel.read_bytes()).hexdigest()=='d5f4bbf83dc050271af66baf35738f3800d02c2592b6fec101457122edc4fb38';assert hashlib.sha256(aot.read_bytes()).hexdigest()=='939bf2d6d71a8d47fff652f944addb16857167e9f4ffbabd83c37a7e9a5ae5fa';controls=json.loads((ROOT/'evidence/20261123-cublas-sustained-qualification/protocol.json').read_text())['installed_controls']
 payload='FILES='+repr({wheel.name:base64.b64encode(wheel.read_bytes()).decode(),aot.name:base64.b64encode(aot.read_bytes()).decode()})+'\nCONTROLS='+repr(controls)+'\n'+r'''
import base64,fcntl,hashlib,json,os,subprocess,tempfile,zipfile
from pathlib import Path
base=Path('/home/dministrator/cuda-lab/nka-persistent');base.mkdir(mode=0o700,exist_ok=True);root=Path(tempfile.mkdtemp(prefix='postboot-recovery-',dir=base));lock=open('/tmp/nka-reviewed-gpu-execution.lock','a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
for n,v in FILES.items():p=root/n;p.write_bytes(base64.b64decode(v));p.chmod(0o600)
wheel=next(root.glob('*.whl'));archive=next(root.glob('*.zip'));sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest();assert sha(wheel)=='d5f4bbf83dc050271af66baf35738f3800d02c2592b6fec101457122edc4fb38' and sha(archive)=='939bf2d6d71a8d47fff652f944addb16857167e9f4ffbabd83c37a7e9a5ae5fa';prefix=base/'dev6-d5f4bbf83dc0/prefix';assert not prefix.exists();prefix.parent.mkdir(mode=0o700,exist_ok=False);py='/home/dministrator/cuda-lab/.venv/bin/python';env={k:v for k,v in os.environ.items() if k in ['HOME','PATH','LANG','LD_LIBRARY_PATH']};env.update(OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1')
r=subprocess.run([py,'-m','pip','install','--no-index','--no-deps','--target',str(prefix),str(wheel)],capture_output=True,text=True,env=env,timeout=120);(root/'install.stdout').write_text(r.stdout);(root/'install.stderr').write_text(r.stderr);assert r.returncode==0 and all(sha(prefix/n)==h for n,h in CONTROLS.items())
bundle=base/'operators-cfae38ce5c0f';bundle.mkdir(mode=0o700,exist_ok=False)
with zipfile.ZipFile(archive) as z:
 for name in z.namelist():
  p=Path(name);assert not p.is_absolute() and '..' not in p.parts;target=bundle/p
  if name.endswith('/'):target.mkdir(parents=True,exist_ok=True)
  else:target.parent.mkdir(parents=True,exist_ok=True);target.write_bytes(z.read(name))
m=json.loads((bundle/'aot-manifest.json').read_text());assert m['bundle_id']=='cfae38ce5c0fac56545207fe93588c1193646886fc5952f113fab487ef0be6eb' and all(sha(bundle/n)==h for n,h in m['files'].items())
result={'status':'RESTORED_EXACT_ARCHIVED_BYTES','remote_root':str(root),'installed_prefix':str(prefix),'bundle':str(bundle),'wheel_sha256':sha(wheel),'aot_archive_sha256':sha(archive),'installed_controls_verified':len(CONTROLS),'aot_files_verified':len(m['files']),'aot_manifest_sha256':sha(bundle/'aot-manifest.json'),'persistent_outside_tmp':True,'new_build':False,'new_gpu_evaluations':0,'heldout_executed':False,'new_deepseek_calls':0,'production_release':False};(root/'recovery.json').write_text(json.dumps(result,indent=2));print(json.dumps(result))
'''
 (output/'driver.py').write_text(payload);r=execute_python(payload,300);(output/'transport.stderr').write_text(r.stderr)
 if r.returncode:(output/'transport.stdout').write_text(r.stdout);raise RuntimeError('Inspect existing persistent recovery; no blind reinstall')
 v=json.loads(r.stdout);(output/'result.json').write_text(json.dumps(v,indent=2)+'\n');print(json.dumps(v))
if __name__=='__main__':main()
