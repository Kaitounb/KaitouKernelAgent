"""Install NVIDIA Nsight Compute into a separate extracted tool directory; no system/CUDA/image upgrade."""
import json
from pathlib import Path
from tools.ssh_3060 import ROOT,execute_python

def main():
 out=ROOT/'evidence/20261124-3060-ncu-install';out.mkdir(exist_ok=False)
 payload=r'''
import gzip,hashlib,json,shutil,subprocess,tempfile,time,urllib.request
from pathlib import Path
root=Path(tempfile.mkdtemp(prefix='ncu-tools-',dir=Path.home()/'nka-persistent'));root.chmod(0o700);base='https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/';(root/'location.json').write_text(json.dumps({'root':str(root)}));raw=urllib.request.urlopen(base+'Packages.gz',timeout=60).read();(root/'Packages.gz').write_bytes(raw);packages=gzip.decompress(raw).decode();matches=[]
for block in packages.split('\n\n'):
 fields={line.split(': ',1)[0]:line.split(': ',1)[1] for line in block.splitlines() if ': ' in line and not line.startswith(' ')}
 if fields.get('Package')=='nsight-compute-2025.3.1':matches.append(fields)
assert len(matches)==1,matches;entry=matches[0];assert entry['Architecture']=='amd64';url=base+entry['Filename'].removeprefix('./');plan={'root':str(root),'official_repo':base,'index_sha256':hashlib.sha256(raw).hexdigest(),'package':entry,'url':url,'scope':'extract only; no dpkg install, no production image changes'};(root/'plan.json').write_text(json.dumps(plan,indent=2));target=root/Path(entry['Filename']).name;h=hashlib.sha256();count=0;started=time.monotonic();last=started
with urllib.request.urlopen(url,timeout=60) as src,target.open('xb') as dst:
 while True:
  block=src.read(1024*1024)
  if not block:break
  dst.write(block);h.update(block);count+=len(block)
  if time.monotonic()-last>5:(root/'progress.json').write_text(json.dumps({'bytes':count,'seconds':time.monotonic()-started}));last=time.monotonic()
assert h.hexdigest()==entry['SHA256'] and count==int(entry['Size']);prefix=root/'prefix';r=subprocess.run(['dpkg-deb','-x',str(target),str(prefix)],capture_output=True,text=True,timeout=180);(root/'extract.stdout').write_text(r.stdout);(root/'extract.stderr').write_text(r.stderr);assert r.returncode==0;candidates=list(prefix.glob('opt/nvidia/nsight-compute/*/ncu'))+list(prefix.glob('opt/nvidia/nsight-compute/*/ncu'))+list(prefix.glob('opt/nvidia/nsight-compute/*/ncu'))+list(prefix.glob('opt/nvidia/nsight-compute/*/ncu'));candidates=sorted(set(candidates))
if not candidates:candidates=[p for p in prefix.rglob('ncu') if p.is_file()]
assert len(candidates)==1;exe=candidates[0];r=subprocess.run([str(exe),'--version'],capture_output=True,text=True,timeout=30);assert r.returncode==0;v={'status':'OFFICIAL_NCU_EXTRACTED_VERSION_PASS','root':str(root),'ncu':str(exe),'tool_root':str(exe.parent),'package_archive':str(target),'package_bytes':count,'package_sha256':h.hexdigest(),'package_version':entry['Version'],'version_stdout':r.stdout,'version_stderr':r.stderr,'production_image_changed':False,'system_packages_installed':False,'gpu_work':False};(root/'result.json').write_text(json.dumps(v,indent=2));print(json.dumps(v))
'''
 (out/'driver.py').write_text(payload);r=execute_python(payload,1200);(out/'transport.stderr').write_text(r.stderr);(out/'transport.stdout').write_text(r.stdout);assert r.returncode==0;v=json.loads(r.stdout);(out/'result.json').write_text(json.dumps(v,indent=2));print(json.dumps(v))
if __name__=='__main__':main()
