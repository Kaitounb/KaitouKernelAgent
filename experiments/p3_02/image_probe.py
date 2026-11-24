"""Inventory existing images under non-root/read-only/no-network constraints."""
import argparse,json,base64
from pathlib import Path
from tools.ssh_3070 import execute_python


def main():
    p=argparse.ArgumentParser();p.add_argument('--evidence',type=Path,required=True);a=p.parse_args()
    if a.evidence.exists():p.error('Use a new directory')
    a.evidence.mkdir(parents=True)
    payload=r'''
import json,subprocess
rows=[]
for tag in ['kernel-lab-runner:attention-20261024','kernel-lab-runner:3070']:
 p=subprocess.run(['docker','image','inspect','--format','{{.Id}}',tag],capture_output=True,text=True,timeout=10)
 if p.returncode:rows.append({'tag':tag,'inspect_exit':p.returncode,'stderr':p.stderr});continue
 image=p.stdout.strip()
 script="""
import json,os,sys,platform,shutil
r={'uid':os.getuid(),'python':sys.version,'executable':sys.executable,'libc':platform.libc_ver(),'nvcc':shutil.which('nvcc'),'credential_environment_names_only':[k for k,v in os.environ.items() if v and any(s in k.upper() for s in ['KEY','TOKEN','PASS'])]}
try:
 import torch
 r.update(torch=torch.__version__,torch_cuda=torch.version.cuda,torch_path=torch.__file__)
except Exception as e:r.update(torch_error=type(e).__name__+': '+str(e)[:500])
print(json.dumps(r))
"""
 shell='command -v python || command -v python3; python -c '+__import__('shlex').quote(script)
 cmd=['docker','run','--rm','--pull=never','--network=none','--read-only','--cap-drop=ALL','--security-opt=no-new-privileges','--user','65534:65534','--pids-limit=128','--memory=2g','--cpus=2','--env','HOME=/tmp','--tmpfs','/tmp:rw,nosuid,nodev,size=64m','--entrypoint','/bin/sh',image,'-c',shell]
 q=subprocess.run(cmd,capture_output=True,text=True,timeout=60)
 rows.append({'tag':tag,'image_id':image,'command':cmd,'exit':q.returncode,'stdout':q.stdout,'stderr':q.stderr})
print(json.dumps({'scope':'CPU-only existing image compatibility probe; no GPU execution or image changes','rows':rows}))
'''
    (a.evidence/'probe.py').write_text(payload)
    r=execute_python(payload);(a.evidence/'transport.log').write_text(r.stdout+r.stderr)
    d=json.loads(r.stdout);(a.evidence/'result.json').write_text(json.dumps(d,indent=2)+'\n')
    print(json.dumps(d,indent=2))


if __name__=='__main__':main()
