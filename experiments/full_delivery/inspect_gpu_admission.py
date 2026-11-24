"""Read-only GPU/host privilege inventory; no credentials or environment dump."""
import argparse,json
from pathlib import Path

PAYLOAD = r'''
import json,os,platform,shutil,subprocess
from pathlib import Path
result={'platform':platform.platform(),'uid':os.geteuid(),'container_marker':Path('/.dockerenv').exists()}
for name in ['/proc/self/status','/proc/1/status']:
 p=Path(name)
 if p.exists():result[name]='\n'.join(x for x in p.read_text().splitlines() if x.startswith(('Name:','Uid:','Gid:','CapInh:','CapPrm:','CapEff:','CapBnd:','NoNewPrivs:','Seccomp:')))
for name in ['/proc/driver/nvidia/params','/sys/module/nvidia/parameters/NVreg_RestrictProfilingToAdminUsers']:
 try:result[name]=Path(name).read_text()
 except OSError as e:result[name]={'available':False,'errno':e.errno}
result['admin_interfaces']={n:Path(n).exists() for n in ['/var/run/docker.sock','/run/containerd/containerd.sock','/sys/kernel/security','/proc/driver/nvidia/capabilities/profiler']}
smi=shutil.which('nvidia-smi') or '/usr/lib/wsl/lib/nvidia-smi'
commands=[[smi,'-q'],[smi,'--query-compute-apps=pid,process_name,used_memory','--format=csv'],['sh','-c','findmnt -no TARGET,FSTYPE,OPTIONS / /proc /sys 2>/dev/null']]
for command in commands:
 try:
  r=subprocess.run(command,capture_output=True,text=True,timeout=30);result[' '.join(command)]={'exit':r.returncode,'stdout':r.stdout,'stderr':r.stderr}
 except OSError as e:result[' '.join(command)]={'error':str(e)}
print(json.dumps(result))
'''

def main():
 p=argparse.ArgumentParser();p.add_argument('--target',choices=['3070','4090'],required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
 if a.target=='3070':from tools.ssh_3070 import execute_python
 else:from tools.ssh_server_gpu import execute_python
 r=execute_python(PAYLOAD,120);a.output.mkdir(parents=True,exist_ok=False);(a.output/'transport.stderr').write_text(r.stderr)
 if r.returncode:raise RuntimeError('Read-only inventory transport failed')
 v=json.loads(r.stdout);(a.output/'inventory.json').write_text(json.dumps(v,indent=2)+'\n');print(json.dumps({'target':a.target,'uid':v['uid'],'container_marker':v['container_marker'],'privileges':v['/proc/self/status'],'admin_interfaces':v['admin_interfaces'],'output':str(a.output)}))
if __name__=='__main__':main()
