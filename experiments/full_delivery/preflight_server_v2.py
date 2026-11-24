"""Read-only inventory for the user-supplied next server; no model or GPU benchmark."""
import json
from tools.ssh_server_gpu import ROOT, execute_python


def main():
    output = ROOT / 'evidence/20261123-sm89-v2-preflight'
    output.mkdir(exist_ok=False)
    payload = r'''
import json,os,platform,shutil,subprocess
from pathlib import Path
def run(args,timeout=30):
 try:
  r=subprocess.run(args,capture_output=True,text=True,timeout=timeout)
  return {'args':args,'exit':r.returncode,'stdout':r.stdout[:16000],'stderr':r.stderr[:3000]}
 except Exception as e:return {'args':args,'error':type(e).__name__}
commands={}
for name,args in [
 ('gpu',['nvidia-smi','--query-gpu=uuid,name,driver_version,memory.total,memory.used,utilization.gpu,temperature.gpu,clocks.sm,clocks.mem,pstate,power.limit','--format=csv,noheader']),
 ('compute_processes',['nvidia-smi','--query-compute-apps=pid,process_name,used_memory','--format=csv,noheader']),
 ('nvcc',['nvcc','--version']),
 ('disk',['df','-h','/tmp','/root']),
 ('docker',['docker','version','--format','{{json .Server}}'])]:commands[name]=run(args)
candidates=['/tmp/nka-full-server-dqnrbkzs/venv/bin/python',shutil.which('python3')]
versions={}
for py in dict.fromkeys(p for p in candidates if p and Path(p).exists()):
 versions[py]=run([py,'-c',"import json,platform;v={'python':platform.python_version()};exec(\"try:\\n import torch;v.update(torch=torch.__version__,cuda=torch.version.cuda)\\nexcept ImportError:v.update(torch=None)\");print(json.dumps(v))"])
status=Path('/proc/self/status').read_text();caps={line.split(':')[0]:line.split(':')[1].strip() for line in status.splitlines() if line.startswith(('CapEff:','CapBnd:'))}
paths={name:shutil.which(name) for name in ['python3','nvcc','ncu','compute-sanitizer','docker','unshare','ninja']}
old_roots=[str(p) for p in Path('/tmp').glob('nka-*') if p.is_dir()]
print(json.dumps({'platform':platform.platform(),'os_release':Path('/etc/os-release').read_text(),'boot_id':Path('/proc/sys/kernel/random/boot_id').read_text().strip(),'cpu_count':os.cpu_count(),'capabilities':caps,'container_marker':Path('/.dockerenv').exists(),'commands':commands,'paths':paths,'python_versions':versions,'existing_project_roots':old_roots,'gpu_benchmark_executed':False,'new_model_calls':0,'production_release':False}))
'''
    (output / 'driver.py').write_text(payload)
    r = execute_python(payload, 150, connection_path=ROOT / '.private/server-gpu-next-connection.json')
    (output / 'transport.stderr').write_text(r.stderr)
    if r.returncode:
        (output / 'transport.stdout').write_text(r.stdout)
        raise RuntimeError('Preflight connection failed; inspect transport without changing host trust')
    value = json.loads(r.stdout)
    (output / 'result.json').write_text(json.dumps(value, indent=2) + '\n')
    print(json.dumps(value))


if __name__ == '__main__':
    main()
