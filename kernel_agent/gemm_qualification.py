"""Evidence-bound factory for the canonical SM86 GEMM CLI; storage/worker stay unchanged."""
import hashlib,json,re
from pathlib import Path
from . import registry as store
from .runner import IMAGE,command_result

RUNTIME_FILES={'kernel_agent/cli.py','kernel_agent/gemm_qualification.py','kernel_agent/registry.py','kernel_agent/deployment.py','kernel_agent/runner.py','kernel_agent/pipe_worker.py','kernel_agent/protocol.py','kernel_agent/output.py','kernel_agent/bounded_io.py','kernel_agent/data/gemm_deployment_policy_v2.json'}


def check_target(environment,target):
 image=command_result(['docker','image','inspect',IMAGE,'--format','{{.Id}}']);gpu=command_result(['nvidia-smi','-i','0','--query-gpu=uuid,driver_version,power.limit,temperature.gpu','--format=csv,noheader,nounits']);rows=[x.strip() for x in gpu['stdout'].splitlines() if x.strip()]
 fields=[x.strip() for x in rows[0].split(',')] if len(rows)==1 else []
 if image['exit']!=0 or image['stdout'].strip()!=environment['image_id'] or gpu['exit']!=0 or len(fields)!=4 or fields[:2]!=[target['gpu_uuid'],target['driver_version']]:raise ValueError('qualified GEMM GPU/driver/image binding differs')
 try:valid=float(fields[2])==100 and 0<=float(fields[3])<83
 except ValueError:valid=False
 if not valid:raise ValueError('qualified GEMM requires100W ceiling and GPU temperature below83C')


def registry(root,*,channel='development'):
 catalog=store.index(channel)
 if channel=='qualified':
  c=json.loads(store.read_regular(store.DATA/'gemm_qualification.json',65536));target=c.get('target_binding');files=c.get('runtime_files')
  if c.get('schema_version')!=2 or c.get('source_execution_enabled') is not False or not isinstance(target,dict) or set(target)!={'gpu_uuid','driver_version'} or any(not isinstance(x,str) or not x for x in target.values()) or not target['gpu_uuid'].startswith('GPU-'):raise ValueError('qualified GEMM target binding schema invalid')
  if not isinstance(files,dict) or set(files)!=RUNTIME_FILES or any(not isinstance(h,str) or re.fullmatch('[0-9a-f]{64}',h) is None for h in files.values()):raise ValueError('qualified GEMM runtime fingerprints invalid')
  if c.get('operating_profile')!={'power_limit_watts':100,'clock_policy':'dynamic','maximum_temperature_exclusive_c':83}:raise ValueError('qualified GEMM operating profile invalid')
  base=Path(__file__).resolve().parents[1]
  if any(hashlib.sha256(store.read_regular(base/n,2*1024*1024)).hexdigest()!=h for n,h in files.items()):raise ValueError('qualified GEMM runtime code differs')
  policy=store.read_regular(store.DATA/'gemm_deployment_policy_v2.json',65536)
  if c.get('deployment_policy_sha256')!=hashlib.sha256(policy).hexdigest():raise ValueError('qualified GEMM policy differs')
  check_target(c['environment'],target)
 return store.Registry(root,channel=channel,catalog=catalog)
