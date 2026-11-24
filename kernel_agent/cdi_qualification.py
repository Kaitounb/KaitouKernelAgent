"""Existing Registry extension for evidence-bound, fixed RTX3060/nativeCDI cuBLAS."""
import hashlib,json,re
from pathlib import Path
from .registry import Registry,read_regular,sha
from .cdi_runner import IMAGE,command_result
DATA=Path(__file__).with_name('data')/'cdi'
SCOPE='RTX3060_REVIEWED_FIXED_CUBLAS_CDI'
TARGET={'gpu_uuid':'GPU-0140b895-e4e9-7c67-aba1-c7a953df1847','driver_version':'580.95.05'}
PROFILE={'power_limit_watts':170,'clock_policy':'dynamic','maximum_temperature_exclusive_c':83}
ENVIRONMENT={'image_id':IMAGE,'torch':'2.14.0+cu130','cuda':'13.0','device':'sm86'}
BINARIES={'66fcc80f9122c919f92c72e37a01794a792c91a58b71d5705b160e0bc979dfc3','93fb2f4344521b7e9b246b54d1c30d4b1a8facf731e6385e18f6119e9c49c72c'}
RUNTIME_FILES={'kernel_agent/cdi_cli.py','kernel_agent/cdi_qualification.py','kernel_agent/cdi_runner.py','kernel_agent/registry.py','kernel_agent/deployment.py','kernel_agent/runner.py','kernel_agent/pipe_worker.py','kernel_agent/protocol.py','kernel_agent/output.py','kernel_agent/bounded_io.py','kernel_agent/data/cdi/gemm_contract_v1.json','kernel_agent/data/cdi/gemm_deployment_policy_v2.json'}

def check_environment():
 image=command_result(['docker','image','inspect',IMAGE,'--format','{{.Id}}']);gpu=command_result(['nvidia-smi','-i','0','--query-gpu=uuid,driver_version,power.limit,temperature.gpu','--format=csv,noheader,nounits']);rows=[x.strip() for x in gpu['stdout'].splitlines() if x.strip()];fields=[x.strip() for x in rows[0].split(',')] if len(rows)==1 else []
 if image['exit']!=0 or image['stdout'].strip()!=IMAGE or gpu['exit']!=0 or len(fields)!=4 or fields[:2]!=list(TARGET.values()):raise ValueError('RTX3060 GPU/driver/image binding differs')
 try:valid=float(fields[2])==PROFILE['power_limit_watts'] and 0<=float(fields[3])<PROFILE['maximum_temperature_exclusive_c']
 except ValueError:valid=False
 if not valid:raise ValueError('RTX3060 requires170W profile and temperature below83C')

def index(channel='development'):
 if channel not in ['development','qualified']:raise ValueError('unknown RTX3060 release channel')
 c=None
 if channel=='qualified':
  try:
   raw=read_regular(DATA/'gemm_qualification.json',65536);c=json.loads(raw);value=json.loads(read_regular(DATA/'qualified_releases.json',65536))
  except FileNotFoundError:raise ValueError('qualified RTX3060 release is not packaged; target acceptance remains pending') from None
  flags=['independent_correctness_pass','stability_pass','nonregression_pass','installed_runtime_pass','boundary_pass','recovery_pass','rollback_pass','raw_audit_pass']
  if not isinstance(c,dict) or not isinstance(value,dict) or value.get('qualification_sha256')!=hashlib.sha256(raw).hexdigest() or c.get('schema_version')!=1 or c.get('status')!='QUALIFIED' or c.get('scope')!=SCOPE or c.get('production_release') is not True or c.get('source_execution_enabled') is not False or any(c.get(k) is not True for k in flags):raise ValueError('RTX3060 qualification certificate invalid')
  if c.get('environment')!=ENVIRONMENT or c.get('target_binding')!=TARGET or c.get('operating_profile')!=PROFILE or c.get('gpu_binding')!='nativeCDIdevice0/runc/visiblevoid' or not isinstance(c.get('binary_sha256'),list) or any(not isinstance(x,str) for x in c['binary_sha256']) or set(c['binary_sha256'])!=BINARIES or len(c['binary_sha256'])!=2:raise ValueError('RTX3060 qualification environment invalid')
  files=c.get('runtime_files')
  if not isinstance(files,dict) or set(files)!=RUNTIME_FILES or any(not isinstance(h,str) or re.fullmatch('[0-9a-f]{64}',h) is None for h in files.values()):raise ValueError('RTX3060 runtime fingerprints invalid')
  root=Path(__file__).resolve().parents[1]
  if any(hashlib.sha256(read_regular(root/n,2*1024*1024)).hexdigest()!=h for n,h in files.items()):raise ValueError('RTX3060 qualified runtime code differs')
  if c.get('deployment_policy_sha256')!=sha(DATA/'gemm_deployment_policy_v2.json'):raise ValueError('RTX3060 qualification policy differs')
 else:value=json.loads(read_regular(DATA/'releases.json',65536))
 if not isinstance(value,dict) or value.get('schema_version')!=1 or value.get('target')!='rtx3060-reviewed-cdi' or not isinstance(value.get('versions'),dict) or set(value['versions'])!={'sm86-cublas-original','sm86-cublas-container'} or value.get('default')!='sm86-cublas-container' or value.get('fallback')!='sm86-cublas-original':raise ValueError('RTX3060 catalog invalid')
 for version,meta in value['versions'].items():
  if not isinstance(meta,dict) or meta.get('version')!=version or meta.get('contract_sha256')!=sha(DATA/'gemm_contract_v1.json') or meta.get('backend')!='cublas' or meta.get('production_release') is not (channel=='qualified') or any(meta.get(k)!=v for k,v in ENVIRONMENT.items()) or meta.get('binary_sha256') not in BINARIES:raise ValueError('RTX3060 release contract/environment differs')
 if {m['binary_sha256'] for m in value['versions'].values()}!=BINARIES:raise ValueError('RTX3060 distinct artifacts required')
 return value

def registry(root,*,channel='development'):
 catalog=index(channel);check_environment();return Registry(root,channel=channel,data=DATA,catalog=catalog)
