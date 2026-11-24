"""Evidence-bound admission for reviewed fixed SM86 operators, separate from development."""
import hashlib,json,re
from pathlib import Path
from kernel_agent.registry import read_regular
from kernel_agent.runner import IMAGE,command_result

DATA=Path(__file__).parent
SCOPE='SM86_REVIEWED_FIXED_ATTENTION_MLA'
RUNTIME_FILES={
    'kernel_agent/operators/cli.py','kernel_agent/operators/qualification.py',
    'kernel_agent/operators/runner.py','kernel_agent/operators/pipe_worker.py',
    'kernel_agent/operators/array_io.py','kernel_agent/operators/runtime_contract.py',
    'kernel_agent/operators/flashinfer_backend.py','kernel_agent/operators/partition.py',
    'kernel_agent/bounded_io.py','kernel_agent/protocol.py',
    'kernel_agent/optimization/source_packets.py','kernel_agent/runner.py',
}


def certificate():
    try:
        raw=read_regular(DATA/'qualification.json',65536)
        seal=json.loads(read_regular(DATA/'qualified_catalog.json',65536))
    except FileNotFoundError:
        raise ValueError('qualified SM86 operator release is not packaged; acceptance remains pending') from None
    c=json.loads(raw)
    flags=['independent_correctness_pass','installed_runtime_pass','recovery_pass','rollback_pass','boundary_pass','stability_pass','raw_audit_pass']
    if not isinstance(c,dict) or not isinstance(seal,dict) or seal.get('qualification_sha256')!=hashlib.sha256(raw).hexdigest() or c.get('status')!='QUALIFIED' or c.get('scope')!=SCOPE or c.get('production_release') is not True or c.get('source_execution_enabled') is not False or any(c.get(k) is not True for k in flags):
        raise ValueError('SM86 operator qualification certificate invalid')
    env=c.get('environment');files=c.get('runtime_files');bundle=c.get('bundle_id')
    if not isinstance(bundle,str) or re.fullmatch('[0-9a-f]{64}',bundle) is None or c.get('operations')!=['attention','mla']:
        raise ValueError('SM86 operator qualification scope invalid')
    if not isinstance(env,dict) or set(env)!={'image_id','gpu_uuid','driver_version','torch','cuda','capability'} or env.get('image_id')!=IMAGE or env.get('torch')!='2.14.0+cu130' or env.get('cuda')!='13.0' or env.get('capability')!=[8,6] or not isinstance(env.get('gpu_uuid'),str) or not env['gpu_uuid'].startswith('GPU-') or not isinstance(env.get('driver_version'),str) or not env['driver_version']:
        raise ValueError('SM86 operator qualification environment invalid')
    if not isinstance(files,dict) or set(files)!=RUNTIME_FILES or any(not isinstance(h,str) or re.fullmatch('[0-9a-f]{64}',h) is None for h in files.values()):
        raise ValueError('SM86 operator qualification runtime fingerprints invalid')
    root=DATA.parents[1]
    for name,h in files.items():
        if hashlib.sha256(read_regular(root/name,2*1024*1024)).hexdigest()!=h:
            raise ValueError('SM86 qualified operator runtime code differs')
    return c


def check_environment(expected):
    image=command_result(['docker','image','inspect',IMAGE,'--format','{{.Id}}'])
    gpu=command_result(['nvidia-smi','-i','0','--query-gpu=uuid,driver_version','--format=csv,noheader'])
    rows=[x.strip() for x in gpu['stdout'].splitlines() if x.strip()]
    if image['exit']!=0 or image['stdout'].strip()!=expected['image_id'] or gpu['exit']!=0 or len(rows)!=1 or [x.strip() for x in rows[0].split(',')]!=[expected['gpu_uuid'],expected['driver_version']]:
        raise ValueError('SM86 qualified operator GPU/driver/image binding differs')


def approve(bundle_id,manifest):
    c=certificate()
    if bundle_id!=c['bundle_id'] or any(manifest.get(k)!=c['environment'][k] for k in ['torch','cuda','capability']):
        raise ValueError('SM86 qualification bundle differs')
    check_environment(c['environment'])
    return c


def check_request(c,request):
    if request.get('operation') not in c['operations']:
        raise ValueError('operation is outside qualified release scope')
    if request['operation']=='mla' and (request.get('page_size')!=16 or request.get('splits')!=1):
        raise ValueError('qualified MLA requires reviewed default page16/split1')
