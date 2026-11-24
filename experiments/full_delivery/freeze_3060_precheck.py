"""Freeze trusted new-target public correctness checks before their GPU execution."""
import json
from tools.ssh_3060 import ROOT,execute_python

def main():
 out=ROOT/'evidence/20261124-sm86-3060-counterpart';loc=json.loads((out/'location.json').read_text());payload='LOCATION='+repr(loc)+'\n'+r'''
import hashlib,json,subprocess
from pathlib import Path
loc=LOCATION;root=Path(loc['root']);proof=json.loads((root/'gpu-probe-result.json').read_text());assert proof['status']=='GPU_ABI_BOUNDARY_PASS' and proof['container_isolation_verified'];assert not (root/'precheck-protocol.json').exists()
code='import json,sys;sys.path.insert(0,'+repr(str(root))+');import host;print(json.dumps(host.environment()))'
cmd=['docker','run','--rm','--pull=never','--runtime=runc','--device','nvidia.com/gpu=0','--env','NVIDIA_VISIBLE_DEVICES=void','--network=none','--read-only','--cap-drop=ALL','--security-opt=no-new-privileges','--user',str(loc['uid'])+':'+str(loc['gid']),'--tmpfs','/tmp:rw,nosuid,nodev,size=128m','--mount','type=bind,src='+str(root)+',dst='+str(root)+',readonly','--entrypoint','/opt/venv/bin/python',loc['image_id'],'-B','-c',code];r=subprocess.run(cmd,capture_output=True,text=True,timeout=60);assert r.returncode==0,r.stderr;env=json.loads(r.stdout);assert env['torch']=='2.14.0+cu130' and env['cuda']=='13.0' and env['capability']==[8,6] and env['gpu_identity']=='GPU-d464a726-39c2-9ad5-d856-f5d00c1fa870, NVIDIA GeForce RTX 3060, 580.95.05'
files=[p for p in (root/'prefix').rglob('*') if p.is_file() and '__pycache__' not in p.parts]+[root/n for n in ['host.py','counterpart_support.py','ordinary_host.py','runtime-binding.json','artifact-binding.json','gpu-probe-result.json','workloads/gemm_deployment_stability_v2.json']];files +=[p for p in (root/'probe_runtime').rglob('*') if p.is_file() and '__pycache__' not in p.parts];sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest();controlled={str(p.relative_to(root)):sha(p) for p in files}
protocol={'schema_version':1,'scope':'RTX3060 nativeLinux public fixedcuBLAS counterpart precheck; not production qualification or old3070 rerun','environment':env,'image_id':loc['image_id'],'gpu_binding':'nativeCDIdevice0','host_sha256':sha(root/'host.py'),'controlled_files':controlled,'policy_sha256':sha(root/'workloads/gemm_deployment_stability_v2.json'),'public_qa_expected':320,'sanitizer_output_checks_expected':32,'recovery_fixtures':['crash','finiteCUDAthenhosttimeout'],'new_compilation':False,'new_model_calls':0,'heldout_read':False,'formal_event_segments':0,'production_release':False};(root/'precheck-protocol.json').write_text(json.dumps(protocol,indent=2));receipt={'status':'PRECHECK_FROZEN','root':str(root),'protocol_sha256':sha(root/'precheck-protocol.json'),'environment':env,'controlled_files':len(controlled),'actual_loaded_numerical_libraries':{k:v for k,v in proof['metadata'].items() if k.startswith('lib_')}};(root/'precheck-freeze.json').write_text(json.dumps(receipt,indent=2));print(json.dumps(receipt))
'''
 with (out/'precheck-freeze-driver.py').open('x') as f:f.write(payload)
 r=execute_python(payload,120);(out/'precheck-freeze.stderr').write_text(r.stderr);assert r.returncode==0;v=json.loads(r.stdout);(out/'precheck-freeze.json').write_text(json.dumps(v,indent=2));print(json.dumps(v))
if __name__=='__main__':main()
