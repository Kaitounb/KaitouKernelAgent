"""Freeze before all Event sampling and execute one native trusted SM89 qualification."""
import hashlib,json
from tools.ssh_server_gpu import ROOT,execute_python


def main():
    output=ROOT/'evidence/20261123-sm89-cublas-v2';reservation=output/'qualification-reservation';reservation.mkdir(exist_ok=False)
    location=json.loads((output/'location.json').read_text());pre=json.loads((output/'precheck.json').read_text());assert pre['exit']==0 and pre['result']['status']=='PRECHECK_PASS'
    payload='ROOT='+repr(location['remote_root'])+'\nAUDIT_SHA='+repr(hashlib.sha256((ROOT/'experiments/full_delivery/audit_sm89_cublas.py').read_bytes()).hexdigest())+'\n'+r'''
import hashlib,json,os,subprocess
from pathlib import Path
root=Path(ROOT);assert not (root/'protocol.json').exists() and not any(root.glob('aa-*')) and not any(root.glob('ab-*'))
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
pre=json.loads((root/'precheck-result.json').read_text());assert pre['status']=='PRECHECK_PASS' and pre['qa_checks']==320 and pre['raw_segments']==0
assert len(pre['sanitizers'])==4 and all(r['checks']==8 and r['errors']==0 for r in pre['sanitizers']) and len(pre['recovery'])==2
files={}
for directory in ['prefix','qualification','kernels','upstream','experiments']:
 for p in (root/directory).rglob('*'):
  if p.is_file() and '__pycache__' not in p.parts:files[str(p.relative_to(root))]=sha(p)
build=json.loads((root/'build-result.json').read_text())
for info in build['builds'].values():
 p=Path(info['binary']);assert sha(p)==info['binary_sha256'];files[str(p.relative_to(root))]=sha(p)
protocol={'schema_version':2,'scope':'Trusted fixed SM89 cuBLAS sustained-average stability and nonregression; no free source/NCU/production certificate','remote_root':str(root),'environment':pre['environment'],'policy_sha256':sha(root/'workloads/gemm_deployment_stability_v2.json'),'host_sha256':sha(root/'experiments/full_delivery/sm89_cublas_host.py'),'audit_source_sha256':AUDIT_SHA,'controlled_files':files,'build_result_sha256':sha(root/'build-result.json'),'precheck_result_sha256':sha(root/'precheck-result.json'),'binary_sha256':{r:info['binary_sha256'] for r,info in build['builds'].items()},'aa_sessions':3,'conditional_ab_sessions':3,'workload_count':8,'input_counts':[1,16],'pairs_per_group':9,'graph_launches':32,'target_segment_ms':1000,'pilot':{'samples':3,'replays':4,'formula':'clamp(ceil(4000/median_pilot_elapsed_ms),1,8192)'},'aa_pair_limit':.03,'aa_group_limit':.015,'deployment_nonregression_guard':.97,'gpu_lock':'/tmp/nka-reviewed-gpu-execution.lock','admission':'three consecutive utilization0 before each session; no killing user workloads','automatic_retry':False,'clock_control':'none; ordinary Linux dynamic clocks recorded','container_isolation_verified':False,'heldout_read':False,'new_model_calls':0,'runtime_activation':False,'production_release':False}
(root/'protocol.json').write_text(json.dumps(protocol,indent=2));freeze={'protocol_sha256':sha(root/'protocol.json'),'host_sha256':protocol['host_sha256'],'audit_source_sha256':AUDIT_SHA};(root/'protocol.freeze.json').write_text(json.dumps(freeze,indent=2))
print(json.dumps({'remote_root':str(root),'protocol':protocol,'freeze':freeze}))
'''
    (reservation/'freeze-driver.py').write_text(payload);r=execute_python(payload,120,connection_path=ROOT/'.private/server-gpu-next-connection.json');(reservation/'freeze.stderr').write_text(r.stderr)
    if r.returncode:raise RuntimeError('Freeze failed: inspect reservation, no automatic retry')
    frozen=json.loads(r.stdout);(output/'protocol.json').write_text(json.dumps(frozen['protocol'],indent=2));(output/'protocol.freeze.json').write_text(json.dumps(frozen['freeze'],indent=2));print(json.dumps({'status':'FROZEN_BEFORE_SAMPLING','remote_root':location['remote_root'],'protocol_sha256':frozen['freeze']['protocol_sha256']}),flush=True)
    payload='ROOT='+repr(location['remote_root'])+'\n'+r'''
import json,os,subprocess
from pathlib import Path
root=Path(ROOT);py='/tmp/nka-full-server-dqnrbkzs/venv/bin/python';assert not (root/'qualification-result.json').exists() and not any(root.glob('aa-*'))
env={k:v for k,v in os.environ.items() if k in ['HOME','LANG','LD_LIBRARY_PATH']};env.update(PATH='/usr/local/cuda-12.6/bin:'+str(Path(py).parent)+':/usr/local/bin:/usr/bin:/bin',OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1',PYTHONDONTWRITEBYTECODE='1')
with (root/'qualification.stdout').open('w') as out,(root/'qualification.stderr').open('w') as err:r=subprocess.run([py,str(root/'experiments/full_delivery/sm89_cublas_host.py'),'--root',str(root),'--phase','qualification'],cwd=root,env=env,stdout=out,stderr=err,timeout=3600)
print(json.dumps({'remote_root':str(root),'exit':r.returncode,'result':json.loads((root/'qualification-result.json').read_text()) if (root/'qualification-result.json').exists() else None}))
'''
    (reservation/'run-driver.py').write_text(payload);r=execute_python(payload,3900,connection_path=ROOT/'.private/server-gpu-next-connection.json');(reservation/'transport.stderr').write_text(r.stderr);(reservation/'transport.stdout').write_text(r.stdout)
    if r.returncode:raise RuntimeError('Qualification transport failed; inspect existing sampled identity, no relaunch')
    v=json.loads(r.stdout);(output/'qualification.json').write_text(json.dumps(v,indent=2));print(json.dumps({k:v for k,v in v.items() if k!='result'}|{'result':{k:v['result'][k] for k in ['status','aa_sessions','ab_sessions','raw_segments','deployment_decision','production_release']} if v['result'] else None}))


if __name__=='__main__':main()
