"""Reserve/freeze the first backend-specific deployed cuBLAS final pairing."""
import argparse,base64,hashlib,json,zipfile
from pathlib import Path
from tools.ssh_3070 import ROOT,execute_python


def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def main():
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);a=p.parse_args();a.output.mkdir(exist_ok=False);wheel=ROOT/'artifacts/wheels/nvidia_kernel_agent-0.1.0.dev6-py3-none-any.whl';assert sha(wheel)=='d5f4bbf83dc050271af66baf35738f3800d02c2592b6fec101457122edc4fb38';names=['kernel_agent/optimization/'+n for n in ['source_runner.py','source_worker.py','fixtures.py','source_packets.py','pair_worker.py','source_timing.py','evaluate.py','cases.py','identity.py']]+['kernel_agent/protocol.py','kernel_agent/bounded_io.py','kernel_agent/runner.py','kernel_agent/operators/measurement.py','kernel_agent/optimization/assets/workloads/gemm_development_v1.json','kernel_agent/optimization/assets/workloads/gemm_contract_v1.json'];host=ROOT/'experiments/full_delivery/cublas_locked_qualification_host.py'
    with zipfile.ZipFile(wheel) as z:controls={n:hashlib.sha256(z.read(n)).hexdigest() for n in names}
    protocol={'repair_evidence_sha256':sha(ROOT/'evidence/20261123-admission-repair/3070-clock-load-verification.json'),'original_failed_protocol_sha256':'8cc50f882ef18a00e538d0cf4b78cabebff7ba2e5bad115c55c23c372d7da0da','audit_source_sha256':sha(ROOT/'experiments/full_delivery/audit_cublas_locked_qualification.py'),'clock_environment':{'sm_lock_mhz':1200,'memory_lock_requested_mhz':7001,'loaded_p2_memory_reported_mhz':6800,'clock_monitor_interval_seconds':.5,'loaded_utilization_minimum':90,'loaded_samples_minimum':30,'sm_allowed_range_mhz':[1185,1215],'memory_allowed_range_mhz':[6798,6802],'temperature_max_exclusive_c':80,'phase_margin_start_seconds':5,'phase_margin_end_seconds':2,'restore_on_exit':True},'scope':'One frozen cuBLAS qualification after native Windows SM1200/memory7001 clock repair; preserves original failed unlocked identity, thresholds and timing path','installed_prefix':'/tmp/nka-operators-dev6-izkosqud/prefix','installed_controls':controls,'evaluator_sha256':sha(host),'backend':'cublas','default_binary_sha256':'66fcc80f9122c919f92c72e37a01794a792c91a58b71d5705b160e0bc979dfc3','fallback_binary_sha256':'93fb2f4344521b7e9b246b54d1c30d4b1a8facf731e6385e18f6119e9c49c72c','graph_launches':32,'pilot':{'replays':4,'samples':3,'selection':'median','target_segment_ms':250,'formula':'clamp(ceil(1000/median_pilot_elapsed_ms),1,2048)'},'pairs_per_group':9,'aa_sessions':3,'conditional_ab_sessions':3,'input_counts':[1,16],'workloads':'all eight frozen development workloads','aa_pair_limit':.02,'aa_group_limit':.015,'deployment_nonregression_guard':.97,'optimization_screening_gate':'original assess unchanged GM>=1.05, interval lower>1.02, each guard>=.97; report separately, no promotion from this qualification','gpu_lock':'/tmp/nka-reviewed-gpu-execution.lock','admission':'three consecutive utilization0 before every session','automatic_retry':False,'heldout_read':False,'new_model_calls':0,'runtime_activation':False,'production_release':False};(a.output/'protocol.json').write_text(json.dumps(protocol,indent=2));(a.output/'host.py').write_bytes(host.read_bytes());(a.output/'protocol.freeze.json').write_text(json.dumps({'protocol_sha256':sha(a.output/'protocol.json'),'evaluator_sha256':sha(host)},indent=2));files={'protocol.json':(a.output/'protocol.json').read_bytes(),'host.py':host.read_bytes()}
    payload='FILES='+repr({n:base64.b64encode(v).decode() for n,v in files.items()})+'\n'+r'''
import base64,json,os,subprocess,tempfile
from pathlib import Path
root=Path(tempfile.mkdtemp(prefix='nka-cublas-final-locked-'))
for n,v in FILES.items():(root/n).write_bytes(base64.b64decode(v))
env={k:v for k,v in os.environ.items() if k in ['HOME','PATH','LANG','LD_LIBRARY_PATH']};env.update(OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1')
with (root/'host.stdout').open('w') as out,(root/'host.stderr').open('w') as err:r=subprocess.run(['/home/dministrator/cuda-lab/.venv/bin/python',str(root/'host.py'),'--root',str(root)],cwd=root,env=env,stdout=out,stderr=err,timeout=3600)
result=json.loads((root/'result.json').read_text()) if (root/'result.json').exists() else {'status':'FAIL_BEFORE_RESULT','exit':r.returncode,'production_release':False};print(json.dumps({'remote_root':str(root),'exit':r.returncode,'result':result}))
'''
    (a.output/'driver.py').write_text(payload);r=execute_python(payload,4200);(a.output/'transport.stderr').write_text(r.stderr);(a.output/'transport.stdout').write_text(r.stdout)
    if r.returncode:raise RuntimeError('transport failed; do not relaunch sampled identity')
    wire=json.loads(r.stdout);(a.output/'result.json').write_text(json.dumps(wire,indent=2));print(json.dumps(wire))


if __name__=='__main__':main()
