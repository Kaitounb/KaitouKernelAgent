"""One registered8-block3070 NVML mechanism check after the3060 counterpart finishes."""
import base64,hashlib,json,zipfile
from pathlib import Path
from tools.ssh_3070 import ROOT,execute_python

def main(evidence_name='20261124-sm86-nvml-mechanism-utf8'):
 counterpart=json.loads((ROOT/'evidence/20261124-sm86-3060-counterpart/qualification-launch-result.json').read_text());assert counterpart['exit']==0 and counterpart['result']['status'] in ['PASS_DEPLOYMENT_STABILITY_AND_GUARDS','AA_REJECTED','AB_GUARD_REJECTED','GPU_BUSY_UNSAMPLED_REMAINDER']
 out=ROOT/'evidence'/evidence_name;out.mkdir(exist_ok=False)
 source=ROOT/'experiments/full_delivery/sm86_nvml_diagnostic_host.py';ordinary=ROOT/'experiments/full_delivery/gemm_ordinary_host.py';public=ROOT/'evidence/20261123-sm86-gemm-ordinary-qualification/segmented-archive/remote/oracle-ledgers.json';prior=ROOT/'evidence/20261123-cublas-hags-qualification/live-parent-continuation/segmented-archive/remote/aa-0/report.json';report=json.loads(prior.read_text());row=next(r for r in report['candidate_aa'] if r['workload']=='dev-03' and r['input_count']==16);assert row['replays']==1348
 prefix=json.loads((ROOT/'evidence/20261123-sm86-operators-qualified-installed/location.json').read_text())['prefix'];wheel=ROOT/'artifacts/wheels/nvidia_kernel_agent-0.1.1-py3-none-any.whl'
 with zipfile.ZipFile(wheel) as z:controls={n:hashlib.sha256(z.read(n)).hexdigest() for n in z.namelist() if not n.endswith('/') and n.startswith('kernel_agent/')}
 plan={'schema_version':1,'scope':'fixed8-block monitor-ON/OFF causal mechanism check on3070 after3060; not formal qualification or retry-to-pass','prefix':prefix,'installed_controls':controls,'host_sha256':hashlib.sha256(source.read_bytes()).hexdigest(),'default_sha256':'66fcc80f9122c919f92c72e37a01794a792c91a58b71d5705b160e0bc979dfc3','fallback_sha256':'93fb2f4344521b7e9b246b54d1c30d4b1a8facf731e6385e18f6119e9c49c72c','physical_environment':json.loads((ROOT/'evidence/20261123-sm86-gemm-ordinary-hags/runtime.json').read_text()),'block_order':['on','off','off','on','off','on','on','off'],'fixed_replays':1348,'replay_source':str(prior.relative_to(ROOT)),'replay_source_sha256':hashlib.sha256(prior.read_bytes()).hexdigest(),'observed_prior_pilot_elapsed_ms':row['pilot_elapsed_ms'],'shape_mnk':row['shape_mnk'],'input_count':16,'pairs_per_block':9,'expected_segments':144,'clock_policy':'nativeNVSMI fixed1200MHz/memoryrequest7001, restorebothfinally','telemetry_on':'sameWSL NVML .5s, querylatencyrecorded','telemetry_off':'none insideworker run','timing_additions':'CPU perf_counter, CLOCK_MONOTONIC_RAW,Unix wall markers; CUDAEvent sequenceunchanged','production_release':False,'qualification_gate_applied':False,'new_model_calls':0,'heldout_read':False,'automatic_retry':False,'old_failures_preserved':True}
 files={'host.py':source.read_bytes(),'ordinary_host.py':ordinary.read_bytes(),'public-ledgers.json':public.read_bytes(),'plan.json':json.dumps(plan,indent=2).encode()};(out/'plan.json').write_bytes(files['plan.json']);(out/'source-hashes.json').write_text(json.dumps({n:hashlib.sha256(b).hexdigest() for n,b in files.items()},indent=2))
 payload='FILES='+repr({n:base64.b64encode(b).decode() for n,b in files.items()})+'\n'+r'''
import base64,hashlib,json,os,subprocess,tempfile,time
from pathlib import Path
root=Path(tempfile.mkdtemp(prefix='nka-cublas-final-nvml-'));root.chmod(0o700)
for n,b in FILES.items():(root/n).write_bytes(base64.b64decode(b))
print(json.dumps({'staged_root':str(root),'gpu_execution_started':False}),flush=True)
env={k:v for k,v in os.environ.items() if k in ['PATH','HOME','LD_LIBRARY_PATH','LANG','WSL_INTEROP','WSL_DISTRO_NAME']};env.update(OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1',PYTHONDONTWRITEBYTECODE='1');start=time.monotonic()
code=None;failure=None;restores=[]
try:
 with (root/'parent.stdout').open('w') as o,(root/'parent.stderr').open('w') as e:r=subprocess.run(['/home/dministrator/cuda-lab/.venv/bin/python','-B',str(root/'host.py'),'--root',str(root)],stdout=o,stderr=e,timeout=600,env=env);code=r.returncode
except Exception as e:failure=repr(e)
finally:
 # Remove only workers whose immutable artifact mount belongs to this run.
 q=subprocess.run(['docker','ps','-aq','--filter','name=nka-source-'],capture_output=True,text=True,timeout=30)
 for cid in q.stdout.split():
  item=subprocess.run(['docker','inspect',cid],capture_output=True,text=True,timeout=30)
  if item.returncode==0 and any(x.get('Source','').startswith(str(root)+'/block-') and x.get('Source','').endswith('/artifact') for x in json.loads(item.stdout)[0].get('Mounts',[])):subprocess.run(['docker','rm','-f',cid],capture_output=True,text=True,timeout=30)
 smi='/mnt/c/Windows/System32/nvidia-smi.exe'
 for option in ['--reset-gpu-clocks','--reset-memory-clocks']:
  q=subprocess.run([smi,'-i','0',option],capture_output=True,text=True,timeout=30);restores.append({'option':option,'exit':q.returncode,'stdout':q.stdout,'stderr':q.stderr})
 (root/'outer-clock-restoration.json').write_text(json.dumps(restores,indent=2))
 q=subprocess.run([smi,'--query-gpu=uuid,power.limit,utilization.gpu,temperature.gpu,clocks.sm,clocks.mem','--format=csv,noheader,nounits'],capture_output=True,text=True,timeout=30);(root/'postflight.json').write_text(json.dumps({'exit':q.returncode,'stdout':q.stdout,'stderr':q.stderr,'restore_commands':restores},indent=2))
value={'exit':code,'failure':failure,'root':str(root),'seconds':time.monotonic()-start,'result':json.loads((root/'result.json').read_text()) if (root/'result.json').exists() else None,'stderr_tail':(root/'parent.stderr').read_text()[-3000:]};(root/'launch-result.json').write_text(json.dumps(value,indent=2));print(json.dumps(value))
'''
 (out/'driver.py').write_text(payload);r=execute_python(payload,660);(out/'transport.stderr').write_text(r.stderr);(out/'transport.stdout').write_text(r.stdout);assert r.returncode==0;values=[json.loads(x) for x in r.stdout.splitlines()];v=values[-1];(out/'launch-result.json').write_text(json.dumps(v,indent=2));print(json.dumps(v));assert v['exit']==0 and v['result'] and v['result']['gpu_reset_exit']==v['result']['memory_reset_exit']==0
if __name__=='__main__':main()
