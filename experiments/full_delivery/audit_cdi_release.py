"""Independent CPU replay of3060 wheel/installed files, frames, FP64 and rollback receipts."""
import argparse,ast,hashlib,io,json,math,re,zipfile
from pathlib import Path
import numpy as np
from kernel_agent.protocol import decode_frame
from kernel_agent.bounded_io import confirmed_removed
ROOT=Path(__file__).resolve().parents[2]
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def audit(folder):
 a=folder/'segmented-archive';root=a/'remote';manifest=json.loads((a/'execution.json').read_text())['manifest'];assert sha(a/'raw.zip')==json.loads((a/'collection.json').read_text())['archive_sha256'];assert all(sha(root/n)==m['sha256'] and (root/n).stat().st_size==m['bytes'] for n,m in manifest.items());assert all(sha(root/n)==h for n,h in json.loads((folder/'source-hashes.json').read_text()).items());result=json.loads((root/'result.json').read_text());loc=result['location'];validation=json.loads((root/'validation/result.json').read_text());version=result['version'];qualified=result['phase']=='qualified';diagnostic_failure=qualified and result['status']==validation['status']=='FAIL';assert (diagnostic_failure or result['status']==validation['status']=='PASS') and validation['cpu_tests']==54 and validation['current_prefix']==loc['prefix'];wheel=ROOT/'artifacts/wheels'/('nvidia_kernel_agent-'+version+'-py3-none-any.whl');assert sha(wheel)==result['wheel_sha256']==sha(root/'dist'/wheel.name)
 with zipfile.ZipFile(wheel) as z:
  tree=ast.parse(z.read('kernel_agent/cdi_qualification.py'));bound_target=next(ast.literal_eval(n.value) for n in tree.body if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='TARGET' for t in n.targets))
  for n in z.namelist():
   if n.startswith('kernel_agent/') and not n.endswith('/'):assert hashlib.sha256(z.read(n)).hexdigest()==sha(root/'prefix'/n)
 admission=json.loads((root/'gpu-admission.json').read_text());assert len(admission)==3
 for row in admission:
  fields=[x.strip() for x in row['stdout'].strip().split(',')];assert row['exit']==0 and fields[:2]==[bound_target['gpu_uuid'],bound_target['driver_version']] and fields[2]=='0' and float(fields[3])==170
 tests=(root/'validation/cpu-tests.stderr').read_text();assert re.search(r'Ran 54 tests',tests) and 'OK' in tests
 x=np.load(root/'validation/a.npy',allow_pickle=False);y=np.load(root/'validation/b.npy',allow_pickle=False);target=x.astype(np.float64)@y.astype(np.float64);assert np.array_equal(target,np.load(root/'validation/reference.npy',allow_pickle=False));outputs=0;attempts=0;faults=[]
 for rp in root.rglob('record.json'):
  record=json.loads(rp.read_text());job=rp.parent;cmd=record['command']
  for flag in ['--network=none','--sig-proxy=false','--read-only','--log-driver=none','--cap-drop=ALL','--security-opt=no-new-privileges','--pids-limit=32','--memory=2g','--memory-swap=2g','--cpus=2','--runtime=runc']:assert cmd.count(flag)==1
  assert cmd[cmd.index('--user')+1]=='65534:65534' and cmd[cmd.index('--device')+1]=='nvidia.com/gpu=0' and 'NVIDIA_VISIBLE_DEVICES=void' in cmd and '--gpus' not in cmd and loc['image_id'] in cmd;assert cmd.count('--mount')==2 and all('readonly' in cmd[i+1] and 'docker.sock' not in cmd[i+1] for i,c in enumerate(cmd) if c=='--mount');assert record['cleanup']['removed'] and confirmed_removed(record['cleanup']['inspect_after_cleanup']) and record['protected_files_unchanged'];assert all(sha(job/n)==h for n,h in record['protected_sha256'].items());assert np.array_equal(np.load(job/'inputs/a.npy',allow_pickle=False),x) and np.array_equal(np.load(job/'inputs/b.npy',allow_pickle=False),y)
  for stream in ['stdout','stderr']:assert (job/'capture'/(stream+'.bin')).stat().st_size==record['stored_bytes'][stream]<=record['limits'][stream]
  attempts+=1
  if record['case'] in ['crash','timeout']:
   assert record['status']=='REJECT';faults.append(record['case'])
   if record['case']=='timeout':assert record['reason']=='timeout' and b'NKA_FINITE_CUDA_READY' in (job/'capture/stderr.bin').read_bytes() and record['cleanup']['host_pids_absent_after_cleanup']
   continue
  assert record['status']=='OUTPUT_VALID' and record['exit']==0 and record['reason'] is None;meta,payload=decode_frame(job/'capture/stdout.bin',x.shape[0]*y.shape[1]*4+16384);assert meta==record['metadata'] and payload==(job/'result.npy').read_bytes();actual=np.load(io.BytesIO(payload),allow_pickle=False);assert actual.dtype==np.float32 and np.all(np.isfinite(actual)) and np.all(np.abs(actual-target)<=3e-5+1e-4*np.abs(target));outputs+=1
  if record['case']=='boundaries':
   if diagnostic_failure:assert meta['host_pid_visible'] is True and not (root/'validation/boundary-summary.json').exists() and 'host_pid_visible' in (root/'validation.stderr').read_text()
   assert meta['uid']==65534 and meta['interfaces']==['lo'] and all(meta[k] is True for k in ['capabilities_zero','no_new_privs','artifact_write_denied','inputs_write_denied','external_connect_denied']) and all(meta[k] is False for k in ['docker_socket_visible','control_canary_in_env','host_private_visible']);assert meta['pids_max']=='32' and meta['memory_max']==str(2*1024**3)
 if qualified:
  supplement_path=ROOT/'evidence/20261124-3060-ga-pid-supplement/verification.json';supplement=json.loads(supplement_path.read_text());assert supplement['status']=='AUDIT_PASS_ACTUAL_HOST_PID_SUPPLEMENT' and supplement['actual_native_host_pid_verified'] and supplement['wheel_sha256']==sha(wheel)
 expected=9 if qualified else 7;assert outputs==expected and attempts==expected+2 and set(faults)=={'crash','timeout'};assert validation['gemm_gpu_outputs']==expected-1 and validation['gemm_gpu_attempts']==attempts-1
 checks=validation['checks'];assert all(c['production_release'] is (qualified and c['label']!='package-previous') for c in checks);assert next(c for c in checks if c['label']=='original')['version']=='sm86-cublas-original';assert next(c for c in checks if c['label']=='restored')['version']=='sm86-cublas-container'
 if qualified:assert {'package-previous','package-current'}<=set(c['label'] for c in checks) and {'actual driver mismatch','runtime policy tamper'}<=set(validation['negatives'])
 value={'status':'AUDIT_PASS_QUALIFIED_INSTALL' if qualified else 'AUDIT_PASS_PREVIEW','version':version,'raw_files_verified':len(manifest),'cpu_tests':54,'gpu_outputs':outputs,'gpu_attempts':attempts,'expected_faults_rejected':2,'boundary_pass':True,'rollback_pass':True,'actual_package_rollback_pass':qualified,'original_validation_status':validation['status'],'pid_diagnostic_failure_preserved':diagnostic_failure,'pid_isolation_evidence':'Qualified GA actual-native-host supplement' if qualified else 'RC2 actual-native-host supplement','boundary_supplement_sha256':sha(supplement_path) if qualified else None,'wheel_sha256':sha(wheel),'image_id':loc['image_id'],'target_binding':bound_target,'new_model_calls':0,'new_event_samples':0,'production_release':qualified};(folder/'verification.json').write_text(json.dumps(value,indent=2));return value
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('folder',type=Path);a=p.parse_args();print(json.dumps(audit(a.folder.resolve())))
