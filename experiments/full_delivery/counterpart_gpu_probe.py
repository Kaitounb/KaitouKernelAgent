"""Actual low-privilege worker/ABI probe of the new image; public input only."""
import fcntl,hashlib,json,os,sys
from pathlib import Path
import numpy as np
lock=open('/tmp/nka-reviewed-gpu-execution.lock','a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
root=Path(sys.argv[1]);prefix=root/'prefix';sys.path.insert(0,str(prefix));binding=json.loads((root/'runtime-binding.json').read_text());host=json.loads((root/'probe-cdi-host.json').read_text())
import probe_runtime.runner as r
r.IMAGE=binding['image_id']
digest='66fcc80f9122c919f92c72e37a01794a792c91a58b71d5705b160e0bc979dfc3';binary=prefix/'kernel_agent/data/artifacts'/(digest+'.so');assert hashlib.sha256(binary.read_bytes()).hexdigest()==digest
rng=np.random.default_rng(3060);a=(rng.standard_normal((17,23))*.25).astype(np.float16);b=(rng.standard_normal((23,19))*.25).astype(np.float16);marker=root/'public-control-marker';marker.write_text('not mounted into worker')
actual,record=r.DockerRunner(binary,digest).run(a,b,root/'gpu-probe-cdi',case='boundaries',host_private_path=str(marker),host_pid=host['pid']);assert actual is not None and record['cleanup']['removed'] and record['protected_files_unchanged'];m=record['metadata'];assert m['uid']==65534 and all(m[k] is True for k in ['capabilities_zero','no_new_privs','artifact_write_denied','inputs_write_denied','external_connect_denied']) and m['interfaces']==['lo'] and all(m[k] is False for k in ['docker_socket_visible','control_canary_in_env','host_private_visible','host_pid_visible']);reference=a.astype(np.float64)@b.astype(np.float64);error=np.abs(actual.astype(np.float64)-reference);assert np.all(error<=3e-5+1e-4*np.abs(reference));v={'status':'GPU_ABI_BOUNDARY_PASS','image_id':binding['image_id'],'max_abs_error':float(error.max()),'metadata':m,'container_isolation_verified':True,'gpu_outputs':1,'new_model_calls':0,'production_release':False};(root/'gpu-probe-result.json').write_text(json.dumps(v,indent=2));print(json.dumps(v))
