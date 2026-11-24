"""One public-input supplement, using an actual native host PID supplied by the outer SSH controller."""
import argparse,fcntl,hashlib,json,os
from pathlib import Path
import numpy as np

def main():
 p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--host-pid',type=int,required=True);a=p.parse_args();root=a.root;work=root/'pid-supplement';lock=open('/tmp/nka-reviewed-gpu-execution.lock','a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
 from kernel_agent.cdi_qualification import registry
 from kernel_agent.cdi_runner import DockerRunner
 store=registry(work/'store');store.initialize();binary,meta,_=store.select();rng=np.random.default_rng(13162);x=(rng.standard_normal((17,23))*.25).astype(np.float16);y=(rng.standard_normal((23,19))*.25).astype(np.float16);ref=x.astype(np.float64)@y.astype(np.float64)
 for n,v in [('a',x),('b',y),('reference',ref)]:np.save(work/(n+'.npy'),v)
 marker=work/'public-control-marker';marker.write_text('not mounted into worker');actual,record=DockerRunner(binary,meta['binary_sha256']).run(x,y,work/'probe',case='boundaries',host_private_path=str(marker),host_pid=a.host_pid);assert actual is not None and np.all(np.abs(actual-ref)<=3e-5+1e-4*np.abs(ref));m=record['metadata'];assert m['uid']==65534 and m['interfaces']==['lo'] and all(m[k] is True for k in ['capabilities_zero','no_new_privs','artifact_write_denied','inputs_write_denied','external_connect_denied']) and all(m[k] is False for k in ['docker_socket_visible','control_canary_in_env','host_private_visible','host_pid_visible']);assert record['cleanup']['removed'] and record['protected_files_unchanged'];v={'status':'ACTUAL_HOST_PID_BOUNDARY_PASS','host_pid_argument':a.host_pid,'controller_container_pid':os.getpid(),'metadata':m,'gpu_outputs':1,'independent_dataset_rerun':False,'frozen_runtime_changed':False,'new_event_samples':0,'new_model_calls':0,'production_release':False};(work/'probe-result.json').write_text(json.dumps(v,indent=2));print(json.dumps(v))
if __name__=='__main__':main()
