"""Real installed GEMM preview/qualified execution, fallback, artifact and package rollback."""
import argparse,fcntl,json,os,subprocess,sys,time
from pathlib import Path
import numpy as np


def main():
 p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--phase',choices=['preview','qualified'],required=True);p.add_argument('--previous-prefix',type=Path,required=True);p.add_argument('--ops-bundle',type=Path,required=True);a=p.parse_args();root=a.root;work=root/'validation';work.mkdir(exist_ok=False);empty=work/'empty';empty.mkdir();link=root/'current';link.symlink_to(root/'prefix');env={k:v for k,v in os.environ.items() if k in ['HOME','PATH','LANG','LD_LIBRARY_PATH']};env.update(PYTHONPATH=str(link),OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',PYTHONDONTWRITEBYTECODE='1',NKA_CONTROL_CANARY='public-release-check');commands=[];checks=[];channel='qualified' if a.phase=='qualified' else 'development';production=channel=='qualified';store=work/'store';lock=open('/tmp/nka-reviewed-gpu-execution.lock','a');fcntl.flock(lock,fcntl.LOCK_EX)
 def cmd(label,args,success=True):
  start=time.monotonic();r=subprocess.run([sys.executable,'-B',*args],cwd=empty,env=env,capture_output=True,text=True,timeout=200);(work/(label+'.stdout')).write_text(r.stdout);(work/(label+'.stderr')).write_text(r.stderr);commands.append({'label':label,'exit':r.returncode,'wall_seconds':time.monotonic()-start,'args':args});(work/'commands.json').write_text(json.dumps(commands,indent=2));assert (r.returncode==0)==success,label;return r
 rng=np.random.default_rng(2611);x=(rng.standard_normal((17,23))*.25).astype(np.float16);y=(rng.standard_normal((23,19))*.25).astype(np.float16);np.save(work/'a.npy',x);np.save(work/'b.npy',y);np.save(work/'reference.npy',x.astype(np.float64)@y.astype(np.float64));target=np.load(work/'reference.npy',allow_pickle=False)
 def run(label,fault=None,selected_store=store,selected_channel=channel):
  out=work/(label+'.npy');args=['-m','kernel_agent.cli','run','--root',str(selected_store),'--channel',selected_channel,'--a',str(work/'a.npy'),'--b',str(work/'b.npy'),'--output',str(out)]
  if fault:args+=['--inject-failure',fault]
  r=cmd(label,args);v=json.loads(r.stdout);actual=np.load(out,allow_pickle=False);assert v['status']=='PASS' and v['production_release'] is (selected_channel=='qualified') and actual.dtype==np.float32 and np.all(np.abs(actual.astype(np.float64)-target)<=3e-5+1e-4*np.abs(target));assert all(rec['cleanup']['removed'] and rec['protected_files_unchanged'] for rec in v['attempts']);checks.append({'label':label,'production_release':v['production_release'],'attempts':len(v['attempts']),'fallback':v['fallback_used'],'version':v['executed_version'],'binary_sha256':v['binary_sha256'],'run_directory':v['run_directory'],'max_abs_error':float(np.abs(actual.astype(np.float64)-target).max())});(work/'checks.json').write_text(json.dumps(checks,indent=2));return v
 status='FAIL';negatives=[];power_changed=False;power_records=[]
 def power(args):
  r=subprocess.run(['/mnt/c/Windows/System32/nvidia-smi.exe',*args],capture_output=True,text=True,timeout=30);power_records.append({'args':args,'exit':r.returncode,'stdout':r.stdout,'stderr':r.stderr});return r
 def settle(watts):
  consecutive=0
  for _ in range(15):
   r=power(['--query-gpu=power.limit','--format=csv,noheader,nounits']);consecutive=consecutive+1 if r.returncode==0 and float(r.stdout.strip())==watts else 0
   if consecutive==3:return
   time.sleep(.2)
  raise ValueError('power limit did not settle')
 try:
  if production:
   r=power(['--query-gpu=power.limit','--format=csv,noheader,nounits']);assert r.returncode==0 and float(r.stdout.strip())==240
   power_changed=True;assert power(['-pl','100']).returncode==0;settle(100)
  expected_version='0.1.2' if production else '0.1.2rc1';r=cmd('version',['-c','import kernel_agent;print(kernel_agent.__version__)']);assert r.stdout.strip()==expected_version
  modules=['test_registry','test_deployment_policy','test_qualified_channel','test_sm89_runtime','test_sm89_qualified','test_operator_qualification','test_gemm_deployment_policy','test_gemm_target_binding'];code="import sys,unittest;sys.path.extend(["+repr(str(root/'source/tests'))+','+repr(str(root/'source'))+"]);r=unittest.TextTestRunner(verbosity=2).run(unittest.TestLoader().loadTestsFromNames("+repr(modules)+"));raise SystemExit(not r.wasSuccessful())";cmd('cpu-tests',['-c',code])
  if not production:
   r=cmd('qualified-missing',['-m','kernel_agent.cli','status','--root',str(work/'missing'),'--channel','qualified'],False);assert 'not packaged' in r.stderr;negatives.append('missing certificate')
  cmd('init',['-m','kernel_agent.cli','init','--root',str(store),'--channel',channel]);run('default');assert run('crash','crash')['fallback_used'];timeout_result=run('timeout','timeout');assert timeout_result['fallback_used'] and b'NKA_FINITE_CUDA_READY' in (Path(timeout_result['run_directory'])/'attempt-0/capture/stderr.bin').read_bytes()
  catalog=json.loads((root/'prefix/kernel_agent/data/releases.json').read_text());active=catalog['default'];original=catalog['fallback'];binary=store/'versions'/active/'nka_gemm_v1.so';saved=binary.read_bytes();binary.chmod(0o644);binary.write_bytes(b'reviewed corruption fixture');assert run('corrupt-active')['fallback_used'];binary.write_bytes(saved);binary.chmod(0o444)
  cmd('activate-original',['-m','kernel_agent.cli','activate','--root',str(store),'--channel',channel,'--version',original]);assert run('original')['executed_version']==original;cmd('rollback',['-m','kernel_agent.cli','rollback','--root',str(store),'--channel',channel]);assert run('restored')['executed_version']==active
  # Actual application pointer, explicitly using development for the preview.
  tmp=root/'next';tmp.symlink_to(a.previous_prefix,target_is_directory=True);os.replace(tmp,link);previous=cmd('previous-version',['-c','import kernel_agent;print(kernel_agent.__version__)']).stdout.strip();previous_store=work/'previous-store';cmd('previous-init',['-m','kernel_agent.cli','init','--root',str(previous_store)]);run('package-previous',selected_store=previous_store,selected_channel='development');tmp.symlink_to(root/'prefix',target_is_directory=True);os.replace(tmp,link);assert cmd('current-version',['-c','import kernel_agent;print(kernel_agent.__version__)']).stdout.strip()==expected_version;run('package-current')
  if production:
   code="from kernel_agent.gemm_qualification import check_target;from kernel_agent.registry import DATA;import json;c=json.loads((DATA/'gemm_qualification.json').read_text());check_target(c['environment'],{**c['target_binding'],'driver_version':'invalid'})";r=cmd('wrong-driver',['-c',code],False);assert 'binding differs' in r.stderr;negatives.append('wrong driver')
   try:
    assert power(['-pl','240']).returncode==0;settle(240)
    r=cmd('wrong-power-profile',['-m','kernel_agent.cli','status','--root',str(store),'--channel','qualified'],False);assert 'requires100W' in r.stderr;negatives.append('actual240W refused')
   finally:
    assert power(['-pl','100']).returncode==0;settle(100)
   policy=root/'prefix/kernel_agent/data/gemm_deployment_policy_v2.json';saved_policy=policy.read_bytes()
   try:
    policy.write_bytes(saved_policy+b'\n');r=cmd('runtime-file-tamper',['-m','kernel_agent.cli','status','--root',str(store),'--channel','qualified'],False);assert 'runtime code differs' in r.stderr;negatives.append('runtime policy file tamper')
   finally:policy.write_bytes(saved_policy)
  # Old qualified operator channel must remain real and usable in this package.
  op_results=[]
  for op in ['attention','mla']:
   folder=work/('operator-'+op);cmd('example-'+op,[str(root/'source/examples/sm86_qualified_operators.py'),'--operation',op,'--work',str(folder),'--generate-only']);cmd('operator-'+op,['-m','kernel_agent.operators.cli','run','--channel','qualified','--bundle',str(a.ops_bundle),'--request',str(folder/'request.json'),'--inputs',str(folder/'inputs'),'--output',str(folder/'outputs'),'--work',str(folder/'run')]);v=json.loads((folder/'run/execution.json').read_text());assert v['production_release'] is True and v['status']=='PASS';op_results.append({'operation':op,'attempts':len(v['attempts'])})
  from kernel_agent.gemm_qualification import registry as factory
  from kernel_agent.runner import DockerRunner
  bound=factory(store,channel=channel);binary,meta,_=bound.select();private_marker=work/'not-mounted';private_marker.write_text('public-boundary-marker');actual,record=DockerRunner(binary,meta['binary_sha256']).run(x,y,work/'boundary',case='boundaries',host_private_path=str(private_marker),host_pid=os.getpid());m=record['metadata'];assert actual is not None and np.all(np.abs(actual.astype(np.float64)-target)<=3e-5+1e-4*np.abs(target));assert record['cleanup']['removed'] and record['protected_files_unchanged'] and m['uid']==65534 and all(m[k] is True for k in ['capabilities_zero','no_new_privs','artifact_write_denied','inputs_write_denied','external_connect_denied']) and m['interfaces']==['lo'] and all(m[k] is False for k in ['docker_socket_visible','control_canary_in_env','host_private_visible','host_pid_visible']);(work/'boundary-summary.json').write_text(json.dumps({'status':'PASS','gpu_outputs':1,'metadata':m},indent=2))
  status='PASS'
 finally:
  if power_changed:
   power(['-pl','240']);settle(240)
  (work/'power-control.json').write_text(json.dumps({'records':power_records,'restored_watts':240 if power_changed else None},indent=2))
  result={'status':status,'version':'0.1.2' if production else '0.1.2rc1','gemm_gpu_outputs':len(checks),'gemm_gpu_attempts':sum(x['attempts'] for x in checks),'operator_outputs':locals().get('op_results',[]),'negatives':negatives,'checks':checks,'commands':commands,'current_prefix':str(link.resolve()),'production_release':production and status=='PASS','new_model_calls':0,'new_event_samples':0};(work/'result.json').write_text(json.dumps(result,indent=2));print(json.dumps({k:v for k,v in result.items() if k not in ['checks','commands']}))

if __name__=='__main__':main()
