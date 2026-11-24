"""Final installed evidence-backed channel checks and actual release/preview package rollback."""
import argparse,fcntl,json,os,shutil,subprocess,sys,time
from pathlib import Path
import numpy as np
from experiments.full_delivery.sm86_operator_release_validate import make_cases


def main():
 p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--preview',type=Path,required=True);a=p.parse_args();root=a.root;work=root/'validation';work.mkdir(exist_ok=False);empty=work/'empty';empty.mkdir();prefix=root/'prefix';bundle=root/'sdk';active=root/'current';active.symlink_to(prefix,target_is_directory=True);bid=json.loads((bundle/'aot-manifest.json').read_text())['bundle_id'];cases=make_cases(bid);commands=[];checks=[];negative=[];env={k:v for k,v in os.environ.items() if k in ['HOME','PATH','LANG','LD_LIBRARY_PATH']};env.update(PYTHONPATH=str(active),PYTHONDONTWRITEBYTECODE='1',OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',NKA_CONTROLLER_CANARY='public-verification-canary');lock=open('/tmp/nka-reviewed-gpu-execution.lock','a');fcntl.flock(lock,fcntl.LOCK_EX)
 def command(label,args,success=True,selected=active):
  e={**env,'PYTHONPATH':str(selected)};start=time.monotonic();r=subprocess.run([sys.executable,'-B',*args],cwd=empty,env=e,capture_output=True,text=True,timeout=200);(work/(label+'.stdout')).write_text(r.stdout);(work/(label+'.stderr')).write_text(r.stderr);commands.append({'label':label,'exit':r.returncode,'wall_seconds':time.monotonic()-start,'prefix':str(selected),'args':args});(work/'commands.json').write_text(json.dumps(commands,indent=2));assert (r.returncode==0)==success,label;return r
 def run_case(label,case,channel='qualified',fault=None):
  _,request,arrays,targets=case;folder=work/label;folder.mkdir();inputs=folder/'inputs';inputs.mkdir();references=folder/'references';references.mkdir();(folder/'request.json').write_text(json.dumps(request))
  for n,v in arrays.items():np.save(inputs/n,v,allow_pickle=False)
  for n,v in targets.items():np.save(references/n,v,allow_pickle=False)
  args=['-m','kernel_agent.operators.cli','run','--channel',channel,'--bundle',str(bundle),'--request',str(folder/'request.json'),'--inputs',str(inputs),'--output',str(folder/'outputs'),'--work',str(folder/'run')]
  if fault:args+=['--inject-failure',fault]
  command(label,args);execution=json.loads((folder/'run/execution.json').read_text());assert execution['status']=='PASS' and execution['production_release'] is (channel=='qualified') and execution['fallback_used']==bool(fault) and len(execution['attempts'])==(2 if fault else 1)
  for rec in execution['attempts']:assert rec['cleanup']['removed'] and rec['protected_unchanged']
  errors={}
  for n,target in targets.items():
   actual=np.load(folder/'outputs'/n,allow_pickle=False);finite=np.isfinite(target);delta=np.abs(actual.astype(np.float64)[finite]-target[finite]);assert np.array_equal(np.isneginf(actual),np.isneginf(target)) and np.all(delta<=.002+.002*np.abs(target[finite]));errors[n]=float(delta.max()) if delta.size else 0.
  checks.append({'label':label,'channel':channel,'production_release':channel=='qualified','attempts':len(execution['attempts']),'max_abs_errors':errors});(work/'checks.json').write_text(json.dumps(checks,indent=2))
 status='FAIL'
 try:
  modules=['test_registry','test_deployment_policy','test_qualified_channel','test_sm89_runtime','test_sm89_qualified','test_operator_qualification'];command('cpu-tests',['-c',"import sys,unittest;sys.path.append("+repr(str(root/'source/tests'))+");r=unittest.TextTestRunner(verbosity=2).run(unittest.TestLoader().loadTestsFromNames("+repr(modules)+"));raise SystemExit(not r.wasSuccessful())"])
  inspect=command('qualified-inspect',['-m','kernel_agent.operators.cli','inspect','--channel','qualified','--bundle',str(bundle)]);meta=json.loads(inspect.stdout);assert meta['status']=='qualified-runtime' and meta['production_release'] is True and meta['scope']=='SM86_REVIEWED_FIXED_ATTENTION_MLA'
  for i in [0,1,3]:run_case(cases[i][0],cases[i])
  for i,fault in [(0,'crash'),(3,'timeout')]:run_case(cases[i][0]+'-'+fault,cases[i],fault=fault)
  # The actual launch pointer switches to the immutable preview and back.
  for version,selected,channel in [('rc2',a.preview,'development'),('ga',prefix,'qualified')]:
   tmp=root/'next';tmp.symlink_to(selected,target_is_directory=True);os.replace(tmp,active);r=command('version-'+version,['-c','import kernel_agent;print(kernel_agent.__version__)']);assert r.stdout.strip()==('0.1.1rc2' if version=='rc2' else '0.1.1')
   for i in [0,3]:run_case('package-'+version+'-'+cases[i][0],cases[i],channel=channel)
  # Real qualified data/binding is used for each CPU-only rejection, no fake certificate.
  bad=work/'bad-scope';bad.mkdir();request={**cases[3][1],'splits':4};(bad/'request.json').write_text(json.dumps(request));args=['-m','kernel_agent.operators.cli','run','--channel','qualified','--bundle',str(bundle),'--request',str(bad/'request.json'),'--inputs',str(bad/'absent'),'--output',str(bad/'outputs'),'--work',str(bad/'run')];r=command('qualified-nondefault-reject',args,False);assert 'page16/split1' in r.stderr and not (bad/'run').exists();negative.append('nondefault MLA')
  code="from kernel_agent.operators.qualification import certificate,check_environment;c=certificate();e={**c['environment'],'driver_version':'deliberately-unqualified'};check_environment(e)";r=command('real-driver-mismatch',['-c',code],False);assert 'binding differs' in r.stderr;negative.append('real driver mismatch')
  for label,name in [('startup','sitecustomize.py'),('bytecode','__pycache__/unknown.cpython-312.pyc')]:
   copied=work/('bad-sdk-'+label);shutil.copytree(bundle,copied)
   for d in [copied,*copied.rglob('*')]:
    if d.is_dir():d.chmod(0o755)
   extra=copied/name;extra.parent.mkdir(exist_ok=True);extra.write_text('raise RuntimeError("unlisted startup fixture must never execute")\n');r=command('unlisted-'+label,['-m','kernel_agent.operators.cli','inspect','--channel','qualified','--bundle',str(copied)],False);assert 'unlisted' in r.stderr;negative.append(label)
  tampered=work/'tampered-prefix';shutil.copytree(prefix,tampered);p=tampered/'kernel_agent/operators/partition.py';p.write_bytes(p.read_bytes()+b'\n# CPU tamper rejection fixture\n');r=command('qualified-source-tamper',['-m','kernel_agent.operators.cli','inspect','--channel','qualified','--bundle',str(bundle)],False,tampered);assert 'runtime code differs' in r.stderr;negative.append('runtime tamper')
  status='PASS'
 finally:
  result={'status':status,'version':'0.1.1','scope':'SM86_REVIEWED_FIXED_ATTENTION_MLA','gpu_outputs':len(checks),'gpu_attempts':sum(c['attempts'] for c in checks),'qualified_outputs':sum(c['production_release'] for c in checks),'real_qualification_negative_checks':negative,'checks':checks,'commands':commands,'current_prefix_after':str(active.resolve()),'new_model_calls':0,'new_event_samples':0,'production_release':status=='PASS'};(work/'result.json').write_text(json.dumps(result,indent=2));print(json.dumps({k:v for k,v in result.items() if k not in ['checks','commands']}))

if __name__=='__main__':main()
