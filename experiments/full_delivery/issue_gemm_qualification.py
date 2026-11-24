"""Fail-closed certificate producer; never execute until actual stability/install audits pass."""
import hashlib,json,zipfile
from pathlib import Path
from kernel_agent.gemm_qualification import RUNTIME_FILES
from tools.verify_gemm_release_freeze import verify
ROOT=Path(__file__).resolve().parents[2]
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main():
 paths={'stability':'evidence/20261124-sm86-gemm-power-qualification/verification.json','installation':'evidence/20261124-sm86-gemm-0.1.2rc1-installed/verification.json','independent':'evidence/20261122-gemm-independent-acceptance/verification.json','thermal_diagnostic':'evidence/20261124-sm86-gemm-thermal/verification.json','original_ordinary_rejection':'evidence/20261123-sm86-gemm-ordinary-qualification/verification.json','numeric_freeze':'releases/gemm-sm86-rc1/freeze.json'};v={k:json.loads((ROOT/n).read_text()) for k,n in paths.items()}
 s=v['stability'];assert s['status']=='AUDIT_PASS_QUALIFICATION_PASS' and s['aa_sessions']==s['ab_sessions']==3 and s['qa_checks']==320 and s['raw_segments']==1728 and s['max_aa_pair_deviation']<.03 and s['max_aa_group_deviation']<.015 and s['minimum_nonregression_geomean']>=.97
 assert v['installation']['status']=='AUDIT_PASS_PREVIEW' and v['installation']['installed_cpu_tests']==49 and v['installation']['total_gpu_outputs']==11 and v['installation']['total_gpu_attempts']==13
 assert v['independent']['status']=='PASS' and v['independent']['independent_outputs']==16 and v['independent']['freeze_id']==verify(ROOT/'releases/gemm-sm86-rc1')['freeze_id']
 assert v['thermal_diagnostic']['status']=='THERMAL_DIAGNOSTIC_PASS_NOT_QUALIFICATION' and v['original_ordinary_rejection']['status']=='AUDIT_PASS_QUALIFICATION_REJECT'
 f=v['numeric_freeze'];catalog=json.loads((ROOT/'kernel_agent/data/releases.json').read_text());binary=[catalog['versions'][catalog[k]]['binary_sha256'] for k in ['default','fallback']];assert binary==[f['deployment_artifacts'][k] for k in ['default','fallback']]
 parent=ROOT/next(n for n in f['files'] if n.endswith('.whl'));numeric=['kernel_agent/data/gemm_contract_v1.json','kernel_agent/pipe_worker.py','kernel_agent/protocol.py','kernel_agent/output.py']+['kernel_agent/data/artifacts/'+h+'.so' for h in binary]
 with zipfile.ZipFile(parent) as z:
  for n in numeric:assert (ROOT/n).read_bytes()==z.read(n)
 assert not (ROOT/'artifacts/wheels/nvidia_kernel_agent-0.1.2-py3-none-any.whl').exists() and not (ROOT/'kernel_agent/data/gemm_qualification.json').exists()
 for name in ['pyproject.toml','kernel_agent/__init__.py']:
  p=ROOT/name;t=p.read_text();assert '0.1.2rc1' in t;p.write_text(t.replace('0.1.2rc1','0.1.2'))
 meta=catalog['versions'][catalog['default']];environment={k:meta[k] for k in ['image_id','torch','cuda','device']};c={'schema_version':2,'status':'QUALIFIED','scope':'SM86_FIXED_IMAGE_GEMM_CUBLAS','backend':'cublas','production_release':True,'source_execution_enabled':False,'independent_correctness_pass':True,'recovery_pass':True,'final_pairing_pass':True,'raw_audit_pass':True,'environment':environment,'target_binding':{'gpu_uuid':'GPU-859dd34e-59f1-959a-2546-ca027062f9bd','driver_version':'591.74'},'operating_profile':{'power_limit_watts':100,'clock_policy':'dynamic','maximum_temperature_exclusive_c':83},'binary_sha256':binary,'runtime_files':{n:sha(ROOT/n) for n in RUNTIME_FILES},'deployment_policy_sha256':sha(ROOT/'kernel_agent/data/gemm_deployment_policy_v2.json'),'stability_protocol_sha256':s['protocol_sha256'],'evidence':{n:sha(ROOT/n) for n in paths.values()},'independent_inheritance':{'freeze_id':f['freeze_id'],'unchanged_numeric_files':{n:sha(ROOT/n) for n in numeric},'heldout_reexecuted':False},'performance_scope':'100W ceiling/current HAGS-On, sustained Graph batch average only; no optimization gain or single-request tail SLA','new_model_calls':0}
 raw=json.dumps(c,indent=2).encode()+b'\n';data=ROOT/'kernel_agent/data';(data/'gemm_qualification.json').write_bytes(raw);catalog['qualification_sha256']=hashlib.sha256(raw).hexdigest()
 for m in catalog['versions'].values():m['production_release']=True;m['status']='qualified-fixed-cublas100W'
 (data/'qualified_releases.json').write_text(json.dumps(catalog,indent=2)+'\n');out=ROOT/'evidence/20261124-sm86-gemm-qualification';out.mkdir(exist_ok=False);receipt={'status':'REAL_EVIDENCE_CERTIFICATE_ISSUED_FINAL_INSTALL_PENDING','qualification_sha256':hashlib.sha256(raw).hexdigest(),'performance_scope':c['performance_scope'],'production_deployment_completed':False};(out/'issue.json').write_text(json.dumps(receipt,indent=2));print(json.dumps(receipt))
if __name__=='__main__':main()
