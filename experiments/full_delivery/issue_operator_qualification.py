"""Issue SM86 operator qualification only after real archived acceptance passes."""
import hashlib,json
from pathlib import Path
from kernel_agent.operators.qualification import RUNTIME_FILES,SCOPE
ROOT=Path(__file__).resolve().parents[2]


def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()

def main():
 paths={'installation':'evidence/20261123-sm86-operators-rc1/verification.json','hardened_installation':'evidence/20261123-sm86-operators-rc2/verification.json','stability':'evidence/20261123-sm86-operator-deployment-shared-input/verification.json','original_rejection':'evidence/20261123-sm86-operator-deployment-stability/verification.json','attention_independent':'evidence/20261123-attention-sm86-independent/verification.json','mla_independent':'evidence/20261123-mla-independent-acceptance/verification.json','numeric_parent':'evidence/20261123-sm86-operators-rc1/numeric-source-binding.json'};v={k:json.loads((ROOT/n).read_text()) for k,n in paths.items()}
 assert v['hardened_installation']['status']=='AUDIT_PASS_HARDENED_RC' and v['hardened_installation']['installed_cpu_tests']==40 and v['hardened_installation']['installed_gpu_outputs']==2
 assert v['installation']['status']=='AUDIT_PASS_INSTALLED_RC' and v['installation']['installed_gpu_outputs']==13 and v['installation']['installed_cpu_tests']==37
 assert v['stability']['status']=='AUDIT_PASS_DEPLOYMENT_QUALIFIED' and v['stability']['qualification_pass'] is True and v['stability']['aa_sessions']==v['stability']['ab_sessions']==3 and v['stability']['raw_segments_verified']==1080
 assert v['attention_independent']['status']=='AUDIT_PASS' and v['attention_independent']['gpu_calls']==v['attention_independent']['numpy_fp64_outputs_verified']==8
 assert v['mla_independent']['status']=='PASS' and v['mla_independent']['gpu_invocations']==8 and v['mla_independent']['independently_checked_output_tensors']==16
 assert v['numeric_parent']['status']=='UNCHANGED_FROM_ACCEPTED_DEV6' and all(sha(ROOT/n)==h for n,h in v['numeric_parent']['files'].items())
 assert not (ROOT/'artifacts/wheels/nvidia_kernel_agent-0.1.1-py3-none-any.whl').exists()
 # Presentation-only qualification scope; numerical/worker code and contracts stay fixed.
 assert "'scope':certificate['scope']" in (ROOT/'kernel_agent/operators/cli.py').read_text()
 for name in ['pyproject.toml','kernel_agent/__init__.py']:
  p=ROOT/name;s=p.read_text();assert '0.1.1rc2' in s;p.write_text(s.replace('0.1.1rc2','0.1.1'))
 environment={'image_id':'sha256:ce00f6dc2eea43dc508b90b548f624646d643a92d363aa715fdb5a3b4f49290a','gpu_uuid':'GPU-859dd34e-59f1-959a-2546-ca027062f9bd','driver_version':'591.74','torch':'2.14.0+cu130','cuda':'13.0','capability':[8,6]}
 c={'schema_version':1,'status':'QUALIFIED','scope':SCOPE,'production_release':True,'source_execution_enabled':False,'operations':['attention','mla'],'qualified_mla_configuration':{'page_size':16,'splits':1},'bundle_id':v['numeric_parent']['aot_bundle_id'],'environment':environment,'independent_correctness_pass':True,'installed_runtime_pass':True,'recovery_pass':True,'rollback_pass':True,'boundary_pass':True,'stability_pass':True,'raw_audit_pass':True,'runtime_files':{n:sha(ROOT/n) for n in RUNTIME_FILES},'evidence':{n:sha(ROOT/n) for n in paths.values()},'stability_protocol_sha256':v['stability']['protocol_sha256'],'performance_scope':'ordinary clocks/current HAGS-On, sustained Graph batch average; no single-request tail SLA or optimization gain claim','inheritance':'MLA independent acceptance belongs to same AOT bundle and unchanged numeric/contract files; original sealed data not reexecuted','new_model_calls':0}
 data=ROOT/'kernel_agent/operators';raw=json.dumps(c,indent=2).encode()+b'\n';(data/'qualification.json').write_bytes(raw);(data/'qualified_catalog.json').write_text(json.dumps({'schema_version':1,'scope':SCOPE,'qualification_sha256':hashlib.sha256(raw).hexdigest()},indent=2)+'\n');out=ROOT/'evidence/20261123-sm86-operator-qualification';out.mkdir(exist_ok=False);(out/'issue.json').write_text(json.dumps({'status':'REAL_EVIDENCE_CERTIFICATE_ISSUED_FINAL_INSTALL_PENDING','qualification_sha256':hashlib.sha256(raw).hexdigest(),'parent_rc1_wheel_sha256':v['installation']['wheel_sha256'],'runtime_files':c['runtime_files'],'new_model_calls':0},indent=2));print(json.dumps({'status':'REAL_EVIDENCE_CERTIFICATE_ISSUED_FINAL_INSTALL_PENDING','qualification_sha256':hashlib.sha256(raw).hexdigest()}))

if __name__=='__main__':main()
