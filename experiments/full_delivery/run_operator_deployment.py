"""Freeze then execute a new SM86 fixed-SDK deployment target, not a retry of old search."""
import base64,hashlib,json
from tools.ssh_3070 import ROOT,execute_python
from kernel_agent.operators.qualification import RUNTIME_FILES
from experiments.full_delivery.operator_deployment_cases import WORKLOADS


def main():
 out=ROOT/'evidence/20261123-sm86-operator-deployment-stability';out.mkdir(exist_ok=False);installed=ROOT/'evidence/20261123-sm86-operators-rc1';deployment=json.loads((installed/'raw-archive/remote/persistent-deployment.json').read_text());hardware=json.loads((ROOT/'evidence/20261123-sm86-operators-hags/runtime.json').read_text());assert hardware['status']=='HAGS_ENABLED'
 names=['experiments/full_delivery/operator_deployment_cases.py','experiments/full_delivery/operator_deployment_paired.py','experiments/full_delivery/operator_deployment_host.py'];files={n:(ROOT/n).read_bytes() for n in names}
 protocol={'schema_version':1,'scope':'SM86 HAGS-On fixed Attention/MLA deployment, sustained Graph average only; not optimization gains or tail SLA','new_target_reason':'Attention/MLA current post-reboot HAGS-On deployment with pinned immutable rc1/AOT; old pre-reboot MLA search AA refusal remains unchanged. Added attention targets and one-second fixed pilot policy frozen before sampling.','hardware':hardware,'installed_prefix':deployment['prefix'],'baseline_prefix':'/home/dministrator/cuda-lab/nka-persistent/dev6-d5f4bbf83dc0/prefix','sdk_prefix':deployment['bundle'],'bundle_id':deployment['inspect']['bundle_id'],'image_id':'sha256:ce00f6dc2eea43dc508b90b548f624646d643a92d363aa715fdb5a3b4f49290a','torch':'2.14.0+cu130','cuda':'13.0','workloads':WORKLOADS,'input_counts':[1,16],'input_generator':'NumPy PCG64/default_rng(1801+slot), FP64 standard_normal*0.25 clipped[-4,4] toFP16; page permutation default_rng(811)','aa_sessions':3,'ab_sessions':3,'pairs_per_group':9,'graph_calls':32,'warmup_seconds':1,'pilot_replays':4,'pilot_samples':3,'pilot_formula':'max(1,min(8192,ceil(4000/median(pilot_ms))))','target_segment_ms':1000,'aa_pair_deviation_limit_exclusive':.02,'aa_group_geomean_deviation_limit_exclusive':.015,'nonregression_minimum_geomean':.97,'idle_admission':'three consecutive utilization==0; at most ten queries before each fresh process','failure_policy':'no retry/resampling of sampled failure; stop before AB if full AA fails; keep all raw evidence','reference_scope':'trusted fixed-SDK diagnostic controller uses CPU FP64 references; no generated/free source, separate deployed Docker worker boundary is not inferred from timing','clock_policy':'ordinary clocks, no clock/HAGS/power changes','event_scope':'complete captured adapter CUDA core including split plan.run/stack; excludes plan/ref/H2D/control/publish','source_files':{n:hashlib.sha256(b).hexdigest() for n,b in files.items()},'host_sha256':hashlib.sha256(files[names[-1]]).hexdigest(),'runtime_files':{n:hashlib.sha256((ROOT/n).read_bytes()).hexdigest() for n in RUNTIME_FILES},'new_model_calls':0,'production_release':False}
 protocol['image_id']='sha256:ce00f6dc2eea43dc508b90b548f624646d643a92d363aa715fdb5a3b4f49290a';(out/'protocol.json').write_text(json.dumps(protocol,indent=2));files['protocol.json']=(out/'protocol.json').read_bytes()
 payload='FILES='+repr({n:base64.b64encode(b).decode() for n,b in files.items()})+'\n'+r'''
import base64,json,subprocess,tempfile,os
from pathlib import Path
root=Path(tempfile.mkdtemp(prefix='nka-sm86-ops-stability-'));root.chmod(0o700)
for n,b in FILES.items():
 p=root/('protocol.json' if n=='protocol.json' else 'source/'+n);p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes(base64.b64decode(b))
protocol=json.loads((root/'protocol.json').read_text());(root/'sdk').symlink_to(protocol['sdk_prefix'],target_is_directory=True);location={'remote_root':str(root),'protocol_sha256':__import__('hashlib').sha256((root/'protocol.json').read_bytes()).hexdigest()};(root/'location.json').write_text(json.dumps(location,indent=2));print(json.dumps(location))
'''
 (out/'stage-driver.py').write_text(payload);r=execute_python(payload,90);(out/'stage.stderr').write_text(r.stderr);assert r.returncode==0;value=json.loads(r.stdout);(out/'location.json').write_text(json.dumps(value,indent=2));print(json.dumps(value))
if __name__=='__main__':main()
