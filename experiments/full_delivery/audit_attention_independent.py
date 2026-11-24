"""CPU replay of new private Attention data, public output is hashes and aggregates only."""
import hashlib,json,zipfile
from pathlib import Path
import numpy as np
from kernel_agent.protocol import decode_frame
from kernel_agent.optimization.source_packets import unpack
from experiments.full_delivery.audit_sm86_operator_release import oracle
ROOT=Path(__file__).resolve().parents[2]


def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()

def main():
 out=ROOT/'evidence/20261123-attention-sm86-independent';private=ROOT/'.private/attention-sm86-rc1-independent-v1';raw=private/'remote';summary=json.loads((out/'summary.json').read_text());freeze=json.loads((out/'freeze.json').read_text());assert summary['status']=='PASS' and summary['correct_outputs']==summary['executed_gpu_calls']==8 and sha(private/'raw.zip')==summary['archive_sha256']
 with zipfile.ZipFile(private/'raw.zip') as z:
  manifest=json.loads(z.read('archive-manifest.json'))
  for name,row in manifest.items():assert sha(raw/name)==row['sha256'] and (raw/name).stat().st_size==row['bytes']
 assert sha(raw/'manifest.json')==freeze['manifest_sha256'] and sha(raw/'protocol.json')==freeze['protocol_sha256'];cases=sorted(raw.glob('case-*'));assert len(cases)==8;max_error=0.;attempts=0
 for case in cases:
  req=json.loads((case/'request.json').read_text());arrays={p.name:np.load(p,allow_pickle=False) for p in (case/'inputs').glob('*.npy')};target=oracle(req,arrays)['output.npy'];saved=np.load(case/'references/output.npy',allow_pickle=False);assert np.allclose(saved,target,atol=1e-10,rtol=1e-10);actual=np.load(case/'outputs/output.npy',allow_pickle=False);delta=np.abs(actual.astype(np.float64)-target);assert np.isfinite(actual).all() and np.all(delta<=.002+.002*np.abs(target));max_error=max(max_error,float(delta.max()));e=json.loads((case/'run/execution.json').read_text());assert e['status']=='PASS' and e['production_release'] is False and not e['fallback_used'] and len(e['attempts'])==1;rec=e['attempts'][0];attempts+=1;assert rec['cleanup']['removed'] and rec['protected_unchanged'];m=rec['metadata'];assert m['uid']==m['boundary_gid']==65534 and all(m['boundary_'+k] is True for k in ['cap_eff_zero','cap_bnd_zero','no_new_privs','network_only_loopback','root_readonly','mounted_files_readonly','host_control_absent','credentials_env_absent'])
  for name,h in rec['protected_hashes'].items():assert sha(case/'run/attempt-0'/name)==h
  assert m['input_hashes']==[hashlib.sha256(arrays[n].tobytes()).hexdigest() for n in sorted(arrays)]
  limit=actual.nbytes+16384;meta,payload=decode_frame(case/'run/attempt-0/capture/stdout.bin',limit+1024*1024+1024);assert meta==m;report,files=unpack(payload,{'output.npy':limit});assert report=={'format':'reviewed-native-operator-v1','output_names':['output.npy']} and files['output.npy']==(case/'run/attempt-0/output.npy').read_bytes()
 result={'status':'AUDIT_PASS','scope':'fresh server-local synthetic Attention acceptance; private exact values remain unseen by model','private_files_verified':len(manifest),'gpu_calls':attempts,'numpy_fp64_outputs_verified':len(cases),'maximum_absolute_error':max_error,'manifest_sha256':freeze['manifest_sha256'],'protocol_sha256':freeze['protocol_sha256'],'archive_sha256':summary['archive_sha256'],'wheel_sha256':freeze['wheel_sha256'],'exact_values_exposed_to_model':False,'old_private_manifest_read':False,'new_model_calls':0,'new_event_samples':0,'production_release':False};(out/'verification.json').write_text(json.dumps(result,indent=2));print(json.dumps(result))
if __name__=='__main__':main()
