"""Independent CPU replay of preregistered operator timing and deterministic public inputs."""
import argparse,hashlib,json,math,statistics,zipfile
from pathlib import Path
import numpy as np
from kernel_agent.operators.measurement import assess
from kernel_agent.optimization.identity import group_log_ratio
from experiments.full_delivery.operator_deployment_cases import WORKLOADS,inputs
from experiments.full_delivery.audit_sm86_operator_release import oracle
ROOT=Path(__file__).resolve().parents[2]


def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()

def main():
 parser=argparse.ArgumentParser();parser.add_argument('--evidence',default='evidence/20261123-sm86-operator-deployment-stability',choices=['evidence/20261123-sm86-operator-deployment-stability','evidence/20261123-sm86-operator-deployment-shared-input']);args=parser.parse_args();out=ROOT/args.evidence;archive=out/'raw-archive';raw=archive/'remote';c=json.loads((archive/'collection.json').read_text());assert sha(archive/'raw.zip')==c['archive_sha256']
 with zipfile.ZipFile(archive/'raw.zip') as z:
  files=json.loads(z.read('manifest.json'))
  for n,row in files.items():assert sha(raw/n)==row['sha256'] and (raw/n).stat().st_size==row['bytes']
 p=json.loads((raw/'protocol.json').read_text());assert sha(raw/'protocol.json')==json.loads((out/'location.json').read_text())['protocol_sha256'] and p==json.loads((out/'protocol.json').read_text());assert p['workloads']==WORKLOADS and p['hardware']['status']=='HAGS_ENABLED' and p['aa_pair_deviation_limit_exclusive']==.02 and p['aa_group_geomean_deviation_limit_exclusive']==.015
 for n,h in p['source_files'].items():assert sha(raw/'source'/n)==h
 installed=ROOT/'evidence/20261123-sm86-operators-rc1/raw-archive/remote';manifest=json.loads((installed/'sdk/aot-manifest.json').read_text());assert p['bundle_id']==manifest['bundle_id']
 for n,h in p['runtime_files'].items():assert sha(installed/'prefix'/n)==h
 aa=[];ab=[];cache={};segment_count=0;event_ms=0.;output_arrays=0;max_error=0.;max_pair=0.;max_group=0.;groups=[]
 def target(w,slot):
  key=(w['id'],slot)
  if key not in cache:
   data=inputs(w,slot);host,extra=(data,{}) if w['operation']=='attention' else data
   names=['q.npy','k.npy','v.npy'] if w['operation']=='attention' else ['q_abs.npy','q_rope.npy','cache.npy','rope_cache.npy'];arrays=dict(zip(names,host));request={**w,'mask':False}
   if w['operation']=='mla':arrays.update({'indptr.npy':np.array(extra['indptr'],np.int32),'indices.npy':np.array(extra['indices'],np.int32)})
   expected=oracle(request,arrays);expected=[expected[n] for n in (['output.npy'] if w['operation']=='attention' else ['z.npy','lse.npy'])];cache[key]=([hashlib.sha256(x.tobytes()).hexdigest() for x in host],expected)
  return cache[key]
 for mode,reports in [('aa',aa),('ab',ab)]:
  for session in range(3):
   path=raw/(mode+'-'+str(session)+'.json')
   if not path.exists():continue
   r=json.loads(path.read_text());assert r['mode']==mode and r['session']==session and r['torch']==p['torch'] and r['cuda']==p['cuda'] and r['capability']==[8,6]
   if r['status']!='PASS':continue
   assert len(r['rows'])==10 and {(row['workload'],row['input_count']) for row in r['rows']}=={(w['id'],n) for w in WORKLOADS for n in [1,16]};reports.append(r)
   for row in r['rows']:
    if p.get('input_storage_policy')=='shared-physical-read-only-ring':assert row['shared_input_storage'] is True and row['input_device_ptrs']['incumbent']==row['input_device_ptrs']['candidate']
    w=next(w for w in WORKLOADS if w['id']==row['workload']);count=row['input_count'];mean,dev=group_log_ratio(row,session)
    if mode=='aa':max_pair=max(max_pair,dev);max_group=max(max_group,abs(math.exp(mean)-1))
    pilot=row['pilot_elapsed_ms'];assert len(pilot)==3 and all(math.isfinite(v) and v>0 for v in pilot) and row['replays']==max(1,min(8192,math.ceil(4000/sorted(pilot)[1]))) and row['timed_calls_per_segment']==32*row['replays']
    assert row['capture_order']==(['incumbent','candidate'] if session%2==0 else ['candidate','incumbent']) and row['qa_after_pass'] and row['inputs_and_guards_unchanged']
    assert row['native_hashes'] and all(manifest['files'].get(n)==h for n,h in row['native_hashes'].items());segment_count+=len(row['segments']);event_ms+=sum(s['elapsed_ms'] for s in row['segments']);folder=raw/row['output_folder']
    for role in ['incumbent','candidate']:
     assert row['input_hashes'][role]==[target(w,i)[0] for i in range(count)] and len(row['output_hashes'][role])==32 and len(row['qa_before_errors'][role])==count
     for slot in range(count):
      hashes=[]
      for j,expected in enumerate(target(w,slot)[1]):
       actual=np.load(folder/(role+'-'+str(slot)+'-'+str(j)+'.npy'),allow_pickle=False);hashes.append(hashlib.sha256(actual.tobytes()).hexdigest());finite=np.isfinite(expected);delta=np.abs(actual.astype(np.float64)[finite]-expected[finite]);assert np.array_equal(np.isneginf(actual),np.isneginf(expected)) and np.all(delta<=2e-3+2e-3*np.abs(expected[finite]));max_error=max(max_error,float(delta.max()) if delta.size else 0.);output_arrays+=1
      for index in range(slot,32,count):assert row['output_hashes'][role][index]==hashes
    groups.append({'mode':mode,'session':session,'workload':w['id'],'input_count':count,'geomean':math.exp(mean),'max_pair_deviation':dev})
 result=json.loads((raw/'result.json').read_text());d=assess(aa,ab,WORKLOADS,[w['id'] for w in WORKLOADS]);assert result['decision']==d and result['raw_segments']==segment_count and result['source_and_native_unchanged'] and result['protocol_sha256']==sha(raw/'protocol.json')
 passed=result['status']=='PASS_DEPLOYMENT_STABILITY_NONREGRESSION';minimum=min((min(v['workload_speedups'].values()) for v in d['policies']),default=None)
 if passed:assert len(aa)==len(ab)==3 and segment_count==1080 and max_pair<.02 and max_group<.015 and d['aa_status']=='PASS' and minimum>=.97
 status='AUDIT_PASS_DEPLOYMENT_QUALIFIED' if passed else 'AUDIT_PASS_QUALIFICATION_REJECT_OR_INCOMPLETE'
 audit={'status':status,'qualification_pass':passed,'protocol_sha256':sha(raw/'protocol.json'),'archive_sha256':c['archive_sha256'],'raw_files_verified':len(files),'aa_sessions':len(aa),'ab_sessions':len(ab),'raw_segments_verified':segment_count,'event_seconds':event_ms/1000.,'maximum_aa_pair_deviation':max_pair,'maximum_aa_group_geomean_deviation':max_group,'minimum_nonregression_geomean':minimum,'independent_numpy_fp64_output_arrays':output_arrays,'maximum_absolute_error':max_error,'deterministic_input_sets_recomputed':len(cache),'groups':groups,'new_model_calls':0,'old_sealed_reexecuted':False,'production_release':False,'performance_scope':p['scope']};(out/'verification.json').write_text(json.dumps(audit,indent=2));print(json.dumps({k:v for k,v in audit.items() if k!='groups'}))

if __name__=='__main__':main()
