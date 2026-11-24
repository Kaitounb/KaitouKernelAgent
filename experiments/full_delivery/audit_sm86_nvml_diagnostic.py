"""Replay all8 diagnostic blocks and both clocks; no formal qualification gate."""
import argparse,hashlib,importlib.util,json,math,statistics,sys,tempfile,types,zipfile
from pathlib import Path
from experiments.p3_01.audit import boundary,witness
from experiments.p3_01.cases import make_cases,case_key
from experiments.full_delivery.gemm_ordinary_host import support_files
from kernel_agent.optimization.identity import group_log_ratio
from kernel_agent.protocol import decode_frame
ROOT=Path(__file__).resolve().parents[2]
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def audit(folder):
 archive=folder/'segmented-archive';root=archive/'remote';execution=json.loads((archive/'execution.json').read_text());assert sha(archive/'raw.zip')==json.loads((archive/'collection.json').read_text())['archive_sha256'];assert all(sha(root/n)==m['sha256'] and (root/n).stat().st_size==m['bytes'] for n,m in execution['manifest'].items());assert all(sha(root/n)==h for n,h in json.loads((folder/'source-hashes.json').read_text()).items());plan=json.loads((root/'plan.json').read_text());protocol=json.loads((root/'protocol.json').read_text());pair_sha=sha(root/'protocol.json');result=json.loads((root/'result.json').read_text());assert result['protocol_sha256']==pair_sha and result['formal_qualification'] is False and result['production_release'] is False and result['heldout_executed'] is False
 assert result['status']=='DIAGNOSTIC_COMPLETE' and result['gpu_reset_exit']==result['memory_reset_exit']==0
 assert all(protocol[k]==v for k,v in plan.items());assert all(sha(root/n)==h for n,h in protocol['controlled_support_files'].items());assert sha(root/'host.py')==plan['host_sha256'];assert sha(ROOT/plan['replay_source'])==plan['replay_source_sha256'];assert plan['block_order']==['on','off','off','on','off','on','on','off'] and plan['fixed_replays']==1348
 with tempfile.TemporaryDirectory() as td:
  temp=Path(td);prefix=temp/'prefix';prefix.mkdir()
  with zipfile.ZipFile(ROOT/'artifacts/wheels/nvidia_kernel_agent-0.1.1-py3-none-any.whl') as z:z.extractall(prefix)
  assert all(sha(prefix/n)==h for n,h in plan['installed_controls'].items())
  stub=types.ModuleType('ordinary_host');stub.support_files=support_files;prior=sys.modules.get('ordinary_host');sys.modules['ordinary_host']=stub
  try:
   spec=importlib.util.spec_from_file_location('nvml_replay_host',root/'host.py');module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module);module.support(temp,prefix,1348)
  finally:
   if prior is None:del sys.modules['ordinary_host']
   else:sys.modules['ordinary_host']=prior
  for p in (temp/'qualification').rglob('*'):
   if p.is_file():assert sha(p)==sha(root/'qualification'/p.relative_to(temp/'qualification'))
 workloads,contract,_=make_cases(ROOT);workload=next(w for w in workloads if w['id']=='dev-03');ledger=json.loads((root/'public-ledgers.json').read_text())['candidate'];blocks=[];all_segments=0;event_ms=0.;clock_pairs=[]
 for block,mode in enumerate(plan['block_order']):
  job=root/('block-'+str(block)+'-'+mode);record=json.loads((job/'record.json').read_text());boundary(record,job);assert all(sha(job/n)==h for n,h in record['protected_hashes'].items());req=json.loads((job/'inputs/request.json').read_text());assert req['mode']=='aa' and req['session']==block and req['cases']==contract and req['workloads']==[workload] and req['pair_sha256']==pair_sha and req['candidate_sha256']==plan['default_sha256'] and req['incumbent_sha256']==plan['fallback_sha256'];assert req['diagnostic_backends']=={'incumbent':'cublas','candidate':'cublas'}
  meta,payload=decode_frame(job/'capture/stdout.bin',4*1024*1024);report=json.loads(payload);assert report==json.loads((job/'report.json').read_text()) and meta['uid']==65534 and meta['pair_sha256']==pair_sha and report['module_identity_distinct'] and report['function_identity_distinct']
  for key in ['checks_before','checks_after']:
   assert len(report[key])==len(contract)
   for value,case in zip(report[key],contract):witness(value,case,ledger)
  assert len(report['candidate_aa'])==1;row=report['candidate_aa'][0];assert row['input_count']==16 and row['workload']=='dev-03' and row['shape_mnk']==workload['shape_mnk'] and row['replays']==row['fixed_diagnostic_replays']==1348 and row['timed_calls_per_segment']==32*1348;assert row['observed_calibration_replays']==max(1,min(8192,math.ceil(4000/sorted(row['pilot_elapsed_ms'])[1])))
  values=[ledger[case_key({'shape_mnk':workload['shape_mnk'],'seed':701+i,'distribution':'random','offset_guard':False})] for i in range(16)];assert row['input_hashes']==[v['input_hashes'] for v in values]
  for role in ['incumbent','candidate']:assert row['output_hashes'][role]==[values[i%16]['output_sha256'] for i in range(32)]
  host=[];raw=[];cross=[];offset=[]
  for pair in range(9):
   s={x['label']:x for x in row['segments'] if x['pair']==pair};device=s['incumbent']['elapsed_ms']/s['candidate']['elapsed_ms'];h=s['incumbent']['host_wall_ms']/s['candidate']['host_wall_ms'];rw=s['incumbent']['raw_wall_ms']/s['candidate']['raw_wall_ms'];host.append(abs(h-1));raw.append(abs(rw-1));cross.append(abs(device-h));unix=(s['incumbent']['wall_end']-s['incumbent']['wall_begin'])/(s['candidate']['wall_end']-s['candidate']['wall_begin']);clock_pairs.append({'event_raw_diff':abs(device-rw),'event_perf_diff':abs(device-h),'event_unix_diff':abs(device-unix)})
   for v in s.values():
    for k in ['host_wall_ms','raw_wall_ms','cpu_enqueue_ms','cpu_wait_ms','wall_begin','wall_end','elapsed_query_ms']:assert math.isfinite(v[k]) and v[k]>0
    assert abs(v['host_wall_ms']-v['cpu_enqueue_ms']-v['cpu_wait_ms'])<1e-6;offset.append(v['elapsed_ms']/v['host_wall_ms']-1)
  samples=json.loads((root/('block-'+str(block)+'-nvml.json')).read_text());assert not samples if mode=='off' else samples and all(x['exit']==0 for x in samples)
  blocks.append({'block':block,'monitor':mode,'max_event_pair_deviation':group_log_ratio(row,block)[1],'max_host_pair_deviation':max(host),'max_raw_pair_deviation':max(raw),'max_event_host_ratio_difference':max(cross),'nvml_samples':len(samples),'max_nvml_query_ms':max((x['query_wall_ms'] for x in samples),default=None)});all_segments+=len(row['segments']);event_ms+=sum(s['elapsed_ms'] for s in row['segments'])
 assert blocks==result['blocks'] and all_segments==144
 restorations=json.loads((root/'outer-clock-restoration.json').read_text());assert len(restorations)==2 and all(x['exit']==0 for x in restorations)
 modes={m:{'blocks':4,'max_event_pair_deviation':max(b['max_event_pair_deviation'] for b in blocks if b['monitor']==m),'max_host_pair_deviation':max(b['max_host_pair_deviation'] for b in blocks if b['monitor']==m),'max_raw_pair_deviation':max(b['max_raw_pair_deviation'] for b in blocks if b['monitor']==m),'max_event_host_ratio_difference':max(b['max_event_host_ratio_difference'] for b in blocks if b['monitor']==m)} for m in ['on','off']}
 cross_clock={'pairs':len(clock_pairs),'max_event_raw_pair_ratio_difference':max(x['event_raw_diff'] for x in clock_pairs),'max_event_perf_pair_ratio_difference':max(x['event_perf_diff'] for x in clock_pairs),'max_event_unix_pair_ratio_difference':max(x['event_unix_diff'] for x in clock_pairs)}
 archived_cross=json.loads((folder/'cross-clock.json').read_text());assert all(archived_cross[k]==v for k,v in cross_clock.items())
 value={'status':'AUDIT_PASS_DIAGNOSTIC_COMPLETE','raw_files_verified':len(execution['manifest']),'raw_segments':all_segments,'measured_event_seconds':event_ms/1000,'modes':modes,'cross_clock_replay':cross_clock,'blocks':blocks,'protocol_sha256':pair_sha,'scope':'fixed8-block3070 public mechanism check, no fullworkload AA/AB or certificate','formal_qualification':False,'production_release':False,'new_model_calls':0,'heldout_executed':False};(folder/'verification.json').write_text(json.dumps(value,indent=2));return value
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('folder',type=Path);a=p.parse_args();print(json.dumps(audit(a.folder.resolve())))
