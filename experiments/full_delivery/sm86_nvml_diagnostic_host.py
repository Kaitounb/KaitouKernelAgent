"""User-requested3070 mechanism diagnostic: fixed8 blocks, no promotion or qualification."""
import argparse,base64,fcntl,hashlib,json,math,statistics,subprocess,sys,threading,time,traceback
from pathlib import Path
from ordinary_host import support_files as original_support

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def once(s,old,new):
 if s.count(old)!=1:raise ValueError('diagnostic adaptation anchor changed: '+old[:80])
 return s.replace(old,new)
def support(root,prefix,replays):
 s=original_support(root,prefix);p=s/'pair_worker.py';text=p.read_text();assert text.count('for count in [1,16]')==2;p.write_text(text.replace('for count in [1,16]','for count in [16]'))
 p=s/'source_timing.py';text=p.read_text();anchor='        replays=max(1,min(8192,math.ceil(4000/sorted(pilot_elapsed_ms)[1])))';text=once(text,anchor,anchor+'\n        observed_calibration_replays=replays;replays='+str(replays))
 old='                start.record(stream)\n                for _ in range(replays):graphs[label].replay()\n                end.record(stream);end.synchronize();elapsed=start.elapsed_time(end)'
 new='''                wall_begin=time.time();raw_begin=time.clock_gettime_ns(time.CLOCK_MONOTONIC_RAW);cpu_begin=time.perf_counter()
                start.record(stream)
                for _ in range(replays):graphs[label].replay()
                end.record(stream);submitted=time.perf_counter();end.synchronize();finished=time.perf_counter();raw_end=time.clock_gettime_ns(time.CLOCK_MONOTONIC_RAW);wall_end=time.time()
                query_begin=time.perf_counter();elapsed=start.elapsed_time(end);query_ms=(time.perf_counter()-query_begin)*1000'''
 text=once(text,old,new);text=once(text,"'elapsed_ms':elapsed}","'elapsed_ms':elapsed,'host_wall_ms':(finished-cpu_begin)*1000,'raw_wall_ms':(raw_end-raw_begin)/1e6,'cpu_enqueue_ms':(submitted-cpu_begin)*1000,'cpu_wait_ms':(finished-submitted)*1000,'wall_begin':wall_begin,'wall_end':wall_end,'elapsed_query_ms':query_ms}");text=once(text,"'target_segment_ms':1000}","'target_segment_ms':1000,'fixed_diagnostic_replays':"+str(replays)+",'observed_calibration_replays':observed_calibration_replays}");p.write_text(text)
def main():
 p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);a=p.parse_args();root=a.root;plan=json.loads((root/'plan.json').read_text());prefix=Path(plan['prefix']);assert sha(Path(__file__))==plan['host_sha256'];assert all(sha(prefix/n)==h for n,h in plan['installed_controls'].items());lock=open('/tmp/nka-reviewed-gpu-execution.lock','a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);sys.path[:0]=[str(root),str(prefix)];smi='/mnt/c/Windows/System32/nvidia-smi.exe';commands=[]
 def run_control(args):
  start=time.time();r=subprocess.run([smi,*args],capture_output=True,text=True,timeout=30);commands.append({'args':args,'exit':r.returncode,'stdout':r.stdout,'stderr':r.stderr,'wall_begin':start,'wall_end':time.time()});(root/'clock-control.json').write_text(json.dumps(commands,indent=2));return r
 power=run_control(['--query-gpu=power.limit','--format=csv,noheader,nounits']);assert power.returncode==0 and float(power.stdout.strip())==240
 identity=run_control(['--query-gpu=uuid,driver_version','--format=csv,noheader']);assert identity.returncode==0 and identity.stdout.strip()=='GPU-859dd34e-59f1-959a-2546-ca027062f9bd, 591.74'
 # Only previously-authorized GPU games/monitor utilities; no system/service/browser targets.
 ps=r"""[Console]::OutputEncoding=[System.Text.UTF8Encoding]::new($false);$OutputEncoding=[Console]::OutputEncoding;$names=@('RTSS','MSIAfterburner','HWiNFO64','GPU-Z','GPUZ','Taskmgr','GameBar','GameBarFTServer','GameBarPresenceWriter','wallpaper32','wallpaper64','RDR2','PlayRDR2');$out=@();foreach($n in $names){foreach($p in @(Get-Process -Name $n -ErrorAction SilentlyContinue)){$r=[ordered]@{name=$p.ProcessName;id=$p.Id;stopped=$false};try{Stop-Process -Id $p.Id -ErrorAction Stop;$r.stopped=$true}catch{$r.error=$_.Exception.GetType().Name};$out+=,$r}};[ordered]@{processes=$out;boot_utc=(Get-CimInstance Win32_OperatingSystem).LastBootUpTime.ToUniversalTime().ToString('o')}|ConvertTo-Json -Depth 4 -Compress"""
 encoded=base64.b64encode(ps.encode('utf-16le')).decode();r=subprocess.run(['/mnt/c/Windows/System32/WindowsPowerShell/v1.0/powershell.exe','-NoProfile','-NonInteractive','-EncodedCommand',encoded],capture_output=True,timeout=45);(root/'powershell.stdout.bin').write_bytes(r.stdout);(root/'powershell.stderr.bin').write_bytes(r.stderr);assert r.returncode==0;cleanup=json.loads(r.stdout.decode('utf-8-sig'));(root/'authorized-process-cleanup.json').write_text(json.dumps(cleanup,indent=2));assert cleanup['boot_utc']==plan['physical_environment']['windows_boot_utc'],'boot changed: obtain new actualHAGS before GPU diagnostic'
 idle=[];consecutive=0
 for _ in range(10):
  r=run_control(['--query-gpu=utilization.gpu','--format=csv,noheader,nounits']);idle.append(r.stdout);consecutive=consecutive+1 if r.returncode==0 and int(r.stdout.strip())==0 else 0
  if consecutive==3:break
  time.sleep(1)
 assert consecutive==3,'GPU busy: no diagnostic timing started'
 support(root,prefix,plan['fixed_replays']);protocol={**plan,'actual_process_cleanup':cleanup,'idle_admission':idle,'controlled_support_files':{str(p.relative_to(root)):sha(p) for p in (root/'qualification').rglob('*') if p.is_file()},'production_release':False};(root/'protocol.json').write_text(json.dumps(protocol,indent=2));pair_sha=sha(root/'protocol.json')
 from qualification.source_runner import SourceRunner
 from kernel_agent.optimization.cases import make_cases,case_key
 from kernel_agent.optimization.evaluate import witness
 from kernel_agent.optimization.identity import group_log_ratio
 workloads,contract,_=make_cases(prefix/'kernel_agent/optimization/assets');workload=next(w for w in workloads if w['id']=='dev-03');ledgers=json.loads((root/'public-ledgers.json').read_text());ledger=ledgers['candidate'];d=prefix/'kernel_agent/data/artifacts'/(plan['default_sha256']+'.so');b=prefix/'kernel_agent/data/artifacts'/(plan['fallback_sha256']+'.so');runner=SourceRunner(d,sha(d),incumbent=b,incumbent_sha256=sha(b),pair_sha256=pair_sha);blocks=[];status='DIAGNOSTIC_INCOMPLETE'
 try:
  assert run_control(['-i','0','--lock-gpu-clocks=1200,1200']).returncode==0;assert run_control(['-i','0','--lock-memory-clocks=7001,7001']).returncode==0
  before=run_control(['-q','-d','POWER,TEMPERATURE,PERFORMANCE,CLOCK']);(root/'physical-before.txt').write_text(before.stdout)
  for block,mode in enumerate(plan['block_order']):
   stop=threading.Event();samples=[]
   def monitor():
    while not stop.is_set():
     t=time.time();m=time.perf_counter();r=subprocess.run(['nvidia-smi','--query-gpu=timestamp,utilization.gpu,clocks.sm,clocks.mem,temperature.gpu,power.draw,memory.used','--format=csv,noheader,nounits'],capture_output=True,text=True,timeout=10);samples.append({'wall_begin':t,'wall_end':time.time(),'query_wall_ms':(time.perf_counter()-m)*1000,'exit':r.returncode,'stdout':r.stdout,'stderr':r.stderr});stop.wait(.5)
   thread=threading.Thread(target=monitor,daemon=True)
   if mode=='on':thread.start()
   path=root/('block-'+str(block)+'-'+mode)
   try:r,record=runner.run({'mode':'aa','session':block,'cases':contract,'workloads':[workload],'diagnostic_backends':{'incumbent':'cublas','candidate':'cublas'}},path)
   finally:
    stop.set()
    if thread.ident:thread.join(15)
    (root/('block-'+str(block)+'-nvml.json')).write_text(json.dumps(samples,indent=2))
   assert not thread.is_alive() and r and record['status']=='WITNESSES_READY';assert len(r['candidate_aa'])==1;row=r['candidate_aa'][0];assert row['input_count']==16 and row['replays']==plan['fixed_replays']
   for key in ['checks_before','checks_after']:
    for value,case in zip(r[key],contract):witness(value,case,ledger)
   values=[ledger[case_key({'shape_mnk':workload['shape_mnk'],'seed':701+i,'distribution':'random','offset_guard':False})] for i in range(16)];assert row['input_hashes']==[v['input_hashes'] for v in values]
   for role in ['incumbent','candidate']:assert row['output_hashes'][role]==[values[i%16]['output_sha256'] for i in range(32)]
   host=[];raw=[];cross=[]
   for pair in range(9):
    s={x['label']:x for x in row['segments'] if x['pair']==pair};device=s['incumbent']['elapsed_ms']/s['candidate']['elapsed_ms'];h=s['incumbent']['host_wall_ms']/s['candidate']['host_wall_ms'];rw=s['incumbent']['raw_wall_ms']/s['candidate']['raw_wall_ms'];host.append(abs(h-1));raw.append(abs(rw-1));cross.append(abs(device-h))
   blocks.append({'block':block,'monitor':mode,'max_event_pair_deviation':group_log_ratio(row,block)[1],'max_host_pair_deviation':max(host),'max_raw_pair_deviation':max(raw),'max_event_host_ratio_difference':max(cross),'nvml_samples':len(samples),'max_nvml_query_ms':max((x['query_wall_ms'] for x in samples),default=None)});(root/'progress.json').write_text(json.dumps(blocks,indent=2));run_control(['--query-gpu=clocks.sm,clocks.mem,temperature.gpu,utilization.gpu','--format=csv,noheader,nounits'])
  status='DIAGNOSTIC_COMPLETE';after=run_control(['-q','-d','POWER,TEMPERATURE,PERFORMANCE,CLOCK']);(root/'physical-after.txt').write_text(after.stdout)
 except Exception:(root/'failure.log').write_text(traceback.format_exc())
 finally:
  gpu_reset=run_control(['-i','0','--reset-gpu-clocks']).returncode;memory_reset=run_control(['-i','0','--reset-memory-clocks']).returncode
 result={'status':status,'protocol_sha256':pair_sha,'blocks':blocks,'gpu_reset_exit':gpu_reset,'memory_reset_exit':memory_reset,'formal_qualification':False,'production_release':False,'new_model_calls':0,'heldout_executed':False};(root/'result.json').write_text(json.dumps(result,indent=2));print(json.dumps(result))
if __name__=='__main__':main()
