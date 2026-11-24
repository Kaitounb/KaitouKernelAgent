"""Trusted fixed-SDK deployment timing, adapted from the accepted mla_paired two-Graph path.

No generated source is loaded. Runtime Docker boundaries are checked separately.
"""
import argparse,hashlib,importlib.util,json,math,sys,time,traceback
from pathlib import Path
import numpy as np
import torch
from kernel_agent.operators import flashinfer_backend as current,references as ref
from experiments.full_delivery.operator_deployment_cases import WORKLOADS,inputs


def thash(value):return hashlib.sha256(value.detach().cpu().contiguous().numpy().tobytes()).hexdigest()

def compare(value,target):
 value=value.detach().cpu().double();finite=torch.isfinite(target);assert torch.equal(torch.isneginf(value),torch.isneginf(target));delta=(value[finite]-target[finite]).abs();assert bool((delta<=2e-3+2e-3*target[finite].abs()).all());return float(delta.max()) if delta.numel() else 0.

def expected(w,host,extra):
 values=[torch.from_numpy(v) for v in host]
 if w['operation']=='attention':return [ref.attention(*(v.permute(0,2,1,3) for v in values),causal=w['causal']).permute(0,2,1,3)]
 return list(ref.mla_core(values[0],values[1],torch.from_numpy(extra['dense_cache']),torch.from_numpy(extra['dense_rope']),w['lengths'],192**-.5))


def paired(w,count,session,mode,root,baseline):
 started=time.time();cpu=[];extras=[];targets=[];folder=root/(mode+'-'+str(session)+'-outputs')/(w['id']+'-'+str(count));folder.mkdir(parents=True)
 for slot in range(count):
  data=inputs(w,slot);host,extra=(data,{}) if w['operation']=='attention' else data;cpu.append(host);extras.append(extra);targets.append(expected(w,host,extra))
 order=['incumbent','candidate'] if session%2==0 else ['candidate','incumbent'];modules={'incumbent':current if mode=='aa' else baseline,'candidate':current};stream=torch.cuda.Stream();stream.wait_stream(torch.cuda.current_stream());graphs={};rings={};storage={};results={};plans={};qa={};hashes={}
 with torch.no_grad(),torch.cuda.stream(stream):
  for role in order:
   module=modules[role];rings[role]=[];storage[role]=[];qa[role]=[]
   if w['operation']=='mla':
    extra=extras[0];plans[role]=module.SplitMLAPlan(lengths=w['lengths'],indptr=extra['indptr'],indices=extra['indices'],num_pages=len(extra['indices']),page_size=16,heads=w['heads'],scale=192**-.5,device='cuda',splits=1)
   for slot,host in enumerate(cpu):
    ring=[]
    for x in host:
     raw=torch.full((x.size+16,),7,dtype=torch.float16,device='cuda');value=raw[8:-8].view(x.shape);value.copy_(torch.from_numpy(x));ring.append(value);storage[role].append(raw)
    rings[role].append(ring)
   def call(role,slot):
    if w['operation']=='attention':return [modules[role].attention(*rings[role][slot],causal=w['causal'])]
    return list(plans[role].run(*rings[role][slot]))
   for slot in range(count):qa[role].append([compare(x,y) for x,y in zip(call(role,slot),targets[slot])])
   hashes[role]=[[thash(x) for x in ring] for ring in rings[role]]
  for role in order:
   graphs[role]=torch.cuda.CUDAGraph();results[role]=[]
   with torch.cuda.graph(graphs[role],stream=stream):
    for i in range(32):results[role].append(call(role,i%count))
   graphs[role].replay();stream.synchronize()
   for i,out in enumerate(results[role]):
    for x,y in zip(out,targets[i%count]):compare(x,y)
  deadline=time.monotonic()+1.
  while time.monotonic()<deadline:
   for role in order:graphs[role].replay()
   stream.synchronize()
  start,end=[torch.cuda.Event(enable_timing=True) for _ in range(2)];pilots=[]
  for _ in range(3):
   start.record(stream)
   for _ in range(4):graphs[order[0]].replay()
   end.record(stream);end.synchronize();pilots.append(start.elapsed_time(end))
  if any(not math.isfinite(v) or v<=0 for v in pilots):raise ValueError('invalid untimed-pilot Event duration')
  replays=max(1,min(8192,math.ceil(4000/sorted(pilots)[1])));segments=[]
  for pair in range(9):
   sequence=['incumbent','candidate'] if (pair+session)%2==0 else ['candidate','incumbent']
   for position,role in enumerate(sequence):
    start.record(stream)
    for _ in range(replays):graphs[role].replay()
    end.record(stream);end.synchronize();elapsed=start.elapsed_time(end);assert math.isfinite(elapsed) and elapsed>0
    segments.append({'pair':pair,'label':role,'order':position,'elapsed_ms':elapsed,'finished_at':time.time()})
  outputs={}
  for role in order:
   outputs[role]=[]
   for i,out in enumerate(results[role]):
    for x,y in zip(out,targets[i%count]):compare(x,y)
    outputs[role].append([thash(x) for x in out])
    if i<count:
     for j,x in enumerate(out):np.save(folder/(role+'-'+str(i)+'-'+str(j)+'.npy'),x.detach().cpu().numpy(),allow_pickle=False)
   assert [[thash(x) for x in ring] for ring in rings[role]]==hashes[role]
   assert all(bool((v[:8]==7).all()) and bool((v[-8:]==7).all()) for v in storage[role])
  native={str(Path(p).resolve().relative_to((root/'sdk').resolve())):hashlib.sha256(Path(p).read_bytes()).hexdigest() for p in torch.ops.loaded_libraries if Path(p).resolve().is_relative_to((root/'sdk').resolve())};assert native
 return {'workload':w['id'],'operation':w['operation'],'input_count':count,'capture_order':order,'segments':segments,'pilot_elapsed_ms':pilots,'replays':replays,'timed_calls_per_segment':32*replays,'input_hashes':hashes,'output_hashes':outputs,'qa_before_errors':qa,'qa_after_pass':True,'inputs_and_guards_unchanged':True,'native_hashes':native,'started_at':started,'finished_at':time.time(),'peak_allocated_bytes':torch.cuda.max_memory_allocated(),'output_folder':str(folder.relative_to(root))}


def main():
 p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--mode',choices=['aa','ab'],required=True);p.add_argument('--session',type=int,choices=[0,1,2],required=True);a=p.parse_args();torch.set_num_threads(1);spec=importlib.util.spec_from_file_location('kernel_agent.operators.baseline_adapter','/home/dministrator/cuda-lab/nka-persistent/dev6-d5f4bbf83dc0/prefix/kernel_agent/operators/flashinfer_backend.py');baseline=importlib.util.module_from_spec(spec);spec.loader.exec_module(baseline);rows=[];failure=None
 try:
  for w in WORKLOADS:
   for count in [1,16]:
    rows.append(paired(w,count,a.session,a.mode,a.root,baseline));(a.root/(a.mode+'-'+str(a.session)+'.progress.json')).write_text(json.dumps({'rows':rows},indent=2))
 except Exception:failure=traceback.format_exc()
 result={'status':'PASS' if failure is None else 'FAIL','mode':a.mode,'session':a.session,'rows':rows,'failure':failure,'torch':str(torch.__version__),'cuda':torch.version.cuda,'capability':list(torch.cuda.get_device_capability()),'reference_scope':'trusted fixed SDK controller, no generated candidate source','production_release':False};(a.root/(a.mode+'-'+str(a.session)+'.json')).write_text(json.dumps(result,indent=2));print(json.dumps({'status':result['status'],'session':a.session,'mode':a.mode,'rows':len(rows),'failure':failure}))
 if failure:raise SystemExit(1)

if __name__=='__main__':main()
