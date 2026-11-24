"""Bounded profiled small-M diagnostic; never an AA decision or qualification replay."""
import argparse,json,math,sys,time
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--prefix',type=Path,required=True);a=p.parse_args();prefix=a.prefix;sys.path[:0]=[str(prefix/'kernel_agent/optimization'),str(prefix/'kernel_agent'),str(prefix)]
import torch
from fixtures import make_inputs
from source_worker import load
torch.set_num_threads(1);digest='66fcc80f9122c919f92c72e37a01794a792c91a58b71d5705b160e0bc979dfc3';model=load(prefix/'kernel_agent/data/artifacts'/(digest+'.so'),digest).gemm_cublas;rows=[];stream=torch.cuda.Stream();torch.cuda.cudart().cudaProfilerStart()
with torch.no_grad(),torch.cuda.stream(stream):
 for repetition in range(3):
  # Fixed preconditioning represents the large GEMM preceding the small-M row.
  x,y=[v.cuda() for v in make_inputs([1024]*3,701)]
  for _ in range(8):model(x,y)
  stream.synchronize();g=torch.cuda.CUDAGraph()
  with torch.cuda.graph(g,stream=stream):large=[model(x,y) for _ in range(32)]
  deadline=time.monotonic()+5
  while time.monotonic()<deadline:g.replay();stream.synchronize()
  del large,g,x,y
  ring=[tuple(v.cuda() for v in make_inputs([1,4096,512],701+i)) for i in range(16)]
  for x,y in ring:model(x,y)
  stream.synchronize();graphs={};outputs={};capture=['A','B'] if repetition%2==0 else ['B','A']
  for label in capture:
   graphs[label]=torch.cuda.CUDAGraph();outputs[label]=[]
   with torch.cuda.graph(graphs[label],stream=stream):
    for i in range(32):outputs[label].append(model(*ring[i%16]))
  deadline=time.monotonic()+1
  while time.monotonic()<deadline:
   for label in capture:graphs[label].replay()
   stream.synchronize()
  start,end=[torch.cuda.Event(enable_timing=True) for _ in range(2)];pilot=[]
  for _ in range(3):
   start.record(stream)
   for _ in range(4):graphs[capture[0]].replay()
   end.record(stream);end.synchronize();pilot.append(start.elapsed_time(end))
  replays=max(1,min(8192,math.ceil(4000/sorted(pilot)[1])))
  for pair in range(9):
   for label in (['A','B'] if (pair+repetition)%2==0 else ['B','A']):
    name=f'dev03/count16/repeat{repetition}/pair{pair}/{label}';torch.cuda.nvtx.range_push(name);wall=time.time();before=time.perf_counter();start.record(stream)
    for _ in range(replays):graphs[label].replay()
    end.record(stream);submitted=time.perf_counter();end.synchronize();finished=time.perf_counter();torch.cuda.nvtx.range_pop()
    rows.append({'range':name,'repetition':repetition,'pair':pair,'label':label,'replays':replays,'gpu_event_ms':start.elapsed_time(end),'cpu_enqueue_ms':(submitted-before)*1000,'cpu_wait_ms':(finished-submitted)*1000,'wall_begin':wall,'wall_end':time.time()})
  del outputs,graphs,ring
 torch.cuda.synchronize()
torch.cuda.cudart().cudaProfilerStop();(a.root/'diagnostic-segments.json').write_text(json.dumps({'status':'DIAGNOSTIC_COMPLETE','scope':'Profiled native cuBLAS/100W small-M diagnostic, not container qualification','rows':rows,'production_release':False},indent=2))
