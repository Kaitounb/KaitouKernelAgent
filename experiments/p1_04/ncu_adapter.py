# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""CUDA collector adapted from pinned KernelAgent; original CSV/parser/prompt reused."""
import csv
import hashlib
import importlib.util
import io
import json
import math
import os
from pathlib import Path
import subprocess
import signal
import sys
import tempfile
import time

ROOT=Path(__file__).resolve().parents[2]
UPSTREAM=ROOT/'provenance/upstream-reviews/20261118/meta-pytorch__KernelAgent/files/kernel_perf_agent/kernel_opt/profiler/ncu_profiler.py'
SCHEMA=UPSTREAM.parent.parent/'diagnose_prompt/metric_schema.py'
HARDWARE_METRICS=[
 'sm__cycles_active.avg','sm__warps_active.avg.pct_of_peak_sustained_active',
 'sm__pipe_tensor_cycles_active.avg.pct_of_peak_sustained_elapsed',
 'dram__bytes_read.sum','dram__bytes_write.sum','dram__throughput.avg.pct_of_peak_sustained_elapsed',
 'lts__t_sector_hit_rate.pct',
 'smsp__warp_issue_stalled_long_scoreboard_per_warp_active.pct',
 'smsp__warp_issue_stalled_short_scoreboard_per_warp_active.pct',
 'smsp__warp_issue_stalled_barrier_per_warp_active.pct','gpu__time_duration.sum']
LAUNCH_METRICS=['launch__registers_per_thread','launch__shared_mem_per_block_allocated',
 'launch__occupancy_limit_registers','launch__occupancy_limit_blocks','launch__occupancy_limit_shared_mem',
 'launch__grid_size','launch__block_size','launch__waves_per_multiprocessor']
UNSUPPORTED_PROBE='smsp__warp_issue_stalled_memory_dependency_per_warp_active.pct'


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_upstream():
    # The supplemental source record/lock archives this exact original file.
    if sha(UPSTREAM)!='44e65a3a1fea674972e31717246ff98b807f49b488548835c21ff7e373f3f41a':
        raise ValueError('pinned KernelAgent profiler drift')
    spec=importlib.util.spec_from_file_location('nka_original_ncu',UPSTREAM)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    return module


def load_schema():
    if sha(SCHEMA)!='92f9d73856c0ea1cd8fe36a3e67337d3947d3c255a4207242a884eace24c5489':
        raise ValueError('pinned metric schema drift')
    spec=importlib.util.spec_from_file_location('nka_original_metric_schema',SCHEMA)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    return module


def select_metrics(query_text):
    selected=[m for m in HARDWARE_METRICS if m in query_text]
    unavailable={m:'not enumerated by this GPU/NCU metric query' for m in [*HARDWARE_METRICS,UNSUPPORTED_PROBE] if m not in query_text}
    # launch__ metrics are software launch statistics, not enumerated by --query-metrics.
    return selected+LAUNCH_METRICS,unavailable


def profile_cuda(script,workdir,tag,metrics,ncu,timeout=180,kernel_regex='.*gemm_wmma.*'):
    """Retain upstream collector's subprocess/CSV design, expose CUDA-specific controls."""
    workdir=Path(workdir);raw=workdir/(tag+'.csv');report=workdir/(tag+'.ncu-rep')
    if raw.exists() or report.exists():raise FileExistsError('use a new profile tag')
    cmd=[str(ncu),'--csv','--page=raw','--print-units=base','--kernel-name-base=demangled',
         '--kernel-name=regex:'+kernel_regex,'--target-processes=all','--replay-mode=kernel',
         '--profile-from-start=off','--cache-control=all','--clock-control=none',
         '--log-file='+str(raw.resolve()),'--export='+str(report.resolve()),
         '--metrics='+','.join(metrics),'--launch-count=3',sys.executable,str(Path(script).resolve())]
    env={k:v for k,v in os.environ.items() if k in ['HOME','LANG','PATH','LD_LIBRARY_PATH','PYTHONPATH','CUDA_HOME','TORCH_CUDA_ARCH_LIST','FLASHINFER_WORKSPACE_BASE','FLASHINFER_CUDA_ARCH_LIST','MAX_JOBS','OMP_NUM_THREADS']}
    env['PYTHONPATH']=':'.join(sys.path)
    record={'command':cmd,'cache_control':'all','clock_control':'none','replay':'kernel',
            'scope':'profile-only diagnostic; never a formal Event score','metrics':metrics}
    started=time.monotonic()
    proc=subprocess.Popen(cmd,cwd=workdir,env=env,stdout=subprocess.PIPE,stderr=subprocess.PIPE,text=True,start_new_session=True)
    try:
        stdout,stderr=proc.communicate(timeout=timeout)
        record.update(exit=proc.returncode,stdout=stdout,stderr=stderr)
    except subprocess.TimeoutExpired as error:
        record.update(exit=None,timed_out=True,stdout=(error.stdout or b'').decode(errors='replace') if isinstance(error.stdout,bytes) else error.stdout,
                      stderr=(error.stderr or b'').decode(errors='replace') if isinstance(error.stderr,bytes) else error.stderr)
    finally:
        record['cleanup']=stop_profile_session(proc.pid)
        if proc.poll() is None:
            try:proc.wait(timeout=3)
            except subprocess.TimeoutExpired:record['cleanup']['removed']=False
    record['wall_seconds']=time.monotonic()-started
    (workdir/(tag+'.execution.json')).write_text(json.dumps(record,indent=2)+'\n')
    if not record['cleanup']['removed']:raise RuntimeError('NCU cleanup unknown; stop GPU work and inspect owned session')
    if record.get('exit')!=0:raise RuntimeError('NCU collection failed; original output retained')
    if not raw.is_file() or not report.is_file():raise RuntimeError('NCU CSV/report missing')
    return raw


def stop_profile_session(session_id):
    """Only processes in the collector's new Linux session, including ninja child groups."""
    def members():
        result={}
        for path in Path('/proc').iterdir():
            if not path.name.isdigit():continue
            try:
                raw=(path/'stat').read_text();fields=raw[raw.rfind(')')+2:].split()
                if int(fields[3])==session_id and fields[0]!='Z':result[int(path.name)]=int(fields[2])
            except (OSError,ValueError,IndexError):continue
        return result
    initial=members()
    for sig in [signal.SIGTERM,signal.SIGKILL]:
        current=members()
        if not current:break
        for group in set(current.values()):
            if group==os.getpgrp():raise RuntimeError('refusing to stop controller group')
            try:os.killpg(group,sig)
            except ProcessLookupError:pass
        time.sleep(.2)
    remaining=members();return {'removed':not remaining,'session_id':session_id,'owned_remaining_at_exit':sorted(initial),'remaining_running':sorted(remaining)}


def parse_profile(path,metrics,expected_warps,expected_launches=3,expected_kernel=None):
    """Only shard launch rows; numeric/unit cleaning and JSON conversion stay upstream."""
    upstream=load_upstream()
    lines=Path(path).read_text().splitlines(keepends=True)
    start=next((i for i,line in enumerate(lines) if line.lstrip().startswith(('"ID",','ID,'))),None)
    if start is None:raise ValueError('missing NCU CSV header')
    rows=list(csv.reader(io.StringIO(''.join(lines[start:]))));header=rows[0]
    if 'Kernel Name' not in header:raise ValueError('kernel identity missing')
    id_index=header.index('ID');kernel_index=header.index('Kernel Name')
    data=[r for r in rows[1:] if len(r)==len(header) and r[id_index].isdigit()]
    if len(data)!=expected_launches or len({r[id_index] for r in data})!=len(data):
        raise ValueError('unexpected launch count or duplicate launch IDs')
    kernel_identity=expected_kernel if expected_kernel is not None else f'gemm_wmma<{expected_warps}>'
    if not isinstance(kernel_identity,str) or not kernel_identity or any(kernel_identity not in r[kernel_index] for r in data):
        raise ValueError('unexpected profiled kernel')
    unit_rows=[r for r in rows[1:] if len(r)==len(header) and not r[id_index].strip()]
    units={m:(unit_rows[0][header.index(m)] if unit_rows and m in header else None) for m in metrics}
    frames=[]
    with tempfile.TemporaryDirectory(prefix='nka-ncu-shards-') as temp:
        for i,row in enumerate(data):
            shard=Path(temp)/f'{i}.csv'
            with shard.open('w',newline='') as file:
                writer=csv.writer(file);writer.writerow(header);writer.writerow(row)
            frame=upstream.load_ncu_metrics(shard,columns=metrics,extra_keep=['ID','Kernel Name'],select=upstream.MetricSelectionPolicy.LAST)
            if len(frame)!=1:raise ValueError('upstream parser did not return one launch')
            frames.append(frame)
    table=upstream.pd.concat(frames,ignore_index=True)
    prompt=json.loads(upstream.metrics_to_prompt(table,round_digits=None,keep_cols=['ID',*metrics]))
    launches=[]
    for record in table.to_dict(orient='records'):
        observed={}
        for metric in metrics:
            value=record.get(metric)
            if value is None or not isinstance(value,(int,float)) or not math.isfinite(float(value)):
                observed[metric]={'value':None,'unit':units[metric],'status':'unavailable','reason':'missing or nonfinite in actual CSV'}
            else:observed[metric]={'value':float(value),'unit':units[metric],'status':'observed'}
        launches.append({'id':int(record['ID']),'kernel':record['Kernel Name'],'metrics':observed})
    schema=load_schema()
    labels={key:{'section':section,'label':label,'schema_unit':unit} for section,entries in schema.NCU_METRIC_SECTIONS.items() for label,key,unit in entries if key in metrics}
    return {'upstream_profiler_sha256':sha(UPSTREAM),'upstream_schema_sha256':sha(SCHEMA),'upstream_labels':labels,'csv_sha256':sha(path),'launches':launches,
            'upstream_prompt_metrics':prompt,'scope':'NCU kernel replay; cache-control all; no formal latency/speedup claim'}
