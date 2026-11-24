"""CPU-only replay of source/artifact/archive, independent-output witnesses and A/A decisions.
Numerical FP64 acceptance remains explicitly attributed to the trusted remote host.
"""
import argparse,ast,hashlib,json,math,re,struct
from pathlib import Path
from evaluators.gemm_decision import ROOT,canonical,group_log_ratio,identity,protected_hashes,sha
from experiments.p3_01.cases import make_cases,case_key
from experiments.p3_01.patches import apply_reviewable_patch
from experiments.p3_01.source_packets import unpack
from kernel_agent.protocol import decode_frame
from kernel_agent.build_artifacts import ARCHIVE_LIMIT,decode_artifacts
from kernel_agent.bounded_io import confirmed_removed


def require(ok,message):
    if not ok:raise ValueError(message)


def archive(folder):
    execution=json.loads((folder/'execution.json').read_text())
    for rel,meta in execution['manifest'].items():
        path=Path(rel);require(not path.is_absolute() and '..' not in path.parts,'unsafe evidence path')
        f=folder/'remote'/path;require(f.stat().st_size==meta['bytes'] and sha(f)==meta['sha256'],'evidence integrity mismatch: '+rel)
    for rel,digest in json.loads((folder/'source-hashes.json').read_text()).items():require(sha(folder/'sources'/rel)==digest,'source staging mismatch')
    return execution


def function(path,name):
    return next(n for n in ast.parse(path.read_text()).body if isinstance(n,ast.FunctionDef) and n.name==name)


def source_ast(folder):
    fixtures=folder/'sources/experiments/p3_01/fixtures.py'
    for name in ['make_inputs','guarded']:
        require(ast.dump(function(fixtures,name))==ast.dump(function(ROOT/'evaluators/gemm_regression.py',name)),'original fixture changed: '+name)
    old=function(ROOT/'evaluators/gemm_candidate.py','paired');new=function(folder/'sources/experiments/p3_01/source_timing.py','paired_witness')
    old_with=next(n for n in old.body if isinstance(n,ast.With));new_with=next(n for n in new.body if isinstance(n,ast.With))
    # Only two post-measurement comparisons become hashes; capture/warmup/Events are byte-for-AST identical.
    require(ast.dump(ast.Module(body=old_with.body[:-2],type_ignores=[]))==ast.dump(ast.Module(body=new_with.body[:-2],type_ignores=[])),'timed path changed')
    return {'original_fixture_functions':2,'timed_graph_event_ast_unchanged':True}


def boundary(record,job):
    cmd=record['command']
    for flag in ['--network=none','--sig-proxy=false','--read-only','--log-driver=none','--cap-drop=ALL','--security-opt=no-new-privileges','--pids-limit=32','--memory=2g','--memory-swap=2g','--cpus=2']:
        require(cmd.count(flag)==1,'runtime boundary missing: '+flag)
    require(cmd[cmd.index('--user')+1]=='65534:65534','worker UID changed')
    require(cmd.count('--mount')==2 and all('readonly' in cmd[i+1] for i,x in enumerate(cmd) if x=='--mount'),'writable host mount')
    require(record['protected_unchanged'] and record['cleanup']['removed'] and confirmed_removed(record['cleanup']['after']),'runtime lifecycle/protection failed')
    require(record['exit']==0 and record['reason'] is None,'runtime failed')
    for stream in ['stdout','stderr']:
        size=(job/'capture'/(stream+'.bin')).stat().st_size
        require(size==record['stored_bytes'][stream] and size<=record['limits'][stream],'pipe bound failed')


def npy_digest(raw,shape):
    require(raw[:8]==b'\x93NUMPY\x01\x00','expected standard NPY1')
    n=struct.unpack('<H',raw[8:10])[0];header=ast.literal_eval(raw[10:10+n].decode('ascii'));data=raw[10+n:]
    require(header=={'descr':'<f4','fortran_order':False,'shape':tuple(shape)},'NPY shape/dtype/order mismatch')
    require(len(data)==math.prod(shape)*4,'NPY byte count mismatch')
    require(all(math.isfinite(x[0]) for x in struct.iter_unpack('<f',data)),'nonfinite output')
    return hashlib.sha256(data).hexdigest()


def staged(record,job,folder,binary):
    request=json.loads((job/'inputs/request.json').read_text())
    allowed={'mode','cases','candidate_sha256'}|({'session','workloads'} if request['mode']=='aa' else set())
    require(set(request)==allowed and request['candidate_sha256']==binary,'candidate received unexpected request data')
    expected={
        'artifact/candidate/nka_gemm_v1.so':binary,
        'inputs/request.json':sha(job/'inputs/request.json'),
        'artifact/protocol.py':sha(folder/'sources/kernel_agent/protocol.py'),
        **{'artifact/'+name:sha(folder/'sources/experiments/p3_01'/name) for name in ['source_worker.py','fixtures.py','source_packets.py','source_timing.py']}}
    require(record['protected_hashes']==expected,'mounted worker/request/artifact differs from staged source')


def witness(row,case,ledger):
    require(all(row.get(k)==v for k,v in case.items()),'witness case differs')
    item=ledger[case_key(case)]
    require(row['input_hashes']==item['input_hashes'] and row['output_sha256']==item['output_sha256'],'witness differs from independently checked output')
    guard=[hashlib.sha256(struct.pack('<e',7.0)*34).hexdigest()]*2 if case['offset_guard'] else []
    require(row['guard_hashes']==guard and row['fresh_output'] is True and row['producer_consumer_stream'] is True,'contract witness failed')


def audit(folder,supplement=None):
    folder=Path(folder);sup=Path(supplement) if supplement else None
    first=archive(folder);original=json.loads((ROOT/'evidence/20261122-p1-05-final/bindings.json').read_text())
    require(protected_hashes()==original['harness'] and sha(ROOT/'evidence/20261122-p1-04-feedback/freeze-current.json')==original['freeze_sha256'],'protected original/best changed')
    deployed=json.loads((ROOT/'evidence/20261122-p3-02-install-rollback-dev2/source-hashes.json').read_text())
    # Later CLI subcommands may be added; replay the unchanged default selection/runtime policy.
    for rel in ['kernel_agent/registry.py','kernel_agent/deployment.py','kernel_agent/data/releases.json']:
        require(sha(ROOT/rel)==deployed[rel],'deployed default policy changed: '+rel)
    stats=source_ast(folder);run=folder/'remote/run';result=json.loads((run/'result.json').read_text());binding=result['candidate'];candidate=binding['candidate'];body={k:v for k,v in candidate.items() if k!='id'}
    require(candidate['id']==hashlib.sha256(canonical(body)).hexdigest() and binding['parent_id']==identity(4)['id'],'candidate/parent binding failed')
    raw=ROOT/'evidence/20261122-p3-01-deepseek-source/response.json';response=json.loads(raw.read_text())
    # Provider records response content under its native choices field.
    content=response['choices'][0]['message']['content']
    patched=apply_reviewable_patch(content,(ROOT/'kernels/gemm/wmma_sm86.cu').read_bytes(),recount=True)
    require(sha(raw)==binding['raw_response_sha256'] and patched['source_sha256']==candidate['files']['kernels/gemm/wmma_sm86.cu'],'model/source identity mismatch')
    require(sha(folder/'sources/candidate/wmma_sm86.cu')==patched['source_sha256'],'staged candidate changed')
    build=json.loads((run/'build/record.json').read_text());meta,packet=decode_frame(run/'build/capture/stdout.bin',ARCHIVE_LIMIT);artifacts=decode_artifacts(packet)
    binary=result['binary_sha256'];require(build['binary_sha256']==binary and hashlib.sha256(artifacts['nka_gemm_v1.so']).hexdigest()==binary and sha(run/'build/artifacts/nka_gemm_v1.so')==binary,'fresh binary mismatch')
    require(meta['gpu_available'] is False and '--gpus' not in build['command'] and 'NVIDIA_VISIBLE_DEVICES=void' in build['command'],'builder GPU access')
    require(build['source_unchanged'] and build['deps_unchanged'] and build['cleanup']['removed'],'builder protection/lifecycle failed')
    pins=json.loads((ROOT/'provenance/upstream-reviews/20261121-p1-02/source-record.json').read_text())
    for item in pins:
        if item['repo']=='ScalingIntelligence/KernelBench' and item['path'] in ['src/kernelbench/eval.py','src/kernelbench/timing.py','src/kernelbench/dataset.py']:
            require(item['revision']=='423217d9fda91e0c2d67e4a43bf62f96f6d104f1' and build['source_hashes']['upstream/kernelbench/'+Path(item['path']).name]==item['sha256'],'fixed KernelBench loader pin mismatch')
    require(build['source_hashes']['evaluators/gemm_regression.py']==original['harness']['evaluators/gemm_regression.py'],'fixed CUDA loader harness changed')
    for name in ['wmma_sm86.cu','cublas_sm86.cu','gemm_contract.cuh']:
        require(build['source_hashes']['kernels/gemm/'+name]==sha(folder/'sources/candidate'/name),'compiled source mismatch')
    workloads,contract,cases=make_cases(ROOT);ledger=json.loads((run/'oracle-ledger.json').read_text());require(len(ledger)==160,'oracle ledger incomplete')
    checks=0
    for group in [w['id'] for w in workloads]+['boundaries']:
        selected=[c for c in cases if c['group']==group];job=run/('qa-'+group);report=json.loads((job/'report.json').read_text());record=json.loads((job/'record.json').read_text());boundary(record,job);staged(record,job,folder,binary)
        require(report['binary_sha256']==binary and report['cpu_reference_available'] is False and len(report['checks'])==len(selected),'QA report binding failed')
        limits={c['id']+'.npy':c['shape_mnk'][0]*c['shape_mnk'][1]*4+16384 for c in selected};frame,payload=decode_frame(job/'capture/stdout.bin',sum(limits.values())+2*1024*1024);packet_report,outputs=unpack(payload,limits)
        require(packet_report==report and frame['binary_sha256']==binary,'QA frame differs')
        for row,case in zip(report['checks'],selected):
            data=outputs[case['id']+'.npy'];require(data==(job/'outputs'/(case['id']+'.npy')).read_bytes(),'saved QA output differs')
            require(npy_digest(data,case['shape_mnk'][:2])==ledger[case_key(case)]['output_sha256'],'output ledger differs')
            require(ledger[case_key(case)]['binary_sha256']==binary and math.isfinite(ledger[case_key(case)]['max_abs_error']),'oracle ledger artifact/error invalid')
            witness(row,case,ledger);checks+=1
    require(checks==160 and result['qa_checks']==160,'QA count mismatch')
    current=run;sanitizers=[];aa=[];unstable=[];final=None
    if sup:
        archive(sup);current=sup/'remote/run';link=json.loads((current/'supplement-binding.json').read_text())
        require(link['binary_sha256']==binary and link['previous_result_sha256']==sha(run/'result.json') and link['qa_new_gpu_calls']==0,'supplement changed source/QA')
        require(json.loads((current/'oracle-ledger.json').read_text())==ledger,'supplement CPU recheck differs')
        for tool in ['memcheck','synccheck']:
            job=current/('sanitizer-'+tool);record=json.loads((job/'record.json').read_text());boundary(record,job);staged(record,job,sup,binary)
            require(record['status']=='WITNESSES_READY' and record['source_binary_sha256']==binary,'sanitizer did not produce accepted output')
            require('--log-file' in record['command'] and re.search(r'^========= ERROR SUMMARY: 0 errors\s*$',(job/'capture/stderr.bin').read_text(),re.MULTILINE),'sanitizer zero summary absent')
            report=json.loads((job/'report.json').read_text());selected=[c for c in contract if c['group']=='boundaries' and c['seed']==171]
            require(len(report['checks'])==len(selected),'sanitizer coverage incomplete')
            for row,case in zip(report['checks'],selected):witness(row,case,ledger)
            sanitizers.append({'tool':tool,'checks':len(selected),'status':'PASS'})
        for session in range(3):
            job=current/('aa-'+str(session))
            if not (job/'report.json').exists():continue
            record=json.loads((job/'record.json').read_text());boundary(record,job);staged(record,job,sup,binary);report=json.loads((job/'report.json').read_text())
            require(record['status']=='WITNESSES_READY' and report['binary_sha256']==binary and report['session']==session and report['graph_launches']==32 and report['replays']==12,'AA binding invalid')
            for phase in ['checks_before','checks_after']:
                require(len(report[phase])==32,'AA contract coverage incomplete')
                for row,case in zip(report[phase],contract):witness(row,case,ledger)
            grid={(w['id'],count) for w in workloads for count in [1,16]};rows=report['candidate_aa']
            require(len(rows)==16 and {(r['workload'],r['input_count']) for r in rows}==grid,'AA grid incomplete')
            for row in rows:
                shape=next(w['shape_mnk'] for w in workloads if w['id']==row['workload']);count=row['input_count']
                require(row['shape_mnk']==shape and row['capture_order']==(['incumbent','candidate'] if session%2==0 else ['candidate','incumbent']),'AA shape/order differs')
                values=[ledger[case_key({'shape_mnk':shape,'seed':701+i,'distribution':'random','offset_guard':False})] for i in range(count)]
                require(row['input_hashes']==[v['input_hashes'] for v in values],'AA input witness invalid')
                for label in ['incumbent','candidate']:require(row['output_hashes'][label]==[values[i%count]['output_sha256'] for i in range(32)],'AA output witness invalid')
                mean,deviation=group_log_ratio(row,session);gm=math.exp(mean)
                item={'session':session,'workload':row['workload'],'input_count':count,'geomean':gm,'max_pair_deviation':deviation}
                aa.append(item)
                if deviation>=.02 or abs(gm-1)>=.015:unstable.append(item)
        final=json.loads((current/'result.json').read_text()) if (current/'result.json').exists() else None
    aa_status=('FAIL_STABILITY' if unstable else 'PASS') if len(aa)==48 else 'NOT_ACCEPTED'
    if final:
        require(final['aa_status']==aa_status and final['unstable_groups']==unstable and final['incumbent_after']==identity(4)['id'],'raw decision replay differs')
    elif sup:
        progress=json.loads((current/'progress.json').read_text())
        require(progress['unstable']==unstable,'raw/progress stability decision differs')
        error=(sup/'remote/stderr.log').read_text()
        require("NameError: name 'files' is not defined" in error and len(aa)==48,'unexpected interrupted evaluation; cannot finalize')
    usage=json.loads((ROOT/'evidence/20261122-p3-01-deepseek-source/usage.json').read_text())
    require(usage['status']=='FAILED' and usage['calls_attempted']==1 and usage['usage']['total_tokens']==2232,'original model failure/usage changed')
    return {'status':'PASS','kind':'CPU-only archive/source/NPY/hash-witness/lifecycle/raw-Event replay; FP64 numerical acceptance from independent trusted host',
      **stats,'candidate_id':candidate['id'],'binary_sha256':binary,'fresh_builds':1,'qa_outputs':checks,'sanitizers':sanitizers,'aa_sessions':len(aa)//16,'aa_raw_segments':len(aa)*18,'aa_status':aa_status,'unstable_groups':unstable,
      'max_pair_deviation':max((r['max_pair_deviation'] for r in aa),default=None),'verdict':'REJECT' if aa_status!='PASS' else 'NO_PROMOTION_PENDING_AB','actual_best_unchanged':True,'default_cublas_unchanged':True,
      'original_api_failure_preserved':True,'original_sanitizer_protocol_failure_preserved':result['aa_status']=='NOT_RUN_SANITIZER_NOT_ACCEPTED','source_model_calls':1,'source_tokens':2232,'heldout_executed':False,'performance_claim':False,'production_release':False}


def main():
    p=argparse.ArgumentParser();p.add_argument('--evidence',type=Path,required=True);p.add_argument('--supplement',type=Path);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    if a.output.exists():p.error('Use a new output')
    result=audit(a.evidence,a.supplement);a.output.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result))


if __name__=='__main__':main()
