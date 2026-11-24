"""CPU-only verification of installed dual-loader/FP64 receipts/raw Event diagnostic evidence."""
import argparse,hashlib,json,zipfile
from pathlib import Path
from evaluators.gemm_decision import ROOT,protected_hashes,group_log_ratio
from experiments.p3_01.audit import boundary,npy_digest,witness
from kernel_agent.optimization.cases import case_key
from kernel_agent.protocol import decode_frame
from kernel_agent.optimization.source_packets import unpack


def require(ok,message):
    if not ok:raise ValueError(message)


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def audit(folder):
    folder=Path(folder);run=folder/'remote/run';execution=json.loads((folder/'execution.json').read_text());result=json.loads((run/'result.json').read_text())
    require(execution['exit']==0 and result['status']=='PASS','installed run failed')
    for rel,meta in execution['manifest'].items():
        path=Path(rel);require(not path.is_absolute() and '..' not in path.parts,'invalid archive path');file=folder/'remote'/path
        require(file.stat().st_size==meta['bytes'] and sha(file)==meta['sha256'],'raw file changed: '+rel)
    sources=json.loads((folder/'source-hashes.json').read_text())
    for rel,digest in sources.items():require(sha(folder/'sources'/rel)==digest,'staged source changed')
    original=json.loads((ROOT/'evidence/20261122-p1-05-final/bindings.json').read_text());require(protected_hashes()==original['harness'] and sha(ROOT/'evidence/20261122-p1-04-feedback/freeze-current.json')==original['freeze_sha256'],'frozen original changed')
    wheel=run/'dist'/Path(result['wheel']).name;require(sha(wheel)==result['wheel_sha256'],'wheel differs')
    with zipfile.ZipFile(wheel) as archive:
        names=archive.namelist();require(not any('.private' in n or '__pycache__' in n or n.endswith('.pyc') or 'wrong-output' in n for n in names),'unintended wheel content')
        for rel,digest in sources.items():
            if rel.startswith('kernel_agent/'):require(hashlib.sha256(archive.read(rel)).hexdigest()==digest,'package differs from staged source')
        bundle=archive.read('kernel_agent/optimization/bundle.json');require(hashlib.sha256(bundle).hexdigest()==result['package']['bundle_sha256'],'bundle differs')
        for rel,digest in json.loads(bundle)['sha256'].items():require(hashlib.sha256(archive.read('kernel_agent/optimization/'+rel)).hexdigest()==digest,'bundle member differs')
        releases=json.loads(archive.read('kernel_agent/data/releases.json'));require(releases['default']=='sm86-cublas-container' and len(releases['versions'])==2 and all(v['backend']=='cublas' for v in releases['versions'].values()),'runtime activation/default changed')
    require(all(c['exit']==c['expected_exit'] and c['cwd']==str(Path(execution['remote_work'])/'run/empty') for c in result['commands']),'installed command/cwd failed')
    for label,count in [('cpu-test_optimizer',16),('cpu-test_pair_decision',19)]:
        log=(run/label/'stderr.log').read_text();require('Ran '+str(count)+' tests' in log and log.strip().endswith('OK'),'installed CPU test failure')
    native_wrong=ROOT/'evidence/20261122-p3-02-bounded-builder/remote/run/build-wrong-output/artifacts/nka_gemm_v1.so'
    require(sources['diagnostics/wrong-output.so']==sha(native_wrong),'unreviewed fault artifact')
    bindings=json.loads((run/'diagnostic-bindings.json').read_text());require(bindings['wrong']['candidate_sha256']==sha(native_wrong),'wrong diagnostic source differs')
    qa_verdicts=result['dual_import_qa_outcomes'];require(len(qa_verdicts)==16 and sum(not r['correct'] for r in qa_verdicts)==4 and all(r['correct']==r['expected_correct'] for r in qa_verdicts),'real fault verdict differs')
    oracle=json.loads((run/'diagnostic-oracles.json').read_text());qa_outputs=0;contracts=None
    qa_jobs=['correct-incumbent','correct-candidate','wrong-incumbent','wrong-candidate','sanitizer-memcheck','sanitizer-synccheck','diagnostic-contract']+['diagnostic-diagnostic-'+str(i)+'-'+role for i in [1,2] for role in ['incumbent','candidate']]
    def worker(job,mode):
        record=json.loads((job/'record.json').read_text());boundary(record,job);report=json.loads((job/'report.json').read_text());req=json.loads((job/'inputs/request.json').read_text())
        role=req.get('role','candidate');expected=req['incumbent_sha256'] if mode=='qa' and role=='incumbent' else req['candidate_sha256']
        require(report['binary_sha256']==expected and report['candidate_binary_sha256']==req['candidate_sha256'] and report['incumbent_binary_sha256']==req['incumbent_sha256'] and report['pair_sha256']==req['pair_sha256'],'worker binary/pair binding differs')
        require(report['module_identity_distinct'] and report['function_identity_distinct'] and report['cpu_reference_available'] is False,'dual import not independently bound')
        require(report['module_paths']=={r:'/artifact/'+r+'/nka_gemm_v1.so' for r in ['incumbent','candidate']} and report['module_names']=={r:'_nka_'+r+'.nka_gemm_v1' for r in ['incumbent','candidate']},'actual module path/name differs')
        expected_files={'artifact/'+name:sources['kernel_agent/optimization/'+name] for name in ['pair_worker.py','source_worker.py','source_timing.py','fixtures.py','source_packets.py']}
        expected_files.update({'artifact/protocol.py':sources['kernel_agent/protocol.py'],'artifact/incumbent/nka_gemm_v1.so':req['incumbent_sha256'],'artifact/candidate/nka_gemm_v1.so':req['candidate_sha256'],'inputs/request.json':sha(job/'inputs/request.json')})
        require(record['protected_hashes']==expected_files,'mounted code/input differs');require(report['backends']==req.get('diagnostic_backends',{'incumbent':'template','candidate':'template'}),'backend label differs')
        require(set(req)<= {'mode','role','session','cases','workloads','candidate_sha256','incumbent_sha256','pair_sha256','diagnostic_backends'},'unexpected data/reference in child request')
        return report,record,req
    for name in qa_jobs:
        job=run/name;report,record,req=worker(job,'qa');cases=req['cases']
        if name=='correct-incumbent':contracts=cases
        limits={c['id']+'.npy':c['shape_mnk'][0]*c['shape_mnk'][1]*4+16384 for c in cases};meta,body=decode_frame(job/'capture/stdout.bin',sum(limits.values())+2*1024*1024);raw_report,outputs=unpack(body,limits)
        require(raw_report==report and meta['pair_sha256']==req['pair_sha256'],'raw frame differs from report')
        for row,case in zip(report['checks'],cases):
            raw=outputs[case['id']+'.npy'];require(raw==(job/'outputs'/(case['id']+'.npy')).read_bytes(),'saved NPY differs');digest=npy_digest(raw,case['shape_mnk'][:2]);require(digest==row['output_sha256'],'GPU output byte witness differs');qa_outputs+=1
            if name=='wrong-candidate':require(digest==hashlib.sha256(bytes(case['shape_mnk'][0]*case['shape_mnk'][1]*4)).hexdigest(),'fault no longer returns zero')
            elif name!='correct-candidate' and not name.startswith('sanitizer-'):
                witness(row,case,oracle[req.get('role','candidate')])
        if name.startswith('sanitizer-'):require('ERROR SUMMARY: 0 errors' in (job/'capture/stderr.bin').read_text(),'sanitizer zero-error summary absent')
    require(qa_outputs==92 and result['diagnostic_graph_qa_checks']==64 and result['host_artifact_hash_rejections']==2,'real diagnostic coverage changed')
    count=0;timing_checks=0
    if result['diagnostic_raw_segments']:
        report,record,req=worker(run/'diagnostic-ab','ab');require(report['backends']=={'incumbent':'template','candidate':'cublas'} and report['graph_launches']==32 and report['replays']==12 and report['session']==0,'diagnostic scope changed')
        require(len(report['measurements'])==4,'diagnostic grid incomplete')
        for phase in ['checks_before','checks_after']:
            for role in ['incumbent','candidate']:
                require(len(report[phase][role])==4,'timed contract coverage incomplete')
                for row,case in zip(report[phase][role],contracts):witness(row,case,oracle[role]);timing_checks+=1
        for row in report['measurements']:
            require(row['capture_order']==['incumbent','candidate'],'diagnostic order differs');group_log_ratio(row,0)
            for role in ['incumbent','candidate']:
                values=[oracle[role][case_key({'shape_mnk':row['shape_mnk'],'seed':701+i,'distribution':'random','offset_guard':False})] for i in range(row['input_count'])]
                require(row['input_hashes']==[v['input_hashes'] for v in values] and row['output_hashes'][role]==[values[i%row['input_count']]['output_sha256'] for i in range(32)],'diagnostic graph hashes differ')
            count+=len(row['segments'])
        monitor=json.loads((run/'gpu-monitor.json').read_text());idle=[r for r in monitor if r['phase']=='idle-before-diagnostic'];require(len(idle)>=3 and all(int(r['stdout'].split(',')[1].strip().split()[0])==0 for r in idle[-3:]),'diagnostic idle preflight absent')
    require(count==result['diagnostic_raw_segments'] and result['diagnostic_screening_eligible'] is False and result['real_candidate_aa_sessions']==0 and result['new_model_calls']==0 and result['frozen_best_updated'] is False and result['runtime_activation'] is False,'scope or frozen state changed')
    return {'status':'PASS','kind':'CPU-only wheel/manifest/frame/NPY/host-FP64-receipt/role-witness/lifecycle/raw-Event replay; not a new numerical or performance experiment',
            'version':result['version'],'wheel_sha256':result['wheel_sha256'],'bundle_sha256':result['package']['bundle_sha256'],'installed_cpu_methods':35,'dual_import_fault_qa_outcomes':16,'wrong_output_rejected':4,'fp64_checked_outputs_including_faults':qa_outputs,'sanitizer_checks':8,'sanitizer_errors':0,'artifact_hash_rejections_before_gpu':2,
            'diagnostic_raw_segments':count,'timed_contract_witness_checks':timing_checks,'diagnostic_graph_output_witnesses':256 if count else 0,'screening_eligible':False,'real_candidate_aa_sessions':0,'new_model_calls':0,'best_unchanged':True,'cuBLAS_default':True,'runtime_activation':False,'heldout_executed':False,'formal_performance_claim':False,'production_release':False}


def main():
    p=argparse.ArgumentParser();p.add_argument('--evidence',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    if a.output.exists():p.error('use a new output')
    r=audit(a.evidence);a.output.write_text(json.dumps(r,indent=2)+'\n');print(json.dumps(r))


if __name__=='__main__':main()
