"""CPU-only installed wheel/source/archive/lifecycle/NPY witness audit. Does not import native code."""
import argparse,hashlib,json,math,zipfile
from pathlib import Path
from evaluators.gemm_decision import ROOT,protected_hashes,identity
from experiments.p3_01.audit import boundary,npy_digest,witness
from experiments.p3_01.cases import make_cases,case_key
from kernel_agent.build_artifacts import ARCHIVE_LIMIT,decode_artifacts
from kernel_agent.protocol import decode_frame


def require(value,message):
    if not value:raise ValueError(message)


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def audit(folder):
    folder=Path(folder);remote=folder/'remote';run=remote/'run';execution=json.loads((folder/'execution.json').read_text());result=json.loads((run/'result.json').read_text())
    require(execution['exit']==0 and result['status']=='PASS','installed verification failed')
    for rel,meta in execution['manifest'].items():
        path=Path(rel);require(not path.is_absolute() and '..' not in path.parts,'unsafe archive path')
        f=remote/path;require(f.stat().st_size==meta['bytes'] and sha(f)==meta['sha256'],'whole-file evidence mismatch: '+rel)
    sources=json.loads((folder/'source-hashes.json').read_text())
    for rel,value in sources.items():require(sha(folder/'sources'/rel)==value,'staged source differs: '+rel)
    original=json.loads((ROOT/'evidence/20261122-p1-05-final/bindings.json').read_text());require(protected_hashes()==original['harness'],'protected original drift')
    require(sha(ROOT/'evidence/20261122-p1-04-feedback/freeze-current.json')==original['freeze_sha256'],'actual search freeze changed')
    cmds=result['commands'];require(all(c['exit']==c['expected_exit'] for c in cmds),'CLI unexpected exit')
    require(all(c['cwd']==str(Path(execution['remote_work'])/'run/empty') for c in cmds),'nonempty/repository cwd used')
    wheel=run/'dist'/Path(result['wheel']).name;require(sha(wheel)==result['wheel_sha256'],'wheel hash mismatch')
    with zipfile.ZipFile(wheel) as archive:
        names=archive.namelist();require(not any('.private' in n or '__pycache__' in n or n.endswith('.pyc') for n in names),'private or cached files packaged')
        for name,value in sources.items():
            if Path(name).name=='.DS_Store':
                require(name not in names,'Finder metadata packaged');continue
            if name.startswith('kernel_agent/'):require(hashlib.sha256(archive.read(name)).hexdigest()==value,'wheel source differs: '+name)
        bundle=json.loads(archive.read('kernel_agent/optimization/bundle.json'))
        for rel,digest in bundle['sha256'].items():require(hashlib.sha256(archive.read('kernel_agent/optimization/'+rel)).hexdigest()==digest,'installed bundle member changed')
        releases=json.loads(archive.read('kernel_agent/data/releases.json'));require(releases['default']=='sm86-cublas-container' and len(releases['versions'])==2 and all(x['backend']=='cublas' for x in releases['versions'].values()),'candidate entered deployment allowlist')
    known=json.loads((run/'known-run/run.json').read_text());require(known['status']=='REJECT' and known['known_decision_reused'] and known['calls_reserved']==0 and known['new_external_model_calls']==0,'known failure did not stop')
    strict=json.loads((run/'strict-run/run.json').read_text());require(strict['status']=='REJECT' and 'invalid_proposal' in strict['reason'],'strict format failure changed')
    checked=json.loads((run/'check-run/run.json').read_text());require(checked['status']=='CHECKS_COMPLETE' and checked['aa_status']=='NOT_RUN_CHECKS_ONLY' and checked['new_external_model_calls']==0,'checks scope mismatch')
    evaluation=run/'check-run/evaluation';summary=json.loads((evaluation/'result.json').read_text());binary=summary['binary_sha256'];require(not summary['aa_sessions'] and not list(evaluation.glob('aa-*')) and summary['verdict']=='CORRECTNESS_ONLY','maintenance ran performance/promotion')
    build=json.loads((evaluation/'build/record.json').read_text());metadata,payload=decode_frame(evaluation/'build/capture/stdout.bin',ARCHIVE_LIMIT);files=decode_artifacts(payload)
    require(metadata['gpu_available'] is False and build['cleanup']['removed'] and build['source_unchanged'] and build['deps_unchanged'],'builder isolation/lifecycle failure')
    require('--gpus' not in build['command'] and 'NVIDIA_VISIBLE_DEVICES=void' in build['command'],'builder had GPU access')
    require(hashlib.sha256(files['nka_gemm_v1.so']).hexdigest()==binary and sha(evaluation/'build/artifacts/nka_gemm_v1.so')==binary,'fresh native artifact differs')
    for name,digest in build['artifact_hashes'].items():require(hashlib.sha256(files[name]).hexdigest()==digest,'closed artifact reparse mismatch')
    ledger=json.loads((evaluation/'oracle-ledger.json').read_text());require(len(ledger)==160,'oracle ledger incomplete')
    workloads,contract,cases=make_cases(ROOT);count=0
    for group in [w['id'] for w in workloads]+['boundaries']:
        selected=[c for c in cases if c['group']==group];job=evaluation/('qa-'+group);record=json.loads((job/'record.json').read_text());boundary(record,job);report=json.loads((job/'report.json').read_text())
        require(report['binary_sha256']==binary and report['cpu_reference_available'] is False and len(report['checks'])==len(selected),'QA worker binding differs')
        req=json.loads((job/'inputs/request.json').read_text());require(set(req)=={'mode','cases','candidate_sha256'} and req['cases']==selected,'unexpected worker data/reference')
        for row,case in zip(report['checks'],selected):
            raw=(job/'outputs'/(case['id']+'.npy')).read_bytes();require(npy_digest(raw,case['shape_mnk'][:2])==ledger[case_key(case)]['output_sha256'],'NPY output witness differs')
            witness(row,case,ledger);require(ledger[case_key(case)]['binary_sha256']==binary and math.isfinite(ledger[case_key(case)]['max_abs_error']),'FP64 ledger binding differs');count+=1
    for tool in ['memcheck','synccheck']:
        job=evaluation/('sanitizer-'+tool);record=json.loads((job/'record.json').read_text());boundary(record,job);report=json.loads((job/'report.json').read_text())
        require('ERROR SUMMARY: 0 errors' in (job/'capture/stderr.bin').read_text() and record['status']=='WITNESSES_READY','sanitizer not accepted')
        selected=[c for c in contract if c['group']=='boundaries' and c['seed']==171];require(len(report['checks'])==8,'sanitizer incomplete')
        for row,case in zip(report['checks'],selected):witness(row,case,ledger)
    deployment=result['runtime_deployment'];require(deployment['status']=='PASS' and deployment['backend']=='cublas' and not deployment['promotion_performed'],'default deployment changed')
    require(result['new_model_calls']==0 and result['aa_segments']==0 and not result['heldout_executed'] and not result['production_release'],'scope exceeded')
    return {'status':'PASS','kind':'CPU-only installed wheel/source/ZIP/artifact/NPY/hash-witness/lifecycle audit; numerical FP64 verdict from independent trusted host',
      'version':result['version'],'wheel_sha256':result['wheel_sha256'],'installed_bundle_sha256':result['package']['bundle_sha256'],'archive_files':len(execution['manifest']),'installed_empty_cwd_commands':len(cmds),'qa_outputs':count,'sanitizers':result['sanitizers'],'fresh_candidate_binary_sha256':binary,
      'known_rejection_reused':True,'duplicate_command_no_retry':True,'actual_search_best_unchanged':True,'cuBLAS_default':True,'allowlist_versions':2,'new_model_calls':0,'aa_segments':0,'performance_claim':False,'heldout_executed':False,'production_release':False}


def main():
    p=argparse.ArgumentParser();p.add_argument('--evidence',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    if a.output.exists():p.error('use new audit output')
    value=audit(a.evidence);a.output.write_text(json.dumps(value,indent=2)+'\n');print(json.dumps(value))


if __name__=='__main__':main()
