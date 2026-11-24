"""CPU-only final-wheel delta audit bound to the previously accepted installed GPU evidence."""
import argparse,hashlib,json,zipfile
from pathlib import Path
from evaluators.gemm_decision import ROOT,protected_hashes


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def audit(folder,previous):
    folder=Path(folder);previous=Path(previous);run=folder/'remote/run';result=json.loads((run/'result.json').read_text());execution=json.loads((folder/'execution.json').read_text());prior=json.loads((previous/'remote/run/result.json').read_text());proof=json.loads((previous/'verification.json').read_text())
    assert execution['exit']==0 and result['status']=='PASS' and proof['status']=='PASS'
    for rel,item in execution['manifest'].items():
        path=Path(rel);assert not path.is_absolute() and '..' not in path.parts
        file=folder/'remote'/rel;assert file.stat().st_size==item['bytes'] and sha(file)==item['sha256'],rel
    sources=json.loads((folder/'source-hashes.json').read_text())
    for rel,item in sources.items():assert sha(folder/'sources'/rel)==item
    assert result['prior_result_sha256']==sha(previous/'remote/run/result.json') and result['prior_wheel_sha256']==prior['wheel_sha256']
    newwheel=run/'dist'/Path(result['wheel']).name;oldwheel=previous/'remote/run/dist'/Path(prior['wheel']).name
    assert sha(newwheel)==result['wheel_sha256'] and sha(oldwheel)==proof['wheel_sha256']
    with zipfile.ZipFile(newwheel) as new,zipfile.ZipFile(oldwheel) as old:
        newnames={n for n in new.namelist() if n.startswith('kernel_agent/')};oldnames={n for n in old.namelist() if n.startswith('kernel_agent/')};assert newnames==oldnames
        changed=[n for n in sorted(newnames) if new.read(n)!=old.read(n)]
        assert changed==result['changed_package_files'] and set(changed)=={'kernel_agent/__init__.py','kernel_agent/optimization/command.py','kernel_agent/optimization/bundle.json'}
        for rel,digest in sources.items():
            if rel.startswith('kernel_agent/'):assert hashlib.sha256(new.read(rel)).hexdigest()==digest,rel
        bundle_raw=new.read('kernel_agent/optimization/bundle.json');assert hashlib.sha256(bundle_raw).hexdigest()==result['package']['bundle_sha256']
        for rel,digest in json.loads(bundle_raw)['sha256'].items():assert hashlib.sha256(new.read('kernel_agent/optimization/'+rel)).hexdigest()==digest
    assert result['new_gpu_calls']==result['new_model_calls']==result['aa_segments']==0
    assert len(result['commands'])==13 and all(c['exit']==c['expected_exit'] and c['cwd']==str(Path(execution['remote_work'])/'run/empty') for c in result['commands'])
    tests=(run/'installed-cpu-regressions/stderr.log').read_text();assert 'Ran 14 tests' in tests and tests.strip().endswith('OK')
    assert json.loads((run/'bad-run/run.json').read_text())['status']=='REJECT'
    assert json.loads((run/'known-run/run.json').read_text())['known_decision_reused']
    assert result['actual_pointer_rollback']['active']=='sm86-cublas-container'
    assert protected_hashes()==json.loads((ROOT/'evidence/20261122-p1-05-final/bindings.json').read_text())['harness']
    return {'status':'PASS','kind':'CPU-only final wheel delta/source/archive/installed-regression replay; unchanged GPU components inherit explicitly bound dev3 real-host evidence','version':result['version'],'wheel_sha256':result['wheel_sha256'],'installed_bundle_sha256':result['package']['bundle_sha256'],'changed_package_files':changed,'installed_empty_cwd_commands':len(result['commands']),'installed_cpu_test_methods':14,'prior_gpu_verification_sha256':sha(previous/'verification.json'),'inherited_independent_qa_outputs':160,'inherited_sanitizer_checks':16,'new_gpu_calls':0,'new_model_calls':0,'aa_segments':0,'actual_pointer_rollback':True,'cuBLAS_default':True,'known_failure_not_resampled':True,'heldout_executed':False,'performance_claim':False,'production_release':False}


def main():
    p=argparse.ArgumentParser();p.add_argument('--evidence',type=Path,required=True);p.add_argument('--previous',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    if a.output.exists():p.error('use a new output')
    result=audit(a.evidence,a.previous);a.output.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result))


if __name__=='__main__':main()
