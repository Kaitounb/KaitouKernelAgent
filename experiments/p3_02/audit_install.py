"""Offline wheel/source/CLI-result audit; no model, GPU, credentials or NumPy required."""
import argparse,hashlib,json,zipfile
from pathlib import Path


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def audit(folder):
    folder=Path(folder);remote=folder/'remote';envelope=json.loads((folder/'execution.json').read_text());result=json.loads((remote/'run/result.json').read_text())
    if result['status']!='PASS' or result['failures'] or result['independent_correctness_calls']!=15:raise ValueError('incomplete installed regression')
    for rel,meta in envelope['manifest'].items():
        path=Path(rel)
        if path.is_absolute() or '..' in path.parts or sha(remote/path)!=meta['sha256']:raise ValueError('archive hash changed')
    wheels=list((remote/'dist').glob('*.whl'))
    if len(wheels)!=1:raise ValueError('wheel missing/nonunique')
    hashes=json.loads((folder/'source-hashes.json').read_text())
    with zipfile.ZipFile(wheels[0]) as archive:
        for rel,expected in hashes.items():
            if rel.startswith('kernel_agent/'):
                info=archive.getinfo(rel)
                if info.file_size>8*1024*1024 or hashlib.sha256(archive.read(rel)).hexdigest()!=expected:raise ValueError('wheel content differs from source')
            if not rel.endswith('.so') and sha(folder/'sources'/rel)!=expected:raise ValueError('staged source mismatch')
    commands=json.loads((remote/'commands.json').read_text())
    if [r['tag'] for r in commands] not in [['wheel','install','regression'],['wheel','install','regression','policy-tests']] or any(r['exit']!=0 for r in commands):raise ValueError('build/install incomplete')
    if '--no-build-isolation' not in commands[0]['command'] or '--no-deps' not in commands[0]['command'] or '--no-index' not in commands[1]['command']:raise ValueError('offline install flags changed')
    if not Path(result['installed_module']['module']).is_relative_to(Path(envelope['remote_work'])/'installed'):raise ValueError('module not from installed prefix')
    rows={r['tag']:r for r in result['cases']}
    for tag,row in rows.items():
        if row.get('independent_correctness') is False:raise ValueError('independent correctness failure')
    for tag in ['missing-active','corrupt-active','incompatible-manifest','execution-crash','execution-timeout']:
        row=rows[tag];response=row['response']
        if row['exit'] or not row['independent_correctness'] or not response['fallback_used'] or response['executed_version']!='sm86-cublas-original' or response['production_release'] or response['promotion_performed']:raise ValueError('fallback verdict mismatch')
        if tag.startswith('execution-') and len(response['attempts'])!=2:raise ValueError('execution fallback was not a fresh second worker')
    if rows['rollback-to-original']['response']['pointer']['active']!='sm86-cublas-original' or rows['rolled-back-original']['response']['binary_sha256']!='93fb2f4344521b7e9b246b54d1c30d4b1a8facf731e6385e18f6119e9c49c72c':raise ValueError('rollback identity mismatch')
    for tag in ['corrupt-fallback-fails-closed','reject-nonfinite','reject-out-of-domain']:
        if rows[tag]['exit']==0:raise ValueError('negative control accepted')
    if not result['pointer_unchanged_after_failures'] or result['heldout_executed'] or result['production_release']:raise ValueError('state/scope differs')
    policy_path=folder/'cpu-policy-tests.json'
    if not policy_path.exists():policy_path=remote/'cpu-policy-tests.json'
    policy=json.loads(policy_path.read_text())
    if policy['exit']!=0 or 'Ran 4 tests' not in policy['stderr']:raise ValueError('CPU mocked policy tests incomplete')
    return {'status':'PASS','cpu_mocked_policy_tests':4,'kind':'CPU-only wheel/source/hash/CLI lifecycle audit; numerical verdicts from independent host FP64 evaluator',
      'installed_correctness_calls':15,'fallback_scenarios':5,'version_rollback_observed':True,'negative_controls':3,
      'default_backend':'cuBLAS','artifact_promotion':False,'development_only':True,'production_release':False,'heldout_executed':False,
      'model_calls':0,'wheel':str(wheels[0]),'wheel_sha256':sha(wheels[0]),'p3_02_complete':False}


def main():
    p=argparse.ArgumentParser();p.add_argument('--evidence',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    if a.output.exists():p.error('Use a new audit output')
    result=audit(a.evidence);a.output.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result))


if __name__=='__main__':main()
