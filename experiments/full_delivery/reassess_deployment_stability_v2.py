"""Post-observation CPU reanalysis under a user-revised deployment policy; no promotion."""
import hashlib
import json
import math
from pathlib import Path
from kernel_agent.optimization.identity import group_log_ratio

ROOT = Path(__file__).resolve().parents[2]


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def reanalyze(folder, policy):
    archive = folder / 'segmented-archive'
    raw = archive / 'remote'
    execution = json.loads((archive / 'execution.json').read_text())
    collection = json.loads((archive / 'collection.json').read_text())
    assert sha(archive / 'raw.zip') == collection['archive_sha256']
    for n, m in execution['manifest'].items():
        p = raw / n
        assert p.stat().st_size == m['bytes'] and sha(p) == m['sha256']
    protocol_path = raw / 'protocol.json'
    protocol_sha = sha(protocol_path)
    frozen = json.loads((folder / 'protocol.freeze.json').read_text())
    assert protocol_sha == frozen['protocol_sha256']
    protocol = json.loads(protocol_path.read_text())
    assert sha(raw / 'host.py') == frozen['evaluator_sha256'] == protocol['evaluator_sha256']
    result = json.loads((raw / 'result.json').read_text())
    assert result['protocol_sha256'] == protocol_sha and result['production_release'] is False
    workload_path = ROOT / 'workloads/gemm_development_v1.json'
    ids = {w['id'] for w in json.loads(workload_path.read_text())['workloads']}
    expected_grid = {(w, n) for w in ids for n in policy['input_counts']}
    groups = []
    sessions = []
    for path in sorted(raw.glob('aa-*/report.json')):
        report = json.loads(path.read_text())
        session = report['session']
        assert report['mode'] == 'aa' and report['status'] == 'PASS'
        assert report['pair_sha256'] == protocol_sha
        assert report['candidate_binary_sha256'] == protocol['default_binary_sha256']
        assert report['incumbent_binary_sha256'] == protocol['fallback_binary_sha256']
        rows = report['candidate_aa']
        assert len(rows) == len(expected_grid) and {(r['workload'], r['input_count']) for r in rows} == expected_grid
        sessions.append(session)
        for row in rows:
            mean, maximum = group_log_ratio(row, session)
            geomean_deviation = abs(math.exp(mean) - 1)
            groups.append({'session': session, 'workload': row['workload'],
                           'input_count': row['input_count'], 'pair_deviation': maximum,
                           'geomean_deviation': geomean_deviation,
                           'pass_v2': maximum < policy['aa_pair_deviation_limit_exclusive'] and
                                      geomean_deviation < policy['aa_group_geomean_deviation_limit_exclusive']})
    complete = len(sessions) == 3 and set(sessions) == {0, 1, 2}
    raw_aa_segments = len(groups) * 18
    assert result['aa_sessions'] == len(sessions) and result['ab_sessions'] == 0
    assert result['raw_segments'] == raw_aa_segments
    return {'source_folder': str(folder.relative_to(ROOT)), 'source_protocol_sha256': protocol_sha,
            'source_archive_sha256': collection['archive_sha256'],
            'original_status': result['status'], 'original_pair_limit': protocol['aa_pair_limit'],
            'complete_three_aa': complete, 'aa_sessions': sessions,
            'raw_aa_segments': raw_aa_segments, 'ab_sessions': 0,
            'max_pair_deviation': max(g['pair_deviation'] for g in groups),
            'max_group_geomean_deviation': max(g['geomean_deviation'] for g in groups),
            'aa_stability_pass_under_v2': complete and all(g['pass_v2'] for g in groups),
            'groups': groups, 'production_release': False}


def main():
    policy_path = ROOT / 'workloads/gemm_deployment_stability_v2.json'
    policy = json.loads(policy_path.read_text())
    output = ROOT / 'evidence/20261123-deployment-stability-v2'
    output.mkdir(exist_ok=False)
    folders = [ROOT / 'evidence' / n for n in [
        '20261123-cublas-final-qualification', '20261123-cublas-locked-qualification',
        '20261123-cublas-sustained-qualification',
        '20261123-cublas-hags-qualification/live-parent-continuation']]
    values = [reanalyze(folder, policy) for folder in folders]
    result = {'status': 'CPU_RETROSPECTIVE_REANALYSIS_COMPLETE',
              'scope': 'A/A component only, revised after observing data; no prospective confirmation or A/B',
              'policy_id': policy['policy_id'], 'policy_sha256': sha(policy_path),
              'reanalysis_source_sha256': sha(__file__), 'batches': values,
              'old_qualifications_overwritten': False, 'new_gpu_samples': 0,
              'new_model_calls': 0, 'certificate_generated': False, 'production_release': False}
    (output / 'policy.json').write_bytes(policy_path.read_bytes())
    (output / 'reanalysis.py').write_bytes(Path(__file__).read_bytes())
    (output / 'verification.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps({**{k: v for k, v in result.items() if k != 'batches'},
                      'batches': [{k:v for k,v in b.items() if k != 'groups'} for b in values]}))


if __name__ == '__main__':
    main()
