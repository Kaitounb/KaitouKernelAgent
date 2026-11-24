"""CPU-only physical-environment binding plus unchanged sustained gate replay."""
import hashlib
import json
from pathlib import Path
from experiments.full_delivery.audit_cublas_sustained_qualification import audit
ROOT = Path(__file__).resolve().parents[2]


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def verify_archive(folder):
    archive = folder / 'segmented-archive'
    execution = json.loads((archive / 'execution.json').read_text())
    receipt = json.loads((archive / 'collection.json').read_text())
    assert sha(archive / 'raw.zip') == receipt['archive_sha256']
    for name, meta in execution['manifest'].items():
        p = archive / 'remote' / name
        assert p.stat().st_size == meta['bytes'] and sha(p) == meta['sha256']
    return archive / 'remote', receipt


def main():
    folder = ROOT / 'evidence/20261123-cublas-hags-qualification'
    r, collection = verify_archive(folder)
    abort = json.loads((r / 'result.json').read_text())
    assert abort['status'] == 'CLOCK_ENVIRONMENT_REJECTED'
    assert abort['unqualified_original_status'] == 'FAIL_BEFORE_QUALIFICATION_RESULT'
    assert not any(r.glob('*/record.json')) and not any(r.glob('qa-*')) and not any(r.glob('aa-*'))
    commands = json.loads((r / 'clock-control.json').read_text())
    assert len(commands) == 3 and all(x['exit'] == 1 and 'UtilAcceptVsock' in x['stderr'] for x in commands)
    value = {'status': 'AUDIT_PASS_PREFLIGHT_ABORT', 'qa_checks': 0, 'event_segments': 0,
             'raw_files_verified': collection['raw_files_verified'],
             'archive_sha256': collection['archive_sha256'], 'qualification_pass': False,
             'production_release': False}
    (folder / 'verification-preflight-abort.json').write_text(json.dumps(value, indent=2))
    live = folder / 'live-parent-continuation'
    r, _ = verify_archive(live)
    for name in ['protocol.json', 'host.py', 'runtime-witness.json']:
        assert (r / name).read_bytes() == (folder / name).read_bytes()
    protocol = json.loads((r / 'protocol.json').read_text())
    previous_path = ROOT / 'evidence/20261123-cublas-sustained-qualification/protocol.json'
    previous = json.loads(previous_path.read_text())
    allowed = {'previous_sustained_rejection_protocol_sha256', 'installed_prefix', 'scope',
               'physical_environment', 'launch_controller_sha256'}
    assert {k for k in protocol.keys() | previous.keys() if protocol.get(k) != previous.get(k)} == allowed
    assert protocol['previous_sustained_rejection_protocol_sha256'] == sha(previous_path)
    witness = json.loads((r / 'runtime-witness.json').read_text())
    physical = protocol['physical_environment']
    assert sha(r / 'runtime-witness.json') == physical['runtime_witness_sha256']
    assert physical['hags_enabled'] is True and witness['actual_adapter']['hags_enabled'] is True
    proof = json.loads((r / 'launch-environment.json').read_text())
    assert proof['status'] == 'PASS' and proof['actual_hags_enabled'] is True and proof['live_parent'] is True
    assert proof['boot_utc'] == witness['windows_boot_utc'] == physical['windows_boot_utc']
    assert proof['gpu_identity'] == physical['gpu_identity'] == witness['gpu_identity']
    assert proof['runtime_witness_sha256'] == physical['runtime_witness_sha256']
    assert sha(ROOT / 'evidence/20261123-hags-postboot-recovery/result.json') == physical['recovery_receipt_sha256']
    assert sha(ROOT / 'evidence/20261123-hags-postboot-clock-probe/result.json') == physical['finite_clock_probe_sha256']
    for name in ['protocol.freeze.json', 'protocol.json', 'host.py']:
        target = live / name
        if target.exists(): assert target.read_bytes() == (folder / name).read_bytes()
        else: target.write_bytes((folder / name).read_bytes())
    actual = json.loads((r / 'result.json').read_text())
    if actual['status'] == 'GPU_BUSY_UNSAMPLED_REMAINDER':
        from experiments.full_delivery.audit_cublas_hags_partial import audit as partial_audit
        result = partial_audit(live)
    else:
        result = audit(live)
    value = {'status': result['status'], 'qualification': result,
             'physical_environment_binding': 'PASS', 'preflight_abort_preserved': True,
             'prior_preflight_event_segments': 0, 'protocol_and_evaluator_unchanged_on_continuation': True,
             'heldout_executed': False, 'new_model_calls': 0, 'certificate_generated': False,
             'production_release': False}
    (folder / 'verification.json').write_text(json.dumps(value, indent=2))
    print(json.dumps(value))


if __name__ == '__main__':
    main()
