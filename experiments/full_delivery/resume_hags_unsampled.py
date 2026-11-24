"""Continue only a proven zero-QA/zero-Event identity with a live Windows/WSL SSH parent."""
import base64
import hashlib
import json
from pathlib import Path
from tools.ssh_3070 import ROOT, execute_python


def main():
    folder = ROOT / 'evidence/20261123-cublas-hags-qualification'
    output = folder / 'live-parent-continuation'
    output.mkdir(exist_ok=False)
    archive = folder / 'segmented-archive'
    execution = json.loads((archive / 'execution.json').read_text())
    assert all(hashlib.sha256((archive / 'remote' / n).read_bytes()).hexdigest() == m['sha256']
               for n, m in execution['manifest'].items())
    result = json.loads((archive / 'remote/result.json').read_text())
    assert result['unqualified_original_status'] == 'FAIL_BEFORE_QUALIFICATION_RESULT'
    assert not list((archive / 'remote').glob('*/record.json'))
    assert not list((archive / 'remote').glob('qa-*'))
    assert not list((archive / 'remote').glob('aa-*'))
    commands = json.loads((archive / 'remote/clock-control.json').read_text())
    assert len(commands) == 3 and all(x['exit'] == 1 and 'UtilAcceptVsock' in x['stderr'] for x in commands)
    receipt = {'status': 'UNSAMPLED_TRANSPORT_REPAIR', 'prior_archive_sha256':
               json.loads((archive / 'collection.json').read_text())['archive_sha256'],
               'prior_qa_checks': 0, 'prior_event_segments': 0,
               'protocol_unchanged': True, 'evaluator_unchanged': True,
               'repair': 'Retain foreground SSH/WSL parent for Windows executable interop; no detached GPU controller',
               'prior_failure_preserved': True, 'automatic_retry': False}
    (output / 'repair.json').write_text(json.dumps(receipt, indent=2))
    files = {n: base64.b64encode((folder / n).read_bytes()).decode()
             for n in ['protocol.json', 'host.py', 'runtime-witness.json']}
    setup = 'FILES=' + repr(files) + '\n' + r'''
import base64,json,tempfile
from pathlib import Path
root=Path(tempfile.mkdtemp(prefix='nka-cublas-final-hags-live-'))
for n,v in FILES.items():(root/n).write_bytes(base64.b64decode(v))
print(json.dumps({'remote_root':str(root)}))
'''
    r = execute_python(setup, 60)
    assert r.returncode == 0
    remote = json.loads(r.stdout)['remote_root']
    (output / 'location.json').write_text(json.dumps({'remote_root': remote}, indent=2))
    print(json.dumps({'status': 'CONTINUING_UNSAMPLED_PROTOCOL', 'remote_root': remote}), flush=True)
    payload = 'ROOT=' + repr(remote) + '\n' + r'''
import fcntl,hashlib,json,os,subprocess
from pathlib import Path
root=Path(ROOT);p=json.loads((root/'protocol.json').read_text());w=json.loads((root/'runtime-witness.json').read_text())
lock=open('/tmp/nka-reviewed-gpu-execution.lock','a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
script="[Console]::OutputEncoding=[System.Text.Encoding]::UTF8; @{BootUtc=(Get-CimInstance Win32_OperatingSystem).LastBootUpTime.ToUniversalTime().ToString('o')} | ConvertTo-Json -Compress"
b=subprocess.run(['/mnt/c/Windows/System32/WindowsPowerShell/v1.0/powershell.exe','-NoProfile','-NonInteractive','-Command',script],capture_output=True,text=True,timeout=30)
assert b.returncode==0 and json.loads(b.stdout)['BootUtc']==w['windows_boot_utc']
raw=Path('/mnt/c/Windows/Temp')/Path(w['remote_raw_path']).name
assert hashlib.sha256(raw.read_bytes()).hexdigest()==w['raw_sha256']
g=subprocess.run(['/usr/lib/wsl/lib/nvidia-smi','--query-gpu=uuid,driver_version,name','--format=csv,noheader'],capture_output=True,text=True,timeout=15)
assert g.returncode==0 and g.stdout.strip()==w['gpu_identity']
image='sha256:ce00f6dc2eea43dc508b90b548f624646d643a92d363aa715fdb5a3b4f49290a'
d=subprocess.run(['docker','image','inspect',image,'--format','{{.Id}}'],capture_output=True,text=True,timeout=30)
assert d.returncode==0 and d.stdout.strip()==image
env={k:v for k,v in os.environ.items() if k in ['HOME','PATH','LANG','LD_LIBRARY_PATH','WSL_INTEROP','WSL_DISTRO_NAME']};env.update(OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1')
proof={'status':'PASS','boot_utc':w['windows_boot_utc'],'gpu_identity':g.stdout.strip(),'actual_hags_enabled':True,'image_id':image,'runtime_witness_sha256':hashlib.sha256((root/'runtime-witness.json').read_bytes()).hexdigest(),'live_parent':True}
(root/'launch-environment.json').write_text(json.dumps(proof,indent=2));fcntl.flock(lock,fcntl.LOCK_UN);lock.close()
with (root/'host.stdout').open('w') as out,(root/'host.stderr').open('w') as err:
 r=subprocess.run(['/home/dministrator/cuda-lab/.venv/bin/python',str(root/'host.py'),'--root',str(root)],cwd=root,env=env,stdin=subprocess.DEVNULL,stdout=out,stderr=err,timeout=3600)
result=json.loads((root/'result.json').read_text()) if (root/'result.json').exists() else {'status':'FAIL_BEFORE_RESULT','exit':r.returncode}
print(json.dumps({'remote_root':str(root),'exit':r.returncode,'result':result}))
'''
    (output / 'driver.py').write_text(payload)
    r = execute_python(payload, 4200)
    (output / 'transport.stdout').write_text(r.stdout)
    (output / 'transport.stderr').write_text(r.stderr)
    if r.returncode:
        raise RuntimeError('Transport failed: collect existing identity, never relaunch sampled protocol')
    value = json.loads(r.stdout)
    (output / 'result.json').write_text(json.dumps(value, indent=2))
    print(json.dumps(value))


if __name__ == '__main__':
    main()
