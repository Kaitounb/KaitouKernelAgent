"""Freeze one new physical HAGS environment; reuse the sustained evaluator unchanged."""
import argparse
import base64
import hashlib
import json
from pathlib import Path
from tools.ssh_3070 import ROOT, execute_python


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    a.output.mkdir(exist_ok=False)
    previous = ROOT / 'evidence/20261123-cublas-sustained-qualification/protocol.json'
    runtime = ROOT / 'evidence/20261123-hags-after-user-reboot/runtime.json'
    recovery = ROOT / 'evidence/20261123-hags-postboot-recovery/result.json'
    probe = ROOT / 'evidence/20261123-hags-postboot-clock-probe/result.json'
    witness = json.loads(runtime.read_text())
    restored = json.loads(recovery.read_text())
    assert witness['status'] == 'HAGS_ENABLED' and witness['actual_adapter']['hags_enabled']
    assert restored['status'] == 'RESTORED_EXACT_ARCHIVED_BYTES'
    assert json.loads(probe.read_text())['status'] == 'PASS'
    host = ROOT / 'experiments/full_delivery/cublas_sustained_qualification_host.py'
    assert sha(host) == '7a10943be10156a18fcd49b85da941db36910ff52452cbb544d3947aa6232676'
    protocol = json.loads(previous.read_text())
    protocol.update(
        previous_sustained_rejection_protocol_sha256=sha(previous),
        installed_prefix=restored['installed_prefix'],
        scope='One new physically verified HAGS-on boot; identical cuBLAS binaries, sustained evaluator, sampling and thresholds; previous HAGS-off rejections immutable',
        physical_environment={
            'hags_enabled': True, 'windows_boot_utc': witness['windows_boot_utc'],
            'gpu_identity': witness['gpu_identity'],
            'runtime_witness_sha256': sha(runtime),
            'recovery_receipt_sha256': sha(recovery),
            'finite_clock_probe_sha256': sha(probe),
        },
        launch_controller_sha256=sha(__file__),
    )
    (a.output / 'protocol.json').write_text(json.dumps(protocol, indent=2))
    (a.output / 'host.py').write_bytes(host.read_bytes())
    (a.output / 'runtime-witness.json').write_bytes(runtime.read_bytes())
    (a.output / 'protocol.freeze.json').write_text(json.dumps({
        'protocol_sha256': sha(a.output / 'protocol.json'), 'evaluator_sha256': sha(host)
    }, indent=2))
    files = {n: base64.b64encode((a.output / n).read_bytes()).decode()
             for n in ['protocol.json', 'host.py', 'runtime-witness.json']}
    payload = 'FILES=' + repr(files) + '\n' + r'''
import base64,fcntl,hashlib,json,os,re,subprocess,tempfile
from pathlib import Path
lock=open('/tmp/nka-reviewed-gpu-execution.lock','a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
root=Path(tempfile.mkdtemp(prefix='nka-cublas-final-hags-'))
for n,v in FILES.items():(root/n).write_bytes(base64.b64decode(v))
w=json.loads((root/'runtime-witness.json').read_text());p=json.loads((root/'protocol.json').read_text())
raw=Path('/mnt/c/Windows/Temp')/Path(w['remote_raw_path']).name
assert hashlib.sha256(raw.read_bytes()).hexdigest()==w['raw_sha256']
rawtext=raw.read_bytes().decode('utf-16' if raw.read_bytes().startswith(b'\xff\xfe') else 'utf-8',errors='replace')
assert re.search(r'Hardware Scheduling:.*Supported:True Enabled:True',rawtext)
script="[Console]::OutputEncoding=[System.Text.Encoding]::UTF8; @{BootUtc=(Get-CimInstance Win32_OperatingSystem).LastBootUpTime.ToUniversalTime().ToString('o')} | ConvertTo-Json -Compress"
r=subprocess.run(['/mnt/c/Windows/System32/WindowsPowerShell/v1.0/powershell.exe','-NoProfile','-NonInteractive','-Command',script],capture_output=True,text=True,timeout=30)
assert r.returncode==0 and json.loads(r.stdout)['BootUtc']==w['windows_boot_utc']
g=subprocess.run(['/usr/lib/wsl/lib/nvidia-smi','--query-gpu=uuid,driver_version,name','--format=csv,noheader'],capture_output=True,text=True,timeout=15)
assert g.returncode==0 and g.stdout.strip()==w['gpu_identity']
image='sha256:ce00f6dc2eea43dc508b90b548f624646d643a92d363aa715fdb5a3b4f49290a'
d=subprocess.run(['docker','image','inspect',image,'--format','{{.Id}}'],capture_output=True,text=True,timeout=30)
assert d.returncode==0 and d.stdout.strip()==image
proof={'status':'PASS','boot_utc':w['windows_boot_utc'],'gpu_identity':g.stdout.strip(),'actual_hags_enabled':True,'image_id':image,'runtime_witness_sha256':hashlib.sha256((root/'runtime-witness.json').read_bytes()).hexdigest()}
assert proof['runtime_witness_sha256']==p['physical_environment']['runtime_witness_sha256']
(root/'launch-environment.json').write_text(json.dumps(proof,indent=2))
env={k:v for k,v in os.environ.items() if k in ['HOME','PATH','LANG','LD_LIBRARY_PATH']};env.update(OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1')
fcntl.flock(lock,fcntl.LOCK_UN);lock.close()
with (root/'host.stdout').open('w') as out,(root/'host.stderr').open('w') as err:
 child=subprocess.Popen(['/home/dministrator/cuda-lab/.venv/bin/python',str(root/'host.py'),'--root',str(root)],cwd=root,env=env,stdin=subprocess.DEVNULL,stdout=out,stderr=err,start_new_session=True)
receipt={'status':'LAUNCHED_ONCE','remote_root':str(root),'pid':child.pid,'protocol_sha256':hashlib.sha256((root/'protocol.json').read_bytes()).hexdigest(),'launch_environment':proof,'new_model_calls':0,'heldout_executed':False,'production_release':False}
(root/'launch.json').write_text(json.dumps(receipt,indent=2));print(json.dumps(receipt))
'''
    (a.output / 'driver.py').write_text(payload)
    r = execute_python(payload, 180)
    (a.output / 'transport.stdout').write_text(r.stdout)
    (a.output / 'transport.stderr').write_text(r.stderr)
    if r.returncode:
        raise RuntimeError('Inspect reserved identity; never automatically relaunch')
    receipt = json.loads(r.stdout)
    (a.output / 'launch.json').write_text(json.dumps(receipt, indent=2) + '\n')
    print(json.dumps(receipt))


if __name__ == '__main__':
    main()
