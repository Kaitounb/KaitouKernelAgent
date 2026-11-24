"""Read-only progress for the single foreground HAGS qualification."""
import json
from tools.ssh_3070 import ROOT, execute_python


def main():
    folder = ROOT / 'evidence/20261123-cublas-hags-qualification'
    root = json.loads((folder / 'live-parent-continuation/location.json').read_text())['remote_root']
    payload = 'ROOT=' + repr(root) + '\n' + r'''
import json,time
from pathlib import Path
p=Path(ROOT)
records=[]
for f in p.glob('*/record.json'):
 r=json.loads(f.read_text());records.append({'job':f.parent.name,'status':r.get('status'),'wall_seconds':r.get('wall_seconds')})
telemetry=json.loads((p/'clock-telemetry.json').read_text()) if (p/'clock-telemetry.json').exists() else []
print(json.dumps({'root':ROOT,'completed_jobs':sorted(records,key=lambda x:x['job']),'incomplete_jobs':[x.name for x in p.iterdir() if x.is_dir() and x.name.startswith(('qa-','aa-','ab-')) and not (x/'record.json').exists()],'telemetry_samples':len(telemetry),'latest_telemetry':telemetry[-1:] ,'stderr':(p/'host.stderr').read_text()[-2500:] if (p/'host.stderr').exists() else '', 'result':json.loads((p/'result.json').read_text()) if (p/'result.json').exists() else None,'observed_wall':time.time()}))
'''
    r = execute_python(payload, 60)
    assert r.returncode == 0
    v = json.loads(r.stdout)
    out = folder / 'progress'
    out.mkdir(exist_ok=True)
    (out / f"live-{int(v['observed_wall'])}.json").write_text(json.dumps(v, indent=2))
    print(json.dumps(v))


if __name__ == '__main__':
    main()
