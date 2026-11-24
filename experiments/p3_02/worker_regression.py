"""Run bounded trusted GPU fixtures in low-privilege Docker; archive explicit evidence only."""
import argparse,base64,hashlib,json
from pathlib import Path
from tools.ssh_3070 import execute_python


def main():
    p=argparse.ArgumentParser();p.add_argument('--evidence',type=Path,required=True);a=p.parse_args()
    if a.evidence.exists():p.error('Use a new evidence directory')
    a.evidence.mkdir(parents=True)
    sources={name:Path(__file__).with_name(name).read_bytes() for name in ['worker.py','supervisor.py']}
    for name,data in sources.items():(a.evidence/name).write_bytes(data)
    encoded={name:base64.b64encode(data).decode() for name,data in sources.items()}
    payload='SOURCES='+repr(encoded)+'\n'+r'''
import base64,json,subprocess,tempfile,stat,hashlib
from pathlib import Path
root=Path(tempfile.mkdtemp(prefix='nka-p3-02-runtime-'));code=root/'code';code.mkdir()
for name,data in SOURCES.items():(code/name).write_bytes(base64.b64decode(data))
work=root/'run'
cmd=['/home/dministrator/cuda-lab/.venv/bin/python',str(code/'supervisor.py'),'--work',str(work)]
p=subprocess.run(cmd,capture_output=True,text=True,timeout=900)
files={};entries=[]
if work.exists():
 for item in work.rglob('*'):
  rel=item.relative_to(work)
  if rel.parts[0] in ['control','artifact']:continue
  info=item.lstat()
  if stat.S_ISLNK(info.st_mode):entries.append({'path':str(rel),'type':'symlink','target':str(item.readlink())});continue
  if not stat.S_ISREG(info.st_mode):continue
  if item.name not in ['result.json','result.partial.json','stdout.log','stderr.log','worker.json','ready.json','request.json','result.npy']:continue
  if info.st_size>8*1024*1024:raise ValueError('evidence file too large')
  data=item.read_bytes();files[str(rel)]=base64.b64encode(data).decode();entries.append({'path':str(rel),'type':'file','bytes':len(data),'sha256':hashlib.sha256(data).hexdigest()})
print(json.dumps({'remote_work':str(root),'command':cmd,'exit':p.returncode,'stdout':p.stdout,'stderr':p.stderr,'files':files,'entries':entries}))
'''
    (a.evidence/'remote-driver.py').write_text(payload)
    r=execute_python(payload,timeout=1000)
    # Raw transport contains fixture data only; no credential/control files are collected.
    (a.evidence/'transport.log').write_text(r.stdout+r.stderr)
    if r.returncode:raise RuntimeError('remote transport failed; see transport.log')
    result=json.loads(r.stdout);run=a.evidence/'run';run.mkdir()
    for name,data in result.pop('files').items():
        rel=Path(name)
        if rel.is_absolute() or '..' in rel.parts:raise ValueError('invalid evidence path')
        target=run/rel;target.parent.mkdir(parents=True,exist_ok=True);target.write_bytes(base64.b64decode(data))
    (a.evidence/'execution.json').write_text(json.dumps(result,indent=2)+'\n')
    (a.evidence/'sources.json').write_text(json.dumps({name:hashlib.sha256(data).hexdigest() for name,data in sources.items()},indent=2)+'\n')
    print(json.dumps({k:result[k] for k in ['remote_work','exit','stdout','stderr']}))
    if result['exit']:raise RuntimeError('supervisor failed')
    report=json.loads((run/'result.json').read_text());print(json.dumps({'status':report['status'],'cases':len(report['cases']),'failures':report['failures']}))


if __name__=='__main__':main()
