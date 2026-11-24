"""Record reviewed package resources; no downloads or private files, no execution."""
import hashlib,json
from pathlib import Path

root=Path(__file__).resolve().parents[1];package=root/'kernel_agent/optimization'
old=json.loads((package/'bundle.json').read_text())
if 'files' in old:(package/'reuse.json').write_text(json.dumps(old,indent=2)+'\n')
files={str(p.relative_to(package)):hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(package.rglob('*')) if p.is_file() and '__pycache__' not in p.parts and p.name not in ['bundle.json'] and p.suffix!='.pyc'}
(package/'bundle.json').write_text(json.dumps({'schema_version':1,'sha256':files},indent=2)+'\n')
print(json.dumps({'files':len(files),'bundle_sha256':hashlib.sha256((package/'bundle.json').read_bytes()).hexdigest()}))
