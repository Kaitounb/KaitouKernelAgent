"""Prepare a source-only GitHub checkout; frozen local releases/raw evidence remain untouched."""
import hashlib,json,re,shutil
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
DEST=ROOT/'.private/github-KaitouKernelAgent'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main():
 assert not DEST.exists();DEST.mkdir(mode=0o700);original={};excluded={'__pycache__','.git','node_modules','history'}
 def copy(p):
  rel=p.relative_to(ROOT)
  if p.is_symlink() or any(x in excluded for x in rel.parts) or p.name=='.DS_Store' or p.suffix in {'.pyc','.pyo'}:return
  assert p.stat().st_size<20*1024**2
  q=DEST/rel;q.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(p,q);original[str(rel)]=sha(p)
 for name in ['kernel_agent','experiments','tests','tools','kernels','evaluators','workloads','include','src','examples','provenance','docs','releases']:
  for p in (ROOT/name).rglob('*'):
   if p.is_file():copy(p)
 for name in ['pyproject.toml','PLAN.md','TASKS.md','SOURCES.md']:copy(ROOT/name)
 # Original runtime artifacts are small reviewed DSOs; multi-GB images/wheels/raw archives are never traversed.
 evidence=[
 'evidence/20261124-3060-next-qualification/verification.json',
 'evidence/20261124-3060-0.1.3rc2-installed/verification.json',
 'evidence/20261124-3060-next-independent/summary.json',
 'evidence/20261124-3060-next-independent/verification.json',
 'evidence/20261124-3060-0.1.3-installed/verification.json',
 'evidence/20261124-3060-ga-pid-supplement/verification.json',
 'evidence/20261124-3060-active-entry/verification.json',
 'evidence/20261124-3060-ncu-smoke-v2/verification.json',
 'evidence/20261124-3060-ncu-smoke-v2/parsed.json',
 'evidence/20261124-3060-ncu-smoke-v2/segmented-archive/remote/cublas.csv',
 'evidence/20261124-3060-final-package-review/result.json',
 'evidence/20261124-3060-ncu-closeout-review/result.json']
 for name in evidence:copy(ROOT/name)
 # These are summaries only; private independent manifests/arrays/credentials are intentionally absent.
 ignore='''.private/
.env
.env.*
!.env.example
.venv/
__pycache__/
*.py[cod]
.DS_Store
.pytest_cache/
.ruff_cache/
.cache/
build/
dist/
*.egg-info/
runs/
/artifacts/
/references/
*.partial
*.docker.tar.gz
''';(DEST/'.gitignore').write_text(ignore)
 (DEST/'THIRD_PARTY_NOTICES.md').write_text('''# Third-party code and attribution

This project reuses pinned components and reviewed SDK interfaces. See [the adoption record](provenance/UPSTREAM_REUSE.md) and [upstream lock](provenance/upstream-lock.json) for commits, symbols and adaptation scope. Original copyright headers, LICENSE and NOTICE files are retained alongside copied upstream files.

Core sources include KernelAgent (Apache-2.0), KernelBench (MIT), ForgeKernel (Apache-2.0), KDA (its documented code/assets license split), and FlashInfer-Bench (Apache-2.0). Operator references and bundled SDK wrappers retain their respective upstream license texts. These entries describe their own files; they do not imply every upstream framework is fully integrated.

No blanket repository-wide license is added in this publication. A file's existing license and upstream notices continue to apply. NVIDIA tools, CUDA/PyTorch libraries, large runtime images and private evaluation material are not redistributed here.
''')
 (DEST/'docs/GITHUB_PUBLICATION.md').write_text('''# Source publication scope

This is a curated source snapshot of the completed, scoped engineering delivery. It includes the Python runtime/agent, CUDA sources and reviewed small extension binaries, workload contracts, tests, experiment controllers, pinned upstream attribution and selected acceptance summaries. [Project closeout](PROJECT_CLOSEOUT.md) defines exactly what was verified and what remains deferred.

The original local workspace remains the full audit archive. Large Docker images, wheel/SDK recovery archives, downloaded NVIDIA tools, raw run directories, private independent workload manifests and SSH/API credentials are excluded from Git. References with incomplete redistribution terms are also excluded. Links into those omitted local archives document provenance and may not resolve in this source-only checkout; their SHA commitments and verified counts are retained in the published receipts. Selected NCU CSV/parsed metrics are included.

Qualified channels are bound to the original GPU UUID/driver/image/profile. Installing this source snapshot on a different machine does not establish that machine's production qualification. The GPU-specific controllers are one-shot historical experiment/release tools: configure your own environment and use a new output namespace. Do not replay old private acceptance data or silently weaken the gates. Runtime source files and included reviewed binaries retain their original hashes.

Local server credentials belong only in private files or environment variables; none are shipped. Public examples must use your own fresh development inputs. Original historical failures and raw evidence remain preserved locally.
''')
 # Gate publication on exact known connection passwords and common token/private-key formats.
 secrets=set();connection_files=list((ROOT/'.private').glob('*connection*.json'))+[ROOT/'.private/3070-connection.txt']
 for p in connection_files:
  if not p.is_file():continue
  text=p.read_text()
  if p.suffix=='.json':
   v=json.loads(text)
   for k in ['password','api_key','token']:
    if isinstance(v.get(k),str) and len(v[k])>=6:secrets.add(v[k].encode())
  else:
   for value in re.findall(r'密码[：:]\s*(\S+)',text):
    if len(value)>=6:secrets.add(value.encode())
 patterns=[rb'sk-[A-Za-z0-9_-]{16,}',rb'gh[pousr]_[A-Za-z0-9_]{20,}',rb'github_pat_[A-Za-z0-9_]{30,}',rb'-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----',rb'AKIA[A-Z0-9]{16}'];findings=[]
 for p in DEST.rglob('*'):
  if not p.is_file():continue
  b=p.read_bytes()
  if any(s in b for s in secrets) or any(re.search(q,b) for q in patterns):findings.append(str(p.relative_to(DEST)))
 report={'status':'PUBLICATION_SCAN_PASS' if not findings else 'PUBLICATION_SCAN_REJECT','files':len([p for p in DEST.rglob('*') if p.is_file()]),'bytes':sum(p.stat().st_size for p in DEST.rglob('*') if p.is_file()),'findings_paths_only':findings,'private_credentials_uploaded':False,'private_workload_manifests_uploaded':False,'large_runtime_archives_uploaded':False,'source_hashes':original};out=ROOT/'.private/github-publication-review.json';out.write_text(json.dumps(report,indent=2));out.chmod(0o600);assert not findings,'Publication scan rejected; inspect private review paths only';print(json.dumps({k:v for k,v in report.items() if k!='source_hashes'}))
if __name__=='__main__':main()
