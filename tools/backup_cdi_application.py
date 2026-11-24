"""Package the audited public GA deployment and RC2 recovery prefix; CPU member-by-member restore check."""
import gzip,hashlib,io,json,tarfile
from pathlib import Path,PurePosixPath
ROOT=Path(__file__).resolve().parents[1]
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main():
 accepted=ROOT/'evidence/20261124-3060-active-entry';assert json.loads((accepted/'verification.json').read_text())['status']=='AUDIT_PASS_ACTIVE_ENTRY';loc=json.loads((ROOT/'evidence/20261124-3060-0.1.3-installed/location.json').read_text());previous=ROOT/'evidence/20261124-3060-0.1.3rc2-installed/segmented-archive/remote';current=accepted/'segmented-archive/remote';out=ROOT/'evidence/20261124-3060-application-backup';out.mkdir(exist_ok=False);target=ROOT/'artifacts/runtime-images/rtx3060-0.1.3-application.tar.gz';assert not target.exists();files={}
 for base,name,prefix_only in [(current,Path(loc['root']).name,False),(previous,Path(loc['previous_prefix']).parent.name,True)]:
  for p in sorted(base.rglob('*')):
   if p.is_file() and not p.is_symlink() and (not prefix_only or p.relative_to(base).parts[0]=='prefix'):
    relative=Path(name)/p.relative_to(base);files[str(relative)]={'source':p,'bytes':p.stat().st_size,'sha256':sha(p),'mode':0o700 if p.name in ['kernel-agent-3060','kernel-agent-3060-switch'] else (0o444 if 'prefix' in p.relative_to(base).parts else 0o600)}
 manifest={'schema_version':1,'scope':'Public GA deployment plus installed RC2 prefix; no private acceptance data or credentials','parent_directory':'/home/featurize/nka-persistent','ga_root':loc['root'],'current_prefix':loc['prefix'],'previous_prefix':loc['previous_prefix'],'files':{n:{k:v for k,v in m.items() if k!='source'} for n,m in files.items()},'current_symlink_requires_recreation':True}
 with target.open('xb') as f,gzip.GzipFile(fileobj=f,mode='wb',mtime=0,filename='') as compressed,tarfile.open(fileobj=compressed,mode='w|') as tar:
  data=json.dumps(manifest,indent=2).encode();i=tarfile.TarInfo('recovery-manifest.json');i.size=len(data);i.mode=0o600;tar.addfile(i,io.BytesIO(data))
  for n,m in files.items():
   i=tarfile.TarInfo(n);i.size=m['bytes'];i.mode=m['mode'];i.mtime=0
   with m['source'].open('rb') as r:tar.addfile(i,r)
 seen=set()
 with tarfile.open(target,'r|gz') as tar:
  for member in tar:
   n=PurePosixPath(member.name);assert member.isfile() and not n.is_absolute() and '..' not in n.parts and str(n) not in seen;seen.add(str(n));data=tar.extractfile(member).read()
   if str(n)=='recovery-manifest.json':assert json.loads(data)==manifest
   else:m=files[str(n)];assert len(data)==m['bytes'] and hashlib.sha256(data).hexdigest()==m['sha256'] and member.mode==m['mode']
 assert seen==set(files)|{'recovery-manifest.json'};target.chmod(0o444);v={'status':'APPLICATION_BACKUP_ALL_MEMBERS_SHA_AND_MODE_PASS','archive':str(target),'archive_sha256':sha(target),'bytes':target.stat().st_size,'files_verified':len(files),'remote_runtime_launch_already_verified':True,'private_data_included':False,'credentials_included':False,'gpu_execution':False};(out/'manifest.json').write_text(json.dumps(manifest,indent=2));(out/'result.json').write_text(json.dumps(v,indent=2));print(json.dumps(v))
if __name__=='__main__':main()
