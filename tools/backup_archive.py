"""Standard streaming tar/gzip backup with file SHA receipts and bounded transfer parts."""
import gzip,hashlib,json,os,stat,tarfile,time
from pathlib import Path,PurePosixPath
BLOCK=1024*1024

def file_sha(path):
 h=hashlib.sha256()
 with Path(path).open('rb') as f:
  while b:=f.read(BLOCK):h.update(b)
 return h.hexdigest()

def snapshot(roots,output,excluded=(),part_bytes=64*1024*1024):
 output=Path(output);output.mkdir(mode=0o700,parents=True,exist_ok=False);excluded=[Path(p) for p in excluded];entries={};memo={};skipped=[];start=time.time()
 def skip(p):return p.name in ['__pycache__','.DS_Store'] or any(p==e or p.is_relative_to(e) for e in excluded)
 def paths(root):
  if skip(root):return
  yield root
  if root.is_dir() and not root.is_symlink():
   for p in sorted(root.iterdir()):yield from paths(p)
 archive=output/'archive.tar.gz'
 with archive.open('xb') as f:
  archive.chmod(0o600)
  with gzip.GzipFile(filename='',mode='wb',fileobj=f,compresslevel=1,mtime=0) as gz,tarfile.open(fileobj=gz,mode='w|',format=tarfile.PAX_FORMAT,dereference=False) as tar:
   for source,arcroot in roots:
    source=Path(source)
    for p in paths(source):
     name=str(PurePosixPath(arcroot)/p.relative_to(source).as_posix());assert name not in entries and not PurePosixPath(name).is_absolute() and '..' not in PurePosixPath(name).parts
     before=p.lstat();kind='file' if stat.S_ISREG(before.st_mode) else 'dir' if stat.S_ISDIR(before.st_mode) else 'symlink' if stat.S_ISLNK(before.st_mode) else 'special';row={'kind':kind,'mode':stat.S_IMODE(before.st_mode),'mtime_ns':before.st_mtime_ns}
     if kind=='special':skipped.append(name);continue
     info=tar.gettarinfo(str(p),name)
     if kind=='file':
      key=(before.st_dev,before.st_ino,before.st_size,before.st_mtime_ns);digest=memo.get(key)
      if digest is None:digest=file_sha(p);memo[key]=digest
      row.update(bytes=before.st_size,sha256=digest)
      if info.isreg():
       with p.open('rb') as data:tar.addfile(info,data)
      else:assert info.islnk();tar.addfile(info)
      after=p.stat();assert (after.st_size,after.st_mtime_ns,after.st_ino)==(before.st_size,before.st_mtime_ns,before.st_ino),str(p)+' changed during backup'
     else:
      if kind=='symlink':row['link']=os.readlink(p)
      tar.addfile(info)
     entries[name]=row
 manifest={'schema_version':1,'created_unix':start,'roots':[{'source':str(s),'archive_root':str(a)} for s,a in roots],'excluded_roots':[str(p) for p in excluded],'transient_names_excluded':['__pycache__','.DS_Store'],'special_files_skipped':skipped,'entries':entries,'archive_sha256':file_sha(archive),'archive_bytes':archive.stat().st_size,'regular_files':sum(v['kind']=='file' for v in entries.values()),'logical_file_bytes':sum(v.get('bytes',0) for v in entries.values()),'production_release':False};(output/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n');(output/'manifest.json').chmod(0o600)
 parts=[]
 if part_bytes:
  directory=output/'parts';directory.mkdir(mode=0o700)
  with archive.open('rb') as f:
   index=0
   while block:=f.read(part_bytes):
    path=directory/f'part-{index:05d}';path.write_bytes(block);path.chmod(0o600);parts.append({'name':path.name,'bytes':len(block),'sha256':hashlib.sha256(block).hexdigest()});index+=1
 transfer={'archive_sha256':manifest['archive_sha256'],'archive_bytes':manifest['archive_bytes'],'manifest_sha256':file_sha(output/'manifest.json'),'parts':parts};(output/'transfer.json').write_text(json.dumps(transfer,indent=2)+'\n');(output/'transfer.json').chmod(0o600);return {'archive_sha256':manifest['archive_sha256'],'archive_bytes':manifest['archive_bytes'],'regular_files':manifest['regular_files'],'members':len(entries),'logical_file_bytes':manifest['logical_file_bytes'],'manifest_sha256':transfer['manifest_sha256'],'parts':len(parts)}

def verify(folder):
 folder=Path(folder);manifest=json.loads((folder/'manifest.json').read_text());archive=folder/'archive.tar.gz';assert file_sha(archive)==manifest['archive_sha256'] and archive.stat().st_size==manifest['archive_bytes'];seen={};bytes_checked=0
 with tarfile.open(archive,mode='r|gz') as tar:
  for member in tar:
   path=PurePosixPath(member.name);assert not path.is_absolute() and '..' not in path.parts and member.name not in seen;row=manifest['entries'][member.name]
   if row['kind']=='file':
    if member.islnk():assert member.linkname in seen and seen[member.linkname]==row['sha256'];digest=seen[member.linkname]
    else:
     assert member.isreg() and member.size==row['bytes'];data=tar.extractfile(member);h=hashlib.sha256()
     while b:=data.read(BLOCK):h.update(b)
     digest=h.hexdigest();bytes_checked+=row['bytes']
    assert digest==row['sha256'];seen[member.name]=digest
   elif row['kind']=='symlink':assert member.issym() and member.linkname==row['link'];seen[member.name]=None
   else:assert row['kind']=='dir' and member.isdir();seen[member.name]=None
 assert set(seen)==set(manifest['entries']);result={'status':'PASS','scope':'CPU whole-archive and every regular file/hardlink/symlink tar receipt replay; no execution of backed-up code','members_verified':len(seen),'regular_files_verified':manifest['regular_files'],'archive_sha256':manifest['archive_sha256'],'archive_bytes':manifest['archive_bytes'],'physical_file_bytes_rehashed':bytes_checked,'production_release':False};(folder/'verification.json').write_text(json.dumps(result,indent=2)+'\n');(folder/'verification.json').chmod(0o600);return result
