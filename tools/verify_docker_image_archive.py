"""Streaming standard Docker/OCI archive integrity checks, without extracting or executing files."""
import hashlib,json,re,tarfile,zlib
from pathlib import PurePosixPath

def verify(path,image_id):
 assert re.fullmatch('sha256:[0-9a-f]{64}',image_id);members={};parsed={};by_digest={};total=0
 with tarfile.open(path,'r|gz') as tar:
  for member in tar:
   name=PurePosixPath(member.name);assert not name.is_absolute() and '..' not in name.parts;name=str(name)
   if member.isdir():continue
   assert member.isfile() and name not in members and len(members)<20000
   file=tar.extractfile(member);h=hashlib.sha256();diff=hashlib.sha256();saved=bytearray() if member.size<=4*1024*1024 else None;decoder=None;first=True;unpacked=0
   while True:
    data=file.read(1024*1024)
    if not data:break
    h.update(data);total+=len(data);assert total<64*1024**3
    if saved is not None:saved.extend(data)
    if first:decoder=zlib.decompressobj(16+zlib.MAX_WBITS) if data[:2]==b'\x1f\x8b' else None;first=False
    if decoder is None:diff.update(data)
    else:
     while data:
      value=decoder.decompress(data,4*1024*1024);diff.update(value);unpacked+=len(value);assert unpacked<64*1024**3;data=decoder.unconsumed_tail
   if decoder is not None:assert decoder.eof and not decoder.unused_data
   digest='sha256:'+h.hexdigest();members[name]={'digest':digest,'size':member.size,'diff_id':'sha256:'+diff.hexdigest()};by_digest.setdefault(digest,name)
   if name.startswith('blobs/sha256/'):assert name.split('/')[-1]==h.hexdigest()
   if saved is not None:
    try:parsed[name]=json.loads(saved)
    except (UnicodeDecodeError,json.JSONDecodeError):pass
 def descriptor(d):
  assert isinstance(d,dict) and re.fullmatch('sha256:[0-9a-f]{64}',d.get('digest','')) and type(d.get('size')) is int;name=by_digest[d['digest']];assert members[name]['size']==d['size'];return name
 visited=set();runnable=[]
 def walk(d):
  name=descriptor(d)
  if name in visited:return
  visited.add(name);node=parsed.get(name)
  if isinstance(node,dict) and 'manifests' in node:
   assert node.get('schemaVersion')==2 and isinstance(node['manifests'],list)
   for child in node['manifests']:walk(child)
  elif isinstance(node,dict) and 'config' in node and 'layers' in node:
   config_name=descriptor(node['config']);visited.add(config_name);config=parsed.get(config_name);layers=[descriptor(x) for x in node['layers']];visited.update(layers)
   if isinstance(config,dict) and config.get('os')=='linux' and config.get('architecture')=='amd64':
    assert config['rootfs']['type']=='layers';assert [members[n]['diff_id'] for n in layers]==config['rootfs']['diff_ids'];runnable.append({'config_digest':members[config_name]['digest'],'layers':len(layers)})
 # New image stores address OCI indices/manifests; require the expected root reachable from index.json.
 if 'index.json' in parsed:
  index=parsed['index.json'];visited.add('index.json');assert index.get('schemaVersion')==2
  for d in index['manifests']:walk(d)
  assert any(members[name]['digest']==image_id for name in visited)
 else:
  rows=parsed.get('manifest.json');assert isinstance(rows,list) and len(rows)==1;row=rows[0];config_name=str(PurePosixPath(row['Config']));config=parsed[config_name];layers=[str(PurePosixPath(n)) for n in row['Layers']];assert members[config_name]['digest']==image_id;assert config['rootfs']['type']=='layers' and [members[n]['diff_id'] for n in layers]==config['rootfs']['diff_ids'];assert config['os']=='linux' and config['architecture']=='amd64';runnable.append({'config_digest':image_id,'layers':len(layers)})
 assert len(runnable)==1
 return {'status':'ARCHIVE_IMAGE_GRAPH_AND_LAYER_DIFFIDS_PASS','image_id':image_id,'tar_members_verified':len(members),'descriptor_nodes_verified':len(visited),'layers_verified':runnable[0]['layers'],'config_digest':runnable[0]['config_digest'],'uncompressed_archive_bytes':total}
