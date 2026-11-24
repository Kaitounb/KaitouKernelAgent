"""CPU synthetic restore-integrity fixtures, never GPU qualification evidence."""
import gzip,hashlib,io,json,tarfile,tempfile,unittest
from pathlib import Path
from tools.verify_docker_image_archive import verify

def raw(v):return json.dumps(v,separators=(',',':')).encode()
def digest(data):return 'sha256:'+hashlib.sha256(data).hexdigest()
def archive(path,files):
 with tarfile.open(path,'w:gz') as t:
  for name,data in files.items():i=tarfile.TarInfo(name);i.size=len(data);t.addfile(i,io.BytesIO(data))
def fixture(oci):
 layer=b'public-layer-tar-bytes';config=raw({'os':'linux','architecture':'amd64','rootfs':{'type':'layers','diff_ids':[digest(layer)]}});files={}
 if not oci:
  name=digest(config).split(':')[1]+'.json';files={name:config,'layer/layer.tar':layer,'manifest.json':raw([{'Config':name,'Layers':['layer/layer.tar']}])};return files,digest(config)
 def add(data,mt):
  d={'digest':digest(data),'size':len(data),'mediaType':mt};files['blobs/sha256/'+d['digest'].split(':')[1]]=data;return d
 cfg=add(config,'application/vnd.oci.image.config.v1+json');lay=add(gzip.compress(layer,mtime=0),'application/vnd.oci.image.layer.v1.tar+gzip');manifest=add(raw({'schemaVersion':2,'config':cfg,'layers':[lay]}),'application/vnd.oci.image.manifest.v1+json');index=add(raw({'schemaVersion':2,'manifests':[manifest]}),'application/vnd.oci.image.index.v1+json');files['index.json']=raw({'schemaVersion':2,'manifests':[index]});return files,index['digest']
class ArchiveTests(unittest.TestCase):
 def test_legacy_config_and_oci_index(self):
  with tempfile.TemporaryDirectory() as td:
   for oci in [False,True]:
    files,image=fixture(oci);p=Path(td)/str(oci);archive(p,files);self.assertEqual(verify(p,image)['layers_verified'],1)
 def test_direct_root_index(self):
  with tempfile.TemporaryDirectory() as td:
   files,image=fixture(True);files['index.json']=files['blobs/sha256/'+image.split(':')[1]];p=Path(td)/'direct';archive(p,files);self.assertEqual(verify(p,image)['layers_verified'],1)
 def test_corrupt_blob_rejected(self):
  with tempfile.TemporaryDirectory() as td:
   files,image=fixture(True);key=next(n for n,v in files.items() if v.startswith(b'\x1f\x8b'));files[key]=gzip.compress(b'wrong-layer',mtime=0);p=Path(td)/'bad';archive(p,files)
   with self.assertRaises(AssertionError):verify(p,image)
 def test_wrong_root_rejected(self):
  with tempfile.TemporaryDirectory() as td:
   files,image=fixture(True);p=Path(td)/'bad';archive(p,files)
   with self.assertRaises(AssertionError):verify(p,'sha256:'+'0'*64)
 def test_parent_path_rejected(self):
  with tempfile.TemporaryDirectory() as td:
   files,image=fixture(False);files['../escape']=b'bad';p=Path(td)/'bad';archive(p,files)
   with self.assertRaises(AssertionError):verify(p,image)
if __name__=='__main__':unittest.main()
