"""Fetch only public pinned operator implementation/notice files, recording exact revisions and SHA."""
import hashlib,json,urllib.request
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]

def get(url):
    req=urllib.request.Request(url,headers={'User-Agent':'NvidiaKernelAgent-source-audit'});return urllib.request.urlopen(req,timeout=45).read()


def main():
    destination=ROOT/'provenance/operator-upstreams';destination.mkdir(exist_ok=True);records=[]
    for repo,tag,paths in [('pytorch/pytorch','v2.7.1',['LICENSE','aten/src/ATen/native/cuda/PersistentSoftmax.cuh','aten/src/ATen/native/cuda/SoftMax.cu']),('flashinfer-ai/flashinfer','v0.2.8',['LICENSE','flashinfer/mla.py','flashinfer/prefill.py','flashinfer/jit/attention.py','include/flashinfer/attention/mla.cuh','tests/test_batch_mla.py'])]:
        ref=json.loads(get('https://api.github.com/repos/'+repo+'/git/ref/tags/'+tag));obj=ref['object']
        if obj['type']=='tag':obj=json.loads(get(obj['url']))['object']
        revision=obj['sha'];folder=destination/(repo.replace('/','__')+'-'+revision[:12]);folder.mkdir(exist_ok=True)
        (folder/'ref.json').write_text(json.dumps({'repository':repo,'tag':tag,'revision':revision},indent=2)+'\n')
        for path in paths:
            try:data=get('https://raw.githubusercontent.com/'+repo+'/'+revision+'/'+path)
            except Exception as error:records.append({'repository':repo,'revision':revision,'path':path,'status':'UNAVAILABLE','error_type':type(error).__name__});continue
            target=folder/path;target.parent.mkdir(parents=True,exist_ok=True);target.write_bytes(data);records.append({'repository':repo,'revision':revision,'tag':tag,'path':path,'local_path':str(target.relative_to(ROOT)),'sha256':hashlib.sha256(data).hexdigest(),'bytes':len(data),'status':'FETCHED_NOT_GPU_VALIDATED'})
    (destination/'source-record.json').write_text(json.dumps(records,indent=2)+'\n');print(json.dumps([{k:r[k] for k in ['repository','revision','path','status']} for r in records]))


if __name__=='__main__':main()
