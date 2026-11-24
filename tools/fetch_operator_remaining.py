"""Retry missing public source via GitHub contents API; preserve original failed fetch record."""
import base64,hashlib,json,urllib.request
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
REPO='flashinfer-ai/flashinfer';REV='3f8317c67ac129facb8525b2fb581be42f634459'


def main():
    destination=ROOT/'provenance/operator-upstreams'/('flashinfer-ai__flashinfer-'+REV[:12]);records=[]
    for path in ['LICENSE','flashinfer/mla.py','flashinfer/prefill.py','flashinfer/jit/attention/__init__.py','include/flashinfer/attention/mla.cuh','tests/test_batch_mla_page.py']:
        request=urllib.request.Request('https://api.github.com/repos/'+REPO+'/contents/'+path+'?ref='+REV,headers={'User-Agent':'NvidiaKernelAgent-source-audit'})
        try:
            response=json.load(urllib.request.urlopen(request,timeout=60));data=base64.b64decode(response['content']);target=destination/path;target.parent.mkdir(parents=True,exist_ok=True);target.write_bytes(data)
            records.append({'repository':REPO,'revision':REV,'path':path,'local_path':str(target.relative_to(ROOT)),'sha256':hashlib.sha256(data).hexdigest(),'bytes':len(data),'status':'FETCHED_NOT_GPU_VALIDATED'})
        except Exception as error:records.append({'path':path,'status':'UNAVAILABLE','error_type':type(error).__name__})
    (destination/'fetch-v2.json').write_text(json.dumps(records,indent=2)+'\n');print(json.dumps(records))


if __name__=='__main__':main()
