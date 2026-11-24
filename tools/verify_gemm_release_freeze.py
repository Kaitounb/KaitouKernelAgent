"""Verify frozen public candidate/protocol without reading evaluator-held workload values."""
import argparse,hashlib,json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]


def verify(folder):
    folder=Path(folder);freeze=json.loads((folder/'freeze.json').read_text());body={k:v for k,v in freeze.items() if k!='freeze_id'}
    encoded=json.dumps(body,sort_keys=True,separators=(',',':'),allow_nan=False).encode()
    if hashlib.sha256(encoded).hexdigest()!=freeze['freeze_id']:raise ValueError('release freeze identity differs')
    for rel,digest in freeze['files'].items():
        p=Path(rel)
        if p.is_absolute() or '..' in p.parts or hashlib.sha256((ROOT/p).read_bytes()).hexdigest()!=digest:raise ValueError('frozen file changed: '+rel)
    protocol=json.loads((folder/'protocol.json').read_text())
    if protocol['freeze_id']!=freeze['freeze_id'] or protocol['dataset_commitment_sha256']!=freeze['independent_commitment']['sha256'] or protocol['performance_sampling'] or not protocol['one_shot']:raise ValueError('release protocol binding differs')
    return {'status':'PASS','candidate_id':freeze['candidate_id'],'freeze_id':freeze['freeze_id'],'protocol_sha256':hashlib.sha256((folder/'protocol.json').read_bytes()).hexdigest(),'heldout_read':False,'production_release':False}


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--freeze',type=Path,default=ROOT/'releases/gemm-sm86-rc1');a=p.parse_args();print(json.dumps(verify(a.freeze)))
