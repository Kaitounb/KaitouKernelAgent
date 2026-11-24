"""Trusted, path-free bindings for two separately built Torch/CUDA artifacts."""
import hashlib,json,re
from pathlib import Path
from .identity import ROOT,canonical,identity,sha
from .resources import verify_bundle

ROLES=('incumbent','candidate')
WORKERS=('pair_worker.py','source_worker.py','source_timing.py','fixtures.py','source_packets.py','protocol.py')


def worker_hashes():
    folder=Path(__file__).resolve().parent
    return {name:sha(folder/name if name!='protocol.py' else folder.parent/name) for name in WORKERS}


def pair_id(descriptor):return hashlib.sha256(canonical(descriptor)).hexdigest()


def make_descriptor(public,build):
    protocol=json.loads((ROOT/'workloads/gemm_measurement_v1.json').read_text())
    descriptor={'schema_version':1,'parent':identity(4),'candidate':public['candidate'],'binary_sha256':{'incumbent':protocol['initial_template']['binary_sha256'],'candidate':build['binary_sha256']},
                'candidate_source_sha256':{name:build['source_hashes']['kernels/gemm/'+name] for name in ['wmma_sm86.cu','cublas_sm86.cu','gemm_contract.cuh']},
                'protocol_sha256':sha(ROOT/'workloads/gemm_measurement_v1.json'),'bundle_sha256':verify_bundle(),'worker_sha256':worker_hashes(),
                'environment':{'capability':[8,6],'torch':'2.14.0+cu130','cuda':'13.0'}}
    if public['parent_id']!=descriptor['parent']['id']:raise ValueError('stale source parent')
    validate_descriptor(descriptor);return descriptor


def validate_descriptor(value):
    if set(value)!={'schema_version','parent','candidate','binary_sha256','candidate_source_sha256','protocol_sha256','bundle_sha256','worker_sha256','environment'} or value['schema_version']!=1:raise ValueError('pair descriptor schema mismatch')
    parent=identity(4);candidate=value['candidate'];body={k:v for k,v in candidate.items() if k!='id'}
    if value['parent']!=parent or candidate['id']!=hashlib.sha256(canonical(body)).hexdigest():raise ValueError('pair source parent/identity mismatch')
    if set(candidate)!=set(parent) or candidate['template']!=parent['template'] or candidate['warps']!=4 or set(candidate['files'])!=set(parent['files']):raise ValueError('pair candidate contract fields differ')
    for path,digest in candidate['files'].items():
        if not re.fullmatch('[0-9a-f]{64}',digest):raise ValueError('source digest invalid')
        if path!='kernels/gemm/wmma_sm86.cu' and digest!=parent['files'][path]:raise ValueError('pair protected source/contract changed')
    expected_sources={name:candidate['files']['kernels/gemm/'+name] for name in ['wmma_sm86.cu','cublas_sm86.cu','gemm_contract.cuh']}
    if value['candidate_source_sha256']!=expected_sources:raise ValueError('compiled/source identity differs')
    protocol=json.loads((ROOT/'workloads/gemm_measurement_v1.json').read_text())
    if set(value['binary_sha256'])!=set(ROLES) or value['binary_sha256']['incumbent']!=protocol['initial_template']['binary_sha256'] or any(not re.fullmatch('[0-9a-f]{64}',x) for x in value['binary_sha256'].values()):raise ValueError('unbound baseline/candidate binary')
    if value['protocol_sha256']!=sha(ROOT/'workloads/gemm_measurement_v1.json') or value['bundle_sha256']!=verify_bundle() or value['worker_sha256']!=worker_hashes():raise ValueError('pair evaluator/protocol drift')
    if value['environment']!={'capability':[8,6],'torch':'2.14.0+cu130','cuda':'13.0'}:raise ValueError('unsupported pair environment')
