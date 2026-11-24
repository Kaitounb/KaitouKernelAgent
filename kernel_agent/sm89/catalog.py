"""Packaged SM89 allowlist; arbitrary CUDA paths and source proposals are not accepted."""
import hashlib,json,re
from pathlib import Path
from kernel_agent.registry import Registry,read_regular,sha

DATA=Path(__file__).with_name('data')


def index(channel='development'):
    if channel not in ['development','qualified']:raise ValueError('unknown SM89 release channel')
    certificate=None
    if channel=='qualified':
        try:
            raw=read_regular(DATA/'gemm_qualification.json',65536);certificate=json.loads(raw);value=json.loads(read_regular(DATA/'qualified_releases.json',65536))
        except FileNotFoundError:raise ValueError('qualified SM89 release is not packaged; target acceptance remains pending') from None
        required=['independent_correctness_pass','stability_pass','nonregression_pass','installed_runtime_pass','boundary_pass','rollback_pass','raw_audit_pass']
        if not isinstance(certificate,dict) or certificate.get('status')!='QUALIFIED' or certificate.get('scope')!='SM89_REVIEWED_FIXED_CUBLAS' or certificate.get('production_release') is not True or certificate.get('source_execution_enabled') is not False or any(certificate.get(k) is not True for k in required) or not isinstance(value,dict) or value.get('qualification_sha256')!=hashlib.sha256(raw).hexdigest():raise ValueError('SM89 qualification certificate invalid')
        environment=certificate.get('environment');binary=certificate.get('binary_sha256')
        if not isinstance(environment,dict) or set(environment)!={'gpu_identity','torch','cuda','device','public_runtime_manifest_sha256','python_sha256'} or any(not isinstance(x,str) or not x for x in environment.values()) or not isinstance(binary,list) or len(binary)!=2 or any(not isinstance(x,str) or re.fullmatch('[0-9a-f]{64}',x) is None for x in binary) or len(set(binary))!=2:raise ValueError('SM89 qualification environment invalid')
        runtime_files=certificate.get('runtime_files')
        required_files={'kernel_agent/sm89/pipe_worker.py','kernel_agent/sm89/restrictions.py','kernel_agent/sm89/runner.py','kernel_agent/deployment.py','kernel_agent/registry.py','kernel_agent/protocol.py','kernel_agent/bounded_io.py','kernel_agent/output.py','kernel_agent/runner.py'}
        if not isinstance(runtime_files,dict) or set(runtime_files)!=required_files or any(not isinstance(h,str) or re.fullmatch('[0-9a-f]{64}',h) is None for h in runtime_files.values()) or any(re.fullmatch('[0-9a-f]{64}',environment[k]) is None for k in ['public_runtime_manifest_sha256','python_sha256']):raise ValueError('SM89 qualification runtime fingerprints invalid')
    else:value=json.loads(read_regular(DATA/'releases.json',65536))
    if not isinstance(value,dict) or value.get('schema_version')!=1 or value.get('target')!='sm89-reviewed-native' or not isinstance(value.get('versions'),dict) or set(value['versions'])!={'sm89-cublas-current','sm89-cublas-baseline'}:raise ValueError('SM89 catalog invalid')
    versions=value['versions']
    if any(not isinstance(value.get(k),str) or value[k] not in versions for k in ['default','fallback']) or value['default']==value['fallback']:raise ValueError('SM89 version pointers invalid')
    digests=[]
    for version,meta in versions.items():
        if not isinstance(meta,dict) or meta.get('version')!=version or meta.get('contract_sha256')!=sha(DATA/'gemm_contract_v1.json') or meta.get('backend')!='cublas' or meta.get('production_release') is not (channel=='qualified'):raise ValueError('SM89 release contract invalid')
        if {k:meta.get(k) for k in ['torch','cuda','device','execution']}!={'torch':'2.7.1+cu126','cuda':'12.6','device':'sm89','execution':'reviewed-native'}:raise ValueError('SM89 release environment invalid')
        digest=meta.get('binary_sha256')
        if not isinstance(digest,str) or re.fullmatch('[0-9a-f]{64}',digest) is None:raise ValueError('SM89 artifact digest invalid')
        digests.append(digest)
    if len(set(digests))!=2:raise ValueError('SM89 artifacts are not distinct')
    if certificate is not None:
        if set(digests)!=set(certificate['binary_sha256']) or any(certificate['environment'][k]!=versions[value['default']][k] for k in ['torch','cuda','device']):raise ValueError('SM89 qualification artifacts/environment differ')
        for name,digest in certificate.get('runtime_files',{}).items():
            if not isinstance(name,str) or Path(name).is_absolute() or '..' in Path(name).parts or sha(Path(__file__).resolve().parents[2]/name)!=digest:raise ValueError('SM89 qualified runtime code differs')
        value['qualified_environment']=certificate['environment']
    return value


def registry(root,*,channel='development'):
    return Registry(root,channel=channel,data=DATA,catalog=index(channel))
