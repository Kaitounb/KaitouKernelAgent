"""Offline single-CUDA-file patch guard using Git's existing patch engine; no native execution."""
import hashlib,os,subprocess,tempfile
from pathlib import Path
from .identity import strict_json

TARGET='kernels/gemm/wmma_sm86.cu'


def apply_reviewable_patch(content,base,*,recount=False):
    proposal=strict_json(content)
    if not isinstance(proposal,dict) or set(proposal)!= {'schema_version','parent_id','patch','reason'}:raise ValueError('source proposal fields invalid')
    if type(proposal['schema_version']) is not int or proposal['schema_version']!=1 or proposal['parent_id']!='current':raise ValueError('source proposal parent/schema invalid')
    reason=proposal['reason'];patch=proposal['patch']
    if not isinstance(reason,str) or not 1<=len(reason)<=1000 or not isinstance(patch,str) or not 1<=len(patch.encode())<=32768 or '\x00' in patch:raise ValueError('source proposal size/type invalid')
    lines=patch.splitlines()
    for prefix,expected in [('diff --git ',f'diff --git a/{TARGET} b/{TARGET}'),('--- ',f'--- a/{TARGET}'),('+++ ',f'+++ b/{TARGET}')]:
        if [line for line in lines if line.startswith(prefix)]!=[expected]:raise ValueError('only one fixed CUDA path accepted')
    forbidden=('new file mode ','deleted file mode ','old mode ','new mode ','rename from ','rename to ','copy from ','copy to ','GIT binary patch','Binary files ')
    if any(line.startswith(forbidden) for line in lines):raise ValueError('file type/mode/rename changes forbidden')
    base_text=base.decode('utf-8');prefix=base_text.split('template<int WARPS>',1)[0];suffix=base_text.split('\ntorch::Tensor gemm_template(',1)[1]
    with tempfile.TemporaryDirectory(prefix='nka-reviewed-patch-') as name:
        root=Path(name);target=root/TARGET;target.parent.mkdir(parents=True);target.write_bytes(base);target.chmod(0o644)
        env={k:v for k,v in os.environ.items() if k in ['PATH','LANG']};env.update(GIT_CONFIG_NOSYSTEM='1',GIT_CONFIG_GLOBAL='/dev/null')
        command=['git','-c','core.hooksPath=/dev/null','apply','--no-index','--whitespace=error']
        if recount:command.append('--recount')
        for flags in [['--check'],[]]:
            result=subprocess.run(command+flags,input=patch,cwd=root,env=env,text=True,capture_output=True,timeout=10)
            if result.returncode:raise ValueError('git apply rejected source patch: '+result.stderr[:512])
        if target.is_symlink() or sorted(str(p.relative_to(root)) for p in root.rglob('*') if p.is_file())!=[TARGET]:raise ValueError('unexpected patch result files')
        changed=target.read_bytes()
    if changed==base or len(changed)>256*1024:raise ValueError('source change missing/too large')
    text=changed.decode('utf-8')
    if not text.startswith(prefix) or text.count('\ntorch::Tensor gemm_template(')!=1 or text.split('\ntorch::Tensor gemm_template(',1)[1]!=suffix:raise ValueError('license/includes/host launcher changed')
    return {'proposal':proposal,'updated_source':changed,'base_sha256':hashlib.sha256(base).hexdigest(),'source_sha256':hashlib.sha256(changed).hexdigest(),
            'status':'PATCH_APPLIED_NOT_EXECUTED','trusted_source_review_required':True,'patch_recounted':recount}
