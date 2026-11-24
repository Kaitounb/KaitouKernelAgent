"""Development deployment: validated reviewed artifact, one trusted cuBLAS fallback, no promotion."""
import json,os,stat,uuid
from pathlib import Path
import numpy as np
from .registry import Registry
from .runner import DockerRunner,validate_inputs


def load_input(path):
    fd=os.open(path,os.O_RDONLY|os.O_NOFOLLOW|os.O_NONBLOCK)
    with os.fdopen(fd,'rb') as file:
        info=os.fstat(file.fileno())
        if not stat.S_ISREG(info.st_mode) or info.st_size>4096*4096*2+16384:raise ValueError('input type/size unsupported')
        version=np.lib.format.read_magic(file)
        if version==(1,0):shape,fortran,dtype=np.lib.format.read_array_header_1_0(file,max_header_size=10000)
        elif version==(2,0):shape,fortran,dtype=np.lib.format.read_array_header_2_0(file,max_header_size=10000)
        else:raise ValueError('unsupported input numpy format')
        if len(shape)!=2 or any(type(d) is not int or not 1<=d<=4096 for d in shape) or fortran or dtype!=np.dtype('float16'):raise ValueError('FP16 contiguous rank-2 dimensions 1..4096 required')
        if info.st_size-file.tell()!=shape[0]*shape[1]*2:raise ValueError('input payload mismatch')
        file.seek(0);return np.load(file,allow_pickle=False,max_header_size=10000)


def execute(root,a,b,*,inject_failure=None,channel='development',registry_factory=None,runner_factory=None):
    if inject_failure not in [None,'crash','timeout']:raise ValueError('unsupported reviewed fault')
    validate_inputs(a,b)
    registry=(registry_factory or Registry)(root,channel=channel);binary,meta,reason=registry.select();attempts=[]
    run=registry.root/'runs'/uuid.uuid4().hex;run.mkdir(mode=0o700)
    fallback_used=reason is not None
    selected_version=meta['version']
    actual=None
    for attempt in range(2):
        try:
            runner=(runner_factory or DockerRunner)(binary,meta['binary_sha256'])
            actual,record=runner.run(a,b,run/('attempt-'+str(attempt)),case=inject_failure if attempt==0 and inject_failure else 'cublas')
        except (OSError,ValueError,RuntimeError) as error:
            record={'status':'REJECT','rejection':str(error)[:512]};actual=None
        record.update(version=meta['version'],binary_sha256=meta['binary_sha256']);attempts.append(record)
        if actual is not None:break
        if record.get('cleanup') and not record['cleanup'].get('removed'):
            reason='cleanup_unconfirmed';break
        if attempt or reason is not None:break
        reason='active_execution_failed: '+record.get('rejection','unknown')
        binary,meta=registry.fallback();fallback_used=True
    result={'status':'PASS' if actual is not None else 'FAIL','deployment':'qualified-runtime' if channel=='qualified' else 'development-only','production_release':meta['production_release'],'channel':channel,'backend':'cublas',
            'selected_version':selected_version,'executed_version':meta['version'],'binary_sha256':meta['binary_sha256'],
            'fallback_used':fallback_used,'fallback_reason':reason,'run_directory':str(run),'attempts':attempts,'promotion_performed':False}
    (run/'execution.json').write_text(json.dumps(result,indent=2)+'\n')
    if actual is None:raise RuntimeError('GEMM failed after bounded attempts; see '+str(run/'execution.json'))
    return actual,result
