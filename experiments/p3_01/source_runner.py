"""Bounded reviewed-source QA/A-A worker; references and validated-output ledger stay on host."""
import hashlib,json,os,re,shutil,uuid
from pathlib import Path
from kernel_agent.bounded_io import capture,confirmed_removed
from kernel_agent.protocol import HEADER,META_LIMIT,decode_frame
from kernel_agent.runner import IMAGE,digest,command_result
from experiments.p3_01.source_packets import REPORT_LIMIT,unpack


class SourceRunner:
    def __init__(self,binary,sha256):
        self.binary=Path(binary);self.sha256=sha256
        if digest(self.binary)!=sha256:raise ValueError('source artifact mismatch')

    def run(self,request,work,*,sanitizer=None):
        if request['mode'] not in ['qa','aa']:raise ValueError('source mode unsupported')
        cases=request['cases']
        if len(cases)>32 or len({case['id'] for case in cases})!=len(cases):raise ValueError('case count/ids invalid')
        limits={case['id']+'.npy':case['shape_mnk'][0]*case['shape_mnk'][1]*4+16384 for case in cases}
        if any(not case['id'].isalnum() or len(case['id'])>16 for case in cases):raise ValueError('case id invalid')
        payload_limit=sum(limits.values())+REPORT_LIMIT+(len(limits)+1)*512 if request['mode']=='qa' else 4*1024*1024
        if payload_limit>96*1024*1024:raise ValueError('source payload cap exceeded')
        work=Path(work).resolve();work.mkdir(parents=True,exist_ok=False);work.chmod(0o700);artifact=work/'artifact';artifact.mkdir(mode=0o755);inputs=work/'inputs';inputs.mkdir(mode=0o755)
        native=artifact/'candidate';native.mkdir();shutil.copyfile(self.binary,native/'nka_gemm_v1.so')
        for name in ['source_worker.py','fixtures.py','source_packets.py','source_timing.py']:shutil.copyfile(Path(__file__).with_name(name),artifact/name)
        shutil.copyfile(Path(__file__).resolve().parents[2]/'kernel_agent/protocol.py',artifact/'protocol.py')
        req={**request,'candidate_sha256':self.sha256};(inputs/'request.json').write_text(json.dumps(req))
        protected={str(path.relative_to(work)):digest(path) for folder in [artifact,inputs] for path in folder.rglob('*') if path.is_file()}
        name='nka-source-'+uuid.uuid4().hex[:12]
        cmd=['docker','run','--name',name,'--pull=never','--network=none','--sig-proxy=false','--read-only','--log-driver=none','--cap-drop=ALL','--security-opt=no-new-privileges','--user','65534:65534','--init',
             '--pids-limit=32','--memory=2g','--memory-swap=2g','--cpus=2','--gpus','device=0','--ulimit','fsize=16777216:16777216','--ulimit','core=0:0',
             '--env','HOME=/tmp','--env','PYTHONDONTWRITEBYTECODE=1','--env','OMP_NUM_THREADS=1','--env','MKL_NUM_THREADS=1','--env','OPENBLAS_NUM_THREADS=1',
             '--tmpfs','/tmp:rw,nosuid,nodev,noexec,size=128m','--mount',f'type=bind,src={artifact},dst=/artifact,readonly','--mount',f'type=bind,src={inputs},dst=/inputs,readonly']
        if sanitizer:
            if sanitizer not in ['memcheck','synccheck']:raise ValueError('sanitizer invalid')
            cmd+=['--entrypoint','/usr/local/cuda/bin/compute-sanitizer',IMAGE,'--tool',sanitizer,'--error-exitcode','86','--target-processes','all','--log-file','/dev/stderr','/opt/venv/bin/python']
        else:cmd+=['--entrypoint','/opt/venv/bin/python',IMAGE]
        cmd+=['/artifact/source_worker.py','--request','/inputs/request.json']
        def cleanup(reason):
            info={'state':command_result(['docker','inspect','--format','{{json .State}}',name])}
            if reason:info['top']=command_result(['docker','top',name,'-eo','pid,ppid,args'])
            info['remove']=command_result(['docker','rm','-f',name]);after=command_result(['docker','inspect',name]);info['after']=after;info['removed']=confirmed_removed(after);return info
        record=capture(cmd,work/'capture',timeout=240 if sanitizer else 180,stdout_limit=HEADER.size+META_LIMIT+payload_limit,stderr_limit=1024*1024,cleanup=cleanup)
        record.update(command=cmd,mode=request['mode'],source_binary_sha256=self.sha256,sanitizer=sanitizer,protected_hashes=protected,protected_unchanged=all(digest(work/path)==sha for path,sha in protected.items()),status='REJECT')
        report=None
        if record['exit']==0 and record['reason'] is None and record['cleanup']['removed'] and record['protected_unchanged']:
            try:
                if sanitizer and not re.search(r'^========= ERROR SUMMARY: 0 errors\s*$',(work/'capture/stderr.bin').read_text(errors='replace'),re.MULTILINE):raise ValueError('sanitizer zero-error summary missing')
                meta,payload=decode_frame(work/'capture/stdout.bin',payload_limit)
                if meta!={'kind':'source-eval-v1','mode':request['mode'],'uid':65534,'binary_sha256':self.sha256}:raise ValueError('source frame metadata mismatch')
                if request['mode']=='qa':
                    report,files=unpack(payload,limits);outputs=work/'outputs';outputs.mkdir()
                    for name,data in files.items():(outputs/name).write_bytes(data)
                else:
                    if len(payload)>4*1024*1024:raise ValueError('AA report too large')
                    report=json.loads(payload)
                if report.get('status')!='PASS' or report.get('binary_sha256')!=self.sha256 or report.get('capability')!=[8,6] or report.get('cpu_reference_available') is not False:raise ValueError('source report contract invalid')
                record['status']='WITNESSES_READY';(work/'report.json').write_text(json.dumps(report,indent=2)+'\n')
            except (OSError,ValueError,EOFError,RecursionError) as error:record['rejection']=str(error)[:512];report=None
        else:record['rejection']=record['reason'] or 'worker_or_cleanup_failure'
        (work/'record.json').write_text(json.dumps(record,indent=2)+'\n');return report,record
