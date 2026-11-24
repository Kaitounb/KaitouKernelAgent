"""Explicit trusted-native adaptation of the frozen Docker worker; no generated CUDA accepted."""
import importlib.util
from pathlib import Path


def replace_once(text,old,new):
    if text.count(old)!=1:raise ValueError('Upstream adaptation anchor changed: '+old[:70])
    return text.replace(old,new)


def support_files(root,prefix):
    spec=importlib.util.spec_from_file_location('sustained_original',root/'experiments/full_delivery/cublas_sustained_qualification_host.py')
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    support=module.support_files(root,prefix)
    worker=(support/'pair_worker.py').read_text()
    worker=replace_once(worker,"torch.__version__!='2.14.0+cu130' or torch.cuda.get_device_capability(0)!=(8,6)","torch.__version__!='2.7.1+cu126' or torch.cuda.get_device_capability(0)!=(8,9)")
    worker=replace_once(worker,"'/artifact/'+role+'/nka_gemm_v1.so'","str(Path(req['artifact_root'])/role/'nka_gemm_v1.so')")
    worker=replace_once(worker,"modules['incumbent'].gemm_template is modules['candidate'].gemm_template","modules['incumbent'].gemm_cublas is modules['candidate'].gemm_cublas")
    worker=replace_once(worker,"models={role:((lambda x,y,module=module:module.gemm_cublas(x,y)) if backends[role]=='cublas' else (lambda x,y,module=module:module.gemm_template(x,y,4))) for role,module in modules.items()}","models={role:(lambda x,y,module=module:module.gemm_cublas(x,y)) for role,module in modules.items()}")
    worker=replace_once(worker,"'capability':[8,6]","'capability':[8,9]")
    worker=replace_once(worker,"    digests={'incumbent'",'''    # Fixed reviewed fault fixtures, never model-supplied code.
    if req.get('reviewed_failure')=='crash':raise RuntimeError('reviewed native crash fixture')
    if req.get('reviewed_failure')=='timeout':
        import time
        torch.cuda._sleep(200000);torch.cuda.synchronize();time.sleep(60)
    digests={'incumbent' ''')
    # The replacement deliberately preserves the original dictionary continuation.
    (support/'pair_worker.py').write_text(worker)
    runner=(support/'source_runner.py').read_text()
    runner=replace_once(runner,'import hashlib,json,os,re,shutil,uuid','import hashlib,json,os,re,shutil,uuid,sys,signal,time')
    runner=replace_once(runner,"req={**request,'candidate_sha256':self.sha256}","req={**request,'candidate_sha256':self.sha256,'artifact_root':str(artifact)}")
    start=runner.index("        name='nka-source-'")
    end=runner.index('        record=capture(',start)
    native='''        # Trusted fixed cuBLAS only: this is not an isolation boundary.
        cmd=[sys.executable,'-E','-s',str(artifact/worker),'--request',str(inputs/'request.json')]
        if sanitizer:
            if sanitizer not in ['memcheck','synccheck']:raise ValueError('sanitizer invalid')
            cmd=['/usr/local/cuda-12.6/bin/compute-sanitizer','--tool',sanitizer,'--error-exitcode','86','--target-processes','all','--log-file','/dev/stderr',*cmd]
        def cleanup(reason):
            # capture() has already waited for/stopped its process. Check only this exact request.
            def owned():
                pids=[]
                for proc in Path('/proc').iterdir():
                    if not proc.name.isdigit():continue
                    try:
                        args=(proc/'cmdline').read_bytes().split(b'\\0')
                        if str(inputs/'request.json').encode() in args and b'--request' in args:pids.append(int(proc.name))
                    except (OSError,ProcessLookupError):pass
                return pids
            before=owned()
            for sig in [signal.SIGTERM,signal.SIGKILL]:
                for pid in owned():
                    try:
                        if os.getpgid(pid)==pid:os.killpg(pid,sig)
                        else:os.kill(pid,sig)
                    except ProcessLookupError:pass
                if not owned():break
                time.sleep(.2)
            after=owned()
            return {'removed':not after,'observed_owned_pids':before,'remaining_owned_pids':after,'scope':'exact reviewed native request only; no container isolation'}
'''
    runner=runner[:start]+native+runner[end:]
    runner=replace_once(runner,'timeout=240 if sanitizer else 600',"timeout=240 if sanitizer else (8 if request.get('reviewed_failure')=='timeout' else 600)")
    runner=replace_once(runner,"'uid':65534","'uid':os.getuid()")
    runner=replace_once(runner,"report.get('capability')!=[8,6]","report.get('capability')!=[8,9]")
    runner=replace_once(runner,"status='REJECT')","status='REJECT',execution_boundary='trusted-native-fixed-cublas-only',container_isolation_verified=False)")
    (support/'source_runner.py').write_text(runner)
    return support
