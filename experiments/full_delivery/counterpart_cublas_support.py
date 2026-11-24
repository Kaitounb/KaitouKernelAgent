"""Reuse validated Docker/Graph support; target image and fixed fault fixtures only."""
import json
from ordinary_host import support_files as original_support

def replace_once(s,old,new):
 if s.count(old)!=1:raise ValueError('adaptation anchor changed: '+old[:80])
 return s.replace(old,new)
def support_files(root,prefix):
 support=original_support(root,prefix);config=json.loads((root/'runtime-binding.json').read_text());image=config['image_id'];assert image.startswith('sha256:') and len(image)==71
 worker=(support/'pair_worker.py').read_text();worker=replace_once(worker,"    digests={'incumbent'",'''    # Reviewed finite failure fixtures reused from the accepted native precheck.
    if req.get('reviewed_failure')=='crash':raise RuntimeError('reviewed counterpart crash fixture')
    if req.get('reviewed_failure')=='timeout':
        import time
        torch.cuda._sleep(200000);torch.cuda.synchronize();sys.stderr.write('NKA_FINITE_CUDA_READY\\n');sys.stderr.flush();time.sleep(60)
    digests={'incumbent' ''');(support/'pair_worker.py').write_text(worker)
 runner=(support/'source_runner.py').read_text();runner=replace_once(runner,'from kernel_agent.runner import IMAGE,digest,command_result','from kernel_agent.runner import IMAGE,digest,command_result\nIMAGE='+repr(image));runner=replace_once(runner,"['docker','run','--name',name,'--pull=never'","['docker','run','--name',name,'--label',"+repr('nka.counterpart='+root.name)+",'--pull=never'");runner=replace_once(runner,"'--gpus','device=0'","'--runtime=runc','--device','nvidia.com/gpu=0','--env','NVIDIA_VISIBLE_DEVICES=void'");runner=replace_once(runner,'timeout=240 if sanitizer else 600',"timeout=240 if sanitizer else (8 if request.get('reviewed_failure')=='timeout' else 600)");(support/'source_runner.py').write_text(runner);return support
