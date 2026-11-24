"""Bounded native profiler, no model client, reference, shell-generated code or host output."""
import json,os,tempfile
from pathlib import Path
from experiments.p1_04.ncu_adapter import profile_cuda
from kernel_agent.protocol import encode_frame
from kernel_agent.optimization.source_packets import pack


def main():
    req=json.loads(Path('/inputs/request.json').read_text());work=Path(tempfile.mkdtemp(dir='/tmp',prefix='profile-'));failure=None
    try:profile_cuda('/artifact/kernel_agent/optimization/profile_target.py',work,'native',req['metrics'],'/opt/nvidia/nsight-compute/2026.3.1/ncu',timeout=180,kernel_regex=req['kernel_regex'])
    except (RuntimeError,OSError,ValueError) as error:failure=str(error)[:512]
    paths={'execution.json':work/'native.execution.json','metrics.csv':work/'native.csv','report.ncu-rep':work/'native.ncu-rep','validation.json':Path('/tmp/profile-validation.json')};limits={'execution.json':1024*1024,'metrics.csv':2*1024*1024,'report.ncu-rep':12*1024*1024,'validation.json':65536};files={}
    for name,path in paths.items():
        if path.exists():
            if path.stat().st_size>limits[name]:raise ValueError('profile artifact exceeds bound')
            files[name]=path.read_bytes()
    meta={'status':'PROFILE_READY' if failure is None else 'PROFILE_FAILED','uid':os.getuid(),'binary_sha256':req['binary_sha256'],'failure':failure,'artifact_names':sorted(files),'cpu_reference_available':False};payload=pack({'format':'bounded-ncu-v1','files':sorted(files)},files);__import__('sys').stdout.buffer.write(encode_frame(meta,payload));__import__('sys').stdout.buffer.flush()


if __name__=='__main__':main()
