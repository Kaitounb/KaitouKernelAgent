"""One experiment transport, adapted from experiments/p0_04/run_3070.py.

Requires the existing authorized 21022 tunnel; no credential enters argv/logs.
"""
import argparse
import base64
import json
import os
from pathlib import Path
import re
import subprocess
import sys


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--evidence", type=Path)
    ap.add_argument("--case", choices=["original", "shared-tail", "naive", "wmma"], default="original")
    args = ap.parse_args()
    root = Path(__file__).resolve().parents[2]
    default_name = "20261120-p1-01"
    if args.case != "original":
        default_name += f"-{args.case}"
    dest = args.evidence.resolve() if args.evidence else root/"evidence"/default_name
    if dest.exists():
        raise FileExistsError("Choose a new evidence directory; do not overwrite runs")
    learning = root/"kernels/gemm/learning/kernelbench"
    paths = {"model_new_ex_tiled_matmul.py":learning/"model_new_ex_tiled_matmul.py",
             "verify_kernelbench.py":Path(__file__).with_name("verify_kernelbench.py")}
    if args.case == "naive":
        paths["naive_matmul.cu"] = learning/"naive_matmul.cu"
    elif args.case == "wmma":
        paths["wmma_tile.cu"] = learning/"wmma_tile.cu"
    files = {name:base64.b64encode(path.read_bytes()).decode() for name,path in paths.items()}
    dest.mkdir(parents=True)
    for name,path in paths.items():
        target = dest/"sources"/name; target.parent.mkdir(parents=True,exist_ok=True)
        target.write_bytes(path.read_bytes())
    for name in ["LICENSE.kernelbench","SOURCE.json","tiled_matmul_reading.cu"]:
        (dest/"sources"/name).write_bytes((learning/name).read_bytes())
    if args.case == "wmma":
        nvidia = root/"provenance/upstream-reviews/20261120-p1-01-wmma"
        for name in ["LICENSE", "source-record.json", "cudaTensorCoreGemm.cu"]:
            (dest/"sources"/name).write_bytes((nvidia/name).read_bytes())
    (dest/"sources/run_3070.py").write_bytes(Path(__file__).read_bytes())
    line = next(x for x in (root/".private/3070-connection.txt").read_text().splitlines()
                if "Windows" in x and "密码" in x)
    user = re.search(r"账号\s*([^，,\s]+)",line).group(1)
    env = os.environ.copy()
    env["SSHPASS"] = re.search(r"密码[：:]\s*(\S+)",line).group(1)
    payload = f'''import base64,hashlib,json,os,pathlib,shutil,subprocess,sys,tempfile
work=pathlib.Path(tempfile.mkdtemp(prefix="nvidia-kernel-agent-p1-01-"))
files=json.loads({json.dumps(json.dumps(files))})
for name,data in files.items(): (work/name).write_bytes(base64.b64decode(data))
env={{k:v for k,v in os.environ.items() if k in ['HOME','LANG','PATH','LD_LIBRARY_PATH']}}
env.update(CUDA_HOME='/usr/local/cuda-13.0',TORCH_CUDA_ARCH_LIST='8.6',MAX_JOBS='1',TORCH_EXTENSIONS_DIR=str(work/'torch-extensions'),PYTHONUNBUFFERED='1')
env['PATH']='/usr/local/cuda-13.0/bin:'+env.get('PATH','')
cmd=[sys.executable,str(work/'verify_kernelbench.py'),'--case',{args.case!r}]
p=subprocess.run(cmd,cwd=work,env=env,text=True,capture_output=True,timeout=300)
(work/'stdout.log').write_text(p.stdout);(work/'stderr.log').write_text(p.stderr)
metadata=[]
for file in (work/'torch-extensions').rglob('*'):
 if file.is_file(): metadata.append({{'path':str(file.relative_to(work)),'bytes':file.stat().st_size,'sha256':hashlib.sha256(file.read_bytes()).hexdigest()}})
(work/'build-artifacts.json').write_text(json.dumps(metadata,indent=2)+'\\n')
artifact_paths=[work/x for x in ['result.json','stdout.log','stderr.log','build-artifacts.json'] if (work/x).exists()]
artifact_paths += [f for f in (work/'torch-extensions').rglob('*') if f.is_file() and f.suffix in ['.ninja','.cpp','.cu']]
if {args.case!r} == 'wmma' and p.returncode == 0:
 binary=work/'torch-extensions/wmma_tile/wmma_tile.so'
 inspect=[env['CUDA_HOME']+'/bin/cuobjdump','--dump-sass',str(binary)]
 sass=subprocess.run(inspect,cwd=work,env=env,text=True,capture_output=True,timeout=25)
 (work/'sass.log').write_text(sass.stdout)
 (work/'sass-inspection.json').write_text(json.dumps({{'command':inspect,'exit':sass.returncode,'stderr':sass.stderr}},indent=2))
 artifact_paths.extend([work/'sass.log',work/'sass-inspection.json'])
artifacts={{str(f.relative_to(work)):base64.b64encode(f.read_bytes()).decode() for f in artifact_paths}}
print('P1_01_RESULT='+json.dumps({{'cwd':str(work),'command':cmd,'exit':p.returncode,'artifacts':artifacts}}))
sys.exit(p.returncode)
'''
    command = ["/opt/homebrew/bin/sshpass","-e","ssh","-p","21022","-o","StrictHostKeyChecking=yes",
               "-o","ConnectTimeout=10",user+"@127.0.0.1",
               "wsl -d Ubuntu -u root -- /home/dministrator/cuda-lab/.venv/bin/python -"]
    p = subprocess.run(command,input=payload,text=True,capture_output=True,env=env,timeout=330)
    (dest/"transport.log").write_text(p.stdout+p.stderr)
    records = [line.split("=",1)[1] for line in p.stdout.splitlines() if line.startswith("P1_01_RESULT=")]
    if not records:
        print("No result envelope; inspect transport.log",file=sys.stderr)
        return p.returncode or 1
    record = json.loads(records[-1])
    for name,data in record["artifacts"].items():
        path = (dest/"run"/name).resolve()
        if not path.is_relative_to((dest/"run").resolve()):
            raise ValueError("unsafe artifact path")
        path.parent.mkdir(parents=True,exist_ok=True); path.write_bytes(base64.b64decode(data))
    (dest/"execution.json").write_text(json.dumps({k:record[k] for k in ["cwd","command","exit"]},indent=2)+"\n")
    print(json.dumps({k:record[k] for k in ["cwd","command","exit"]},indent=2))
    if (dest/"run/result.json").exists():
        result = json.loads((dest/"run/result.json").read_text())
        print(json.dumps({"status":result["status"],"checks":len(result["checks"]),"error":result.get("error")},indent=2))
    return p.returncode or record["exit"]


if __name__ == "__main__":
    sys.exit(main())
