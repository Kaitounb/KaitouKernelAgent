"""Local authorized transport; password read privately, only supplied via child env."""
import argparse
import base64
import json
import os
from pathlib import Path
import re
import shutil
import socket
import subprocess
import sys


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--evidence",type=Path)
    args=ap.parse_args()
    source=Path(__file__).resolve().parent
    root=source.parents[1]
    dest=args.evidence.resolve() if args.evidence else root/"evidence/20261117-p0-04"
    if dest.exists():
        raise FileExistsError("Use a new evidence directory; original runs are immutable")
    lines=(root/".private/3070-connection.txt").read_text().splitlines()
    windows=next(x for x in lines if "Windows" in x and "密码" in x)
    account=lambda x: re.search(r"账号\s*([^，,\s]+)",x).group(1)
    password=lambda x: re.search(r"密码[：:]\s*(\S+)",x).group(1)
    sshpass="/opt/homebrew/bin/sshpass"
    try:
        with socket.create_connection(("127.0.0.1",21022),timeout=1):
            pass
    except OSError:
        cloud=next(x for x in lines if "腾讯云" in x and "密码" in x and x!=windows)
        host=re.search(r"\b(?:\d{1,3}\.){3}\d{1,3}\b",cloud).group(0)
        port=re.search(r"端口\s*(\d+)",cloud).group(1)
        frp=re.search(r"本地转发端口\s*(\d+)",windows).group(1)
        env=os.environ.copy(); env["SSHPASS"]=password(cloud)
        cmd=[sshpass,"-e","ssh","-M","-S","/tmp/nvidia-kernel-agent-p0-04-3070.sock","-fNT",
             "-L",f"127.0.0.1:21022:127.0.0.1:{frp}","-p",port,"-o","StrictHostKeyChecking=yes",
             "-o","ExitOnForwardFailure=yes","-o","ConnectTimeout=10","-o","ServerAliveInterval=30",
             "-o","ServerAliveCountMax=3",account(cloud)+"@"+host]
        p=subprocess.run(cmd,env=env,capture_output=True,text=True,timeout=30)
        if p.returncode:
            print(p.stderr,file=sys.stderr)
            return p.returncode
        print("Restored authorized local SSH tunnel",flush=True)
    files={str(p.relative_to(source)):base64.b64encode(p.read_bytes()).decode()
           for p in sorted(source.rglob("*")) if p.is_file() and p.suffix in [".py",".json"] and "__pycache__" not in p.parts}
    dest.mkdir(parents=True)
    for name,data in files.items():
        target=dest/"sources"/name; target.parent.mkdir(parents=True,exist_ok=True)
        target.write_bytes(base64.b64decode(data))
    payload=f'''import base64,json,pathlib,subprocess,sys,tempfile
files=json.loads({json.dumps(json.dumps(files))})
work=pathlib.Path(tempfile.mkdtemp(prefix="nvidia-kernel-agent-p0-04-"))
for name,data in files.items():
 p=work/name; p.parent.mkdir(parents=True,exist_ok=True); p.write_bytes(base64.b64decode(data))
cmd=[sys.executable,str(work/"supervisor.py"),"--output",str(work/"results")]
p=subprocess.Popen(cmd,cwd=work,text=True,stdout=subprocess.PIPE,stderr=subprocess.PIPE)
try:
 stdout,stderr=p.communicate(timeout=300)
except subprocess.TimeoutExpired:
 p.terminate()
 try: stdout,stderr=p.communicate(timeout=5)
 except subprocess.TimeoutExpired:
  p.kill(); stdout,stderr=p.communicate()
artifacts={{str(f.relative_to(work/"results")):base64.b64encode(f.read_bytes()).decode() for f in (work/"results").rglob("*") if f.is_file()}}
print("P0_04_RESULT="+json.dumps({{"command":cmd,"cwd":str(work),"exit":p.returncode,"stdout":stdout,"stderr":stderr,"artifacts":artifacts}}))
sys.exit(p.returncode)
'''
    env=os.environ.copy(); env["SSHPASS"]=password(windows)
    cmd=[sshpass,"-e","ssh","-p","21022","-o","StrictHostKeyChecking=yes","-o","ConnectTimeout=10",
         account(windows)+"@127.0.0.1","wsl -d Ubuntu -u root -- /home/dministrator/cuda-lab/.venv/bin/python -"]
    p=subprocess.run(cmd,input=payload,text=True,capture_output=True,env=env,timeout=330)
    (dest/"transport.log").write_text(p.stdout+p.stderr)
    envelopes=[x.split("=",1)[1] for x in p.stdout.splitlines() if x.startswith("P0_04_RESULT=")]
    if not envelopes:
        print("No experiment envelope; inspect preserved transport.log",file=sys.stderr)
        return p.returncode or 1
    data=json.loads(envelopes[-1])
    out=dest/"run"; out.mkdir()
    for name,blob in data["artifacts"].items():
        target=(out/name).resolve()
        if not target.is_relative_to(out.resolve()):
            raise ValueError("unsafe artifact path")
        target.parent.mkdir(parents=True,exist_ok=True)
        target.write_bytes(base64.b64decode(blob))
    print(json.dumps({k:data[k] for k in ["command","cwd","exit","stdout","stderr"]},indent=2))
    return p.returncode or data["exit"]


if __name__=="__main__":
    sys.exit(main())
