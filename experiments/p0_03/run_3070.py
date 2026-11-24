"""Use the already-authorized local SSH tunnel; credentials never enter argv/logs."""
import argparse
import base64
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--phase", choices=["calibrate", "measure"], required=True)
    ap.add_argument("--session", type=int, default=0)
    ap.add_argument("--evidence", type=Path)
    args = ap.parse_args()
    source = Path(__file__).resolve().parent
    root = source.parents[1]
    dest = args.evidence.resolve() if args.evidence else root / "evidence/20261117-p0-03"
    dest.mkdir(parents=True, exist_ok=True)
    private = (root / ".private/3070-connection.txt").read_text()
    line = next(x for x in private.splitlines() if "Windows" in x and "密码" in x)
    user = re.search(r"账号\s*([^，,\s]+)", line).group(1)
    password = re.search(r"密码[：:]\s*(\S+)", line).group(1)
    files = {name: base64.b64encode((source/name).read_bytes()).decode() for name in ["kernel.py", "measure.py", "task.json"]}
    output = "calibration.json" if args.phase == "calibrate" else f"session-{args.session}.json"
    if (dest/output).exists() or (dest/(output+".transport.log")).exists():
        raise FileExistsError("Preserve existing evidence; select a new --evidence directory")
    snapshot = dest / "sources" / output.removesuffix(".json")
    snapshot.mkdir(parents=True, exist_ok=True)
    for name in ["kernel.py", "measure.py", "task.json", "run_3070.py"]:
        shutil.copyfile(source/name, snapshot/name)
    # The bootstrap is data sent over stdin, never shell-interpolated.
    payload = f'''import base64,json,os,pathlib,subprocess,sys,tempfile
files=json.loads({json.dumps(json.dumps(files))})
work=pathlib.Path(tempfile.mkdtemp(prefix="nvidia-kernel-agent-p0-03-"))
for name,data in files.items(): (work/name).write_bytes(base64.b64decode(data))
cmd=[sys.executable,"measure.py","--phase",{args.phase!r},"--session",{str(args.session)!r},"--output",{output!r}]
p=subprocess.run(cmd,cwd=work,text=True,capture_output=True,timeout=180)
print("P0_03_RESULT="+json.dumps({{"command":cmd,"cwd":str(work),"exit":p.returncode,"stdout":p.stdout,"stderr":p.stderr,"result":json.loads((work/{output!r}).read_text()) if (work/{output!r}).exists() else None}}))
sys.exit(p.returncode)
'''
    env = os.environ.copy()
    env["SSHPASS"] = password
    cmd = ["/opt/homebrew/bin/sshpass", "-e", "ssh", "-p", "21022", "-o", "StrictHostKeyChecking=yes",
           "-o", "ConnectTimeout=10", user+"@127.0.0.1",
           "wsl -d Ubuntu -u root -- /home/dministrator/cuda-lab/.venv/bin/python -"]
    p = subprocess.run(cmd, input=payload, text=True, capture_output=True, env=env, timeout=210)
    (dest/(output+".transport.log")).write_text(p.stdout+p.stderr)
    lines = [x for x in p.stdout.splitlines() if x.startswith("P0_03_RESULT=")]
    if not lines:
        print("No result; inspect transport log", file=sys.stderr)
        return 1
    envelope = json.loads(lines[-1].split("=",1)[1])
    if envelope["result"] is not None:
        (dest/output).write_text(json.dumps(envelope["result"],indent=2)+"\n")
    print(json.dumps({k: envelope[k] for k in ["command","cwd","exit","stdout","stderr"]},indent=2))
    return p.returncode or envelope["exit"]


if __name__ == "__main__":
    sys.exit(main())
