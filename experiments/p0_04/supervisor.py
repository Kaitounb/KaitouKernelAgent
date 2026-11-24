"""Serial WSL/Linux supervisor: bounded workers, immutable evidence, parent-owned best."""
import argparse
import copy
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

from common import decide, dump, paired_stats, source_hashes


def utc():
    return datetime.now(timezone.utc).isoformat()


def group_members(pgid):
    members = []
    for stat in Path("/proc").glob("[0-9]*/stat"):
        try:
            parts = stat.read_text().rsplit(")",1)[1].split()
            if int(parts[2]) == pgid:
                members.append({"pid":int(stat.parent.name),"state":parts[0]})
        except (FileNotFoundError,ProcessLookupError,PermissionError):
            continue
    return members


def terminate_group(proc, grace):
    record = {"pgid":proc.pid,"before":group_members(proc.pid),"signals":[]}
    if record["before"]:
        try:
            os.killpg(proc.pid,signal.SIGTERM)
            record["signals"].append("SIGTERM")
        except ProcessLookupError:
            pass
    try:
        proc.wait(timeout=grace)
    except subprocess.TimeoutExpired:
        pass
    if any(x["state"] not in ["Z","X"] for x in group_members(proc.pid)):
        try:
            os.killpg(proc.pid,signal.SIGKILL)
            record["signals"].append("SIGKILL")
        except ProcessLookupError:
            pass
    proc.wait(timeout=grace+2)
    record["after"] = group_members(proc.pid)
    record["no_live_members"] = not any(x["state"] not in ["Z","X"] for x in record["after"])
    return record


def telemetry():
    result=[]
    for cmd in [
        ["nvidia-smi","--query-gpu=name,uuid,driver_version,temperature.gpu,clocks.sm,clocks.mem,power.draw,utilization.gpu","--format=csv"],
        ["nvidia-smi","--query-compute-apps=pid,process_name,used_gpu_memory","--format=csv"],
    ]:
        p=subprocess.run(cmd,capture_output=True,text=True,timeout=15)
        result.append({"command":cmd,"exit":p.returncode,"stdout":p.stdout,"stderr":p.stderr})
    return {"utc":utc(),"queries":result}


def main():
    def stop(signum, frame):
        raise SystemExit(128+signum)
    for sig in [signal.SIGTERM,signal.SIGINT,signal.SIGHUP]:
        signal.signal(sig,stop)
    ap=argparse.ArgumentParser()
    ap.add_argument("--output",type=Path,required=True)
    args=ap.parse_args()
    root=Path(__file__).resolve().parent
    out=args.output.resolve()
    out.mkdir(parents=True,exist_ok=False)
    task=json.loads((root/"task.json").read_text())
    expected=source_hashes(root)
    dump(out/"source_hashes.json",expected)
    dump(out/"task.json",task)
    # Set only operational runtime variables, never inherit model/API credentials.
    env={k:v for k,v in os.environ.items() if k in ["PATH","HOME","LANG","LC_ALL","LD_LIBRARY_PATH"]}
    env["PYTHONUNBUFFERED"]="1"
    best=None
    records=[]
    for proposal in task["candidate_order"]:
        cid,source=proposal["id"],proposal["source"]
        dest=out/cid
        dest.mkdir()
        before=copy.deepcopy(best)
        before_bytes=(out/"best.json").read_bytes() if (out/"best.json").exists() else None
        record={"id":cid,"candidate_source":source,"candidate_source_sha256":expected[source],
                "parent_source_sha256":None if best is None else best["candidate_source_sha256"],
                "started_utc":utc(),"transitions":["proposed","running"],"best_before":before,
                "telemetry_before":telemetry()}
        cmd=[sys.executable,str(root/"worker.py"),"--candidate",source,"--output",str(dest)]
        started=time.monotonic()
        timed_out=False
        execution_start=None
        with (dest/"stdout.log").open("w") as stdout,(dest/"stderr.log").open("w") as stderr:
            proc=subprocess.Popen(cmd,cwd=root,env=env,stdout=stdout,stderr=stderr,start_new_session=True)
            try:
                while proc.poll() is None:
                    if execution_start is None and (dest/"ready.json").exists():
                        execution_start=time.monotonic()
                    limit=task["startup_timeout_seconds"] if execution_start is None else task["execution_timeout_seconds"]
                    elapsed=time.monotonic()-(started if execution_start is None else execution_start)
                    if elapsed>limit:
                        timed_out=True
                        record["timeout_phase"]="startup" if execution_start is None else "execution"
                        record["termination"]=terminate_group(proc,task["termination_grace_seconds"])
                        break
                    time.sleep(.02)
            finally:
                if proc.poll() is None or group_members(proc.pid):
                    record["final_cleanup"]=terminate_group(proc,task["termination_grace_seconds"])
        record.update({"command":cmd,"pid":proc.pid,"exit":proc.returncode,"wall_seconds":time.monotonic()-started})
        raw=json.loads((dest/"worker.json").read_text()) if (dest/"worker.json").exists() else {}
        result=copy.deepcopy(raw)
        if timed_out:
            result.update({"status":"timed_out","source_sha256":expected})
        elif proc.returncode != 0:
            result["status"]="worker_error"
        current=source_hashes(root)
        if current != expected:
            result["status"]="source_changed"
        decision=decide(result,current,source,best,task,initializing=cid=="baseline")
        record.update({"worker_status":raw.get("status"),"effective_status":result.get("status"),"decision":decision})
        if result.get("segments"):
            record["statistics"]=paired_stats(result["segments"])
        record["transitions"] += [result.get("status","missing_result"),decision]
        if decision in ["accepted_baseline","accepted"]:
            best={"id":cid,"candidate_source":source,"candidate_source_sha256":current[source],
                  "task_sha256":current["task.json"],"worker_evidence_sha256":hashlib.sha256((dest/"worker.json").read_bytes()).hexdigest()}
            dump(out/"best.json",best)
        else:
            assert best==before
            assert ((out/"best.json").read_bytes() if (out/"best.json").exists() else None)==before_bytes
        record["best_after"]=copy.deepcopy(best)
        record["best_unchanged"]=best==before
        record["finished_utc"]=utc()
        record["telemetry_after"]=telemetry()
        dump(dest/"decision.json",record)
        with (out/"trajectory.jsonl").open("a") as f:
            f.write(json.dumps(record,allow_nan=False)+"\n")
        records.append(record)
        print(json.dumps({"id":cid,"status":result.get("status"),"decision":decision,"best":None if best is None else best["id"]}),flush=True)
        if cid=="baseline" and decision!="accepted_baseline":
            raise RuntimeError("baseline failed; preserve evidence and stop")
        if timed_out and not record["termination"]["no_live_members"]:
            raise RuntimeError("timeout group cleanup failed; stop before next GPU worker")
    dump(out/"summary.json",{"status":"RUN_COMPLETED","records":len(records),"best":best,
        "decisions":{r["id"]:r["decision"] for r in records},"scope":"GPU fault-injection; not performance optimization"})


if __name__=="__main__":
    main()
