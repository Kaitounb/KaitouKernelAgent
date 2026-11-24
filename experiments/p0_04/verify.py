"""Independent CPU audit of archived GPU outcomes and the best-state trajectory."""
import argparse
import hashlib
import json
import math
from pathlib import Path
import statistics


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--evidence",type=Path,default=Path(__file__).resolve().parents[2]/"evidence/20261117-p0-04")
    args=ap.parse_args()
    evidence=args.evidence
    run=evidence/"run"
    hashes=json.loads((run/"source_hashes.json").read_text())
    for name,sha in hashes.items():
        assert hashlib.sha256((evidence/"sources"/name).read_bytes()).hexdigest()==sha,name
    task=json.loads((run/"task.json").read_text())
    # Supervisor also writes a normalized JSON copy; source bytes are hashed above.
    assert task==json.loads((evidence/"sources/task.json").read_text())
    trajectory=[json.loads(x) for x in (run/"trajectory.jsonl").read_text().splitlines()]
    expected={"baseline":"accepted_baseline","wrong_constant":"rejected_validation_failed",
              "input_mutation":"rejected_validation_failed","tail_drop":"rejected_validation_failed",
              "cached_output":"rejected_validation_failed","delayed_mutation":"rejected_post_validation_failed",
              "runtime_error":"rejected_worker_error","host_timeout":"rejected_timed_out",
              "recovery":"rejected_duplicate_source"}
    assert [x["id"] for x in trajectory]==[x["id"] for x in task["candidate_order"]]
    best=json.loads((run/"best.json").read_text())
    assert best["id"]=="baseline" and best["candidate_source_sha256"]==hashes["baseline.py"]
    assert hashlib.sha256((run/"baseline/worker.json").read_bytes()).hexdigest()==best["worker_evidence_sha256"]
    rows=[]
    raw={}
    total_segments=0
    for entry in trajectory:
        cid=entry["id"]
        assert entry==json.loads((run/cid/"decision.json").read_text())
        assert entry["decision"]==expected[cid],(cid,entry["decision"])
        if cid=="baseline":
            assert entry["best_before"] is None and entry["best_after"]==best
        else:
            assert entry["best_before"]==entry["best_after"]==best and entry["best_unchanged"]
            assert entry["parent_source_sha256"]==hashes["baseline.py"]
        result=json.loads((run/cid/"worker.json").read_text())
        raw[cid]=result
        assert result["source_sha256"]==hashes
        assert result["candidate_source"]==entry["candidate_source"]
        assert entry["candidate_source_sha256"]==hashes[entry["candidate_source"]]
        for check in result["checks_pre"]+result["checks_post"]:
            assert check["input_unchanged"]==(check["input_sha256_before"]==check["input_sha256_after"])
            assert check["output_exact"]==(check["output_sha256"]==check["reference_sha256"])
            assert check["pass"]==all(check[k] for k in ["input_unchanged","output_exact","guards_unchanged"])
            if check["pass"]:
                assert check["max_abs_error"]==check["max_stable_relative_error"]==check["normalized_rms_error"]==0
        if result["segments"]:
            assert len(result["segments"])==18
            assert result["source_sha256_after"]==hashes
            pairmap={}
            for x in result["segments"]:
                expected_order=["baseline","candidate"] if x["pair"]%2==0 else ["candidate","baseline"]
                assert x["label"]==expected_order[x["order"]]
                assert x["calls"]==512 and x["elapsed_ms"]>0 and math.isfinite(x["elapsed_ms"])
                assert x["label"] not in pairmap.setdefault(x["pair"],{})
                pairmap[x["pair"]][x["label"]]=x["elapsed_ms"]
            assert sorted(pairmap)==list(range(9))
            ratios=[pairmap[i]["baseline"]/pairmap[i]["candidate"] for i in range(9)]
            ratio=math.exp(statistics.mean(math.log(x) for x in ratios))
            assert math.isclose(ratio,entry["statistics"]["geomean_ratio"],rel_tol=1e-12)
            total_segments+=len(result["segments"])
        rows.append({"id":cid,"worker_status":result["status"],"decision":entry["decision"],
                     "pre_checks":len(result["checks_pre"]),"segments":len(result["segments"]),
                     "best_unchanged":entry["best_unchanged"],"wall_seconds":entry["wall_seconds"]})

    assert not raw["wrong_constant"]["checks_pre"][0]["output_exact"]
    check=raw["input_mutation"]["checks_pre"][0]
    assert check["output_exact"] and not check["input_unchanged"] and check["guards_unchanged"]
    assert raw["tail_drop"]["checks_pre"][0]["nonfinite_count"]==1
    cached=raw["cached_output"]["checks_pre"]
    assert len(cached)==2 and cached[0]["pass"] and not cached[1]["output_exact"]
    delayed=raw["delayed_mutation"]
    assert len(delayed["checks_pre"])==5 and all(x["pass"] for x in delayed["checks_pre"])
    assert delayed["checks_post"][0]["pass"] and not delayed["checks_post"][1]["input_unchanged"]
    assert len(delayed["segments"])==18
    for cid in ["baseline","recovery"]:
        assert len(raw[cid]["checks_pre"])==5 and len(raw[cid]["checks_post"])==2
        assert all(x["pass"] for x in raw[cid]["checks_pre"]+raw[cid]["checks_post"])
    timeout=next(x for x in trajectory if x["id"]=="host_timeout")
    assert timeout["timeout_phase"]=="execution" and timeout["termination"]["no_live_members"]
    assert len(timeout["termination"]["before"])>=2
    assert "SIGTERM" in timeout["termination"]["signals"]
    marker=json.loads((run/"host_timeout/stdout.log").read_text().strip())
    assert marker["gpu_synchronized"] and marker["child_pid"] in [x["pid"] for x in timeout["termination"]["before"]]
    assert not any(x["state"] not in ["Z","X"] for x in timeout["termination"]["after"])
    summary=json.loads((run/"summary.json").read_text())
    assert summary["best"]==best and summary["decisions"]==expected and summary["records"]==9
    report={"status":"INDEPENDENT_AUDIT_PASS","evidence_type":"CPU_RECOMPUTATION_OF_GPU_EVIDENCE",
            "worker_attempts_on_gpu_environment":9,"faulty_candidates_rejected":7,"raw_segments":total_segments,
            "final_best":"baseline","timeout_process_group_no_live_members":True,"rows":rows}
    (evidence/"verification.json").write_text(json.dumps(report,indent=2)+"\n")
    print(json.dumps(report,indent=2))


if __name__=="__main__":
    main()
