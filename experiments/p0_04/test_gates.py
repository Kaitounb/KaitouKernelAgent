"""CPU simulations of the acceptance gate; never presented as GPU evidence."""
import argparse
import copy
import json
from pathlib import Path

from common import decide, dump


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--output",type=Path,required=True)
    args=ap.parse_args()
    task=json.loads((Path(__file__).parent/"task.json").read_text())
    hashes={"baseline.py":"base_hash","candidate.py":"new_hash","task.json":"task_hash"}
    best={"candidate_source_sha256":"base_hash"}
    valid={"status":"validated_and_measured","source_sha256":hashes,"source_sha256_after":hashes,
        "checks_pre":[{"pass":True} for _ in task["correctness_cases"]],"checks_post":[{"pass":True},{"pass":True}],
        "segments":[{"pair":i,"label":label,"elapsed_ms":ms} for i in range(9) for label,ms in [("baseline",12.),("candidate",8.)]]}
    tests=[]

    def run(name,value,expected,current=None):
        got=decide(value,hashes if current is None else current,"candidate.py",best,task)
        assert got==expected,(name,got,expected)
        tests.append({"name":name,"decision":got,"pass":True})

    # A legitimate speedup must be able to pass, so rejecting failures is not vacuous.
    run("valid_faster_can_promote",copy.deepcopy(valid),"accepted")
    for status in ["validation_failed","post_validation_failed","worker_error","timed_out"]:
        value=copy.deepcopy(valid); value["status"]=status
        run("fast_claim_cannot_override_"+status,value,"rejected_"+status)
    value=copy.deepcopy(valid); value["checks_pre"][0]["pass"]=False
    run("fast_claim_cannot_override_failed_check",value,"rejected_failed_checks")
    value=copy.deepcopy(valid); value["checks_post"]=[]
    run("missing_post_check",value,"rejected_missing_checks")
    changed=dict(hashes); changed["candidate.py"]="changed_after_run"
    run("changed_source_invalidates_old_evidence",copy.deepcopy(valid),"rejected_source_mismatch",changed)
    value=copy.deepcopy(valid); value["source_sha256_after"]={}
    run("source_changed_inside_worker",value,"rejected_source_mismatch")
    value=copy.deepcopy(valid); value["segments"][0]["elapsed_ms"]=float("nan")
    run("nan_timing_rejected",value,"rejected_invalid_measurement")
    value=copy.deepcopy(valid); value["segments"].pop()
    run("incomplete_pairs_rejected",value,"rejected_invalid_measurement")
    value=copy.deepcopy(valid)
    for row in value["segments"]: row["elapsed_ms"]=10.
    run("valid_equal_speed_rejected",value,"rejected_no_stable_gain")
    dump(args.output,{"status":"PASS","evidence_type":"CPU_SIMULATION_ONLY","test_count":len(tests),"tests":tests})
    print(json.dumps({"status":"PASS","CPU_simulation_tests":len(tests)}))


if __name__=="__main__":
    main()
