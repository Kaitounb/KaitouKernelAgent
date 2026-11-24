"""CPU-only independent recomputation and evidence consistency checks."""
import argparse
import hashlib
import json
import math
from pathlib import Path
import random
import statistics


def quantile(values, p):
    values = sorted(values)
    x = (len(values)-1)*p
    a = math.floor(x)
    b = math.ceil(x)
    return values[a]*(b-x)+values[b]*(x-a) if a != b else values[a]


def stats(segments):
    pairs = {}
    for row in segments:
        assert row["label"] not in pairs.setdefault(row["pair"], {})
        pairs[row["pair"]][row["label"]] = row["elapsed_ms"]
        assert row["elapsed_ms"] > 0
        assert math.isclose(row["us_per_call"],row["elapsed_ms"]*1000/row["calls"])
    ratios = [p["A"]/p["A_prime"] for p in pairs.values()]
    logs = [math.log(x) for x in ratios]
    rng = random.Random(3070)
    boot = [math.exp(statistics.mean(rng.choices(logs,k=len(logs)))) for _ in range(10000)]
    return {"pair_ratios": ratios, "geomean_ratio": math.exp(statistics.mean(logs)),
            "bootstrap_95_percentile": [quantile(boot,.025),quantile(boot,.975)],
            "max_pair_deviation_percent": max(abs(r-1) for r in ratios)*100,
            "median_us_per_call": statistics.median(x["us_per_call"] for x in segments),
            "segment_ms_range": [min(x["elapsed_ms"] for x in segments),max(x["elapsed_ms"] for x in segments)]}


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--evidence",type=Path,default=Path(__file__).resolve().parents[2]/"evidence/20261117-p0-03")
    args=ap.parse_args()
    results=[]
    hashes=[]
    pids=[]
    for session in range(3):
        raw=json.loads((args.evidence/f"session-{session}.json").read_text())
        assert raw["status"] == "PASS" and raw["session"] == session and raw["phase"] == "measure"
        hashes.append(raw["source_sha256"])
        pids.append(raw["pid"])
        for f,h in raw["source_sha256"].items():
            data=(args.evidence/"sources"/f"session-{session}"/f).read_bytes()
            assert hashlib.sha256(data).hexdigest() == h, (session,f)
        task=json.loads((args.evidence/"sources"/f"session-{session}"/"task.json").read_text())
        for policy in raw["policies"]:
            assert len(policy["segments"]) == 18
            assert policy["graph_replays_per_segment"] == task["graph_replays_per_segment"]
            for row in policy["segments"]:
                assert row["calls"] == task["graph_launches"]*task["graph_replays_per_segment"]
                expected=["A","A_prime"] if (row["pair"]+session)%2==0 else ["A_prime","A"]
                assert row["label"] == expected[row["order"]]
            for c in policy["checks"]:
                assert c["input_unchanged"] and c["output_exact"] and c["guards_unchanged"]
                assert c["input_sha256_before"] == c["input_sha256_after"]
                assert c["max_abs_error"] == c["max_stable_relative_error"] == c["normalized_rms_error"] == 0
            results.append({"session": session,"policy": policy["policy"],**stats(policy["segments"])})
    assert all(h == hashes[0] for h in hashes)
    assert len(set(pids)) == 3
    summary={"status":"RECOMPUTED_PASS","independent_process_pids":pids,"source_sha256":hashes[0],
             "total_segments":108,"total_pairs":54,"results":results,
             "inference_limit":"Bootstrap resamples 9 pairs within one process only; exploratory, not a post-selection confidence guarantee or a cross-workload noise bound."}
    (args.evidence/"summary.json").write_text(json.dumps(summary,indent=2)+"\n")
    print("session policy median_us geomean_ratio bootstrap_95 max_pair_deviation_percent")
    for r in results:
        print(r["session"],r["policy"],f'{r["median_us_per_call"]:.4f}',f'{r["geomean_ratio"]:.6f}',
              [round(x,6) for x in r["bootstrap_95_percentile"]],f'{r["max_pair_deviation_percent"]:.4f}')


if __name__ == "__main__":
    main()
