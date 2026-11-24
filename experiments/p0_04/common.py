"""CPU-only source binding, paired statistics, and deterministic acceptance gate."""
import hashlib
import json
import math
from pathlib import Path
import random
import statistics


def source_hashes(root):
    root = Path(root)
    files = sorted(root.rglob("*.py")) + [root / "task.json"]
    return {str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in files if "__pycache__" not in p.parts}


def dump(path, data):
    path = Path(path)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(data, indent=2, allow_nan=False) + "\n")
    tmp.replace(path)


def paired_stats(rows):
    if len(rows) != 18:
        raise ValueError("need all 9 complete pairs")
    pairs = {}
    for row in rows:
        if row["label"] in pairs.setdefault(row["pair"], {}):
            raise ValueError("duplicate pair member")
        ms = row["elapsed_ms"]
        if not math.isfinite(ms) or ms <= 0:
            raise ValueError("nonpositive/nonfinite timing")
        pairs[row["pair"]][row["label"]] = ms
    if sorted(pairs) != list(range(9)):
        raise ValueError("missing pair")
    ratios = [p["baseline"] / p["candidate"] for p in pairs.values()]
    logs = [math.log(x) for x in ratios]
    rng = random.Random(3070)
    boot = sorted(math.exp(statistics.mean(rng.choices(logs, k=9))) for _ in range(10000))

    def q(p):
        x = (len(boot)-1)*p
        lo, hi = math.floor(x), math.ceil(x)
        return boot[lo] if lo == hi else boot[lo]*(hi-x)+boot[hi]*(x-lo)

    return {"ratios": ratios, "geomean_ratio": math.exp(statistics.mean(logs)),
            "bootstrap_95": [q(.025), q(.975)]}


def decide(result, current_hashes, candidate_source, best, task, initializing=False):
    """Only this trusted gate may change best; candidate claims do not choose it."""
    if result.get("source_sha256") != current_hashes:
        return "rejected_source_mismatch"
    if result.get("status") != "validated_and_measured":
        return "rejected_" + result.get("status", "missing_result")
    if result.get("source_sha256_after") != current_hashes:
        return "rejected_source_mismatch"
    pre, post = result.get("checks_pre", []), result.get("checks_post", [])
    if len(pre) != len(task["correctness_cases"]) or len(post) != 2:
        return "rejected_missing_checks"
    if not all(c.get("pass") is True for c in pre+post):
        return "rejected_failed_checks"
    try:
        stats = paired_stats(result["segments"])
    except (KeyError, ValueError, TypeError, ZeroDivisionError):
        return "rejected_invalid_measurement"
    if initializing:
        return "accepted_baseline" if candidate_source == task["baseline"] else "rejected_invalid_baseline"
    if best is None:
        return "rejected_no_baseline"
    if current_hashes[candidate_source] == best["candidate_source_sha256"]:
        return "rejected_duplicate_source"
    rule = task["promotion"]
    if stats["geomean_ratio"] >= rule["minimum_ratio"] and stats["bootstrap_95"][0] > rule["bootstrap_lower_strictly_above"]:
        return "accepted"
    return "rejected_no_stable_gain"
