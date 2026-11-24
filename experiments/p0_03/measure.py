"""Real CUDA calibration and A/A measurements. No profiler; all segments retained."""
import argparse
import hashlib
import json
import math
import os
import platform
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path

import torch
import triton
from kernel import launch


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def smi():
    commands = [
        ["nvidia-smi", "--query-gpu=name,uuid,driver_version,temperature.gpu,clocks.sm,clocks.mem,power.draw,power.limit,utilization.gpu,memory.used", "--format=csv"],
        ["nvidia-smi", "--query-compute-apps=pid,process_name,used_gpu_memory", "--format=csv"],
    ]
    records = []
    for cmd in commands:
        p = subprocess.run(cmd, text=True, capture_output=True, timeout=15)
        records.append({"command": cmd, "exit": p.returncode, "stdout": p.stdout, "stderr": p.stderr})
    return {"utc": datetime.now(timezone.utc).isoformat(), "queries": records}


def digest(t):
    return hashlib.sha256(t.contiguous().numpy().tobytes()).hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--phase", choices=["calibrate", "measure"], required=True)
    ap.add_argument("--session", type=int, default=0)
    ap.add_argument("--output", required=True)
    args = ap.parse_args()
    torch.set_num_threads(1)
    task = json.loads(Path("task.json").read_text())
    result = {"phase": args.phase, "session": args.session, "started_utc": datetime.now(timezone.utc).isoformat(),
              "pid": os.getpid(), "source_sha256": {f: sha(f) for f in ["kernel.py", "measure.py", "task.json"]},
              "environment": {"python": platform.python_version(), "platform": platform.platform(),
                              "torch": torch.__version__, "triton": triton.__version__, "cuda_runtime": torch.version.cuda,
                              "device": str(torch.cuda.get_device_properties(0)), "clock_lock_requested": False}, "policies": []}
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    stream = torch.cuda.Stream()
    n, guard = task["n"], task["guard_elements_each_side"]
    with torch.cuda.stream(stream):
        for policy, count in task["cache_policies"].items():
            before = smi()
            xs, ys, original, refs = [], [], [], []
            for seed in task["seeds"][:count]:
                g = torch.Generator(device="cpu").manual_seed(seed)
                x = torch.randint(-8192, 8193, (n,), generator=g, dtype=torch.int32).float() / 16
                original.append(x)
                refs.append((x.double() + 1).float())
                padded = torch.full((n + 2*guard,), task["guard_value"], device="cuda")
                padded[guard:guard+n].copy_(x)
                xs.append(padded)
                ys.append(torch.full_like(padded, task["guard_value"]))

            def invoke(i):
                j = i % count
                launch(xs[j][guard:guard+n], ys[j][guard:guard+n], n)

            # Compile and warm before capture; allocations and views are excluded.
            for i in range(task["graph_launches"]):
                invoke(i)
            stream.synchronize()
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph, stream=stream):
                for i in range(task["graph_launches"]):
                    invoke(i)
            start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            # Initialize lazy Events before recording data.
            start.record(stream); end.record(stream); end.synchronize()

            def segment(replays):
                start.record(stream)
                for _ in range(replays):
                    graph.replay()
                end.record(stream)
                end.synchronize()
                ms = start.elapsed_time(end)
                if not math.isfinite(ms) or ms <= 0:
                    raise RuntimeError("invalid Event duration")
                return ms

            until = time.monotonic() + task["warmup_wall_seconds"]
            while time.monotonic() < until:
                graph.replay()
                stream.synchronize()
            item = {"policy": policy, "count": count, "telemetry_before": before}
            if args.phase == "calibrate":
                item["pilot_graph_ms"] = [segment(1) for _ in range(5)]
            else:
                repeats = task["graph_replays_per_segment"]
                if not isinstance(repeats, int) or repeats < 1:
                    raise ValueError("freeze calibration before measuring")
                item["graph_replays_per_segment"] = repeats
                item["segments"] = []
                for pair in range(task["pairs_per_session"]):
                    labels = ["A", "A_prime"] if (pair+args.session) % 2 == 0 else ["A_prime", "A"]
                    for order, label in enumerate(labels):
                        elapsed = segment(repeats)
                        item["segments"].append({"pair": pair, "order": order, "label": label,
                            "elapsed_ms": elapsed, "calls": repeats*task["graph_launches"],
                            "us_per_call": elapsed*1000/(repeats*task["graph_launches"])})
            checks = []
            stream.synchronize()
            for i in range(count):
                x, y = xs[i].cpu(), ys[i].cpu()
                actual = y[guard:guard+n]
                err = actual.double() - refs[i].double()
                ok_x = torch.equal(x[guard:guard+n], original[i])
                ok_y = torch.equal(actual, refs[i])
                ok_guard = all(torch.all(t == task["guard_value"]).item() for t in [x[:guard],x[-guard:],y[:guard],y[-guard:]])
                checks.append({"seed": task["seeds"][i], "input_sha256_before": digest(original[i]),
                    "input_sha256_after": digest(x[guard:guard+n]), "output_sha256": digest(actual),
                    "input_unchanged": ok_x, "output_exact": ok_y, "guards_unchanged": ok_guard,
                    "max_abs_error": err.abs().max().item(),
                    "max_stable_relative_error": (err.abs()/refs[i].double().abs().clamp(min=1e-12)).max().item(),
                    "normalized_rms_error": (err.square().mean().sqrt()/refs[i].double().square().mean().sqrt().clamp(min=1e-12)).item()})
                if not (ok_x and ok_y and ok_guard):
                    raise RuntimeError("correctness/input/guard check failed")
            item["checks"] = checks
            item["telemetry_after"] = smi()
            result["policies"].append(item)
            # Release graph references before moving to another policy.
            del graph
    result["finished_utc"] = datetime.now(timezone.utc).isoformat()
    result["status"] = "PASS"
    Path(args.output).write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({"status": "PASS", "output": args.output, "phase": args.phase, "session": args.session}))


if __name__ == "__main__":
    main()
