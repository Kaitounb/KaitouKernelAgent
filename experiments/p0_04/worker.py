"""Trusted evaluator runs a candidate; parent owns all best-state writes."""
import argparse
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import platform
import time
import traceback

from common import dump, source_hashes


def load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--candidate", required=True)
    ap.add_argument("--output", type=Path, required=True)
    args = ap.parse_args()
    root = Path(__file__).resolve().parent
    task = json.loads((root/"task.json").read_text())
    result = {"status": "starting", "source_sha256": source_hashes(root),
              "candidate_source": args.candidate, "pid": os.getpid(),
              "checks_pre": [], "checks_post": [], "segments": []}
    args.output.mkdir(parents=True, exist_ok=True)

    def save():
        dump(args.output/"worker.json", result)

    save()
    try:
        import torch
        import triton
        torch.set_num_threads(1)
        result["environment"] = {"python": platform.python_version(), "platform": platform.platform(),
            "torch": torch.__version__, "triton": triton.__version__, "cuda_runtime": torch.version.cuda,
            "device": str(torch.cuda.get_device_properties(0)), "uid": os.getuid(),
            "environment_variable_names": sorted(os.environ)}
        candidate = load(root/args.candidate, "candidate_module")
        baseline = load(root/task["baseline"], "baseline_module")
        stream = torch.cuda.Stream()
        guard = task["guard"]

        def make(case):
            n = case["n"]
            g = torch.Generator(device="cpu").manual_seed(case["seed"])
            original = (torch.zeros(n) if case["distribution"] == "zeros" else
                        torch.randint(-8192,8193,(n,),generator=g,dtype=torch.int32).float()/16)
            x = torch.full((n+guard*2,), task["guard_value"], device="cuda")
            y = torch.full_like(x, task["guard_value"])
            x[guard:guard+n].copy_(original)
            y[guard:guard+n].fill_(float("nan"))
            return {"case": case, "original": original, "ref": (original.double()+1).float(),
                    "x": x, "y": y, "xv": x[guard:guard+n], "yv": y[guard:guard+n]}

        def check(buf, label):
            stream.synchronize()
            x, y, original, ref = buf["x"].cpu(), buf["y"].cpu(), buf["original"], buf["ref"]
            xx, yy = x[guard:-guard], y[guard:-guard]
            err = yy.double()-ref.double()
            finite = bool(torch.isfinite(yy).all())
            unchanged, correct = torch.equal(xx,original), torch.equal(yy,ref)
            guards = all(bool(torch.all(t==task["guard_value"])) for t in [x[:guard],x[-guard:],y[:guard],y[-guard:]])
            digest = lambda t: hashlib.sha256(t.contiguous().numpy().tobytes()).hexdigest()
            return {"case": buf["case"], "label": label, "pass": unchanged and correct and guards,
                "output_exact": correct, "input_unchanged": unchanged, "guards_unchanged": guards,
                "output_finite": finite, "nonfinite_count": int((~torch.isfinite(yy)).sum()),
                "input_sha256_before": digest(original), "input_sha256_after": digest(xx),
                "output_sha256": digest(yy), "reference_sha256": digest(ref),
                "max_abs_error": float(err.abs().max()) if finite else None,
                "max_stable_relative_error": float((err.abs()/ref.double().abs().clamp(min=1e-12)).max()) if finite else None,
                "normalized_rms_error": float(err.square().mean().sqrt()/ref.double().square().mean().sqrt().clamp(min=1e-12)) if finite else None}

        # Keep every buffer alive so per-address delayed faults cannot be hidden by allocator reuse.
        with torch.cuda.stream(stream):
            buffers = [make(case) for case in task["correctness_cases"]]
            stream.synchronize()
            dump(args.output/"ready.json", {"pid": os.getpid(), "phase": "before_first_candidate_call"})
            result["status"] = "validating"
            save()
            for buf in buffers:
                candidate.launch(buf["xv"],buf["yv"],buf["case"]["n"])
                row = check(buf,"candidate_pre")
                result["checks_pre"].append(row)
                save()
                if not row["pass"]:
                    result["status"] = "validation_failed"
                    save()
                    return

            case = {"n":task["timing_n"],"seed":task["timing_seed"],"distribution":"random"}
            b, c = make(case), make(case)
            funcs = {"baseline": (baseline.launch,b),"candidate": (candidate.launch,c)}
            # Compile all paths outside graph capture, including repeated-call behavior.
            for fn,buf in funcs.values():
                for _ in range(2):
                    fn(buf["xv"],buf["yv"],case["n"])
            stream.synchronize()
            graphs = {}
            for label,(fn,buf) in funcs.items():
                graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph, stream=stream):
                    for _ in range(task["graph_launches"]):
                        fn(buf["xv"],buf["yv"],case["n"])
                graphs[label] = graph
            until = time.monotonic()+task["warmup_wall_seconds"]
            while time.monotonic()<until:
                for graph in graphs.values():
                    graph.replay()
                stream.synchronize()
            start,end = torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True)
            start.record(stream); end.record(stream); end.synchronize()
            result["status"] = "benchmarking"
            save()
            for pair in range(task["pairs"]):
                labels = ["baseline","candidate"] if pair%2==0 else ["candidate","baseline"]
                for order,label in enumerate(labels):
                    start.record(stream)
                    for _ in range(task["graph_replays"]):
                        graphs[label].replay()
                    end.record(stream); end.synchronize()
                    ms = start.elapsed_time(end)
                    if not math.isfinite(ms) or ms<=0:
                        raise ValueError("invalid Event duration")
                    result["segments"].append({"pair":pair,"order":order,"label":label,"elapsed_ms":ms,
                        "calls":task["graph_launches"]*task["graph_replays"]})
            result["checks_post"] = [check(b,"baseline_post"),check(c,"candidate_post")]
            result["status"] = "validated_and_measured" if all(r["pass"] for r in result["checks_post"]) else "post_validation_failed"
            result["source_sha256_after"] = source_hashes(root)
            save()
    except Exception:
        result["status"] = "worker_error"
        result["exception"] = traceback.format_exc()
        save()
        raise


if __name__ == "__main__":
    main()
