"""Pilot A/A of a trusted GEMM binary, following the P0-03 paired Graph protocol."""
import argparse
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import statistics
import time

import torch


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--binary", type=Path, required=True)
    parser.add_argument("--sha256", required=True)
    parser.add_argument("--session", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists() or args.session < 0:
        parser.error("Use a new output file and a nonnegative session number")
    binary = args.binary.resolve()
    if hashlib.sha256(binary.read_bytes()).hexdigest() != args.sha256:
        raise ValueError("Trusted binary hash mismatch")
    spec = importlib.util.spec_from_file_location("nka_cublas_gemm", binary)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    torch.set_num_threads(1)
    device = torch.device("cuda:0")
    stream = torch.cuda.Stream(device=device)
    graph = torch.cuda.CUDAGraph()
    generator = torch.Generator().manual_seed(173)
    a_cpu, b_cpu = [torch.randn(512, 512, generator=generator).mul_(0.25).half() for _ in range(2)]
    expected = (a_cpu.double() @ b_cpu.double()).float()
    graph_launches = 32
    replays = 12
    segments = []
    with torch.no_grad(), torch.cuda.stream(stream):
        a, b = a_cpu.to(device), b_cpu.to(device)
        for _ in range(4):
            module.gemm_cublas(a, b)
        stream.synchronize()
        with torch.cuda.graph(graph, stream=stream):
            for _ in range(graph_launches):
                last_output = module.gemm_cublas(a, b)

        deadline = time.monotonic() + 0.5
        while time.monotonic() < deadline:
            graph.replay()
            stream.synchronize()

        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        for pair in range(9):
            order = ("A", "A_prime") if (pair + args.session) % 2 == 0 else ("A_prime", "A")
            for position, label in enumerate(order):
                start.record(stream)
                for _ in range(replays):
                    graph.replay()
                end.record(stream)
                end.synchronize()
                elapsed = start.elapsed_time(end)
                if not math.isfinite(elapsed) or elapsed <= 0:
                    raise RuntimeError(f"Invalid Event segment: {elapsed}")
                segments.append({"pair": pair, "order": position, "label": label, "elapsed_ms": elapsed})
        torch.testing.assert_close(last_output.cpu(), expected, atol=3e-5, rtol=1e-4)
        torch.testing.assert_close(a.cpu(), a_cpu, atol=0, rtol=0)
        torch.testing.assert_close(b.cpu(), b_cpu, atol=0, rtol=0)
    result = {
        "status": "PASS",
        "session": args.session,
        "scope": "A/A pilot, same binary and same graph, repeated inputs, no profiler",
        "device": torch.cuda.get_device_name(device),
        "binary_sha256": args.sha256,
        "shape_mnk": [512, 512, 512],
        "graph_launches": graph_launches,
        "replays_per_segment": replays,
        "segments": segments,
        "median_segment_ms": statistics.median(segment["elapsed_ms"] for segment in segments),
        "cpu_reference_and_inputs_pass": True,
    }
    with args.output.open("x") as file:
        json.dump(result, file, indent=2, allow_nan=False)
        file.write("\n")


if __name__ == "__main__":
    main()
