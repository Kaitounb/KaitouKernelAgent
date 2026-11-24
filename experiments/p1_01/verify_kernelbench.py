"""Small teaching checks using the KernelBench load_inline pattern.

This is correctness glue, not KernelBench's full evaluator or a benchmark.
"""
import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import platform
import time

import torch


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def check_address_toy(check):
    check("address_toy", torch.arange(1, 7, dtype=torch.float32).reshape(2, 3),
          torch.arange(1, 13, dtype=torch.float32).reshape(3, 4))


def check_naive(check):
    check_address_toy(check)
    for label, m, n, k in [
        ("both_output_edges", 17, 19, 7),
        ("single_row", 1, 33, 5),
        ("single_column", 33, 1, 5),
    ]:
        a = ((torch.arange(m * k) % 7) - 3).reshape(m, k).float()
        b = ((torch.arange(k * n) % 11) - 5).reshape(k, n).float()
        check(label, a, b)


def check_wmma(check):
    a = torch.zeros((16, 16), dtype=torch.float16)
    b = torch.zeros_like(a)
    a[:, 0] = torch.arange(1, 17, dtype=torch.float16)
    a[:, 15] = 2
    b[0, :] = torch.arange(1, 17, dtype=torch.float16)
    b[15, :] = -1
    expected = [[(r + 1) * (c + 1) - 2 for c in range(16)] for r in range(16)]
    check("hand_tile", a, b, expected=expected)

    linear = torch.arange(256).reshape(16, 16)
    a = (((linear % 17) - 8) / 8).half()
    b = (((linear % 13) - 6) / 4).half()
    check("dense_signed_half", a, b)

    a = torch.zeros((16, 16), dtype=torch.float32)
    a[:, 0] = 1 + 2**-12
    a[:, 1] = 1
    b = torch.zeros((16, 16), dtype=torch.float16)
    b[0, :] = 2048
    b[1, :] = 1
    check("half_input_rounding_float_output", a.half(), b, expected=[[2049] * 16 for _ in range(16)])


def check_shared_tail(check):
    a = torch.zeros((2, 17), dtype=torch.float32)
    b = torch.zeros((17, 2), dtype=torch.float32)
    a[:, [0, 15, 16]] = torch.tensor([[1, 2, 3], [4, 5, 6]], dtype=torch.float32)
    b[[0, 15, 16]] = torch.tensor([[7, 8], [9, 10], [11, 12]], dtype=torch.float32)

    # Preserve both K iterations when isolating each contribution.
    a_first = a.clone()
    a_first[:, 16] = 0
    a_tail = a.clone()
    a_tail[:, :16] = 0

    check("both_k_tiles", a, b, expected=[[58, 64], [139, 154]])
    check("first_k_tile_only", a_first, b, expected=[[25, 28], [73, 82]])
    check("tail_k_tile_only", a_tail, b, expected=[[33, 36], [66, 72]])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--case", choices=["original", "shared-tail", "naive", "wmma"], default="original")
    args = parser.parse_args()
    torch.set_num_threads(1)
    upstream = Path("model_new_ex_tiled_matmul.py")
    upstream_sha = "4d168d4fa4c8d5260d4ac9bccdc229702b13630704a59a409ec7090322c1fd79"
    if sha(upstream) != upstream_sha:
        raise ValueError("upstream teaching source changed")
    source = upstream
    expected_sha = upstream_sha
    if args.case == "naive":
        source = Path("naive_matmul.cu")
        expected_sha = "d64f72a1aaf9774f44d46c0be49fc681c47213c1f929dc1dd67eac0a0f0c7c8b"
    elif args.case == "wmma":
        source = Path("wmma_tile.cu")
        expected_sha = "e2058dfd9098d33d16208071c64bd61fe6f7e75162516ee3bf394fe9eaf6dff1"
    if sha(source) != expected_sha:
        raise ValueError("selected teaching source changed")
    result = {"status": "STARTING", "phase": "compile", "gpu_run": True,
              "performance_claim": False, "source_sha256": sha(source),
              "checker_sha256": sha(__file__), "checks": [],
              "environment": {"torch": torch.__version__, "python": platform.python_version(),
                  "platform": platform.platform(), "device": torch.cuda.get_device_name(0),
                  "capability": list(torch.cuda.get_device_capability(0)), "uid": os.getuid(),
                  "build_environment": {k: os.environ.get(k) for k in
                      ["CUDA_HOME", "TORCH_CUDA_ARCH_LIST", "TORCH_EXTENSIONS_DIR", "MAX_JOBS"]}},
              "reference": "CPU FP64 matmul from original FP32 values, then FP32; atol=rtol=0",
              "input_rule": "random integer [-8,8]/8, exact binary fractions; K<=65; toy small integers",
              "scope": "FP32, positive 2D contiguous inputs on CUDA:0, default stream; no timing"}
    if args.case == "shared-tail":
        result["case"] = args.case
        result["input_rule"] = "Fixed M=N=2, K=17; small integers at K=0,15,16; isolate each K tile using zeros"
    if args.case == "naive":
        result["case"] = args.case
        result["upstream_sha256"] = upstream_sha
        result["input_rule"] = "Address toy plus fixed small integer sequences for rectangular edge cases"
    if args.case == "wmma":
        result["case"] = args.case
        result["reference"] = "CPU FP64 matmul from stored FP16 values, then FP32; atol=rtol=0"
        result["input_rule"] = "Hand-computed sparse tile, dense signed dyadic values, and FP16 input rounding"
        result["scope"] = "16x16 contiguous aligned FP16 inputs, FP32 output on CUDA:0, default stream; no timing"

    def save():
        Path("result.json").write_text(json.dumps(result, indent=2, allow_nan=False)+"\n")

    save()
    try:
        started = time.monotonic()
        if source.suffix == ".cu":
            from torch.utils.cpp_extension import load_inline

            symbol = f"{source.stem}_cuda"
            extension = load_inline(
                source.stem,
                cpp_sources=f"torch::Tensor {symbol}(const torch::Tensor& a, const torch::Tensor& b);",
                cuda_sources=source.read_text(encoding="utf-8"),
                functions=[symbol],
                extra_cuda_cflags=["--ptxas-options=-v"],
                verbose=True,
            )
            model = getattr(extension, symbol)
        else:
            spec = importlib.util.spec_from_file_location("kernelbench_teaching", source)
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)  # Original load_inline builds the extension.
            model = module.ModelNew().eval()
        result["import_and_build_wall_seconds"] = time.monotonic()-started
        result["phase"] = "correctness"
        default = torch.cuda.default_stream(0)
        with torch.no_grad(), torch.cuda.stream(default):
            def check(label, a_cpu, b_cpu, expected=None):
                reference = (a_cpu.double() @ b_cpu.double()).float()
                if expected is not None:
                    torch.testing.assert_close(
                        reference, torch.tensor(expected, dtype=reference.dtype), atol=0, rtol=0
                    )
                a, b = a_cpu.cuda(), b_cpu.cuda()
                outputs = []
                for _ in range(2):
                    out = model(a, b)
                    torch.cuda.synchronize(0)
                    actual = out.cpu()
                    torch.testing.assert_close(actual, reference, atol=0, rtol=0)
                    torch.testing.assert_close(a.cpu(), a_cpu, atol=0, rtol=0)
                    torch.testing.assert_close(b.cpu(), b_cpu, atol=0, rtol=0)
                    outputs.append(actual)
                row = {"label": label, "m": a_cpu.shape[0], "n": b_cpu.shape[1],
                       "k": a_cpu.shape[1], "repeated_calls": 2, "output_exact": True,
                       "inputs_unchanged": True, "max_abs_error": float((outputs[-1]-reference).abs().max()),
                       "output_sha256": hashlib.sha256(outputs[-1].numpy().tobytes()).hexdigest()}
                if label == "address_toy":
                    row.update({"a": a_cpu.tolist(), "b": b_cpu.tolist(), "c": outputs[-1].tolist(),
                        "c_1_2": float(outputs[-1][1,2]), "a_flat_offsets": [3,4,5],
                        "b_flat_offsets": [2,6,10]})
                    assert row["c_1_2"] == 113.0
                if expected is not None or args.case in ("naive", "wmma"):
                    row.update({"a": a_cpu.tolist(), "b": b_cpu.tolist(),
                                "outputs": [actual.tolist() for actual in outputs],
                                "cpu_reference": reference.tolist()})
                if expected is not None:
                    row["hand_reference"] = expected
                result["checks"].append(row)
                save()

            if args.case == "shared-tail":
                check_shared_tail(check)
            elif args.case == "naive":
                check_naive(check)
            elif args.case == "wmma":
                check_wmma(check)
            else:
                check_address_toy(check)
                for m,n,k in [(1,1,1),(16,16,16),(17,19,23),(32,48,64),(31,33,17),(2,4,65)]:
                    for seed in [1701,1702]:
                        g = torch.Generator().manual_seed(seed)
                        a = torch.randint(-8,9,(m,k),generator=g,dtype=torch.int32).float()/8
                        b = torch.randint(-8,9,(k,n),generator=g,dtype=torch.int32).float()/8
                        check(f"random_seed_{seed}",a,b)
                check("zeros", torch.zeros(17,23), torch.zeros(23,19))
        result["source_sha256_after"] = sha(source)
        assert result["source_sha256_after"] == expected_sha
        result["status"] = "PASS"
        result["phase"] = "finished"
        save()
        summary = {"status": "PASS", "input_sets": len(result["checks"]),
                   "calls": len(result["checks"])*2}
        if args.case == "original":
            summary["address_toy_C_1_2"] = 113.0
        print(json.dumps(summary))
    except Exception as exc:
        result["status"] = "FAIL"
        result["error_type"] = type(exc).__name__
        result["error"] = str(exc)
        save()
        raise


if __name__ == "__main__":
    main()
