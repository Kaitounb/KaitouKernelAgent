"""Engineering bring-up for the trusted cuBLAS GEMM via pinned KernelBench APIs."""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import statistics

import torch
from kernelbench import eval as kb_eval
from kernelbench import timing as kb_timing


SOURCE = Path(__file__).resolve().parents[1] / "kernels/gemm/cublas_sm86.cu"
BASELINE = """
from torch import nn
from torch.utils.cpp_extension import load_inline

extension = load_inline(
    name="nka_cublas_gemm",
    cpp_sources="torch::Tensor gemm_cublas(const torch::Tensor& a, const torch::Tensor& b);",
    cuda_sources={cuda_source},
    functions=["gemm_cublas"],
    extra_ldflags=["-lcublas"],
    extra_include_paths=[{include_dir}],
    verbose=True,
)

class ModelNew(nn.Module):
    def forward(self, a, b):
        return extension.gemm_cublas(a, b)
"""
SHAPES = [(1, 1, 1), (1, 117, 65), (17, 19, 23), (127, 205, 129), (512, 512, 512)]
ATOL = 3e-5
RTOL = 1e-4


class Reference(torch.nn.Module):
    def forward(self, a, b):
        return (a.double() @ b.double()).float()


def inputs(shape, generator=None):
    m, n, k = shape
    a = torch.randn(m, k, generator=generator).mul_(0.25).half()
    b = torch.randn(k, n, generator=generator).mul_(0.25).half()
    return a, b


def check_shapes(model, device):
    rows = []
    stream = torch.cuda.Stream(device=device)
    for seed in (171, 172):
        generator = torch.Generator().manual_seed(seed)
        for shape in SHAPES:
            a_cpu, b_cpu = inputs(shape, generator)
            expected = (a_cpu.double() @ b_cpu.double()).float()
            with torch.cuda.stream(stream):
                a, b = a_cpu.to(device), b_cpu.to(device)
                actual = model(a, b)
                stream.synchronize()
                actual_cpu = actual.cpu()
                torch.testing.assert_close(a.cpu(), a_cpu, rtol=0, atol=0)
                torch.testing.assert_close(b.cpu(), b_cpu, rtol=0, atol=0)
            torch.testing.assert_close(actual_cpu, expected, rtol=RTOL, atol=ATOL)
            delta = (actual_cpu - expected).double()
            rows.append({
                "shape_mnk": list(shape),
                "seed": seed,
                "max_abs_error": delta.abs().max().item(),
                "normalized_rms_error": (delta.norm() / expected.double().norm().clamp_min(1e-12)).item(),
                "inputs_unchanged": True,
                "output_dtype": str(actual.dtype),
                "stream": "non-default",
            })
    return rows


def run(output):
    device = torch.device("cuda:0")
    torch.set_num_threads(1)
    source = SOURCE.read_bytes()
    context = {}
    candidate = kb_eval.load_custom_model(BASELINE.format(cuda_source=repr(source.decode("utf-8")), include_dir=repr(str(SOURCE.parent))), context)
    if candidate is None:
        raise RuntimeError("KernelBench did not load ModelNew")
    model = candidate().eval()
    result = kb_eval.run_and_check_correctness(
        Reference(), model, lambda: inputs((17, 19, 23)),
        metadata={}, num_correct_trials=3, device=device, backend="cuda",
        precision=torch.float16, seed=1701,
    )
    if not result.compiled or not result.correctness:
        raise RuntimeError(f"KernelBench correctness failed: {result.model_dump_json()}")

    with torch.no_grad():
        checks = check_shapes(model, device)
        shape = SHAPES[-1]
        samples = kb_timing.time_execution_with_cuda_event(
            model, [x.to(device) for x in inputs(shape, torch.Generator().manual_seed(173))],
            num_warmup=4, num_trials=12, discard_first=0, verbose=False, device=device,
        )
    if len(samples) != 12 or any(not math.isfinite(value) or value <= 0 for value in samples):
        raise RuntimeError(f"Invalid CUDA Event data: {samples}")

    binary = Path(context["extension"].__file__)
    report = {
        "status": "PASS",
        "scope": "Trusted cuBLAS engineering bring-up",
        "environment": {
            "device": torch.cuda.get_device_name(device),
            "capability": list(torch.cuda.get_device_capability(device)),
            "torch": torch.__version__,
            "torch_cuda": torch.version.cuda,
            "uid": os.getuid(),
        },
        "contract": {"input": "FP16", "accumulation": "FP32", "output": "FP32", "layout": "row-major contiguous"},
        "source_sha256": hashlib.sha256(source).hexdigest(),
        "kernelbench_eval_sha256": hashlib.sha256(Path(kb_eval.__file__).read_bytes()).hexdigest(),
        "kernelbench_timing_sha256": hashlib.sha256(Path(kb_timing.__file__).read_bytes()).hexdigest(),
        "binary": {"path": str(binary), "sha256": hashlib.sha256(binary.read_bytes()).hexdigest()},
        "kernelbench_correctness": result.model_dump(),
        "cpu_reference": {"atol": ATOL, "rtol": RTOL, "cases": checks},
        "timing": {"method": "KernelBench unpaired CUDA Event", "shape_mnk": list(shape),
                   "samples_ms": samples, "median_ms": statistics.median(samples)},
        "formal_performance_claim": False,
    }
    with output.open("x", encoding="utf-8") as file:
        json.dump(report, file, indent=2, allow_nan=False)
        file.write("\n")
    print(f"PASS: wrote {output}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True, help="Result JSON at a new path")
    args = parser.parse_args()
    if args.output.exists():
        parser.error("--output already exists")
    run(args.output)


if __name__ == "__main__":
    main()
