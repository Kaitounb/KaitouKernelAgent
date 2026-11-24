"""Trusted P1-02 GEMM contract checks; never accepts arbitrary candidate source."""
import argparse
import hashlib
import json
import math
from pathlib import Path
import traceback

import torch
from kernelbench import eval as kb_eval

ROOT = Path(__file__).resolve().parents[1]
CONTRACT = ROOT / 'workloads/gemm_contract_v1.json'
DEVELOPMENT = ROOT / 'workloads/gemm_development_v1.json'


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_trusted(warps):
    folder = ROOT / 'kernels/gemm'
    source = (folder / 'cublas_sm86.cu').read_text() + '\n' + (folder / 'wmma_sm86.cu').read_text()
    wrapper = f'''
from torch import nn
from torch.utils.cpp_extension import load_inline
extension = load_inline(name="nka_gemm_v1", cpp_sources="""
torch::Tensor gemm_cublas(const torch::Tensor& a, const torch::Tensor& b);
torch::Tensor gemm_template(const torch::Tensor& a, const torch::Tensor& b, int64_t warps);
""", cuda_sources={source!r}, functions=["gemm_cublas", "gemm_template"],
extra_include_paths=[{str(folder)!r}], extra_ldflags=["-lcublas"],
extra_cuda_cflags=["-lineinfo"], verbose=True)
class ModelNew(nn.Module):
    def forward(self, a, b):
        return extension.gemm_template(a, b, {warps})
'''
    context = {}
    cls = kb_eval.load_custom_model(wrapper, context)
    if cls is None:
        raise RuntimeError('KernelBench failed to load trusted template')
    return context['extension'], cls().eval()


def make_inputs(shape, seed, distribution='random'):
    m, n, k = shape
    g = torch.Generator().manual_seed(seed)
    values = []
    for dims in ((m, k), (k, n)):
        if distribution == 'random':
            x = (torch.randn(dims, generator=g) * .25).clamp_(-4, 4)
        elif distribution == 'zero':
            x = torch.zeros(dims)
        elif distribution == 'signed4':
            x = (torch.arange(math.prod(dims)).reshape(dims) % 2 * 2 - 1).float() * 4
        else:
            raise ValueError(distribution)
        values.append(x.half())
    return values


def guarded(x, device):
    # Sentinel storage on both sides, including an intentionally unaligned pointer.
    raw = torch.full((x.numel() + 34,), 7.0, dtype=x.dtype, device=device)
    view = raw[17:-17].view(x.shape)
    view.copy_(x)
    return raw, view


def check_case(model, shape, seed, distribution='random', offset=False):
    cpu = make_inputs(shape, seed, distribution)
    expected = (cpu[0].double() @ cpu[1].double()).float()
    stream = torch.cuda.Stream()
    guards = []
    with torch.no_grad(), torch.cuda.stream(stream):
        # Queued producers make honoring the current stream observable.
        torch.cuda._sleep(200000)
        if offset:
            pairs = [guarded(x, 'cuda:0') for x in cpu]
            guards = [p[0] for p in pairs]
            a, b = [p[1] for p in pairs]
        else:
            a, b = [x.to('cuda:0') for x in cpu]
        actual = model(a, b)
        if not isinstance(actual, torch.Tensor):
            raise AssertionError('output is not Tensor')
        if actual.dtype != torch.float32 or actual.device != a.device or list(actual.shape) != list(expected.shape) or not actual.is_contiguous():
            raise AssertionError('output contract')
        if torch._C._is_alias_of(actual, a) or torch._C._is_alias_of(actual, b):
            raise AssertionError('output aliases input storage')
        consumed = actual.clone()  # consumer must see completed candidate on this stream
        out = consumed.cpu()
        snapshots = [a.cpu(), b.cpu()]
        guard_copies = [x.cpu() for x in guards]
    stream.synchronize()
    for snapshot, original in zip(snapshots, cpu):
        torch.testing.assert_close(snapshot, original, atol=0, rtol=0, equal_nan=False)
    for raw in guard_copies:
        if not bool((raw[:17] == 7).all() and (raw[-17:] == 7).all()):
            raise AssertionError('input guard overwritten')
    if not bool(torch.isfinite(out).all()):
        raise AssertionError('nonfinite output')
    torch.testing.assert_close(out, expected, atol=3e-5, rtol=1e-4, equal_nan=False)
    delta = (out - expected).double()
    return {'shape_mnk': list(shape), 'seed': seed, 'distribution': distribution,
            'offset_guard': offset, 'max_abs_error': delta.abs().max().item(),
            'max_stable_relative_error': (delta.abs() / expected.double().abs().clamp_min(1e-6)).max().item(),
            'normalized_rms_error': (delta.norm() / expected.double().norm().clamp_min(1e-12)).item(),
            'input_immutable': True, 'stream': 'explicit producer/candidate/consumer'}


class Reference(torch.nn.Module):
    def forward(self, a, b):
        return (a.double() @ b.double()).float()


class Fault:
    def __init__(self, name, baseline):
        self.name, self.baseline, self.cached = name, baseline, None
        self.side = torch.cuda.Stream() if name == 'wrong-stream' else None

    def __call__(self, a, b):
        if self.name == 'host-error':
            raise RuntimeError('injected host failure')
        if self.name == 'pollution':
            a.zero_()
        out = self.baseline(a, b)
        if self.name == 'constant':
            return torch.zeros_like(out)
        if self.name == 'tail':
            out[-1, :] = 0
        if self.name == 'cache':
            if self.cached is None:
                self.cached = out.clone()
            return self.cached
        if self.name == 'dtype':
            return out.half()
        if self.name == 'shape':
            return out.flatten()
        if self.name == 'nan':
            out.fill_(float('nan'))
        if self.name == 'wrong-stream':
            delayed = torch.zeros_like(out)
            # Intentionally omit an output dependency, while preserving input readiness.
            self.side.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(self.side):
                torch.cuda._sleep(20000000)
                delayed.copy_(out)
            return delayed
        return out


def run(args, report):
    torch.set_num_threads(1)
    extension, template = load_trusted(args.warps)
    baseline = extension.gemm_cublas
    report.update(environment={'device': torch.cuda.get_device_name(0), 'capability':list(torch.cuda.get_device_capability(0)),
                               'torch':torch.__version__, 'cuda':torch.version.cuda},
                  binary={'path':extension.__file__, 'sha256':sha(Path(extension.__file__))})
    report['sources'] = {str(p.relative_to(ROOT)):sha(p) for p in [CONTRACT, DEVELOPMENT, Path(__file__),
                         ROOT/'kernels/gemm/gemm_contract.cuh', ROOT/'kernels/gemm/cublas_sm86.cu', ROOT/'kernels/gemm/wmma_sm86.cu', Path(kb_eval.__file__)] if p.is_relative_to(ROOT)}
    report['kernelbench_eval_sha256'] = sha(Path(kb_eval.__file__))
    kb = kb_eval.run_and_check_correctness(Reference(), template,
            lambda: make_inputs((32, 48, 64), 1701), metadata={}, num_correct_trials=3,
            device=torch.device('cuda:0'), backend='cuda', precision=torch.float16, seed=1701)
    report['kernelbench_correctness'] = kb.model_dump()
    if not kb.compiled or not kb.correctness:
        raise AssertionError('KernelBench correctness')
    development = json.loads(DEVELOPMENT.read_text())['workloads']
    boundaries = [((1,1,1),'random',False), ((17,19,23),'random',True), ((48,32,80),'random',False),
                  ((32,48,64),'random',True), ((16,16,16),'zero',False), ((32,48,64),'signed4',False)]
    report['checks'] = []
    for name, model in [('cublas',baseline), ('template',template)]:
        cases = [] if args.sanitizer_smoke else [(x['shape_mnk'],'random',False) for x in development]
        for shape, distribution, offset in cases + boundaries:
            for seed in ((171,) if args.sanitizer_smoke else (171,172)):
                row = check_case(model, shape, seed, distribution, offset)
                row['implementation'] = name
                report['checks'].append(row)
    report['invalid_inputs'] = []
    for name, model in [('cublas',baseline), ('template',template)]:
        a, b = [x.cuda() for x in make_inputs((16,16,16),171)]
        invalid = [('dtype',a.float(),b),('rank',a.flatten(),b),('stride',a.t(),b),
                   ('zero',a[:0],b),('cpu',a.cpu(),b),('K',a,b[:8]),
                   ('too-large',torch.empty((4097,1),device='cuda',dtype=torch.float16),b[:1])]
        for case, x, y in invalid:
            try:
                model(x, y)
            except RuntimeError as error:
                report['invalid_inputs'].append({'implementation':name,'case':case,'rejected':True,'reason':str(error).splitlines()[0]})
            else:
                raise AssertionError(f'accepted invalid input: {name}/{case}')
    report['faults'] = []
    if not args.sanitizer_smoke:
        for fault_name in ['constant','pollution','tail','cache','dtype','shape','nan','wrong-stream','host-error']:
            fault = Fault(fault_name, baseline)
            try:
                for seed in (171,172):
                    check_case(fault, (17,19,23), seed, offset=True)
            except (AssertionError,RuntimeError) as error:
                report['faults'].append({'case':fault_name,'rejected':True,'reason':str(error)[:800],
                                         'incumbent':'trusted cuBLAS, unchanged; no promotion performed'})
            else:
                raise AssertionError(f'fault escaped: {fault_name}')
            finally:
                torch.cuda.synchronize()
        report['recovery'] = check_case(baseline,(32,48,64),173)
    report['status'] = 'PASS'


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--output',type=Path,required=True)
    ap.add_argument('--warps',type=int,choices=[1,2,4,8],default=4)
    ap.add_argument('--sanitizer-smoke',action='store_true')
    args = ap.parse_args()
    if args.output.exists():
        ap.error('Use a new result path')
    report = {'status':'FAIL','warps':args.warps,'scope':'trusted development and boundary regression; no heldout execution',
              'formal_performance_claim':False,'checks':[]}
    try:
        run(args,report)
    except Exception:
        report['error'] = traceback.format_exc()
        raise
    finally:
        args.output.write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')


if __name__ == '__main__':
    main()
