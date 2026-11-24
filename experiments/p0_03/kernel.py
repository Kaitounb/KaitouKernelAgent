"""P0-03: one masked FP32 load/add/store; independent of the evaluator."""
import triton
import triton.language as tl


@triton.jit
def add_one(X, Y, N: tl.constexpr, BLOCK: tl.constexpr):
    i = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    x = tl.load(X + i, i < N, other=0)
    tl.store(Y + i, x + 1.0, i < N)


def launch(x, y, n):
    add_one[(triton.cdiv(n, 256),)](x, y, n, 256, num_warps=4)
