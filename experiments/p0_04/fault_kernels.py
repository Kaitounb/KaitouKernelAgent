"""Intentional defects, not optimization candidates for production use."""
import triton
import triton.language as tl


@triton.jit
def faulty(X, Y, N: tl.constexpr, MODE: tl.constexpr, BLOCK: tl.constexpr):
    i = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    x = tl.load(X + i, i < N, other=0)
    if MODE == 0:
        tl.store(Y + i, 0.0, i < N)
    else:
        tl.store(Y + i, x + 1.0, i < N)
        # Only input element zero is corrupted; output uses the old value.
        tl.store(X + i, x + 16.0, i == 0)


def launch_fault(x, y, n, mode):
    faulty[(triton.cdiv(n, 256),)](x, y, n, mode, 256, num_warps=4)
