"""Original verified make_inputs/guarded bodies; no reference implementation."""
import math
import torch


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
