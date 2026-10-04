"""Mamba-1 equations with frame-local states and standard dt initialization.

Reference backend is portable and slower than CUDA mamba-ssm 2.2.2.
Parameter names match that implementation for backend-independent checkpoints.
"""
import math
import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint


class ReferenceMamba(nn.Module):
    def __init__(self, d_model, d_state=16, d_conv=4, expand=2, dt_rank=16):
        super().__init__()
        self.d_inner, self.d_state, self.dt_rank = d_model * expand, d_state, dt_rank
        # Input and output projections include bias.
        self.in_proj = nn.Linear(d_model, 2 * self.d_inner, bias=True)
        self.conv1d = nn.Conv1d(self.d_inner, self.d_inner, d_conv, groups=self.d_inner, padding=d_conv - 1)
        self.x_proj = nn.Linear(self.d_inner, dt_rank + 2 * d_state, bias=False)
        self.dt_proj = nn.Linear(dt_rank, self.d_inner)
        nn.init.uniform_(self.dt_proj.weight, -dt_rank**-0.5, dt_rank**-0.5)
        dt = torch.exp(torch.rand(self.d_inner) * math.log(100) + math.log(0.001))
        with torch.no_grad():
            self.dt_proj.bias.copy_(dt + torch.log(-torch.expm1(-dt)))
        self.A_log = nn.Parameter(torch.arange(1, d_state + 1).float().log().repeat(self.d_inner, 1))
        self.D = nn.Parameter(torch.ones(self.d_inner))
        self.out_proj = nn.Linear(self.d_inner, d_model, bias=True)
        self.A_log._no_weight_decay = True
        self.D._no_weight_decay = True

    def forward(self, tokens):
        u, z = self.in_proj(tokens).chunk(2, -1)
        u = F.silu(self.conv1d(u.transpose(1, 2))[..., :tokens.shape[1]].transpose(1, 2))
        dt, b, c = self.x_proj(u).split([self.dt_rank, self.d_state, self.d_state], -1)
        # Scan in float32 for low-precision inputs; preserve double for gradcheck.
        scan_dtype = torch.float64 if tokens.dtype == torch.float64 else torch.float32
        dt = F.softplus(self.dt_proj(dt)).to(scan_dtype)
        u_scan, b, c = u.to(scan_dtype), b.to(scan_dtype), c.to(scan_dtype)
        a = -self.A_log.to(scan_dtype).exp()
        state = torch.zeros(tokens.shape[0], self.d_inner, self.d_state, device=tokens.device, dtype=scan_dtype)
        outputs = []
        for t in range(tokens.shape[1]):
            delta = dt[:, t, :, None]
            state = (delta * a).exp() * state + delta * b[:, t, None, :] * u_scan[:, t, :, None]
            outputs.append((state * c[:, t, None, :]).sum(-1) + self.D.to(scan_dtype) * u_scan[:, t])
        y = torch.stack(outputs, 1).to(tokens.dtype) * F.silu(z)
        return self.out_proj(y)


class Block(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.norm = nn.LayerNorm(cfg.d_model)
        kwargs = dict(d_model=cfg.d_model, d_state=cfg.d_state, d_conv=cfg.d_conv, expand=cfg.expand, dt_rank=cfg.dt_rank)
        if cfg.backend == "reference":
            self.mixer = ReferenceMamba(**kwargs)
        else:
            try:
                from mamba_ssm import Mamba
            except ImportError as exc:
                raise RuntimeError("Install mamba-ssm==2.2.2 on Linux/WSL2 with CUDA, or choose reference") from exc
            self.mixer = Mamba(**kwargs, bias=True)

    def forward(self, x):
        return x + self.mixer(self.norm(x))  # No inference_params / cross-frame caches.


class Stack(nn.Module):
    def __init__(self, cfg, depth):
        super().__init__()
        self.blocks = nn.ModuleList([Block(cfg) for _ in range(depth)])
        self.checkpoint_branches = cfg.checkpoint_branches

    def forward(self, x):
        for block in self.blocks:
            x = checkpoint(block, x, use_reentrant=False) if self.training and self.checkpoint_branches and torch.is_grad_enabled() else block(x)
        return x


class _BoundedProjection(torch.autograd.Function):
    @staticmethod
    def forward(ctx, value, lower, upper):
        n = value.shape[-1]
        # Euclidean projection g_i=clamp(v_i-lambda), sum g_i=n.
        v = value.double()
        lo = (v - upper).amin(-1, keepdim=True)
        hi = (v - lower).amax(-1, keepdim=True)
        for _ in range(64):
            mid = (lo + hi) / 2
            s = (v - mid).clamp(lower, upper).sum(-1, keepdim=True)
            lo, hi = torch.where(s > n, mid, lo), torch.where(s > n, hi, mid)
        out = (v - (lo + hi) / 2).clamp(lower, upper).to(value.dtype)
        ctx.save_for_backward((out > lower) & (out < upper))
        return out

    @staticmethod
    def backward(ctx, gradient):
        free, = ctx.saved_tensors
        g = gradient * free
        count = free.sum(-1, keepdim=True).clamp_min(1)
        return (g - g.sum(-1, keepdim=True) / count) * free, None, None


def bounded_gate(logits, cfg):
    if not 0 < cfg.gate_min <= 1 <= cfg.gate_max:
        raise ValueError("infeasible gate constraint")
    alpha = (logits / cfg.gate_temperature).softmax(-1)
    v = 1 + cfg.gate_eta * (logits.shape[-1] * alpha - 1)
    return _BoundedProjection.apply(v, cfg.gate_min, cfg.gate_max)


def sincos2d(height, width, dim):
    if dim % 4:
        raise ValueError("sine-cosine dimension must be a multiple of four")
    y, x = torch.meshgrid(torch.arange(height), torch.arange(width), indexing="ij")
    omega = 1 / (10000 ** (torch.arange(dim // 4).float() / (dim // 4)))
    x, y = x.flatten()[:, None] * omega, y.flatten()[:, None] * omega
    return torch.cat([x.sin(), x.cos(), y.sin(), y.cos()], -1)[None]
