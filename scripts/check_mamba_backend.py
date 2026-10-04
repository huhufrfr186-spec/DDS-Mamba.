"""Optional Linux/CUDA reference-vs-mamba-ssm 2.2.2 forward/gradient check."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import torch
from mamba_ssm import Mamba
from dds_mamba.mamba import ReferenceMamba

torch.manual_seed(2024)
kwargs = dict(d_model=32, d_state=16, d_conv=4, expand=2, dt_rank=2)
reference = ReferenceMamba(**kwargs).cuda()
cuda = Mamba(**kwargs, bias=True).cuda()
cuda.load_state_dict(reference.state_dict(), strict=True)
x = torch.randn(1, 258, 32, device="cuda", requires_grad=True)
y = x.detach().clone().requires_grad_(True)
a, b = reference(x), cuda(y)
torch.testing.assert_close(a, b, atol=1e-4, rtol=1e-4)
a.square().sum().backward(); b.square().sum().backward()
torch.testing.assert_close(x.grad, y.grad, atol=2e-4, rtol=2e-4)
for (name, p), (_, q) in zip(reference.named_parameters(), cuda.named_parameters()):
    torch.testing.assert_close(p.grad, q.grad, atol=2e-3, rtol=2e-3, msg=name)
print("Reference and CUDA Mamba forward/gradients agree within stated tolerances")
