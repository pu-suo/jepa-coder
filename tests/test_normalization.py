import torch
from models.normalization import HybridNorm

norm = HybridNorm(768)
norm.l2_enabled = True

x = torch.randn(10, 768)  # sequence of 10 vectors
out = norm(x)

# Every vector should have unit norm
norms = out.norm(dim=-1)
assert torch.allclose(norms, torch.ones(10), atol=1e-4), f"Norms: {norms}"
print("normalization.py PASSED")
