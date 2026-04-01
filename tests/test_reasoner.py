import torch
import torch.nn.functional as F
from models.reasoner import Reasoner

reasoner = Reasoner(
    vocab_size=49152, 
    dim=768, 
    n_layers=16, 
    n_heads=12, 
    ffn_dim=3072
)
reasoner.hybrid_norm.l2_enabled = True
reasoner.eval()

h = F.normalize(torch.randn(768), dim=-1)
for step in range(10):
    r = reasoner.step(h)
    assert abs(r.norm().item() - 1.0) < 1e-4, f"Step {step}: norm = {r.norm().item()}"
    h = r
print("Norm preservation PASSED over 10 steps")