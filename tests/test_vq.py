import torch
import torch.nn.functional as F
from models.vq import VectorQuantizer

def test_exact_codebook_entry():
    vq = VectorQuantizer(512, 768)
    
    for _ in range(100):
        z = F.normalize(torch.randn(768), dim=-1)
        quantized, idx, loss = vq(z)
        
        # quantized must exactly equal the codebook row
        expected = vq.embedding.weight[idx]
        assert torch.allclose(quantized, expected, atol=1e-6), \
            f"Output doesn't match codebook entry {idx}"
    
    print("PASS: All outputs are exact codebook entries")

def test_gradient_flow():
    vq = VectorQuantizer(16, 64)
    z = F.normalize(torch.randn(64), dim=-1)
    z.requires_grad_(True)
    
    quantized, idx, loss = vq(z)
    
    # Backward through the quantized output (simulating downstream use)
    fake_downstream_loss = quantized.sum()
    fake_downstream_loss.backward()
    
    assert z.grad is not None, "No gradient on z"
    assert z.grad.norm() > 0, "Zero gradient on z"
    
    # Also check commitment loss gradient
    z2 = F.normalize(torch.randn(64), dim=-1)
    z2.requires_grad_(True)
    _, _, loss2 = vq(z2)
    loss2.backward()
    
    assert z2.grad is not None, "No gradient from commitment loss"
    
    print("PASS: Gradients flow through VQ via straight-through estimator")

def test_nearest_neighbor():
    vq = VectorQuantizer(512, 768)
    
    for _ in range(100):
        z = F.normalize(torch.randn(768), dim=-1)
        _, idx, _ = vq(z)
        
        # Verify this is actually the nearest entry
        all_distances = torch.cdist(z.unsqueeze(0), vq.embedding.weight).squeeze()
        true_nearest = all_distances.argmin()
        
        assert idx == true_nearest, \
            f"VQ returned index {idx} but nearest is {true_nearest}"
    
    print("PASS: VQ correctly finds nearest codebook entry")

def test_codebook_normalization():
    vq = VectorQuantizer(512, 768)
    
    # After initialization
    norms = vq.embedding.weight.norm(dim=-1)
    assert torch.allclose(norms, torch.ones(512), atol=1e-4), \
        f"Codebook not normalized after init: norms range [{norms.min():.4f}, {norms.max():.4f}]"
    
    # After EMA updates
    for _ in range(100):
        z = F.normalize(torch.randn(768), dim=-1)
        _, idx, _ = vq(z)
        vq.update_codebook_ema(z, idx)
    
    norms = vq.embedding.weight.norm(dim=-1)
    assert torch.allclose(norms, torch.ones(512), atol=1e-4), \
        f"Codebook not normalized after EMA: norms range [{norms.min():.4f}, {norms.max():.4f}]"
    
    print("PASS: Codebook entries stay on unit hypersphere")

def test_utilization():
    vq = VectorQuantizer(512, 768)
    
    # Initially zero utilization
    assert vq.utilization() == 0.0
    
    # After many random inputs, utilization should grow
    for _ in range(10000):
        z = F.normalize(torch.randn(768), dim=-1)
        vq(z)
    
    util = vq.utilization()
    assert util > 0.5, f"Low utilization after 10K random inputs: {util:.2%}"
    
    print(f"PASS: Utilization = {util:.2%} after 10K random inputs")

if __name__ == "__main__":
    test_exact_codebook_entry()
    test_gradient_flow()
    test_nearest_neighbor()
    test_codebook_normalization()
    test_utilization()
    print("\nAll VQ tests PASSED")