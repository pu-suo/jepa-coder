import torch
import torch.nn as nn
from models.ema_encoder import EMAEncoder

ema = EMAEncoder(vocab_size=1000, dim=768)
tokens = torch.randint(0, 1000, (20,))  # 20 random tokens

target = ema.encode_block(tokens)
assert target.shape == (768,), f"Wrong shape: {target.shape}"
assert abs(target.norm().item() - 1.0) < 1e-4, f"Not unit norm: {target.norm()}"

# Test EMA update
source_emb = nn.Embedding(1000, 768)
initial_dist = (ema.embedding.weight - source_emb.weight).norm().item()
for _ in range(50):
    ema.update(source_emb)
final_dist = (ema.embedding.weight - source_emb.weight).norm().item()
assert final_dist < initial_dist, "EMA didn't converge"
print("ema_encoder.py PASSED")
