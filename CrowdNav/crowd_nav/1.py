import torch

ckpt = torch.load('data/output/rl_model.pth', map_location='cpu')
print("=== Weight shape inspection ===")
for k, v in ckpt.items():
    print(f"{k:40s} {tuple(v.shape)}")

