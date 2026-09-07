import torch
import torch.nn as nn
from mamba_ssm import Mamba

class ValueNet(nn.Module):
    """
    Mamba Value Network: Mamba block + embedding/projection，RL
    """
    def __init__(self, 
                 obs_dim=16,
                 hidden_dim=64,
                 n_blocks=4,
                 state_dim=16,
                 conv_dim=4,
                 expand=2,
                 use_proj=True):
        super().__init__()
        self.use_proj = use_proj

        if self.use_proj:
            self.proj = nn.Linear(obs_dim, hidden_dim)
        else:
            assert obs_dim == hidden_dim

        self.blocks = nn.ModuleList([
            Mamba(
                d_model=hidden_dim,
                d_state=state_dim,
                d_conv=conv_dim,
                expand=expand
            )
            for _ in range(n_blocks)
        ])

        self.head = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        if x.dim() == 2:   # [B, obs_dim] → [B, 1, obs_dim]
            x = x.unsqueeze(1)
        if self.use_proj:
            x = self.proj(x)

        for block in self.blocks:
            x = block(x)  # [B, T, hidden_dim]
        v_t = self.head(x[:, -1, :])  # [B,1]
        return v_t.squeeze(-1)

