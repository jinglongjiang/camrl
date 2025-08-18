import torch.nn as nn
from mamba_ssm import Mamba

class ValueNet(nn.Module):
    """
    Mamba Value Network: 堆叠多层 Mamba block，输出状态价值。
    """
    def __init__(self, hidden_dim, n_blocks=4, state_dim=16, conv_dim=4, expand=2):
        super().__init__()
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

    def forward(self, x):  # x: [B, T+1, hidden_dim]
        for block in self.blocks:
            x = block(x)
        v_t = self.head(x[:, -1, :])
        return v_t  # [B,1]

