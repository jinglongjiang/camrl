import torch
import torch.nn as nn
from mamba_ssm import Mamba

class ValueNet(nn.Module):
    """
    Mamba Value Network: 多层Mamba block + 前后可加embedding/projection层，适配RL输入输出
    """
    def __init__(self, 
                 obs_dim=16,           # 状态输入维度
                 hidden_dim=64,        # Mamba主隐层维度
                 n_blocks=4,           # Mamba block层数
                 state_dim=16,         # d_state参数
                 conv_dim=4,           # d_conv参数
                 expand=2,             # expand参数
                 use_proj=True):       # 是否加投影
        super().__init__()
        self.use_proj = use_proj

        # 前置线性投影（可选，适配任意输入维度）
        if self.use_proj:
            self.proj = nn.Linear(obs_dim, hidden_dim)
        else:
            assert obs_dim == hidden_dim

        # Mamba Block 堆叠
        self.blocks = nn.ModuleList([
            Mamba(
                d_model=hidden_dim,
                d_state=state_dim,
                d_conv=conv_dim,
                expand=expand
            )
            for _ in range(n_blocks)
        ])

        # 输出Head，可是单价值或多输出
        self.head = nn.Linear(hidden_dim, 1)

    def forward(self, x):  # x: [B, T, obs_dim] or [B, obs_dim] (自动适配单步/序列)
        # 自动补batch,seq维度
        if x.dim() == 2:   # [B, obs_dim] → [B, 1, obs_dim]
            x = x.unsqueeze(1)
        if self.use_proj:
            x = self.proj(x)

        for block in self.blocks:
            x = block(x)  # [B, T, hidden_dim]
        # 默认取最后一步输出
        v_t = self.head(x[:, -1, :])  # [B,1]
        return v_t.squeeze(-1)        # [B]（常见RL接口）

