import torch.nn as nn

class StateEncoder(nn.Module):
    """
    Crowd State Encoder: 将长度为 T+1 的状态序列编码成隐藏向量。
    """
    def __init__(self, input_dim, hidden_dim):
        super().__init__()
        self.gru = nn.GRU(input_dim, hidden_dim, batch_first=True)

    def forward(self, seq):  # seq: [B, T+1, input_dim]
        _, h = self.gru(seq)
        return h.squeeze(0)  # [B, hidden_dim]

