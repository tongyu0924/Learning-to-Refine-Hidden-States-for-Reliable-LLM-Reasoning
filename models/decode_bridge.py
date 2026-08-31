import torch
import torch.nn as nn


class DecodeBridge(nn.Module):
    """fdecode(s_d, h0) = (1 - gate) * h0 + gate * proj(s_d)"""

    def __init__(self, state_dim: int, hidden_size: int):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(state_dim, hidden_size), nn.GELU(),
            nn.Linear(hidden_size, hidden_size),
        )
        self.gate_net = nn.Sequential(
            nn.Linear(state_dim, hidden_size),
            nn.Sigmoid(),
        )
        self.layernorm = nn.LayerNorm(hidden_size)

    def forward(self, st: torch.Tensor, h0: torch.Tensor) -> torch.Tensor:
        st_proj = self.proj(st.float())
        gate = self.gate_net(st.float())
        h_decode = (1.0 - gate) * h0.float() + gate * st_proj.unsqueeze(0)
        return self.layernorm(h_decode).to(h0.dtype)
