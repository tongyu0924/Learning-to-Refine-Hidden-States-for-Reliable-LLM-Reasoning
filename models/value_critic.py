import torch
import torch.nn as nn


class ValueCritic(nn.Module):
    """Section 3.4: baseline value function V(s0), trained with MSE loss."""

    def __init__(self, state_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, state_dim), nn.GELU(),
            nn.Linear(state_dim, state_dim // 2), nn.GELU(),
            nn.Linear(state_dim // 2, 1),
        )

    def forward(self, s0: torch.Tensor) -> torch.Tensor:
        return self.net(s0.float()).squeeze(-1)
