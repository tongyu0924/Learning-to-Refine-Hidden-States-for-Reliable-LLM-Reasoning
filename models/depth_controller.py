import torch
import torch.nn as nn
import torch.nn.functional as F


class DepthController(nn.Module):
    """Section 3.4: depth controller pi_d(d | s0). Samples refinement depth from {0, ..., max_depth}."""

    def __init__(self, state_dim: int, max_depth: int = 4):
        super().__init__()
        self.max_depth = max_depth
        self.mlp = nn.Sequential(
            nn.Linear(state_dim, state_dim), nn.GELU(),
            nn.Linear(state_dim, state_dim // 2), nn.GELU(),
            nn.Linear(state_dim // 2, max_depth + 1),
        )

    def forward(self, s0: torch.Tensor):
        logits = self.mlp(s0.float())
        probs = F.softmax(logits, dim=-1)
        dist = torch.distributions.Categorical(probs)
        depth = dist.sample()
        return depth, dist.log_prob(depth), probs, dist.entropy()
