import torch
import torch.nn as nn


class InitialStateExtractor(nn.Module):
    """s0 = fextract(h0). Attention-pooling over h0 followed by MLP projection."""

    def __init__(self, hidden_size: int, state_dim: int):
        super().__init__()
        self.attn_score = nn.Linear(hidden_size, 1, bias=False)
        self.proj = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2), nn.GELU(),
            nn.Linear(hidden_size // 2, state_dim),
            nn.LayerNorm(state_dim),
        )

    def forward(self, h0: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        h0_f = h0.float()
        mask = attention_mask.bool()
        scores = self.attn_score(h0_f).squeeze(-1)
        scores = scores.masked_fill(~mask, float("-inf"))
        weights = torch.softmax(scores, dim=0)
        agg = (h0_f * weights.unsqueeze(-1)).sum(0)
        return self.proj(agg)
