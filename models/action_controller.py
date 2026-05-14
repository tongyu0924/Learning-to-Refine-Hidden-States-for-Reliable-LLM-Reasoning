import torch
import torch.nn as nn
import torch.nn.functional as F


class ActionController(nn.Module):
    """action controller pi_a(a_t | s_t). Outputs gamma (scale), beta (shift), v (direction)."""

    def __init__(self, state_dim: int, action_dim: int = 64, max_scale: float = 1.2):
        super().__init__()
        self.max_scale = max_scale
        mid = state_dim * 2
        self.shared_enc = nn.Sequential(
            nn.Linear(state_dim, mid), nn.GELU(),
            nn.Linear(mid, state_dim), nn.LayerNorm(state_dim),
        )
        self.gamma_mlp = nn.Linear(state_dim, action_dim * 2)
        self.beta_mlp = nn.Linear(state_dim, action_dim * 2)
        self.dir_mlp = nn.Linear(state_dim, action_dim * 2)

    def _sample(self, mlp_out: torch.Tensor, normalize: bool = False):
        mean, log_std = mlp_out.chunk(2, dim=-1)
        log_std = log_std.clamp(-3.0, 1.0)
        dist = torch.distributions.Normal(mean, log_std.exp())
        raw = dist.rsample()
        vec = F.normalize(raw, dim=-1) if normalize else torch.tanh(raw) * self.max_scale
        return vec, dist.log_prob(raw).sum(-1), dist.entropy().sum(-1)

    def forward(self, st: torch.Tensor):
        feat = self.shared_enc(st.float())
        gamma, g_logp, g_entr = self._sample(self.gamma_mlp(feat))
        beta, b_logp, b_entr = self._sample(self.beta_mlp(feat))
        vt, v_logp, v_entr = self._sample(self.dir_mlp(feat), normalize=True)
        return gamma, beta, vt, g_logp + b_logp + v_logp, g_entr + b_entr + v_entr
