import torch
import torch.nn as nn
import torch.nn.functional as F


class HiddenRefiner(nn.Module):
    """
    Section 3.3:
      refine:       h_{t+1} = frefine(h_t, s_t, gamma, beta, v)
      update_state: s_{t+1} = g(s_t, h_{t+1})
    """

    def __init__(self, hidden_size: int, state_dim: int, action_dim: int = 64, dir_scale: float = 0.05):
        super().__init__()
        self.hidden_size = hidden_size
        self.dir_scale = dir_scale
        self.layernorm = nn.LayerNorm(hidden_size)
        self.gamma_proj = nn.Linear(action_dim, hidden_size, bias=False)
        self.beta_proj = nn.Linear(action_dim, hidden_size, bias=True)
        self.dir_proj = nn.Linear(action_dim, hidden_size, bias=False)
        self.h_agg_proj = nn.Linear(hidden_size, state_dim, bias=False)
        self.state_update = nn.Sequential(
            nn.Linear(state_dim * 2, state_dim * 2), nn.GELU(),
            nn.Linear(state_dim * 2, state_dim),
            nn.LayerNorm(state_dim),
        )

    def refine(
        self,
        ht: torch.Tensor,
        st: torch.Tensor,
        gamma: torch.Tensor,
        beta: torch.Tensor,
        vt: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        orig_dtype = ht.dtype
        ht_f = ht.float()
        gamma_h = self.gamma_proj(gamma.float())
        beta_h = self.beta_proj(beta.float())
        v_h = F.normalize(self.dir_proj(vt.float()), dim=-1)

        h_norm = self.layernorm(ht_f)
        h_film = h_norm * (1.0 + gamma_h.unsqueeze(0)) + beta_h.unsqueeze(0)
        h_next = h_film + self.dir_scale * v_h.unsqueeze(0).expand_as(h_film)
        return h_next.to(orig_dtype)

    def update_state(
        self,
        st: torch.Tensor,
        h_next: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        valid = attention_mask.bool()
        h_valid = h_next[valid].float()
        h_agg = h_valid.mean(0) if len(h_valid) > 0 else torch.zeros(self.hidden_size, device=h_next.device)
        h_proj = self.h_agg_proj(h_agg)
        state_in = torch.cat([st.float(), h_proj], dim=-1)
        return self.state_update(state_in) + st.float()
