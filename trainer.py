import csv
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, AutoModelForCausalLM

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from dataset import GSMHardDataset, collate_fn_pad
from models import (
    InitialStateExtractor, DepthController, ActionController,
    HiddenRefiner, DecodeBridge, ValueCritic,
)

logger = logging.getLogger(__name__)


class HiddenRLTrainer:
    def __init__(
        self,
        model_name: str = "google/gemma-2-2b-it",
        device: str = "auto",
        beta: float = 0.1,
        max_depth: int = 4,
        depth_cost: float = 0.01,
        entropy_bonus: float = 0.01,
        depth_rl_weight: float = 0.15,
        critic_weight: float = 0.2,
        action_dim: int = 64,
        state_dim: int = 256,
        hf_token: Optional[str] = None,
    ):
        self.device = (
            torch.device("cuda" if torch.cuda.is_available() else "cpu")
            if device == "auto" else torch.device(device)
        )
        logger.info(f"Device: {self.device}")

        self.beta = beta
        self.max_depth = max_depth
        self.depth_cost = depth_cost
        self.entropy_bonus = entropy_bonus
        self.depth_rl_weight = depth_rl_weight
        self.critic_weight = critic_weight

        kw = {"trust_remote_code": True, **({"token": hf_token} if hf_token else {})}
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, **kw)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.policy_model = AutoModelForCausalLM.from_pretrained(model_name, **kw).to(self.device)
        self.reference_model = AutoModelForCausalLM.from_pretrained(model_name, **kw).to(self.device)
        self.reference_model.eval()
        for p in self.reference_model.parameters():
            p.requires_grad = False

        H = self.policy_model.config.hidden_size
        self.state_extractor = InitialStateExtractor(H, state_dim).to(self.device)
        self.depth_controller = DepthController(state_dim, max_depth).to(self.device)
        self.action_controller = ActionController(state_dim, action_dim).to(self.device)
        self.refiner = HiddenRefiner(H, state_dim, action_dim).to(self.device)
        self.decode_bridge = DecodeBridge(state_dim, H).to(self.device)
        self.critic = ValueCritic(state_dim).to(self.device)

        self._global_step = 0
        self._hist: Dict[str, list] = {
            k: [] for k in ["step", "total", "lm", "kl", "rl", "critic", "depth", "entropy", "reward"]
        }

    # ── log-likelihood & KL ──────────────────────────────────────

    def _logp_and_kl(
        self,
        policy_logits: torch.Tensor,
        ref_logits: torch.Tensor,
        attention_mask: torch.Tensor,
        labels: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        plp = F.log_softmax(policy_logits, dim=-1)
        rlp = F.log_softmax(ref_logits, dim=-1)
        pp = plp.exp()

        shift_plp = plp[:, :-1, :]
        shift_rlp = rlp[:, :-1, :]
        shift_pp = pp[:, :-1, :]
        shift_lbl = labels[:, 1:]
        shift_mask = (shift_lbl != -100) & (attention_mask[:, 1:] == 1)

        valid_lbl = shift_lbl.clamp(0, shift_plp.size(-1) - 1)
        pg = shift_plp.gather(-1, valid_lbl.unsqueeze(-1)).squeeze(-1)
        pm = torch.where(shift_mask, pg, torch.zeros_like(pg))
        cnt = shift_mask.sum(-1).float() + 1e-8

        seq_logp = pm.sum(-1) / cnt
        kl_tok = (shift_pp * (shift_plp - shift_rlp)).sum(-1)
        kl_tok = torch.where(shift_mask, kl_tok, torch.zeros_like(kl_tok))
        seq_kl = kl_tok.sum(-1) / cnt

        seq_logp = torch.where(torch.isfinite(seq_logp), seq_logp, torch.full_like(seq_logp, -5.0))
        seq_kl = torch.where(torch.isfinite(seq_kl), seq_kl, torch.full_like(seq_kl, 0.1))
        return seq_logp, seq_kl

    def _logp_from_state(
        self,
        st: torch.Tensor,
        h0: torch.Tensor,
        ref_logits: torch.Tensor,
        attention_mask: torch.Tensor,
        labels: torch.Tensor,
    ) -> Tuple[float, float]:
        """log p_theta(y* | x, s_t) via fdecode(s_t, h0). Used for per-step reward computation."""
        h_dec = self.decode_bridge(st, h0)
        logits = self.policy_model.lm_head(h_dec.unsqueeze(0))
        lp, kl = self._logp_and_kl(logits, ref_logits, attention_mask, labels)
        return float(lp.item()), float(kl.item())

    # ── shaped return ──────────────────────────────

    def _shaped_return(
        self,
        logp_seq: List[float],
        final_kl: float,
        total_H: float,
        d: int,
    ) -> float:
        if d == 0:
            return -self.beta * final_kl + self.entropy_bonus * total_H

        global_term = -self.beta * final_kl + self.entropy_bonus * total_H
        total_R = 0.0
        for t in range(d):
            delta_t = logp_seq[t + 1] - logp_seq[t]
            r_t = delta_t - self.depth_cost
            R_t = r_t / (t + 1) + (1.0 / d) * global_term
            total_R += R_t
        return total_R

    # ── forward pass for one sample ──────────────────────────────

    def _compute_one_sample(self, inp: Dict, is_train: bool = True) -> Optional[Dict]:
        input_ids = inp["input_ids"].unsqueeze(0).to(self.device)
        attn_mask = inp["attention_mask"].unsqueeze(0).to(self.device)
        labels = inp["labels"].unsqueeze(0).to(self.device)

        with torch.no_grad():
            ref_logits = self.reference_model(input_ids=input_ids, attention_mask=attn_mask).logits

        policy_out = self.policy_model(input_ids=input_ids, attention_mask=attn_mask, output_hidden_states=True)
        h0 = policy_out.hidden_states[-1][0]  # (L, D) — fixed anchor for fdecode

        s0 = self.state_extractor(h0, attn_mask[0])

        d_sample, depth_logp, _, depth_entropy = self.depth_controller(s0.unsqueeze(0))
        d = max(0, min(int(d_sample.item()), self.max_depth))

        with torch.no_grad():
            logp_s0, _ = self._logp_from_state(s0.detach(), h0.detach(), ref_logits, attn_mask, labels)
        logp_seq = [logp_s0]

        st = s0
        total_action_logp = torch.zeros(1, device=self.device)
        total_entropy = depth_entropy[0]

        ht = h0
        for t in range(d):
            gamma, beta, vt, act_logp, act_entr = self.action_controller(st)
            total_action_logp = total_action_logp + act_logp
            total_entropy = total_entropy + act_entr

            h_next = self.refiner.refine(ht, st, gamma, beta, vt, attn_mask[0])
            st = self.refiner.update_state(st, h_next, attn_mask[0])
            ht = h_next

            with torch.no_grad():
                logp_t, _ = self._logp_from_state(st.detach(), h0.detach(), ref_logits, attn_mask, labels)
            logp_seq.append(logp_t)

        h_final = self.decode_bridge(st, h0)
        final_logits = self.policy_model.lm_head(h_final.unsqueeze(0))
        lm_logp, lm_kl = self._logp_and_kl(final_logits, ref_logits, attn_mask, labels)

        lm_loss = -lm_logp.mean() + self.beta * lm_kl.mean()
        if not torch.isfinite(lm_loss):
            return None

        total_H_val = float(total_entropy.item())
        final_kl_val = float(lm_kl.item())
        traj_reward = self._shaped_return(logp_seq, final_kl_val, total_H_val, d)

        v_s0 = self.critic(s0.detach())
        advantage = float(np.clip(float(traj_reward) - float(v_s0.detach().item()), -5.0, 5.0))
        adv = torch.tensor(advantage, device=self.device, dtype=torch.float32)

        joint_logp = depth_logp[0] + total_action_logp.squeeze()
        rl_loss = -(adv * joint_logp) - self.entropy_bonus * total_entropy

        r_tensor = torch.tensor(float(traj_reward), device=self.device)
        critic_loss = F.mse_loss(v_s0, r_tensor)

        return {
            "lm_loss": lm_loss,
            "rl_loss": rl_loss,
            "critic_loss": critic_loss,
            "lm_logp": float(lm_logp.item()),
            "lm_kl": float(lm_kl.item()),
            "depth": d,
            "entropy": total_H_val,
            "reward": float(traj_reward),
        }

    # ── training loop ────────────────────────────────────────────

    def train(
        self,
        output_dir: str,
        epochs: int = 3,
        batch_size: int = 2,
        lr: float = 1e-6,
        max_length: int = 512,
        val_ratio: float = 0.1,
        seed: int = 42,
        max_samples: int = 5000,
    ):
        dataset = GSMHardDataset(
            self.tokenizer, split="train",
            max_length=max_length, val_ratio=val_ratio, seed=seed,
            max_target_tokens=128, max_samples=max_samples,
        )
        pad_id = int(self.tokenizer.pad_token_id)
        loader = DataLoader(
            dataset, batch_size=batch_size, shuffle=True,
            collate_fn=lambda b: collate_fn_pad(b, pad_id),
        )

        all_modules = [
            (self.policy_model,      lr * 0.1),
            (self.state_extractor,   lr * 2.0),
            (self.depth_controller,  lr * 3.0),
            (self.action_controller, lr * 3.0),
            (self.refiner,           lr * 2.0),
            (self.decode_bridge,     lr * 2.0),
            (self.critic,            lr * 3.0),
        ]
        optimizer = AdamW(
            [{"params": m.parameters(), "lr": l, "weight_decay": 0.01} for m, l in all_modules]
        )
        all_params = [p for m, _ in all_modules for p in m.parameters()]

        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        for epoch in range(epochs):
            for m, _ in all_modules:
                m.train()
            epoch_loss = num_batches = 0

            for step, batch in enumerate(loader):
                results = [r for inp in batch if (r := self._compute_one_sample(inp)) is not None]
                if not results:
                    continue

                lm_loss = torch.stack([r["lm_loss"] for r in results]).mean()
                rl_loss = torch.stack([r["rl_loss"] for r in results]).mean()
                critic_loss = torch.stack([r["critic_loss"] for r in results]).mean()
                total_loss = lm_loss + self.depth_rl_weight * rl_loss + self.critic_weight * critic_loss

                if not torch.isfinite(total_loss):
                    logger.warning("Non-finite total_loss, skipping batch")
                    continue

                optimizer.zero_grad()
                total_loss.backward()
                gnorm = torch.nn.utils.clip_grad_norm_(all_params, 1.0)
                optimizer.step()

                epoch_loss += float(total_loss.item())
                num_batches += 1
                self._global_step += 1

                avg_d = np.mean([r["depth"] for r in results])
                avg_H = np.mean([r["entropy"] for r in results])
                avg_R = np.mean([r["reward"] for r in results])
                self._record(float(total_loss.item()), float(lm_loss.item()),
                             float(rl_loss.item()), float(critic_loss.item()), avg_d, avg_H, avg_R)

                if step % 10 == 0:
                    logger.info(
                        f"Ep{epoch+1}/{epochs} Step{step} | "
                        f"total={total_loss:.4f} lm={lm_loss:.4f} rl={rl_loss:.4f} "
                        f"critic={critic_loss:.4f} reward={avg_R:.4f} "
                        f"depth={avg_d:.2f} entropy={avg_H:.3f} gnorm={gnorm:.4f}"
                    )

            logger.info(f"Epoch {epoch+1} avg_loss={epoch_loss / max(num_batches, 1):.4f}")
            self._save_ckpt(output_path / f"checkpoint-epoch-{epoch+1}")

        self._save_history(output_path)
        self._save_ckpt(output_path / "final_model")
        logger.info(f"Done → {output_path / 'final_model'}")

    # ── evaluation ───────────────────────────────────────────────

    @torch.no_grad()
    def evaluate(
        self,
        split: str = "val",
        max_samples: int = 500,
        max_length: int = 512,
        val_ratio: float = 0.1,
        seed: int = 42,
        max_gen_tokens: int = 64,
    ):
        import re

        dataset = GSMHardDataset(
            self.tokenizer, split=split,
            max_length=max_length, val_ratio=val_ratio, seed=seed,
            max_target_tokens=128, max_samples=None,
        )
        for mod in [self.policy_model, self.state_extractor, self.depth_controller,
                    self.action_controller, self.refiner, self.decode_bridge, self.critic]:
            mod.eval()

        n = min(len(dataset), max_samples)
        tot_logp = tot_depth = tot_H = 0.0
        exact_match = 0

        for i in range(n):
            item = dataset[i]
            r = self._compute_one_sample(item, is_train=False)
            if r is None:
                continue
            tot_logp += r["lm_logp"]
            tot_depth += r["depth"]
            tot_H += r["entropy"]

            input_ids = item["input_ids"].unsqueeze(0).to(self.device)
            attn_mask = item["attention_mask"].unsqueeze(0).to(self.device)

            policy_out = self.policy_model(input_ids=input_ids, attention_mask=attn_mask, output_hidden_states=True)
            h0 = policy_out.hidden_states[-1][0]
            s0 = self.state_extractor(h0, attn_mask[0])
            d_sample, _, _, _ = self.depth_controller(s0.unsqueeze(0))
            d = max(0, min(int(d_sample.item()), self.max_depth))

            st = s0
            ht = h0
            for _ in range(d):
                gamma, beta, vt, _, _ = self.action_controller(st)
                h_next = self.refiner.refine(ht, st, gamma, beta, vt, attn_mask[0])
                st = self.refiner.update_state(st, h_next, attn_mask[0])
                ht = h_next

            h_final = self.decode_bridge(st, h0)
            gen_ids = self.policy_model.generate(
                input_ids=input_ids,
                attention_mask=attn_mask,
                max_new_tokens=max_gen_tokens,
                do_sample=False,
                pad_token_id=self.tokenizer.pad_token_id,
            )
            gen_text = self.tokenizer.decode(gen_ids[0][input_ids.shape[1]:], skip_special_tokens=True)
            nums = re.findall(r"-?\d+(?:\.\d+)?", gen_text.replace(",", ""))
            pred = nums[-1] if nums else ""
            if pred == item["answer"]:
                exact_match += 1

            if i < 5:
                logger.info(f"Sample {i+1}: pred={pred!r} gold={item['answer']!r} logp={r['lm_logp']:.4f} depth={r['depth']}")

        c = max(n, 1)
        em = exact_match / c
        logger.info(f"[Eval:{split}] EM={em:.4f} ({exact_match}/{c}) logp={tot_logp/c:.4f} depth={tot_depth/c:.3f}")
        return {"exact_match": em, "logp": tot_logp / c, "avg_depth": tot_depth / c, "entropy": tot_H / c}

    # ── utilities ────────────────────────────────────────────────

    def _record(self, total, lm, rl, critic, depth, entropy, reward):
        self._hist["step"].append(self._global_step)
        for k, v in [("total", total), ("lm", lm), ("rl", rl), ("critic", critic),
                     ("depth", depth), ("entropy", entropy), ("reward", reward)]:
            self._hist[k].append(v)

    def _save_ckpt(self, path: Path):
        path.mkdir(parents=True, exist_ok=True)
        self.policy_model.save_pretrained(str(path), safe_serialization=True)
        self.tokenizer.save_pretrained(str(path))
        for name, mod in [
            ("state_extractor", self.state_extractor),
            ("depth_controller", self.depth_controller),
            ("action_controller", self.action_controller),
            ("refiner", self.refiner),
            ("decode_bridge", self.decode_bridge),
            ("critic", self.critic),
        ]:
            torch.save(mod.state_dict(), path / f"{name}.pt")

    def _save_history(self, output_path: Path):
        with open(output_path / "loss_history.csv", "w", newline="") as f:
            w = csv.writer(f)
            keys = list(self._hist.keys())
            w.writerow(keys)
            for row in zip(*[self._hist[k] for k in keys]):
                w.writerow(row)
        self._plot(output_path / "loss_curve.png")

    def _plot(self, out_png: Path, ma_w: int = 50):
        def ma(x):
            a = np.array(x, float)
            c = np.convolve(a, np.ones(ma_w) / ma_w, mode="valid")
            return np.concatenate([np.full(len(a) - len(c), np.nan), c])

        steps = np.array(self._hist["step"], float)
        fig, axes = plt.subplots(3, 1, figsize=(12, 12))

        for arr, lbl in [("total", "Total"), ("lm", "LM"), ("rl", "RL"), ("critic", "Critic")]:
            a = np.array(self._hist[arr], float)
            axes[0].plot(steps, a, alpha=0.3, label=lbl)
            axes[0].plot(steps, ma(a), lw=2, label=f"{lbl} MA")
        axes[0].set_title("Losses")
        axes[0].legend()
        axes[0].grid(alpha=0.3)

        d = np.array(self._hist["depth"], float)
        axes[1].plot(steps, d, alpha=0.3)
        axes[1].plot(steps, ma(d), lw=2, label="Depth MA")
        axes[1].set_title("Avg Refinement Depth")
        axes[1].legend()
        axes[1].grid(alpha=0.3)

        r = np.array(self._hist["reward"], float)
        axes[2].plot(steps, r, alpha=0.3)
        axes[2].plot(steps, ma(r), lw=2, label="Shaped Return MA")
        axes[2].set_title("Shaped Trajectory Reward")
        axes[2].legend()
        axes[2].grid(alpha=0.3)

        plt.tight_layout()
        plt.savefig(out_png, dpi=150)
        plt.close()
        logger.info(f"Saved plot: {out_png}")
