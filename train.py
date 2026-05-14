import os
import logging
from trainer import HiddenRLTrainer

logging.basicConfig(level=logging.INFO)

HF_TOKEN = os.environ.get("HF_TOKEN", "")

if __name__ == "__main__":
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

    trainer = HiddenRLTrainer(
        model_name="google/gemma-2-2b-it",
        beta=0.1,
        max_depth=3,
        depth_cost=0.02,
        entropy_bonus=0.01,
        depth_rl_weight=0.15,
        critic_weight=0.2,
        action_dim=64,
        state_dim=256,
        hf_token=HF_TOKEN,
    )

    trainer.train(
        output_dir="./gsm_hard_hidden_rl",
        epochs=15,
        batch_size=2,
        lr=1e-6,
        max_length=512,
        val_ratio=0.1,
        seed=42,
        max_samples=5000,
    )

    trainer.evaluate(
        split="val",
        max_samples=200,
        max_length=512,
        val_ratio=0.1,
        seed=42,
        max_gen_tokens=64,
    )
