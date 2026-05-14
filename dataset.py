import random
import logging
from typing import Dict, List, Optional

import torch
from torch.utils.data import Dataset
from datasets import load_dataset

logger = logging.getLogger(__name__)


class GSMHardDataset(Dataset):
    def __init__(
        self,
        tokenizer,
        split: str = "train",
        max_length: int = 512,
        val_ratio: float = 0.1,
        seed: int = 42,
        max_target_tokens: int = 128,
        max_samples: Optional[int] = 5000,
    ):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.max_target_tokens = max_target_tokens

        raw = load_dataset("reasoning-machines/gsm-hard", split="train")
        data = list(raw)

        rng = random.Random(seed)
        idxs = list(range(len(data)))
        rng.shuffle(idxs)

        if max_samples is not None:
            idxs = idxs[:max_samples]

        val_n = max(1, int(round(len(idxs) * val_ratio)))
        val_set = set(idxs[:val_n])
        train_set = set(idxs[val_n:])

        self.data = [data[i] for i in (train_set if split == "train" else val_set)]
        logger.info(f"GSM-Hard split={split}, samples={len(self.data)}")

    def __len__(self):
        return len(self.data)

    @staticmethod
    def _normalize_answer(answer) -> str:
        ans = str(answer).strip().replace(",", "")
        try:
            f = float(ans)
            ans = str(int(f)) if f == int(f) else str(f)
        except ValueError:
            pass
        return ans

    def _build_prompt(self, question: str) -> str:
        return (
            "Solve the following math problem step by step.\n\n"
            f"Problem:\n{question.strip()}\n\n"
            "Solution (think step by step, then state the final answer):\n"
        )

    def __getitem__(self, idx: int) -> Dict:
        item = self.data[idx]
        question = str(item["input"]).strip()
        answer = self._normalize_answer(item["target"])
        prompt = self._build_prompt(question)
        target_text = f"Answer: {answer}"

        target_ids = self.tokenizer(
            "\n" + target_text,
            add_special_tokens=False,
            truncation=True,
            max_length=self.max_target_tokens,
        )["input_ids"]
        if not target_ids:
            target_ids = [self.tokenizer.eos_token_id]

        prompt_budget = max(1, self.max_length - len(target_ids) - 1)
        prompt_ids = self.tokenizer(
            prompt,
            add_special_tokens=False,
            truncation=True,
            max_length=prompt_budget,
        )["input_ids"]

        eos_id = self.tokenizer.eos_token_id
        input_ids = (prompt_ids + [eos_id] + target_ids)[: self.max_length]
        labels = ([-100] * (len(prompt_ids) + 1) + target_ids)[: self.max_length]

        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "attention_mask": torch.tensor([1] * len(input_ids), dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
            "full_text": prompt + "\n" + target_text,
            "answer": answer,
        }


def collate_fn_pad(batch: List[Dict], pad_token_id: int) -> List[Dict]:
    max_len = max(len(b["input_ids"]) for b in batch)
    out = []
    for b in batch:
        pad = max_len - len(b["input_ids"])
        if pad > 0:
            out.append({
                "input_ids": torch.cat([b["input_ids"], torch.full((pad,), pad_token_id, dtype=torch.long)]),
                "attention_mask": torch.cat([b["attention_mask"], torch.zeros(pad, dtype=torch.long)]),
                "labels": torch.cat([b["labels"], torch.full((pad,), -100, dtype=torch.long)]),
                "full_text": b["full_text"],
                "answer": b["answer"],
            })
        else:
            out.append(b)
    return out
