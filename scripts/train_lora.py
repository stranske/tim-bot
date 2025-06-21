#!/usr/bin/env python
"""
Fine‑tune a causal‑LM with LoRA (PEFT) using the 🤗 Transformers Trainer.

Quick‑start
-----------
accelerate launch scripts/train_lora.py configs/lora_mistral.yaml \
    --resume_from_checkpoint checkpoints/mistral-tim-lora

YAML schema expected
--------------------
base_model:   <HF repo id or local path>
output_dir:   <where to write checkpoints>
train_file:   data/final/train_split.jsonl
val_file:     data/final/val_split.jsonl
lora:         # arguments forwarded to peft.LoraConfig
  r: 64
  alpha: 16
  dropout: 0.05
  target_modules: [q_proj, k_proj, v_proj]
training:     # any transformers.TrainingArguments
  per_device_train_batch_size: 2
  gradient_accumulation_steps: 8
  num_train_epochs: 3
  save_steps: 100
  evaluation_strategy: "steps"
  eval_steps: 100
  logging_steps: 25
  fp16: true

"""
from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Optional

import torch
import yaml
from datasets import load_dataset
from peft import LoraConfig, get_peft_model
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    DataCollatorForLanguageModeling,
    Trainer,
    TrainingArguments,
    set_seed,
)


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

def load_yaml(path: str | os.PathLike) -> dict:
    """Read YAML file into a dict."""
    with open(path, "r") as f:
        return yaml.safe_load(f)


def build_dataset(cfg: dict, tokenizer, max_samples: Optional[int] = None):
    """Tokenise JSONL → HF Dataset suitable for causal‑LM."""

    def tok_fn(batch):
        return tokenizer(batch["text"])

    ds = load_dataset(
        "json",
        data_files={"train": cfg["train_file"], "validation": cfg["val_file"]},
    )

    if max_samples:
        ds["train"] = ds["train"].select(range(min(max_samples, len(ds["train"]))))
        ds["validation"] = ds["validation"].select(
            range(min(max_samples, len(ds["validation"])))
        )

    ds_tok = ds.map(tok_fn, batched=True, num_proc=os.cpu_count(), remove_columns=["text"])
    return ds_tok["train"], ds_tok["validation"]


# ---------------------------------------------------------------------------
# Main training entry
# ---------------------------------------------------------------------------

def main(
    config_path: str | Path,
    max_samples: Optional[int] = None,
    resume_from_checkpoint: Optional[str] = None,
):
    cfg = load_yaml(config_path)

    base_model = cfg["base_model"]
    out_dir = cfg["output_dir"]
    Path(out_dir).mkdir(parents=True, exist_ok=True)

    # Tokeniser
    tok = AutoTokenizer.from_pretrained(base_model, use_fast=True)
    if tok.pad_token is None:
        tok.pad_token = tok.unk_token

    # LoRA‑wrapped model
    lora_cfg = LoraConfig(**cfg["lora"])
    model = AutoModelForCausalLM.from_pretrained(
        base_model, torch_dtype=torch.bfloat16, device_map="auto"
    )
    model = get_peft_model(model, lora_cfg)
    model.print_trainable_parameters()

    # Data
    train_ds, val_ds = build_dataset(cfg, tok, max_samples)
    collator = DataCollatorForLanguageModeling(tok, mlm=False)

    # TrainingArguments (yaml wins, but we set output_dir explicitly)
    training_args = TrainingArguments(output_dir=out_dir, **cfg["training"])

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        data_collator=collator,
    )

    trainer.train(resume_from_checkpoint=resume_from_checkpoint)
    trainer.save_model(out_dir)
    tok.save_pretrained(out_dir)


# ---------------------------------------------------------------------------
# CLI wrapper
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="LoRA fine‑tune helper around HuggingFace Trainer.",
    )
    parser.add_argument("config", help="Path to YAML config file.")
    parser.add_argument(
        "--max_samples",
        type=int,
        default=None,
        help="Truncate datasets to N samples (debug mode).",
    )
    parser.add_argument(
        "--resume_from_checkpoint",
        default=None,
        help="Path to checkpoint folder or adapter root to resume training from.",
    )
    args = parser.parse_args()

    set_seed(0)
    main(args.config, args.max_samples, args.resume_from_checkpoint)
