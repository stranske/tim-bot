#!/usr/bin/env python
"""
train_lora.py — One‑stop LoRA Trainer wrapper (HF Transformers + PEFT)
====================================================================
• Understands short YAML keys → TrainingArguments
• Auto‑detects LoRA rank/alpha on resume (if you like)
• Auto‑picks the latest `checkpoint-*` folder when you pass the adapter root
• Caps tokenizer length to avoid OverflowError

Run (resume the final steps)
---------------------------
accelerate launch scripts/train_lora.py configs/lora_mistral.yaml \
    --resume_from_checkpoint checkpoints/mistral-tim-lora

YAML cheat‑sheet (short keys allowed)
-------------------------------------
base_model:  mistralai/Mistral-7B-Instruct-v0.3
output_dir:  checkpoints/mistral-tim-lora
train_file:  data/final/train_split.jsonl
val_file:    data/final/val_split.jsonl
seq_len:     2048
lora:
  r: 8                  # must match adapter when resuming
  alpha: 16             # mapped → lora_alpha
  dropout: 0.05         # mapped → lora_dropout
  target_modules: [q_proj, k_proj, v_proj]
training:
  batch_size: 2         # per_device_train_batch_size
  grad_accum: 8         # gradient_accumulation_steps
  epochs: 3             # num_train_epochs
  lr: 2e-4              # learning_rate
  save_steps: 100
  eval_steps: 100
  logging_steps: 25
  fp16: true
"""
from __future__ import annotations

import argparse
import json
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
    TrainingArguments,
    Trainer,
    set_seed,
)

DEFAULT_SEQ_LEN = 2048  # sane truncation length

# ---------------------------------------------------------------------------
# Utility mappers
# ---------------------------------------------------------------------------

def load_yaml(path: str | os.PathLike) -> dict:
    with open(path, "r") as fh:
        return yaml.safe_load(fh)


def map_training_keys(d: dict) -> dict:
    tbl = {
        "batch_size": "per_device_train_batch_size",
        "grad_accum": "gradient_accumulation_steps",
        "epochs": "num_train_epochs",
        "lr": "learning_rate",
    }
    return {tbl.get(k, k): v for k, v in d.items() if k != "output_dir"}


def map_lora_keys(d: dict) -> dict:
    out = dict(d)
    if "alpha" in out:
        out["lora_alpha"] = out.pop("alpha")
    if "dropout" in out:
        out["lora_dropout"] = out.pop("dropout")
    return out

# ---------------------------------------------------------------------------
# Dataset helper
# ---------------------------------------------------------------------------

def build_dataset(cfg: dict, tok, max_samples: Optional[int] = None):
    seq_len = cfg.get("seq_len", DEFAULT_SEQ_LEN)
    if not hasattr(tok, "model_max_length") or tok.model_max_length > 1_000_000:
        tok.model_max_length = seq_len

    def tok_fn(batch):
        return tok(batch["text"], truncation=True, max_length=seq_len)

    ds = load_dataset(
        "json",
        data_files={"train": cfg["train_file"], "validation": cfg["val_file"]},
    )
    if max_samples:
        ds["train"] = ds["train"].select(range(min(max_samples, len(ds["train"]))))
        ds["validation"] = ds["validation"].select(range(min(max_samples, len(ds["validation"]))))

    ds_tok = ds.map(tok_fn, batched=True, num_proc=os.cpu_count(), remove_columns=["text"])
    return ds_tok["train"], ds_tok["validation"]

# ---------------------------------------------------------------------------
# Main driver
# ---------------------------------------------------------------------------

def main(cfg_path: str | Path, max_samples: Optional[int], resume: Optional[str]):
    cfg = load_yaml(cfg_path)

    ## Resolve paths --------------------------------------------------------
    base_model = cfg["base_model"]
    out_dir = cfg.get("output_dir") or cfg.get("training", {}).get("output_dir")
    if out_dir is None:
        raise KeyError("output_dir missing in YAML (top‑level or training:)")
    Path(out_dir).mkdir(parents=True, exist_ok=True)

    ## Auto‑sync LoRA params with adapter when resuming --------------------
    if resume and Path(resume, "adapter_config.json").exists():
        cfg["lora"] = json.load(open(Path(resume, "adapter_config.json")))

    ## Tokeniser + model ----------------------------------------------------
    tok = AutoTokenizer.from_pretrained(base_model, use_fast=True)
    if tok.pad_token is None:
        tok.pad_token = tok.unk_token

    model = AutoModelForCausalLM.from_pretrained(base_model, device_map="auto", torch_dtype="auto")
    model = get_peft_model(model, LoraConfig(**map_lora_keys(cfg["lora"])))
    model.print_trainable_parameters()

    ## Dataset --------------------------------------------------------------
    train_ds, val_ds = build_dataset(cfg, tok, max_samples)
    collator = DataCollatorForLanguageModeling(tok, mlm=False)

    ## TrainingArguments ----------------------------------------------------
    training_args = TrainingArguments(
        output_dir=out_dir,
        **map_training_keys(cfg.get("training", {})),
    )

    ## Trainer --------------------------------------------------------------
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        data_collator=collator,
    )

    ## Smart resume ---------------------------------------------------------
    if resume:
        r = Path(resume)
        if r.is_dir() and not (r / "trainer_state.json").exists():
            ckpts = sorted(r.glob("checkpoint-*"), key=lambda p: int(p.name.split("-")[-1]))
            if not ckpts:
                raise FileNotFoundError(f"No checkpoint-* dirs inside {r}")
            resume = str(ckpts[-1])
            print(f"[train_lora] Auto‑resuming from {resume}")
    trainer.train(resume_from_checkpoint=resume)

    trainer.save_model(out_dir)
    tok.save_pretrained(out_dir)

# ---------------------------------------------------------------------------
# CLI ----------------------------------------------------------------------
if __name__ == "__main__":
    p = argparse.ArgumentParser(description="LoRA fine‑tune helper")
    p.add_argument("config", help="Path to YAML config")
    p.add_argument("--max_samples", type=int, default=None, help="Debug: cap dataset size")
    p.add_argument("--resume_from_checkpoint", default=None, help="Adapter root or checkpoint‑#### folder")
    args = p.parse_args()

    set_seed(0)
    main(args.config, args.max_samples, args.resume_from_checkpoint)
