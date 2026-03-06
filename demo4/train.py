#!/usr/bin/env python3
"""
Demo 4: LoRA fine-tuning on Azure ML using PyTorch + Hugging Face.

Same task as demo2 (insurance damage extraction) but runs on Azure ML compute
with Qwen2.5-7B-Instruct, PEFT LoRA, and Transformers. Uses the same JSONL data
format: {"prompt": "...", "completion": "<JSON string>"}.

Run locally: python train.py --data-dir ../demo2/data --output-dir ./outputs
Run on Azure ML: submit_job.py submits this script as a job.
"""

import argparse
import json
import sys
from pathlib import Path

import torch
from datasets import Dataset
from peft import LoraConfig, TaskType, get_peft_model
from transformers import (AutoModelForCausalLM, AutoTokenizer, Trainer,
                          TrainingArguments)

# -----------------------------------------------------------------------------
# Config (aligned with demo2 where possible)
# -----------------------------------------------------------------------------
MODEL_ID = "Qwen/Qwen2.5-7B-Instruct"
DEMO_MODEL_ID = "Qwen/Qwen2.5-0.5B-Instruct"
MAX_SEQ_LENGTH = 512
LORA_R = 16
LORA_ALPHA = 32
LORA_DROPOUT = 0.05
TARGET_MODULES = ["q_proj", "k_proj", "v_proj", "o_proj"]


def load_jsonl(path: Path) -> list[dict]:
    """Load JSONL file; return list of dicts."""
    out = []
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        out.append(json.loads(line))
    return out


def tokenize_and_mask_prompt(
    examples: dict,
    tokenizer,
    max_length: int,
) -> dict:
    """
    Tokenize prompt+completion and create labels with -100 for prompt tokens.
    Only the completion tokens contribute to the loss.
    """
    prompts = examples["prompt"]
    completions = examples["completion"]

    input_ids_list = []
    labels_list = []
    attention_mask_list = []

    for prompt, completion in zip(prompts, completions):
        full_text = prompt + completion
        prompt_enc = tokenizer(
            prompt,
            truncation=True,
            max_length=max_length,
            add_special_tokens=True,
            return_attention_mask=False,
        )
        full_enc = tokenizer(
            full_text,
            truncation=True,
            max_length=max_length,
            add_special_tokens=True,
            return_attention_mask=True,
            padding="max_length",
        )

        prompt_len = len(prompt_enc["input_ids"])
        full_ids = full_enc["input_ids"]
        full_len = len(full_ids)

        # Labels: -100 for prompt, token ids for completion, -100 for padding
        labels = [-100] * full_len
        for i in range(prompt_len, full_len):
            if full_ids[i] != tokenizer.pad_token_id:
                labels[i] = full_ids[i]

        input_ids_list.append(full_ids)
        labels_list.append(labels)
        attention_mask_list.append(full_enc["attention_mask"])

    return {
        "input_ids": input_ids_list,
        "labels": labels_list,
        "attention_mask": attention_mask_list,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="LoRA fine-tune Qwen2.5 for damage extraction")
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("data"),
        help="Directory with train.jsonl, valid.jsonl",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs"),
        help="Where to save adapters (Azure ML uses ./outputs)",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=MODEL_ID,
        help="Hugging Face model id",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=3,
        help="Number of training epochs",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=2,
        help="Per-device train batch size",
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=7e-5,
        help="Learning rate",
    )
    parser.add_argument(
        "--max-length",
        type=int,
        default=MAX_SEQ_LENGTH,
        help="Max sequence length",
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        default=None,
        help="Stop after N training steps (overrides epochs)",
    )
    parser.add_argument(
        "--max-train-samples",
        type=int,
        default=None,
        help="Use only first N training examples",
    )
    parser.add_argument(
        "--demo",
        action="store_true",
        help="~2 min conference demo: 0.5B model, 20 steps, 40 samples",
    )
    args = parser.parse_args()

    if args.demo:
        args.model = DEMO_MODEL_ID
        args.max_steps = 20
        args.max_train_samples = 40

    data_dir = args.data_dir.resolve()
    output_dir = args.output_dir.resolve()
    train_path = data_dir / "train.jsonl"
    valid_path = data_dir / "valid.jsonl"

    if not train_path.exists():
        print(f"Error: {train_path} not found", file=sys.stderr)
        sys.exit(1)
    if not valid_path.exists():
        print(f"Error: {valid_path} not found", file=sys.stderr)
        sys.exit(1)

    print("=== CONFIG ===")
    print(f"  Model:      {args.model}")
    print(f"  Data dir:   {data_dir}")
    print(f"  Output:     {output_dir}")
    print(f"  Epochs:     {args.epochs}")
    print(f"  Batch:      {args.batch_size}")
    if args.max_steps:
        print(f"  Max steps:  {args.max_steps}")
    if args.max_train_samples:
        print(f"  Max samples: {args.max_train_samples}")
    if args.demo:
        print("  Mode:       DEMO (~2 min)")
    print()

    # Load data
    train_data = load_jsonl(train_path)
    valid_data = load_jsonl(valid_path)
    if args.max_train_samples:
        train_data = train_data[: args.max_train_samples]
    train_ds = Dataset.from_list(train_data)
    valid_ds = Dataset.from_list(valid_data)

    # Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    # Tokenize with prompt masking
    def tokenize_fn(examples):
        return tokenize_and_mask_prompt(examples, tokenizer, args.max_length)

    train_ds = train_ds.map(
        tokenize_fn,
        batched=True,
        remove_columns=["prompt", "completion"],
        desc="Tokenizing train",
    )
    valid_ds = valid_ds.map(
        tokenize_fn,
        batched=True,
        remove_columns=["prompt", "completion"],
        desc="Tokenizing valid",
    )

    # Model + LoRA
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
    )
    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=LORA_R,
        lora_alpha=LORA_ALPHA,
        lora_dropout=LORA_DROPOUT,
        target_modules=TARGET_MODULES,
        bias="none",
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    # Data collator: batch our pre-tokenized data (labels already set, -100 for prompt)
    def collate_fn(examples):
        return {
            "input_ids": torch.tensor([e["input_ids"] for e in examples]),
            "labels": torch.tensor([e["labels"] for e in examples]),
            "attention_mask": torch.tensor([e["attention_mask"] for e in examples]),
        }

    # Training
    train_kwargs = {
        "output_dir": str(output_dir),
        "num_train_epochs": 1 if args.max_steps else args.epochs,
        "per_device_train_batch_size": args.batch_size,
        "per_device_eval_batch_size": args.batch_size,
        "learning_rate": args.learning_rate,
        "logging_steps": 5 if args.demo else 10,
        "eval_strategy": "no" if args.demo else "epoch",
        "save_strategy": "steps" if args.max_steps else "epoch",
        "save_total_limit": 1 if args.demo else 2,
        "bf16": True,
        "gradient_checkpointing": True,
        "report_to": "none",
    }
    if args.max_steps:
        train_kwargs["max_steps"] = args.max_steps
        train_kwargs["save_steps"] = args.max_steps
    training_args = TrainingArguments(**train_kwargs)

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=valid_ds,
        data_collator=collate_fn,
    )

    print("=== TRAIN ===")
    trainer.train()
    trainer.save_model(str(output_dir))
    tokenizer.save_pretrained(str(output_dir))
    print(f"Adapters saved to {output_dir}")


if __name__ == "__main__":
    main()
