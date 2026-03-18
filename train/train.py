#!/usr/bin/env python3
"""
QLoRA Fine-tuning Script for Incident Root Cause Analyzer
Uses PEFT + bitsandbytes for efficient 4-bit quantized fine-tuning on Mistral-7B.

Run on DigitalOcean GPU Droplet (A100 40GB recommended):
    python train.py --model_name mistralai/Mistral-7B-v0.1 \
                    --train_file ../data/training_incidents.jsonl \
                    --output_dir ./checkpoints/incident-analyzer-v1
"""

import argparse
import json
import logging
import os
import sys
from dataclasses import dataclass, field
from typing import Optional

import torch
from datasets import Dataset, load_dataset
from peft import (
    LoraConfig,
    TaskType,
    get_peft_model,
    prepare_model_for_kbit_training,
)
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    DataCollatorForSeq2Seq,
    HfArgumentParser,
    TrainerCallback,
    TrainingArguments,
    set_seed,
)
from trl import SFTTrainer

logging.basicConfig(
    format="%(asctime)s | %(levelname)s | %(message)s",
    level=logging.INFO,
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Argument Dataclasses
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class ModelArguments:
    model_name: str = field(
        default="mistralai/Mistral-7B-v0.1",
        metadata={"help": "HuggingFace model ID or local path"},
    )
    use_4bit: bool = field(default=True, metadata={"help": "Use 4-bit quantization (QLoRA)"})
    bnb_4bit_compute_dtype: str = field(default="bfloat16")
    bnb_4bit_quant_type: str = field(default="nf4")
    use_nested_quant: bool = field(default=False)


@dataclass
class LoRAArguments:
    lora_r: int = field(default=64, metadata={"help": "LoRA attention dimension"})
    lora_alpha: int = field(default=128, metadata={"help": "LoRA scaling factor"})
    lora_dropout: float = field(default=0.05)
    target_modules: str = field(
        default="q_proj,v_proj,k_proj,o_proj,gate_proj,up_proj,down_proj",
        metadata={"help": "Comma-separated list of modules to apply LoRA to"},
    )


@dataclass
class DataArguments:
    train_file: str = field(default="../data/training_incidents.jsonl")
    eval_file: Optional[str] = field(default=None)
    max_seq_length: int = field(default=2048)
    prompt_template: str = field(
        default=(
            "### Instruction:\n{instruction}\n\n"
            "### Input:\n{input}\n\n"
            "### Response:\n{output}"
        )
    )


# ─────────────────────────────────────────────────────────────────────────────
# Prompt Formatting
# ─────────────────────────────────────────────────────────────────────────────

def format_prompt(example: dict, template: str, include_response: bool = True) -> str:
    """Format a training example into instruction-tuning prompt."""
    prompt = template.format(
        instruction=example.get("instruction", "Analyze this production incident and identify the root cause."),
        input=example.get("input", ""),
        output=example.get("output", "") if include_response else "",
    )
    return prompt


def load_jsonl_dataset(file_path: str, template: str, tokenizer, max_seq_length: int) -> Dataset:
    """Load and format JSONL dataset for training."""
    logger.info(f"Loading dataset from {file_path}")
    examples = []
    with open(file_path, "r") as f:
        for line in f:
            line = line.strip()
            if line:
                examples.append(json.loads(line))

    logger.info(f"Loaded {len(examples)} examples")

    def format_and_filter(example):
        text = format_prompt(example, template)
        tokens = tokenizer(text, truncation=True, max_length=max_seq_length)
        return {"text": text, "input_ids": tokens["input_ids"]}

    dataset = Dataset.from_list(examples)
    dataset = dataset.map(format_and_filter, remove_columns=dataset.column_names)
    logger.info(f"Dataset formatted: {len(dataset)} examples retained")
    return dataset


# ─────────────────────────────────────────────────────────────────────────────
# Training Callback
# ─────────────────────────────────────────────────────────────────────────────

class LossLoggingCallback(TrainerCallback):
    def __init__(self):
        self.training_losses = []
        self.eval_losses = []

    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs is None:
            return
        if "loss" in logs:
            self.training_losses.append({"step": state.global_step, "loss": logs["loss"]})
            logger.info(f"Step {state.global_step} | Loss: {logs['loss']:.4f} | LR: {logs.get('learning_rate', 0):.2e}")
        if "eval_loss" in logs:
            self.eval_losses.append({"step": state.global_step, "eval_loss": logs["eval_loss"]})
            logger.info(f"Step {state.global_step} | Eval Loss: {logs['eval_loss']:.4f}")

    def save_metrics(self, output_dir: str):
        metrics_path = os.path.join(output_dir, "training_metrics.json")
        with open(metrics_path, "w") as f:
            json.dump({"training_losses": self.training_losses, "eval_losses": self.eval_losses}, f, indent=2)
        logger.info(f"Training metrics saved to {metrics_path}")


# ─────────────────────────────────────────────────────────────────────────────
# Model Loading
# ─────────────────────────────────────────────────────────────────────────────

def load_model_and_tokenizer(model_args: ModelArguments):
    """Load base model with optional 4-bit quantization."""
    compute_dtype = getattr(torch, model_args.bnb_4bit_compute_dtype)

    bnb_config = None
    if model_args.use_4bit:
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=compute_dtype,
            bnb_4bit_quant_type=model_args.bnb_4bit_quant_type,
            bnb_4bit_use_double_quant=model_args.use_nested_quant,
        )
        logger.info("Using 4-bit QLoRA quantization")

    logger.info(f"Loading base model: {model_args.model_name}")
    model = AutoModelForCausalLM.from_pretrained(
        model_args.model_name,
        quantization_config=bnb_config,
        device_map="auto",
        torch_dtype=compute_dtype if not model_args.use_4bit else None,
        trust_remote_code=True,
    )

    if model_args.use_4bit:
        model = prepare_model_for_kbit_training(model)

    logger.info(f"Loading tokenizer: {model_args.model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_args.model_name, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    return model, tokenizer


def apply_lora(model, lora_args: LoRAArguments):
    """Apply LoRA adapters to the model."""
    target_modules = [m.strip() for m in lora_args.target_modules.split(",")]

    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=lora_args.lora_r,
        lora_alpha=lora_args.lora_alpha,
        target_modules=target_modules,
        lora_dropout=lora_args.lora_dropout,
        bias="none",
        inference_mode=False,
    )

    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    return model


# ─────────────────────────────────────────────────────────────────────────────
# Main Training Loop
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="QLoRA fine-tuning for incident analyzer")
    parser.add_argument("--model_name", type=str, default="mistralai/Mistral-7B-v0.1")
    parser.add_argument("--train_file", type=str, default="../data/training_incidents.jsonl")
    parser.add_argument("--eval_file", type=str, default=None)
    parser.add_argument("--output_dir", type=str, default="./checkpoints/incident-analyzer-v1")
    parser.add_argument("--num_epochs", type=int, default=5)
    parser.add_argument("--learning_rate", type=float, default=2e-4)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--grad_accum_steps", type=int, default=4)
    parser.add_argument("--max_seq_length", type=int, default=2048)
    parser.add_argument("--lora_r", type=int, default=64)
    parser.add_argument("--lora_alpha", type=int, default=128)
    parser.add_argument("--warmup_ratio", type=float, default=0.03)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--push_to_hub", action="store_true")
    parser.add_argument("--hub_model_id", type=str, default=None)
    parser.add_argument("--spaces_bucket", type=str, default=None, help="DO Spaces bucket for artifact upload")
    args = parser.parse_args()

    set_seed(args.seed)
    os.makedirs(args.output_dir, exist_ok=True)

    model_args = ModelArguments(model_name=args.model_name)
    lora_args = LoRAArguments(lora_r=args.lora_r, lora_alpha=args.lora_alpha)
    data_args = DataArguments(
        train_file=args.train_file,
        eval_file=args.eval_file,
        max_seq_length=args.max_seq_length,
    )

    # Load model & tokenizer
    model, tokenizer = load_model_and_tokenizer(model_args)
    model = apply_lora(model, lora_args)

    # Load datasets
    train_dataset = load_jsonl_dataset(
        data_args.train_file, data_args.prompt_template, tokenizer, data_args.max_seq_length
    )
    eval_dataset = None
    if data_args.eval_file:
        eval_dataset = load_jsonl_dataset(
            data_args.eval_file, data_args.prompt_template, tokenizer, data_args.max_seq_length
        )

    # Training arguments
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.num_epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum_steps,
        learning_rate=args.learning_rate,
        lr_scheduler_type="cosine",
        warmup_ratio=args.warmup_ratio,
        fp16=not torch.cuda.is_bf16_supported(),
        bf16=torch.cuda.is_bf16_supported(),
        logging_steps=10,
        evaluation_strategy="epoch" if eval_dataset else "no",
        save_strategy="epoch",
        save_total_limit=3,
        load_best_model_at_end=bool(eval_dataset),
        metric_for_best_model="eval_loss" if eval_dataset else None,
        push_to_hub=args.push_to_hub,
        hub_model_id=args.hub_model_id,
        report_to="tensorboard",
        seed=args.seed,
        dataloader_num_workers=4,
        group_by_length=True,
        remove_unused_columns=False,
    )

    loss_callback = LossLoggingCallback()

    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=tokenizer,
        dataset_text_field="text",
        max_seq_length=data_args.max_seq_length,
        callbacks=[loss_callback],
    )

    logger.info("Starting training...")
    logger.info(f"  Training examples: {len(train_dataset)}")
    logger.info(f"  Epochs: {args.num_epochs}")
    logger.info(f"  Batch size: {args.batch_size} × {args.grad_accum_steps} grad accum")
    logger.info(f"  Learning rate: {args.learning_rate}")
    logger.info(f"  Output: {args.output_dir}")

    train_result = trainer.train()

    logger.info("Training complete.")
    logger.info(f"  Final loss: {train_result.training_loss:.4f}")

    # Save model
    final_dir = os.path.join(args.output_dir, "final")
    trainer.save_model(final_dir)
    tokenizer.save_pretrained(final_dir)
    loss_callback.save_metrics(args.output_dir)

    # Save training summary
    summary = {
        "model_name": args.model_name,
        "training_loss": train_result.training_loss,
        "num_epochs": args.num_epochs,
        "num_train_examples": len(train_dataset),
        "lora_r": args.lora_r,
        "lora_alpha": args.lora_alpha,
        "learning_rate": args.learning_rate,
        "output_dir": args.output_dir,
    }
    with open(os.path.join(args.output_dir, "training_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    logger.info(f"Model saved to: {final_dir}")

    # Upload to DigitalOcean Spaces (optional)
    if args.spaces_bucket:
        upload_to_spaces(final_dir, args.spaces_bucket)


def upload_to_spaces(local_dir: str, bucket: str):
    """Upload model artifacts to DigitalOcean Spaces (S3-compatible)."""
    import boto3
    from botocore.config import Config

    endpoint = os.environ.get("SPACES_ENDPOINT", "https://nyc3.digitaloceanspaces.com")
    key = os.environ.get("SPACES_ACCESS_KEY")
    secret = os.environ.get("SPACES_SECRET_KEY")

    if not key or not secret:
        logger.warning("SPACES_ACCESS_KEY / SPACES_SECRET_KEY not set, skipping upload")
        return

    s3 = boto3.client(
        "s3",
        endpoint_url=endpoint,
        aws_access_key_id=key,
        aws_secret_access_key=secret,
        config=Config(signature_version="s3v4"),
    )

    for root, _, files in os.walk(local_dir):
        for fname in files:
            local_path = os.path.join(root, fname)
            s3_key = f"models/incident-analyzer/{os.path.relpath(local_path, local_dir)}"
            logger.info(f"Uploading {local_path} → s3://{bucket}/{s3_key}")
            s3.upload_file(local_path, bucket, s3_key)

    logger.info(f"Model artifacts uploaded to s3://{bucket}/models/incident-analyzer/")


if __name__ == "__main__":
    main()
