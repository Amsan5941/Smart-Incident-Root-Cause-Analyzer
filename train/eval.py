#!/usr/bin/env python3
"""
Evaluation Script for Fine-tuned Incident Root Cause Analyzer
Computes accuracy, F1, ROUGE, and per-category metrics on the held-out test set.
"""

import argparse
import json
import logging
import os
import sys
from collections import defaultdict

import torch
from datasets import Dataset
from peft import AutoPeftModelForCausalLM
from rouge_score import rouge_scorer
from sklearn.metrics import classification_report, f1_score
from tqdm import tqdm
from transformers import AutoTokenizer

logging.basicConfig(format="%(asctime)s | %(levelname)s | %(message)s", level=logging.INFO)
logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
# Category classification helpers
# ─────────────────────────────────────────────────────────────────────────────

CATEGORY_KEYWORDS = {
    "database_n_plus_one": ["n+1", "lazy-load", "n plus 1", "orm lazy"],
    "database_connection_pool": ["connection pool", "pool exhausted", "pool size", "pgbouncer"],
    "memory_leak": ["memory leak", "oom", "heap space", "out of memory", "gc pause"],
    "cpu_hot_loop": ["hot-loop", "cpu spike", "regex backtrack", "thread spinning", "100% cpu"],
    "disk_full": ["disk full", "no space left", "enospc", "disk exhaustion", "inode"],
    "database_missing_index": ["missing index", "seq scan", "sequential scan", "full table scan"],
    "kubernetes_oom": ["oomkill", "oom killed", "crashloopbackoff", "container killed"],
    "cache_stampede": ["cache stampede", "cache miss", "cache hit rate", "thundering herd"],
    "database_deadlock": ["deadlock", "lock ordering", "sharelock", "transaction rollback"],
    "service_timeout_cascade": ["cascading", "circuit breaker", "timeout cascade", "thread pool"],
    "certificate_expiry": ["certificate", "ssl", "tls", "cert expired", "x509"],
    "rate_limit_breach": ["rate limit", "429", "too many requests", "quota exceeded"],
    "config_error": ["missing config", "environment variable", "configmap", "crashloopbackoff"],
    "network_partition": ["network partition", "split-brain", "quorum", "az unreachable"],
    "application_error": ["schema change", "attribute error", "breaking change", "null pointer"],
}


def classify_category(text: str) -> str:
    """Classify predicted root cause into category by keyword matching."""
    text_lower = text.lower()
    for category, keywords in CATEGORY_KEYWORDS.items():
        if any(kw in text_lower for kw in keywords):
            return category
    return "unknown"


def extract_root_cause(output: str) -> str:
    """Extract the ROOT CAUSE line from model output."""
    lines = output.split("\n")
    for line in lines:
        if line.strip().startswith("ROOT CAUSE:"):
            return line.replace("ROOT CAUSE:", "").strip()
    # Fallback: return first non-empty line
    for line in lines:
        if line.strip():
            return line.strip()
    return output.strip()


# ─────────────────────────────────────────────────────────────────────────────
# Inference
# ─────────────────────────────────────────────────────────────────────────────

PROMPT_TEMPLATE = (
    "### Instruction:\n{instruction}\n\n"
    "### Input:\n{input}\n\n"
    "### Response:\n"
)


def run_inference(model, tokenizer, example: dict, max_new_tokens: int = 300, device: str = "cuda") -> str:
    prompt = PROMPT_TEMPLATE.format(
        instruction=example.get("instruction", "Analyze this production incident and identify the root cause."),
        input=example.get("input", ""),
    )
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=1800).to(device)
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=0.1,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )
    generated = outputs[0][inputs["input_ids"].shape[1]:]
    return tokenizer.decode(generated, skip_special_tokens=True).strip()


# ─────────────────────────────────────────────────────────────────────────────
# Metrics
# ─────────────────────────────────────────────────────────────────────────────

def compute_rouge(predictions: list[str], references: list[str]) -> dict:
    scorer = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=True)
    results = defaultdict(list)
    for pred, ref in zip(predictions, references):
        scores = scorer.score(ref, pred)
        for key, value in scores.items():
            results[key].append(value.fmeasure)
    return {k: sum(v) / len(v) for k, v in results.items()}


def compute_category_accuracy(
    true_categories: list[str], predicted_texts: list[str]
) -> tuple[float, dict]:
    pred_categories = [classify_category(t) for t in predicted_texts]
    correct = sum(t == p for t, p in zip(true_categories, pred_categories))
    accuracy = correct / len(true_categories)

    per_category = defaultdict(lambda: {"correct": 0, "total": 0})
    for true, pred in zip(true_categories, pred_categories):
        per_category[true]["total"] += 1
        if true == pred:
            per_category[true]["correct"] += 1

    per_category_acc = {
        cat: v["correct"] / v["total"] for cat, v in per_category.items()
    }
    return accuracy, per_category_acc


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_dir", type=str, required=True, help="Path to fine-tuned model checkpoint")
    parser.add_argument("--test_file", type=str, default="../data/training_incidents_test.jsonl")
    parser.add_argument("--output_file", type=str, default="./eval_results.json")
    parser.add_argument("--max_new_tokens", type=int, default=300)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--limit", type=int, default=None, help="Limit number of test examples")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Using device: {device}")

    # Load model
    logger.info(f"Loading model from {args.model_dir}")
    model = AutoPeftModelForCausalLM.from_pretrained(
        args.model_dir,
        torch_dtype=torch.float16 if device == "cuda" else torch.float32,
        device_map="auto" if device == "cuda" else None,
    )
    model.eval()
    tokenizer = AutoTokenizer.from_pretrained(args.model_dir)
    tokenizer.pad_token = tokenizer.eos_token

    # Load test data
    logger.info(f"Loading test data from {args.test_file}")
    examples = []
    with open(args.test_file) as f:
        for line in f:
            line = line.strip()
            if line:
                examples.append(json.loads(line))

    if args.limit:
        examples = examples[:args.limit]

    logger.info(f"Evaluating on {len(examples)} examples...")

    predictions = []
    references = []
    true_categories = []

    for i, example in enumerate(tqdm(examples, desc="Evaluating")):
        pred = run_inference(model, tokenizer, example, args.max_new_tokens, device)
        predictions.append(pred)
        references.append(example.get("output", ""))
        true_categories.append(example.get("category", "unknown"))

        if i < 3:
            logger.info(f"\n--- Example {i+1} ---")
            logger.info(f"EXPECTED: {extract_root_cause(references[-1])}")
            logger.info(f"PREDICTED: {extract_root_cause(pred)}")

    # Compute metrics
    rouge = compute_rouge(predictions, references)
    accuracy, per_cat_acc = compute_category_accuracy(true_categories, predictions)

    # F1 on category classification
    pred_categories = [classify_category(t) for t in predictions]
    f1_macro = f1_score(true_categories, pred_categories, average="macro", zero_division=0)
    f1_weighted = f1_score(true_categories, pred_categories, average="weighted", zero_division=0)

    results = {
        "num_examples": len(examples),
        "category_accuracy": round(accuracy, 4),
        "f1_macro": round(f1_macro, 4),
        "f1_weighted": round(f1_weighted, 4),
        "rouge": {k: round(v, 4) for k, v in rouge.items()},
        "per_category_accuracy": {k: round(v, 4) for k, v in sorted(per_cat_acc.items())},
        "classification_report": classification_report(
            true_categories, pred_categories, zero_division=0, output_dict=True
        ),
    }

    print("\n" + "=" * 60)
    print("EVALUATION RESULTS")
    print("=" * 60)
    print(f"Category Accuracy:  {accuracy:.1%}")
    print(f"F1 Macro:           {f1_macro:.4f}")
    print(f"F1 Weighted:        {f1_weighted:.4f}")
    print(f"ROUGE-1:            {rouge['rouge1']:.4f}")
    print(f"ROUGE-2:            {rouge['rouge2']:.4f}")
    print(f"ROUGE-L:            {rouge['rougeL']:.4f}")
    print("\nPer-Category Accuracy:")
    for cat, acc in sorted(per_cat_acc.items(), key=lambda x: -x[1]):
        print(f"  {cat:<35} {acc:.1%}")
    print("=" * 60)

    os.makedirs(os.path.dirname(args.output_file) or ".", exist_ok=True)
    with open(args.output_file, "w") as f:
        json.dump(results, f, indent=2)
    logger.info(f"Results saved to {args.output_file}")

    target = 0.80
    if accuracy >= target:
        logger.info(f"✓ Accuracy {accuracy:.1%} meets target of {target:.0%}")
    else:
        logger.warning(f"✗ Accuracy {accuracy:.1%} below target of {target:.0%} – consider more training")


if __name__ == "__main__":
    main()
