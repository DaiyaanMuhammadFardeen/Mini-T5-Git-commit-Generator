#!/usr/bin/env python3
"""
predict.py

Use trained T5 model to predict commit messages from git diffs.
Reads random samples from a parquet dataset and outputs CSV with:
    - actual commit message
    - generated commit message

Features:
- Configurable via command-line arguments
- Progress bars for each step
- Logging for runtime visibility
- End statistics summary
"""

import os
import argparse
import logging
import random
import time
import pandas as pd
from tqdm import tqdm
from transformers import T5ForConditionalGeneration, RobertaTokenizerFast

# ----------------------
# Logging setup
# ----------------------
logging.basicConfig(
    format="%(asctime)s | %(levelname)s | %(message)s",
    level=logging.INFO
)
logger = logging.getLogger(__name__)

# ----------------------
# Argument parser
# ----------------------
def parse_args():
    parser = argparse.ArgumentParser(description="Predict commit messages from git diffs")
    
    parser.add_argument("--model_dir", type=str, required=True,
                        help="Directory containing the trained model and tokenizer")
    parser.add_argument("--input_file", type=str, required=True,
                        help="Parquet file with dataset containing diff_text and message")
    parser.add_argument("--output_file", type=str, default="predictions.csv",
                        help="CSV file to save predictions")
    parser.add_argument("--num_samples", type=int, default=100,
                        help="Number of random samples to predict")
    parser.add_argument("--max_input_length", type=int, default=512,
                        help="Maximum token length for input sequences")
    parser.add_argument("--max_output_length", type=int, default=64,
                        help="Maximum token length for generated commit message")
    parser.add_argument("--batch_size", type=int, default=4,
                        help="Batch size for generation")
    parser.add_argument("--device", type=str, default="cuda",
                        help="Device to run model on (cuda or cpu)")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for reproducibility")
    
    return parser.parse_args()

# ----------------------
# Main
# ----------------------
def main():
    args = parse_args()
    random.seed(args.seed)

    start_time = time.time()

    # Load model + tokenizer
    logger.info("Loading model and tokenizer from %s", args.model_dir)
    tokenizer = RobertaTokenizerFast.from_pretrained(args.model_dir)
    model = T5ForConditionalGeneration.from_pretrained(args.model_dir).to(args.device)
    model.eval()

    # Load dataset
    logger.info("Loading dataset from %s", args.input_file)
    df = pd.read_parquet(args.input_file)

    # Sample random rows
    logger.info("Sampling %d random rows", args.num_samples)
    df_sample = df.sample(n=args.num_samples, random_state=args.seed)

    results = []

    # Process in batches with progress bar
    logger.info("Starting predictions...")
    for i in tqdm(range(0, len(df_sample), args.batch_size), desc="Predicting"):
        batch = df_sample.iloc[i:i+args.batch_size]

        inputs = tokenizer(
            batch["diff_text"].tolist(),
            padding=True,
            truncation=True,
            max_length=args.max_input_length,
            return_tensors="pt"
        ).to(args.device)

        with torch.no_grad():
            outputs = model.generate(
                inputs["input_ids"],
                attention_mask=inputs["attention_mask"],
                max_length=args.max_output_length,
                num_beams=5,
                early_stopping=True
            )

        predictions = tokenizer.batch_decode(outputs, skip_special_tokens=True)

        for actual, pred in zip(batch["message"].tolist(), predictions):
            results.append({"actual": actual, "predicted": pred})

    # Save to CSV
    logger.info("Saving predictions to %s", args.output_file)
    results_df = pd.DataFrame(results)
    results_df.to_csv(args.output_file, index=False)

    # Show statistics
    logger.info("Calculating statistics...")
    exact_matches = sum(r["actual"].strip() == r["predicted"].strip() for r in results)
    exact_match_rate = exact_matches / len(results) * 100

    logger.info("Run completed in %.2f seconds", time.time() - start_time)
    logger.info("Total samples: %d", len(results))
    logger.info("Exact match: %d / %d (%.2f%%)", exact_matches, len(results), exact_match_rate)

    print("\n=== Prediction Summary ===")
    print(f"Samples processed: {len(results)}")
    print(f"Exact match rate: {exact_match_rate:.2f}%")
    print(f"Output saved to: {args.output_file}")

if __name__ == "__main__":
    import torch
    main()

