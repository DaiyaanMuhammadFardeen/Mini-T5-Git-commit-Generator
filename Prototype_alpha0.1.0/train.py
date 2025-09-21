#!/usr/bin/env python3
"""
train_t5_vram_saturate_checkpoints.py

Train a T5-like encoder-decoder from scratch on git diffs -> commit messages.
Memory-safe version optimized for 16GB RAM + 16GB swap.
Uses RobertaTokenizerFast for tokenization.
Now with checkpointing every N steps/samples.
"""

import os
os.environ["TOKENIZERS_PARALLELISM"] = "true"
os.environ["HF_DATASETS_DISABLE_PROGRESS_BARS"] = "1"
os.environ["HF_DATASETS_CACHE"] = "./.hf_cache"

import sys
import argparse
import random
import math
import logging
import warnings
import pprint
import gc
import multiprocessing

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
from datasets import Dataset, concatenate_datasets, load_from_disk
from transformers import (
    T5Config,
    T5ForConditionalGeneration,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    DataCollatorForSeq2Seq,
    RobertaTokenizerFast,
    TrainerCallback   ### ADDED
)
from tokenizers import ByteLevelBPETokenizer

# ---------------- logging & warnings ----------------
warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger(__name__)

# ---------------- CFG ----------------
CFG = {
    # data
    "train_sample_size": -1,
    "diff_col": "diff_text",
    "msg_col": "message",
    "random_seed": 42,

    # tokenizer
    "tokenizer_name_or_path": "roberta-base",
    "tokenizer_num_threads": multiprocessing.cpu_count(),

    # model
    "d_model": 256,
    "d_ff": 512,
    "vocab_size": 32000,
    "encoder_layers": 6,
    "decoder_layers": 6,
    "num_heads": 4,
    "dropout_rate": 0.1,

    # sequence lengths
    "max_source_length": 256,
    "max_target_length": 32,

    # training
    "per_device_train_batch_size": 32,
    "gradient_accumulation_steps": 64,
    "num_train_epochs": 3,
    "learning_rate": 1e-4,
    "weight_decay": 0.01,
    "warmup_steps": 400,
    "fp16": True,
    "save_total_limit": 3,

    # dataloader
    "dataloader_num_workers": 4,
    "dataloader_pin_memory": True,

    # logging / checkpointing
    "logging_steps": 100,
    "save_steps": 50000,   
    "checkpoint_every_n_samples": 100_000, 
    "output_dir": "t5_vram_saturated_experiment",

    # generation
    "generation_num_beams": 4,
    "max_gen_length": 32,

    # preprocessing
    "preproc_num_proc": 6,
    "preproc_batch_size": 2000,

    # dataset cache
    "dataset_cache_dir": "dataset_cache"
}

# ---------------- Utilities ----------------
def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    logger.info(f"Random seed set to {seed}")

# ---------------- Data loading ----------------
def load_and_sample(data_path: str, data_format: str, diff_col: str, msg_col: str, sample_size: int):
    logger.info(f"Loading data from: {data_path} (format={data_format})")
    if data_format.lower() == "parquet":
        df = pd.read_parquet(data_path)
    elif data_format.lower() == "csv":
        df = pd.read_csv(data_path)
    else:
        raise ValueError("data_format must be 'parquet' or 'csv'")

    if diff_col not in df.columns or msg_col not in df.columns:
        raise ValueError(f"Columns not found. Available columns: {list(df.columns)}")

    df = df[[diff_col, msg_col]].dropna().astype(str)

    if sample_size > 0 and sample_size < len(df):
        df_sample = df.sample(n=sample_size, random_state=CFG["random_seed"]).reset_index(drop=True)
    else:
        df_sample = df.reset_index(drop=True)

    logger.info(f"Dataset loaded: total_rows={len(df)}, sampled={len(df_sample)}")
    del df
    gc.collect()
    return df_sample

# ---------------- Dataset preprocessing ----------------
def train_tokenizer_from_scratch(df, cfg, save_dir="my_tokenizer"):
    logger.info("Training new tokenizer from scratch ...")
    os.makedirs(save_dir, exist_ok=True)

    # dump texts for tokenizer training
    text_file = os.path.join(save_dir, "train_texts.txt")
    with open(text_file, "w", encoding="utf-8") as f:
        for diff, msg in zip(df[cfg["diff_col"]], df[cfg["msg_col"]]):
            f.write(str(diff) + "\n")
            f.write(str(msg) + "\n")

    tokenizer = ByteLevelBPETokenizer()
    tokenizer.train(
        files=[text_file],
        vocab_size=cfg["vocab_size"],
        min_frequency=2,
        special_tokens=["<s>", "<pad>", "</s>", "<unk>", "<mask>"]
    )

    tokenizer.save_model(save_dir)
    logger.info(f"Tokenizer trained & saved to {save_dir}")

    # wrap as HF fast tokenizer
    hf_tokenizer = RobertaTokenizerFast.from_pretrained(save_dir)
    return hf_tokenizer

def pandas_to_dataset_memory_safe(df, batch_size=10000):
    chunks = []
    num_chunks = math.ceil(len(df) / batch_size)
    for i in tqdm(range(num_chunks), desc="Converting pandas -> Dataset"):
        start = i * batch_size
        end = min((i + 1) * batch_size, len(df))
        chunk_df = df.iloc[start:end]
        chunks.append(Dataset.from_pandas(chunk_df))
        del chunk_df
        gc.collect()
    ds = concatenate_datasets(chunks) if len(chunks) > 1 else chunks[0]
    del chunks
    gc.collect()
    return ds

def preprocess_function(examples, tokenizer, cfg):
    inputs = examples[cfg["diff_col"]]
    targets = examples[cfg["msg_col"]]
    model_inputs = tokenizer(
        inputs, max_length=cfg["max_source_length"], padding="max_length", truncation=True
    )
    labels = tokenizer(
        text_target=targets, max_length=cfg["max_target_length"], padding="max_length", truncation=True
    )
    model_inputs["labels"] = labels["input_ids"]
    return model_inputs

def preprocess_dataset_memory_safe(df, tokenizer, cfg):
    ds = pandas_to_dataset_memory_safe(df, batch_size=10000)
    batch_size = cfg["preproc_batch_size"]
    num_batches = math.ceil(len(ds) / batch_size)
    all_results = []

    logger.info(f"Tokenizing dataset in {num_batches} batches ...")
    for i in tqdm(range(num_batches), desc="Tokenizing batches"):
        start = i * batch_size
        end = min((i + 1) * batch_size, len(ds))
        batch = ds.select(range(start, end))

        processed_batch = batch.map(
            lambda examples: preprocess_function(examples, tokenizer, cfg),
            batched=True,
            batch_size=batch_size,
            num_proc=1,
            desc=f"Tokenizing batch {i+1}/{num_batches}"
        )

        all_results.append(processed_batch)
        del batch, processed_batch
        gc.collect()

    ds = concatenate_datasets(all_results) if len(all_results) > 1 else all_results[0]
    ds.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])
    del all_results
    gc.collect()
    return ds

def load_or_process_dataset(df, tokenizer, cfg):
    cache_path = os.path.join(cfg["output_dir"], cfg.get("dataset_cache_dir", "dataset_cache"))
    os.makedirs(cache_path, exist_ok=True)
    if os.path.exists(os.path.join(cache_path, "dataset.arrow")):
        logger.info(f"Loading preprocessed dataset from {cache_path} ...")
        ds = load_from_disk(cache_path)
    else:
        logger.info("Preprocessing dataset from scratch ...")
        ds = preprocess_dataset_memory_safe(df, tokenizer, cfg)
        logger.info(f"Saving preprocessed dataset to {cache_path} ...")
        ds.save_to_disk(cache_path)
        gc.collect()
    return ds

# ---------------- Sample checkpoint callback ----------------
class SampleCheckpointCallback(TrainerCallback):  ### ADDED
    def __init__(self, checkpoint_every_n_samples, output_dir):
        self.checkpoint_every_n_samples = checkpoint_every_n_samples
        self.output_dir = output_dir
        self.samples_seen = 0
        self.next_checkpoint = checkpoint_every_n_samples

    def on_step_end(self, args, state, control, model=None, tokenizer=None, **kwargs):
        batch_size = args.per_device_train_batch_size * max(1, torch.cuda.device_count())
        self.samples_seen += batch_size * args.gradient_accumulation_steps
        if self.samples_seen >= self.next_checkpoint:
            ckpt_dir = os.path.join(self.output_dir, f"checkpoint_samples_{self.samples_seen}")
            os.makedirs(ckpt_dir, exist_ok=True)
            model.save_pretrained(ckpt_dir)
            if tokenizer:
                tokenizer.save_pretrained(ckpt_dir)
            logger.info(f"[SampleCheckpoint] Saved at {self.samples_seen} samples -> {ckpt_dir}")
            self.next_checkpoint += self.checkpoint_every_n_samples
        return control

# ---------------- Safe save ----------------
def safe_save(trainer, tokenizer, output_dir):
    try:
        trainer.save_model(output_dir)
        tokenizer.save_pretrained(output_dir)
        logger.info("Saved model & tokenizer successfully.")
    except Exception as e:
        logger.exception("Failed to save model/tokenizer: %s", e)

# ---------------- Main run ----------------
def run(args):
    CFG.update({
        "diff_col": args.diff_col,
        "msg_col": args.msg_col,
        "train_sample_size": args.train_sample_size,
        "output_dir": args.output_dir
    })
    pprint.pprint(CFG)

    set_seed(CFG["random_seed"])
    df_sample = load_and_sample(args.data_path, args.data_format, CFG["diff_col"], CFG["msg_col"], CFG["train_sample_size"])

    tokenizer = RobertaTokenizerFast.from_pretrained(CFG["tokenizer_name_or_path"])
    try:
        tokenizer = RobertaTokenizerFast.from_pretrained(CFG["tokenizer_name_or_path"])
        logger.info(f"Loaded pretrained tokenizer: {CFG['tokenizer_name_or_path']}")
    except Exception:
        logger.warning("No pretrained tokenizer found, falling back to training from scratch.")
        tokenizer = train_tokenizer_from_scratch(df_sample, CFG, save_dir=os.path.join(CFG["output_dir"], "tokenizer"))

    gc.collect()

    model_cfg = T5Config(
        vocab_size=tokenizer.vocab_size,
        d_model=CFG["d_model"],
        d_ff=CFG["d_ff"],
        num_heads=CFG["num_heads"],
        num_layers=CFG["encoder_layers"],
        num_decoder_layers=CFG["decoder_layers"],
        dropout_rate=CFG["dropout_rate"],
    )
    model = T5ForConditionalGeneration(model_cfg)
    model.resize_token_embeddings(tokenizer.vocab_size)
    model.config.decoder_start_token_id = tokenizer.pad_token_id
    model.gradient_checkpointing_enable()
    model.config.use_cache = False

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    logger.info(f"Model created with {sum(p.numel() for p in model.parameters()):,} parameters on {device}")

    ds_train = load_or_process_dataset(df_sample, tokenizer, CFG)
    del df_sample
    gc.collect()

    data_collator = DataCollatorForSeq2Seq(tokenizer, model=model, padding=True, label_pad_token_id=-100)
    training_args = Seq2SeqTrainingArguments(
        output_dir=CFG["output_dir"],
        per_device_train_batch_size=CFG["per_device_train_batch_size"],
        gradient_accumulation_steps=CFG["gradient_accumulation_steps"],
        num_train_epochs=CFG["num_train_epochs"],
        learning_rate=CFG["learning_rate"],
        weight_decay=CFG["weight_decay"],
        warmup_steps=CFG["warmup_steps"],
        logging_steps=CFG["logging_steps"],
        save_total_limit=CFG["save_total_limit"],
        fp16=CFG["fp16"],
        predict_with_generate=False,
        eval_strategy="no",
        save_strategy="steps",   ### CHANGED (was "epoch")
        save_steps=CFG["save_steps"],   ### ADDED
        dataloader_num_workers=CFG["dataloader_num_workers"],
        dataloader_pin_memory=CFG["dataloader_pin_memory"],
        remove_unused_columns=True,
        report_to="none"
    )

    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=ds_train,
        tokenizer=tokenizer,
        data_collator=data_collator,
        callbacks=[SampleCheckpointCallback(CFG["checkpoint_every_n_samples"], CFG["output_dir"])]  ### ADDED
    )

    try:
        logger.info("Starting training ... press Ctrl+C to interrupt and save progress.")

        # --- PATCH START ---
        last_checkpoint = None
        if os.path.isdir(CFG["output_dir"]):
            checkpoints = [d for d in os.listdir(CFG["output_dir"]) if d.startswith("checkpoint")]
            if checkpoints:
                checkpoints = sorted(checkpoints, key=lambda x: int(x.split("-")[-1]))
                last_checkpoint = os.path.join(CFG["output_dir"], checkpoints[-1])

        if last_checkpoint:
            logger.info(f"Resuming training from checkpoint: {last_checkpoint}")
            trainer.train(resume_from_checkpoint=last_checkpoint)
        else:
            logger.info("No checkpoint found, starting fresh training.")
            trainer.train()
    except KeyboardInterrupt:
        logger.warning("KeyboardInterrupt detected — saving progress.")
        safe_save(trainer, tokenizer, CFG["output_dir"])
        sys.exit(0)
    except Exception as e:
        logger.exception("Training failed: %s", e)
        safe_save(trainer, tokenizer, CFG["output_dir"])
        sys.exit(1)

    safe_save(trainer, tokenizer, CFG["output_dir"])
    logger.info("Training completed successfully.")

# ---------------- CLI ----------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, default="commitpack_cleaned.parquet")
    parser.add_argument("--data_format", type=str, default="parquet", choices=["parquet", "csv"])
    parser.add_argument("--diff_col", type=str, default="diff_text")
    parser.add_argument("--msg_col", type=str, default="message")
    parser.add_argument("--train_sample_size", type=int, default=CFG["train_sample_size"])
    parser.add_argument("--output_dir", type=str, default=CFG["output_dir"])
    args = parser.parse_args()
    run(args)

