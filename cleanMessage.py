#!/usr/bin/env python3
"""
cleanCommit.py

Clean commit messages in commitpack_parted.parquet:
- Drop single-word messages without underscores.
- Keep only the first line of each commit message.
- Remove meaningless or auto-generated commit messages.
- Drop rows with empty/garbage messages.
- Save each chunk's cleaned and dropped messages to separate Parquet files.
"""

import pandas as pd
import re
import argparse
import time
import numpy as np
from tqdm import tqdm
from langdetect import detect, LangDetectException
from functools import lru_cache
from multiprocessing import Pool
import os

# ---------- Language detection ----------
@lru_cache(maxsize=100_000)
def detect_lang_safe(text: str) -> str:
    """Safe language detection with caching."""
    try:
        return detect(text)
    except LangDetectException:
        return "unknown"

# ---------- Cleaning helpers ----------
def clean_message(msg: str) -> str:
    if not isinstance(msg, str):
        return ""

    msg = msg.split('\n')[0].strip()  # Keep only first line

    # Remove metadata
    msg = re.sub(r"(git-svn-id:.*|Change-Id:.*|Signed-off-by:.*|Reviewed-by:.*|Co-authored-by:.*|Ticket:.*)",
                 "", msg, flags=re.IGNORECASE).strip()

    # Remove auto prefixes
    msg = re.sub(r"^[^:]+:\s+", "", msg, flags=re.IGNORECASE)

    # Remove bracketed/parenthesized noise
    msg = re.sub(r'\[.*?]', '', msg)
    msg = re.sub(r'\([#_\w]*\d+\)|\b\d{4,}\b', '', msg)

    # Remove emails, URLs, issue refs, hashes
    msg = re.sub(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b', '', msg)
    msg = re.sub(r'https?://[^\s]+|www\.[^\s]+', '', msg)
    msg = re.sub(r'\b(?:#|GH-|JIRA-|Issue\s*#?)[0-9]+\b', '', msg)
    msg = re.sub(r'\b[0-9a-f]{7,40}\b', '', msg)

    # Remove profanities
    msg = re.sub(r'\b(fuck|shit|damn|asshole|bastard|crap)\b', '', msg, flags=re.IGNORECASE)

    # Remove file paths
    msg = re.sub(r'\b[\w/.-]+\.(?:py|js|cpp|h|txt|md|yml|yaml|json|xml)\b', '', msg)

    # Remove timestamps/dates
    msg = re.sub(r'\b\d{4}-\d{2}-\d{2}(?:\s+\d{2}:\d{2}(?::\d{2})?)?\b', '', msg)
    msg = re.sub(r'\b(?:Mon|Tue|Wed|Thu|Fri|Sat|Sun)\s+[A-Za-z]+\s+\d{1,2}\s+\d{2}:\d{2}(?::\d{2})?\s+\d{4}\b', '', msg)

    # Remove inline code
    msg = re.sub(r'`[^`]*`', '', msg)

    # Remove excessive punctuation
    msg = re.sub(r'[.!?-]{2,}', '', msg)

    # Remove non-ASCII characters
    msg = re.sub(r'[^\x00-\x7F]', '', msg)

    return msg.strip()

def is_meaningless(msg: str) -> bool:
    """Heuristic filter for meaningless commit messages."""
    if not msg or msg.strip() == "":
        return True
    msg_lower = msg.lower().strip()

    # Too short or too few words
    if len(msg_lower.split()) <= 2 and "_" not in msg_lower:
        return True
    if len(msg_lower) <= 3:
        return True

    # Language detection (skip for very short strings)
    if len(msg_lower) > 8:
        lang = detect_lang_safe(msg_lower)
        if lang != "en":
            return True

    garbage_keywords = [
        "fixes", "updates", "upgrades", "typo", "typos", "minor",
        "test", "tests", "testing", "merge branch", "merge pull request",
        "merge remote-tracking branch", "merge", "remerge", "sync", "synced",
        "version bump", "bump version", "bump", "release", "released", "releases",
        "initial commit", "first commit", "auto-update", "autoupdate",
        "auto commit", "automated commit", "generated", "regenerate", "regen",
        "chore", "ci", "build", "builds", "build fix", "pipeline",
        "readme", "license", "copyright", "header", "style", "format",
        "whitespace", "indents", "reformat", "code style", "lint", "eslint",
        "prettier", "temp", "temporary", "sample", "examples", "default",
        "nan", "none", "meaningless", "message", "some shit", "changes",
        "stupid", "test commit", "more"
    ]
    for kw in garbage_keywords:
        if msg_lower == kw or msg_lower.startswith(kw):
            return True

    # Version numbers only
    if re.fullmatch(r"v?\d+(\.\d+)*", msg_lower):
        return True

    return False

def process_chunk(args):
    """Process a single chunk of the DataFrame and save to Parquet."""
    chunk, chunk_id, output_dir, dropped_dir = args
    print(f"[+] Process {chunk_id}: Starting {len(chunk):,} rows")
    start_time = time.time()

    tqdm.pandas(desc=f"Chunk {chunk_id}: Cleaning", position=chunk_id)
    chunk["message"] = chunk["message"].progress_map(clean_message)

    # Drop single-word commits
    mask_single = chunk["message"].map(lambda x: len(str(x).split()) == 1 and "_" not in str(x))
    df_single_dropped = chunk[mask_single].reset_index(drop=True)
    chunk = chunk[~mask_single].reset_index(drop=True)

    # Drop meaningless
    tqdm.pandas(desc=f"Chunk {chunk_id}: Filtering", position=chunk_id)
    mask_meaningless = chunk["message"].progress_map(is_meaningless)
    df_dropped = pd.concat([df_single_dropped, chunk[mask_meaningless]]).reset_index(drop=True)
    df_cleaned = chunk[~mask_meaningless].reset_index(drop=True)

    # Drop diff_text from dropped for smaller file
    if "diff_text" in df_dropped.columns:
        df_dropped = df_dropped.drop(columns=["diff_text"])

    # Save cleaned and dropped DataFrames to Parquet
    cleaned_path = os.path.join(output_dir, f"cleaned_chunk_{chunk_id}.parquet")
    dropped_path = os.path.join(dropped_dir, f"dropped_chunk_{chunk_id}.parquet")
    df_cleaned.to_parquet(cleaned_path, index=False)
    df_dropped.to_parquet(dropped_path, index=False)

    print(f"[+] Process {chunk_id}: Saved {len(df_cleaned):,} cleaned rows to {cleaned_path}")
    print(f"[+] Process {chunk_id}: Saved {len(df_dropped):,} dropped rows to {dropped_path}")
    print(f"[+] Process {chunk_id}: Done in {time.time() - start_time:.2f}s")
    return len(df_cleaned), len(df_dropped)

# ---------- Main ----------
def main():
    parser = argparse.ArgumentParser(description="Clean commit messages in commitpack_parted.parquet")
    parser.add_argument("-i", "--input", default="commitpack_parted.parquet", help="Input parquet file")
    parser.add_argument("-o", "--output", default="commitpack_cleaned", help="Output directory for cleaned Parquet files")
    parser.add_argument("-d", "--dropped", default="droppedMessages", help="Output directory for dropped Parquet files")
    parser.add_argument("-p", "--processes", type=int, default=6, help="Number of worker processes")
    args = parser.parse_args()

    start_time = time.time()
    print(f"[+] Starting at {time.ctime()}")

    print(f"[+] Reading {args.input} ...")
    df = pd.read_parquet(args.input)
    print(f"[+] Loaded {len(df):,} rows")

    # Create output directories
    output_dir = os.path.splitext(args.output)[0]  # Remove extension if provided
    dropped_dir = os.path.splitext(args.dropped)[0]  # Remove extension if provided
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(dropped_dir, exist_ok=True)

    # Split into chunks
    chunks = np.array_split(df, args.processes)
    print(f"[+] Split into {len(chunks)} chunks")

    # Parallel processing
    total_cleaned = 0
    total_dropped = 0
    with Pool(processes=args.processes) as pool:
        results = list(tqdm(pool.imap(process_chunk, [(chunk, i, output_dir, dropped_dir) for i, chunk in enumerate(chunks)]),
                            total=len(chunks), desc="Processing all chunks"))
        for cleaned_count, dropped_count in results:
            total_cleaned += cleaned_count
            total_dropped += dropped_count

    print("\n===== Cleaning Report =====")
    print(f"Total messages      : {len(df):,}")
    print(f"Dropped messages    : {total_dropped:,}")
    print(f"Remaining messages  : {total_cleaned:,}")
    print(f"Cleaned files saved : {output_dir}/cleaned_chunk_*.parquet")
    print(f"Dropped files saved : {dropped_dir}/dropped_chunk_*.parquet")
    print(f"Total time          : {time.time() - start_time:.2f}s")
    print("===========================")

if __name__ == "__main__":
    main()
