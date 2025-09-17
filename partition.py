#!/usr/bin/env python3
"""
partition.py

Generate a `diff_text` column that combines code additions, deletions,
modifications, and comments added/removed/modified.
"""

from __future__ import annotations
import argparse
import pandas as pd
import difflib
import re
from typing import List, Optional
from tqdm import tqdm
import os

# ---------- Utilities ----------
def normalize_line(
    line: str,
    lowercase: bool = True,
    collapse_whitespace: bool = True,
    strip: bool = True,
    strip_diff_prefix: bool = True
) -> str:
    if line is None:
        return ""
    s = str(line)
    if strip:
        s = s.strip()
    if strip_diff_prefix:
        if len(s) > 1 and (s[0] in "+-") and not (s.startswith("+++") or s.startswith("---")):
            s = s[1:].lstrip()
    if collapse_whitespace:
        s = s.replace("\t", " ")
        s = re.sub(r" {2,}", " ", s)
    if lowercase:
        s = s.lower()
    return s

def lines_from_text(text: Optional[str]) -> List[str]:
    if text is None:
        return []
    text = str(text).replace("\r\n", "\n")
    return text.split("\n")

def classify_line(line: str, in_multiline: bool) -> tuple[bool, bool]:
    """
    Classify a line as comment or code, with support for triple-quoted blocks (docstrings).
    Returns (is_comment, new_in_multiline_state).
    """
    stripped = line.strip()

    # Already inside a triple-quoted block
    if in_multiline:
        if stripped.endswith('"""') or stripped.endswith("'''"):
            return True, False  # closing block
        return True, True  # still inside block

    # Not inside: check if this starts a block
    if (stripped.startswith('"""') or stripped.startswith("'''")):
        # Single-line triple-quoted (e.g. """ comment """)
        if (stripped.endswith('"""') and len(stripped) > 3) or \
           (stripped.endswith("'''") and len(stripped) > 3):
            return True, False
        return True, True  # entering block

    # Regular single-line comments
    if stripped.startswith(("#", "//", "/*", "*", "--")):
        return True, False

    return False, False

# --- Diff computation with MODIFY detection ---
def compute_diff_text(
    old_text: Optional[str],
    new_text: Optional[str],
    *,
    lowercase: bool,
    collapse_whitespace: bool,
    strip_diff_prefix: bool,
    ignore_blank_lines: bool,
    min_line_length: int
) -> str:
    old_lines = [normalize_line(l,
                                lowercase=lowercase,
                                collapse_whitespace=collapse_whitespace,
                                strip_diff_prefix=strip_diff_prefix)
                 for l in lines_from_text(old_text)]
    new_lines = [normalize_line(l,
                                lowercase=lowercase,
                                collapse_whitespace=collapse_whitespace,
                                strip_diff_prefix=strip_diff_prefix)
                 for l in lines_from_text(new_text)]

    if ignore_blank_lines:
        old_lines = [l for l in old_lines if l and len(l) >= min_line_length]
        new_lines = [l for l in new_lines if l and len(l) >= min_line_length]

    sm = difflib.SequenceMatcher(None, old_lines, new_lines, autojunk=False)
    diff_parts = []

    old_in_multi = False
    new_in_multi = False

    for tag, i1, i2, j1, j2 in sm.get_opcodes():
        if tag == "equal":
            continue
        elif tag == "insert":
            for line in new_lines[j1:j2]:
                is_cmt, new_in_multi = classify_line(line, new_in_multi)
                if is_cmt:
                    diff_parts.append(f"<COMMENT_ADD> {line} </COMMENT_ADD>")
                else:
                    diff_parts.append(f"<ADD> {line} </ADD>")
        elif tag == "delete":
            for line in old_lines[i1:i2]:
                is_cmt, old_in_multi = classify_line(line, old_in_multi)
                if is_cmt:
                    diff_parts.append(f"<COMMENT_REMOVE> {line} </COMMENT_REMOVE>")
                else:
                    diff_parts.append(f"<REMOVE> {line} </REMOVE>")
        elif tag == "replace":
            old_chunk = old_lines[i1:i2]
            new_chunk = new_lines[j1:j2]
            for o, n in zip(old_chunk, new_chunk):
                o_cmt, old_in_multi = classify_line(o, old_in_multi)
                n_cmt, new_in_multi = classify_line(n, new_in_multi)
                if o_cmt or n_cmt:
                    diff_parts.append(f"<COMMENT_MODIFY> {o} → {n} </COMMENT_MODIFY>")
                else:
                    diff_parts.append(f"<MODIFY> {o} → {n} </MODIFY>")
            # handle leftovers
            if len(old_chunk) > len(new_chunk):
                for o in old_chunk[len(new_chunk):]:
                    o_cmt, old_in_multi = classify_line(o, old_in_multi)
                    if o_cmt:
                        diff_parts.append(f"<COMMENT_REMOVE> {o} </COMMENT_REMOVE>")
                    else:
                        diff_parts.append(f"<REMOVE> {o} </REMOVE>")
            elif len(new_chunk) > len(old_chunk):
                for n in new_chunk[len(old_chunk):]:
                    n_cmt, new_in_multi = classify_line(n, new_in_multi)
                    if n_cmt:
                        diff_parts.append(f"<COMMENT_ADD> {n} </COMMENT_ADD>")
                    else:
                        diff_parts.append(f"<ADD> {n} </ADD>")

    return "\n".join(diff_parts)

# ---------- Main processing ----------
def process_dataframe(
    df: pd.DataFrame,
    old_col: str,
    new_col: str,
    msg_col: str,
    *,
    lowercase: bool,
    collapse_whitespace: bool,
    strip_diff_prefix: bool,
    ignore_blank_lines: bool,
    min_line_length: int,
    drop_rows_where_both_empty: bool,
    show_progress: bool
) -> pd.DataFrame:

    if drop_rows_where_both_empty:
        mask_keep = ~((df[old_col].isna() | (df[old_col].astype(str).str.strip() == "")) &
                      (df[new_col].isna() | (df[new_col].astype(str).str.strip() == "")))
        df = df.loc[mask_keep].reset_index(drop=True)

    results = []
    iterator = df.itertuples(index=False)
    if show_progress:
        iterator = tqdm(list(df[[old_col, new_col]].itertuples(index=False)),
                        desc="computing diffs", total=len(df))

    for row in iterator:
        old_val, new_val = row[0], row[1]
        diff_text = compute_diff_text(
            old_val, new_val,
            lowercase=lowercase,
            collapse_whitespace=collapse_whitespace,
            strip_diff_prefix=strip_diff_prefix,
            ignore_blank_lines=ignore_blank_lines,
            min_line_length=min_line_length
        )
        results.append(diff_text)

    df_out = pd.DataFrame({
        "message": df[msg_col],
        "diff_text": results
    })
    return df_out

# ---------- CLI ----------
def parse_args():
    p = argparse.ArgumentParser(description="Generate diff_text with ADD/REMOVE/MODIFY and comment tracking (incl. docstrings)")
    p.add_argument("-i", "--input", required=True, help="Input parquet/csv file (parquet recommended).")
    p.add_argument("-o", "--output", required=True, help="Output path (parquet recommended).")
    p.add_argument("--old-col", default="old_contents", help="Column name for old content.")
    p.add_argument("--new-col", default="new_contents", help="Column name for new content.")
    p.add_argument("--msg-col", default="message", help="Column name for commit messages.")
    p.add_argument("--no-lowercase", action="store_true")
    p.add_argument("--no-collapse-ws", action="store_true")
    p.add_argument("--strip-diff-prefix", action="store_true")
    p.add_argument("--keep-blank-lines", action="store_true")
    p.add_argument("--min-line-length", type=int, default=0)
    p.add_argument("--drop-both-empty", action="store_true")
    p.add_argument("--no-progress", action="store_true")
    return p.parse_args()

def main():
    args = parse_args()

    print(f"[+] reading {args.input} ...")
    try:
        if args.input.lower().endswith(".parquet"):
            df = pd.read_parquet(args.input)
        elif args.input.lower().endswith(".csv"):
            df = pd.read_csv(args.input)
        else:
            df = pd.read_parquet(args.input)
    except Exception as e:
        raise RuntimeError(f"Failed to read input file {args.input}: {e}")

    print(f"[+] rows: {len(df):,}, columns: {list(df.columns)[:10]}{'...' if len(df.columns)>10 else ''}")

    df_out = process_dataframe(
        df,
        old_col=args.old_col,
        new_col=args.new_col,
        msg_col=args.msg_col,
        lowercase=not args.no_lowercase,
        collapse_whitespace=not args.no_collapse_ws,
        strip_diff_prefix=args.strip_diff_prefix,
        ignore_blank_lines=not args.keep_blank_lines,
        min_line_length=args.min_line_length,
        drop_rows_where_both_empty=args.drop_both_empty,
        show_progress=(not args.no_progress)
    )

    print(f"[+] writing to {args.output} ...")
    try:
        if args.output.lower().endswith(".parquet"):
            df_out.to_parquet(args.output, index=False)
        elif args.output.lower().endswith(".csv"):
            df_out.to_csv(args.output, index=False)
        else:
            df_out.to_parquet(args.output, index=False)
    except Exception as e:
        raise RuntimeError(f"Failed to write output file {args.output}: {e}")

    print("[+] done.")
    output_file = args.output
    output_file = args.output
    if os.path.exists(output_file):
        existing_df = pd.read_parquet(output_file)
        df_out = pd.concat([existing_df, df_out], ignore_index=True)
    df_out.to_parquet(output_file, index=False)

if __name__ == "__main__":
    main()
