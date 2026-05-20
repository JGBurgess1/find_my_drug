"""
Extract the top-scoring (most negative docking score) compounds from a large
predictions CSV, reading in chunks to avoid memory overflow on ~69 GB files.

The running buffer strategy: after each 1M-row chunk, merge with the current
top-N buffer and call nsmallest() so we never hold more than 2M rows in RAM.

Usage:
    python 01_extract_top_hits.py \
        --input  predictions.csv \
        --output top_1M_hits.csv \
        --top-n  1000000
"""

import argparse
import os
import sys

import pandas as pd

SCORE_COL  = "predicted_docking_score"
CHUNK_SIZE = 1_000_000


def extract_top_hits(input_path: str, output_path: str, top_n: int) -> None:
    if not os.path.exists(input_path):
        raise FileNotFoundError(f"Input file not found: {input_path}")

    file_size_gb = os.path.getsize(input_path) / 1e9
    print(f"Input      : {input_path}  ({file_size_gb:.1f} GB)")
    print(f"Top-N      : {top_n:,}")
    print(f"Chunk size : {CHUNK_SIZE:,}\n")

    reader = pd.read_csv(input_path, chunksize=CHUNK_SIZE)

    top_buffer      = pd.DataFrame()
    chunks_done     = 0
    total_rows_seen = 0

    for chunk in reader:
        chunks_done     += 1
        total_rows_seen += len(chunk)

        chunk = chunk.dropna(subset=[SCORE_COL])

        top_buffer = pd.concat([top_buffer, chunk], ignore_index=True)

        # Keep only the top_n most negative scores to bound memory usage
        if len(top_buffer) > top_n:
            top_buffer = top_buffer.nsmallest(top_n, SCORE_COL)

        current_cutoff = top_buffer[SCORE_COL].max() if len(top_buffer) else float("inf")
        print(
            f"  chunk {chunks_done:>4}  |  rows: {len(chunk):>10,}  |  "
            f"buffer: {len(top_buffer):>9,}  |  cutoff: {current_cutoff:>8.4f}"
        )

        del chunk

    print(f"\nProcessed {total_rows_seen:,} rows across {chunks_done} chunks.")
    print(f"Writing {len(top_buffer):,} hits to {output_path} ...")

    top_buffer = top_buffer.sort_values(SCORE_COL).reset_index(drop=True)
    top_buffer.to_csv(output_path, index=False)

    best   = top_buffer[SCORE_COL].min()
    cutoff = top_buffer[SCORE_COL].max()
    mean   = top_buffer[SCORE_COL].mean()
    print(f"\nScore statistics (top {len(top_buffer):,}):")
    print(f"  Best   (most negative) : {best:.4f} kcal/mol")
    print(f"  Cutoff (worst kept)    : {cutoff:.4f} kcal/mol")
    print(f"  Mean                   : {mean:.4f} kcal/mol")
    print(f"\nOutput written to: {output_path}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Extract top-N scoring compounds from a large predictions CSV"
    )
    parser.add_argument("--input",  "-i", required=True,
                        help="Path to predictions CSV (~69 GB)")
    parser.add_argument("--output", "-o", default="top_1M_hits.csv",
                        help="Output CSV path (default: top_1M_hits.csv)")
    parser.add_argument("--top-n",  "-n", type=int, default=1_000_000,
                        help="Number of top hits to keep (default: 1,000,000)")
    args = parser.parse_args()

    extract_top_hits(args.input, args.output, args.top_n)


if __name__ == "__main__":
    main()
