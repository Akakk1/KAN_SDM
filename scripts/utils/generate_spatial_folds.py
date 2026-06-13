#!/usr/bin/env python3
"""
Python equivalent of Program/R/generate_spatial_folds.R
Generates spatially blocked folds using a simple but effective checkerboard / systematic grid.
Useful while blockCV/sf are not installed in the current env.

Usage examples:
  python Program/utils/generate_spatial_folds.py \
    --input Program/Data/ginkgo_occurrence.csv \
    --output Program/Data/folds_prelim.csv \
    --block_size 200000 \   # meters
    --k 5 \
    --selection checkerboard \
    --seed 42

Outputs (parallel to the R script):
  - the output csv (original cols + label + fold)
  - *_fold_summary.txt
  - *_map.png (scatter colored by fold)
"""
import argparse
import os
import sys

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def main():
    parser = argparse.ArgumentParser(description="Generate preliminary spatial CV folds (checkerboard-style).")
    parser.add_argument("-i", "--input", required=True, help="Input CSV with decimalLatitude, decimalLongitude (label optional)")
    parser.add_argument("-o", "--output", required=True, help="Output CSV path (will add fold column)")
    parser.add_argument("-b", "--block_size", type=int, default=200000, help="Block size in meters (default 200000 = 200km)")
    parser.add_argument("-k", "--k", type=int, default=5, help="Number of folds (default 5)")
    parser.add_argument("-s", "--selection", default="checkerboard", choices=["checkerboard", "systematic"],
                        help="Assignment method (default checkerboard)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()

    np.random.seed(args.seed)

    if not os.path.exists(args.input):
        print(f"ERROR: input file not found: {args.input}", file=sys.stderr)
        sys.exit(1)

    df = pd.read_csv(args.input)
    print(f"Loaded {len(df)} points from {args.input}")

    # Auto label=1 if missing (presence-only case)
    if "label" not in df.columns:
        df["label"] = 1
        print("No 'label' column — assuming all presences (label=1)")

    # Approximate degrees for block size (rough, good enough for prelim at China mid-lat)
    km_per_deg = 111.1
    block_deg = (args.block_size / 1000.0) / km_per_deg

    lon = df["decimalLongitude"].values
    lat = df["decimalLatitude"].values
    lon_min = float(np.min(lon))
    lat_min = float(np.min(lat))

    lon_bin = np.floor((lon - lon_min) / block_deg).astype(int)
    lat_bin = np.floor((lat - lat_min) / block_deg).astype(int)

    if args.selection == "checkerboard":
        fold = (lon_bin + lat_bin) % args.k
    else:
        # Simple systematic: row-major within the grid
        # (blockCV systematic is more sophisticated; this is a reasonable approx)
        max_lon_bins = int(np.max(lon_bin)) + 1
        fold = (lat_bin * max_lon_bins + lon_bin) % args.k

    df["fold"] = fold

    # Ensure a stable column order (put label + fold at end)
    base_cols = [c for c in df.columns if c not in ("label", "fold")]
    df = df[base_cols + ["label", "fold"]]

    out_dir = os.path.dirname(args.output) or "."
    os.makedirs(out_dir, exist_ok=True)

    df.to_csv(args.output, index=False)
    print(f"Folds saved to {args.output}")

    # Distribution
    print("\nFold distribution:")
    counts = df["fold"].value_counts().sort_index()
    print(counts)
    ctab = pd.crosstab(df["fold"], df["label"])
    print("\nCross tab (fold x label):")
    print(ctab)

    # Summary txt
    summary_file = args.output.replace(".csv", "_fold_summary.txt")
    with open(summary_file, "w") as f:
        f.write(f"Preliminary spatial folds (python grid, selection={args.selection})\n")
        f.write(f"Input: {args.input}\n")
        f.write(f"Block size (m): {args.block_size}\n")
        f.write(f"k: {args.k}\n")
        f.write(f"selection: {args.selection}\n\n")
        f.write("Fold distribution:\n")
        f.write(str(counts) + "\n\n")
        f.write("Cross tab (fold x label):\n")
        f.write(str(ctab) + "\n")
    print(f"Summary saved to {summary_file}")

    # PNG map
    map_file = args.output.replace(".csv", "_map.png")
    plt.figure(figsize=(9, 7))
    cmap = plt.get_cmap("tab10", args.k)
    for f in range(args.k):
        sub = df[df["fold"] == f]
        plt.scatter(sub["decimalLongitude"], sub["decimalLatitude"],
                    s=12, color=cmap(f), label=f"fold {f} (n={len(sub)})", alpha=0.75, linewidths=0)
    plt.xlabel("Longitude (°E)")
    plt.ylabel("Latitude (°N)")
    plt.title(f"Spatial folds (python {args.selection}, ~{args.block_size/1000:.0f}km blocks, k={args.k})\n"
              f"n={len(df)} points | seed={args.seed}")
    plt.legend(loc="best", fontsize=8)
    plt.grid(alpha=0.3, linestyle="--")
    plt.tight_layout()
    plt.savefig(map_file, dpi=150, bbox_inches="tight")
    print(f"Map saved to {map_file}")

    print("\nDone.")

if __name__ == "__main__":
    main()
