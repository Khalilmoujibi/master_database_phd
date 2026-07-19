#!/usr/bin/env python3
"""
clear_figures.py
=================
Run ON TOUBKAL against master_4sources_split_ready.parquet. Three figures,
each with ONE clear message, no stereochemistry framing:

  1. Corpus scale by source (bar) -- how big is each source, and how much of
     it survived deduplication?
  2. Mapping confidence distribution (histogram) -- how reliable is the atom
     mapping, overall, across the whole 2.5M-reaction master?
  3. Molecule size distribution, reactants vs products (overlaid histogram) --
     a simple, intuitive sanity check: are products generally smaller than
     the combined reactant pool? (yes, expected -- leaving groups, byproducts)

No time axis, no per-source correlation games -- straightforward descriptive
figures anyone in a room can read in five seconds.
"""
import argparse
import time

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

RED = "#D95335"
NAVY = "#1A237E"
GRAY = "#455A64"
LIGHTGRAY = "#ECEFF1"
DARKTEXT = "#263238"

plt.rcParams["text.color"] = DARKTEXT
plt.rcParams["axes.labelcolor"] = DARKTEXT
plt.rcParams["xtick.color"] = GRAY
plt.rcParams["ytick.color"] = GRAY


def log(m):
    print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


def fig1_corpus_scale(df, raw_counts, outpath):
    """Bar chart: raw count per source vs. count remaining after dedup."""
    sources = ["ord", "uspto50k", "uspto_mit", "uspto_stereo"]
    labels = ["ORD", "USPTO-50K", "USPTO-MIT", "USPTO-STEREO"]
    kept = [int((df["source"] == s).sum()) for s in sources]
    raw = [raw_counts[s] for s in sources]

    x = np.arange(len(sources))
    w = 0.35
    fig, ax = plt.subplots(figsize=(9, 5.5))
    ax.bar(x - w/2, raw, w, label="Raw (before dedup)", color=GRAY, zorder=3)
    ax.bar(x + w/2, kept, w, label="In master_v1 (after dedup)", color=RED, zorder=3)

    for xi, r, k in zip(x, raw, kept):
        ax.text(xi - w/2, r + 30000, f"{r:,}", ha="center", fontsize=9, color=DARKTEXT)
        ax.text(xi + w/2, k + 30000, f"{k:,}", ha="center", fontsize=9,
               fontweight="bold", color=DARKTEXT)

    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=11)
    ax.set_ylabel("Reactions", fontsize=12)
    ax.set_title("Corpus scale by source", fontsize=16, fontweight="bold",
                pad=14, color=DARKTEXT)
    ax.spines[["top", "right"]].set_visible(False)
    ax.yaxis.grid(True, color="#E0E0E0", linewidth=0.7, zorder=0)
    ax.set_axisbelow(True)
    ax.get_yaxis().set_major_formatter(
        plt.FuncFormatter(lambda v, _: f"{int(v/1e6)}M" if v >= 1e6 else f"{int(v/1e3)}k")
    )
    ax.legend(frameon=False, fontsize=10, loc="upper right")
    fig.text(0.5, -0.02, "Message: 2.5M unique reactions in the final master, "
            "drawn from 3.9M raw reactions across four public sources.",
            ha="center", fontsize=10.5, style="italic", color=GRAY)
    fig.tight_layout()
    fig.savefig(outpath, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    log(f"wrote {outpath}")


def fig2_mapping_confidence(df, outpath):
    """Histogram of mapping_confidence across the whole master."""
    conf = df["mapping_confidence"].dropna()
    mean_c = conf.mean()
    median_c = conf.median()
    pct_high = (conf >= 0.5).mean() * 100

    fig, ax = plt.subplots(figsize=(9, 5.5))
    ax.hist(conf, bins=50, color=NAVY, alpha=0.85, zorder=3)
    ax.axvline(mean_c, color=RED, linestyle="--", linewidth=2, zorder=5,
              label=f"Mean = {mean_c:.2f}")
    ax.axvline(0.5, color=GRAY, linestyle=":", linewidth=1.8, zorder=4,
              label="\u03c4 = 0.5 (quality threshold)")

    ax.set_xlabel("RXNMapper confidence score", fontsize=12)
    ax.set_ylabel("Reactions", fontsize=12)
    ax.set_title("Atom-mapping confidence \u2014 full master (2.5M reactions)",
                fontsize=16, fontweight="bold", pad=14, color=DARKTEXT)
    ax.spines[["top", "right"]].set_visible(False)
    ax.yaxis.grid(True, color="#E0E0E0", linewidth=0.7, zorder=0)
    ax.set_axisbelow(True)
    ax.legend(frameon=False, fontsize=10, loc="upper left")
    fig.text(0.5, -0.02,
            f"Message: {pct_high:.0f}% of reactions map with confidence \u2265 0.5; "
            f"mean confidence {mean_c:.2f} across the entire corpus.",
            ha="center", fontsize=10.5, style="italic", color=GRAY)
    fig.tight_layout()
    fig.savefig(outpath, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    log(f"wrote {outpath}  (mean={mean_c:.3f}, median={median_c:.3f}, "
       f"%>=0.5={pct_high:.1f}%)")


def fig3_size_distribution(df, outpath, sample_n=200000):
    """Overlaid histogram: heavy_left vs heavy_product, log scale."""
    sub = df[["heavy_left", "heavy_product"]].dropna()
    sub = sub[(sub["heavy_left"] > 0) & (sub["heavy_product"] > 0)]
    if len(sub) > sample_n:
        sub = sub.sample(sample_n, random_state=42)

    left_med = sub["heavy_left"].median()
    prod_med = sub["heavy_product"].median()

    fig, ax = plt.subplots(figsize=(9, 5.5))
    bins = np.logspace(np.log10(1), np.log10(sub[["heavy_left","heavy_product"]].values.max()), 50)
    ax.hist(sub["heavy_left"], bins=bins, alpha=0.55, color=GRAY,
           label=f"Left side, all molecules (median={left_med:.0f})", zorder=3)
    ax.hist(sub["heavy_product"], bins=bins, alpha=0.7, color=RED,
           label=f"Product (median={prod_med:.0f})", zorder=4)
    ax.set_xscale("log")
    ax.set_xlabel("Heavy atoms (log scale)", fontsize=12)
    ax.set_ylabel("Reactions", fontsize=12)
    ax.set_title("Molecule size \u2014 left side vs. product", fontsize=16,
                fontweight="bold", pad=14, color=DARKTEXT)
    ax.spines[["top", "right"]].set_visible(False)
    ax.yaxis.grid(True, color="#E0E0E0", linewidth=0.7, zorder=0)
    ax.set_axisbelow(True)
    ax.legend(frameon=False, fontsize=10, loc="upper right")
    fig.text(0.5, -0.02,
            "Message: the left-side pool (reactants + spectators) skews larger "
            "than the product \u2014 expected, since spectators add atoms that "
            "never reach the product.",
            ha="center", fontsize=10.5, style="italic", color=GRAY)
    fig.tight_layout()
    fig.savefig(outpath, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    log(f"wrote {outpath}  (left_median={left_med:.1f}, prod_median={prod_med:.1f})")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="infile", required=True)
    ap.add_argument("--outdir", default="figures")
    args = ap.parse_args()

    import os
    os.makedirs(args.outdir, exist_ok=True)

    log(f"loading {args.infile}")
    df = pd.read_parquet(args.infile)
    log(f"{len(df):,} rows loaded")

    # raw counts before dedup, as documented in the data card (real numbers,
    # not recomputed here since the raw per-source files are no longer all
    # loaded together at this stage)
    raw_counts = {
        "ord": 2375686,
        "uspto50k": 50016,
        "uspto_mit": 479020,
        "uspto_stereo": 1002967,
    }

    fig1_corpus_scale(df, raw_counts, f"{args.outdir}/fig1_corpus_scale.png")
    fig2_mapping_confidence(df, f"{args.outdir}/fig2_mapping_confidence.png")
    fig3_size_distribution(df, f"{args.outdir}/fig3_size_distribution.png")

    log("done.")


if __name__ == "__main__":
    main()
