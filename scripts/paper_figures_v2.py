#!/usr/bin/env python3
"""
paper_figures_v2.py
=====================
Run ON TOUBKAL against master_4sources_split_ready.parquet.

Corrects the previous version's actual defects (not a style guess this time):
  - Serif font -> WRONG. Real ACS/JCIM/Nature Chem figures use a grotesque
    sans-serif (Arial/Helvetica). Switched to Liberation Sans (metric-
    compatible with Arial).
  - Soft, alpha-blended, edge-less bars -> WRONG. Real journal bar charts use
    FLAT solid fills with a thin BLACK outline (edgecolor='black',
    linewidth=0.8) -- this is what gives the "sharp, not soft" look. No
    alpha/transparency anywhere in this version.
  - Sparse tick marks -> real figures use dense minor ticks and tick marks
    pointing IN, not just major labels.
  - Loose default spacing -> tightened, no padding, bbox_inches='tight' with
    a small explicit pad.
  - Numbers on bars in a monospace/tabular-figure style, not default serif.

Same five figures, same real data as before; only the rendering discipline
changed.
"""
import argparse
import time

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import rcParams
from matplotlib.ticker import AutoMinorLocator, LogLocator

rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Liberation Sans", "Arial", "DejaVu Sans"],
    "font.size": 8.5,
    "axes.linewidth": 1.0,
    "axes.labelsize": 9.5,
    "axes.labelweight": "medium",
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "legend.fontsize": 7.5,
    "legend.frameon": False,
    "xtick.direction": "in",
    "ytick.direction": "in",
    "xtick.major.width": 1.0,
    "ytick.major.width": 1.0,
    "xtick.major.size": 4,
    "ytick.major.size": 4,
    "xtick.minor.width": 0.7,
    "ytick.minor.width": 0.7,
    "xtick.minor.size": 2,
    "ytick.minor.size": 2,
    "axes.edgecolor": "black",
    "text.color": "black",
    "axes.labelcolor": "black",
    "xtick.color": "black",
    "ytick.color": "black",
    "savefig.dpi": 600,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "axes.unicode_minus": False,
})

# Flat, high-contrast, print-safe colors -- solid, black-edged, no alpha
BLUE = "#2C5F8A"
RED = "#B5342A"
GRAY = "#8C8C8C"
GREEN = "#4C7A4C"
PURPLE = "#6B4C8A"
BLACK = "#000000"
EDGE = "black"
EDGE_LW = 0.8

SOURCE_COLORS = {"ord": BLUE, "uspto50k": RED, "uspto_mit": GREEN,
                 "uspto_stereo": PURPLE}

DOUBLE_COL = 7.0
SINGLE_COL = 3.35


def log(m):
    print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


def save(fig, outdir, name):
    fig.savefig(f"{outdir}/{name}.pdf", bbox_inches="tight", pad_inches=0.03)
    fig.savefig(f"{outdir}/{name}.png", dpi=600, bbox_inches="tight",
               pad_inches=0.03, facecolor="white")
    plt.close(fig)
    log(f"wrote {name}.pdf / .png")


def panel_label(ax, letter):
    ax.text(-0.15, 1.08, letter, transform=ax.transAxes, fontsize=12,
           fontweight="bold", va="top", ha="right", family="sans-serif")


def sharpen(ax, logy=False, logx=False):
    """Applies the black-frame, inward-tick, dense-minor-tick look that
    reads as a real journal figure rather than a default matplotlib chart."""
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_linewidth(1.0)
        spine.set_color("black")
    ax.tick_params(which="both", top=True, right=True)
    if not logy:
        ax.yaxis.set_minor_locator(AutoMinorLocator(2))
    if not logx:
        ax.xaxis.set_minor_locator(AutoMinorLocator(2))


# ============================================================
# FIGURE 1 — Corpus composition and deduplication
# ============================================================
def figure1(df, raw_counts, outdir):
    fig, axes = plt.subplots(1, 2, figsize=(DOUBLE_COL, 2.7))

    ax = axes[0]
    sources = ["ord", "uspto50k", "uspto_mit", "uspto_stereo"]
    labels = ["ORD", "USPTO-50K", "USPTO-MIT", "USPTO-\nSTEREO"]
    kept = [int((df["source"] == s).sum()) for s in sources]
    raw = [raw_counts[s] for s in sources]
    x = np.arange(len(sources))
    w = 0.36
    ax.bar(x - w/2, raw, w, color=GRAY, edgecolor=EDGE, linewidth=EDGE_LW,
          label="Raw", zorder=3)
    ax.bar(x + w/2, kept, w, color=RED, edgecolor=EDGE, linewidth=EDGE_LW,
          label="Deduplicated", zorder=3)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=7.5)
    ax.set_ylabel("Reactions")
    ax.set_yscale("log")
    ax.yaxis.set_major_locator(LogLocator(base=10, numticks=8))
    sharpen(ax, logy=True)
    ax.legend(loc="upper right", handlelength=1.2)
    panel_label(ax, "a")

    ax2 = axes[1]
    stages = ["Pass 1\nbefore", "Pass 1\nafter", "Pass 2\nbefore", "Pass 2\nafter"]
    vals = [2904722, 1823789, 2826756, 2523137]
    ax2.plot(range(len(stages)), vals, marker="o", markersize=5,
            markerfacecolor=BLUE, markeredgecolor="black", markeredgewidth=0.8,
            color="black", linewidth=1.2, zorder=3)
    for i, v in enumerate(vals):
        ax2.annotate(f"{v/1e6:.2f}M", (i, v), textcoords="offset points",
                    xytext=(0, 9), ha="center", fontsize=7.5,
                    family="monospace")
    ax2.set_xticks(range(len(stages)))
    ax2.set_xticklabels(stages, fontsize=7.5)
    ax2.set_ylabel("Reactions (\u00d710\u2076)")
    ax2.set_ylim(1.6e6, 3.1e6)
    ax2.yaxis.set_major_formatter(lambda v, _: f"{v/1e6:.1f}")
    sharpen(ax2)
    panel_label(ax2, "b")

    fig.tight_layout()
    save(fig, outdir, "figure1_corpus_composition")


# ============================================================
# FIGURE 2 — Atom-mapping quality
# ============================================================
def figure2(df, outdir):
    fig, axes = plt.subplots(1, 2, figsize=(DOUBLE_COL, 2.9))

    sources = ["ord", "uspto50k", "uspto_mit", "uspto_stereo"]
    order_preserved = [26.6, 66.5, 25.7, 27.3]
    matched_smiles = [97.0, 100.0, 99.0, 97.1]

    ax = axes[0]
    x = np.arange(len(sources))
    w = 0.36
    ax.bar(x - w/2, order_preserved, w, color=GRAY, edgecolor=EDGE,
          linewidth=EDGE_LW, label="Fragment order\npreserved", zorder=3)
    ax.bar(x + w/2, matched_smiles, w, color=BLUE, edgecolor=EDGE,
          linewidth=EDGE_LW, label="Matched by\nSMILES", zorder=3)
    ax.set_xticks(x)
    ax.set_xticklabels(["ORD", "50K", "MIT", "STEREO"], fontsize=8)
    ax.set_ylabel("Reactions (%)")
    ax.set_ylim(0, 112)
    sharpen(ax)
    ax.legend(loc="upper left", fontsize=6.8, handlelength=1.2)
    panel_label(ax, "a")

    ax2 = axes[1]
    for s in sources:
        vals = df.loc[df["source"] == s, "mapping_confidence"].dropna()
        if len(vals) > 20000:
            vals = vals.sample(20000, random_state=0)
        ax2.hist(vals, bins=40, histtype="step", linewidth=1.4,
                color=SOURCE_COLORS[s], label=s.upper().replace("_", "-"),
                density=True, zorder=3)
    ax2.axvline(0.5, color="black", linestyle="--", linewidth=1.0, zorder=2)
    ax2.set_xlabel("Mapping confidence")
    ax2.set_ylabel("Density")
    sharpen(ax2)
    ax2.legend(fontsize=6.3, loc="upper left", ncol=1)
    panel_label(ax2, "b")

    fig.tight_layout()
    save(fig, outdir, "figure2_mapping_quality")


# ============================================================
# FIGURE 3 — Role separation validation
# ============================================================
def figure3(outdir):
    fig, axes = plt.subplots(1, 2, figsize=(DOUBLE_COL, 2.7))

    ax = axes[0]
    cats = ["All reactions", "High-confidence\nonly"]
    agree = [93.7, 96.9]
    disagree = [100 - a for a in agree]
    x = np.arange(len(cats))
    ax.bar(x, agree, 0.5, color=BLUE, edgecolor=EDGE, linewidth=EDGE_LW,
          label="Agree", zorder=3)
    ax.bar(x, disagree, 0.5, bottom=agree, color=GRAY, edgecolor=EDGE,
          linewidth=EDGE_LW, label="Disagree", zorder=3)
    for i, a in enumerate(agree):
        ax.text(i, a / 2, f"{a:.1f}", ha="center", va="center",
               fontsize=8.5, color="white", fontweight="bold",
               family="monospace")
    ax.set_xticks(x)
    ax.set_xticklabels(cats, fontsize=8)
    ax.set_ylabel("Reactions (%)")
    ax.set_ylim(0, 100)
    sharpen(ax)
    ax.legend(loc="lower right", fontsize=7)
    panel_label(ax, "a")

    ax2 = axes[1]
    cats2 = ["Genuine reactant\n(59.1%)", "Spectator, hidden\n(40.9%)"]
    vals2 = [59.1, 40.9]
    y = np.arange(len(cats2))
    ax2.barh(y, vals2, color=[BLUE, RED], edgecolor=EDGE, linewidth=EDGE_LW,
            height=0.5, zorder=3)
    for i, v in enumerate(vals2):
        ax2.text(v + 1.5, i, f"{v:.1f}%", va="center", fontsize=8.5,
                family="monospace")
    ax2.set_yticks(y)
    ax2.set_yticklabels(cats2, fontsize=8)
    ax2.set_xlabel("Molecules in raw \u201creactants\u201d column (%)")
    ax2.set_xlim(0, 75)
    sharpen(ax2)
    panel_label(ax2, "b")

    fig.tight_layout()
    save(fig, outdir, "figure3_role_separation")


# ============================================================
# FIGURE 4 — Reaction center and stereo-change statistics
# ============================================================
def figure4(outdir):
    fig, axes = plt.subplots(1, 2, figsize=(DOUBLE_COL, 3.1))

    ax = axes[0]
    labels = ["Before fix", "After fix"]
    vals = [17.0, 1.0]
    ax.bar(labels, vals, color=[GRAY, RED], edgecolor=EDGE, linewidth=EDGE_LW,
          width=0.5, zorder=3)
    for i, v in enumerate(vals):
        ax.text(i, v + 0.5, f"{v:.1f}%", ha="center", fontsize=8.5,
               family="monospace")
    ax.set_ylabel("Empty reaction center (%)")
    ax.set_ylim(0, 20)
    sharpen(ax)
    panel_label(ax, "a")

    ax2 = axes[1]
    cats = ["retained", "created", "destroyed", "inverted", "resolved", "unresolved"]
    vals_3src = np.array([1024973, 141737, 48807, 21538, 16005, 14220])
    vals_stereo = np.array([533162, 81224, 33072, 18012, 12938, 10217])
    totals = vals_3src + vals_stereo
    order = np.argsort(totals)
    cats_sorted = [cats[i] for i in order]
    totals_sorted = totals[order]
    colors_bar = [BLUE if c == "retained" else RED for c in cats_sorted]
    y = np.arange(len(cats_sorted))
    ax2.barh(y, totals_sorted, color=colors_bar, edgecolor=EDGE,
            linewidth=EDGE_LW, height=0.6, zorder=3)
    ax2.set_yticks(y)
    ax2.set_yticklabels(cats_sorted, fontsize=8)
    ax2.set_xlabel("Stereocenters (count)")
    ax2.set_xscale("log")
    sharpen(ax2, logx=True)
    panel_label(ax2, "b")

    fig.tight_layout()
    save(fig, outdir, "figure4_reaction_center_stereo")


# ============================================================
# FIGURE 5 — Molecular size distribution
# ============================================================
def figure5(df, outdir):
    fig, ax = plt.subplots(figsize=(SINGLE_COL, 2.7))
    sub = df[["heavy_left", "heavy_product"]].dropna()
    sub = sub[(sub["heavy_left"] > 0) & (sub["heavy_product"] > 0)]
    if len(sub) > 150000:
        sub = sub.sample(150000, random_state=0)

    bins = np.logspace(0, np.log10(sub.values.max()), 40)
    ax.hist(sub["heavy_left"], bins=bins, histtype="stepfilled",
           facecolor=GRAY, edgecolor="black", linewidth=0.6, alpha=1.0,
           label="Left side (all)", zorder=2)
    ax.hist(sub["heavy_product"], bins=bins, histtype="step", linewidth=1.6,
           color=RED, label="Product", zorder=3)
    ax.set_xscale("log")
    ax.set_xlabel("Heavy atom count")
    ax.set_ylabel("Reactions")
    sharpen(ax, logx=True)
    ax.legend(loc="upper right")

    fig.tight_layout()
    save(fig, outdir, "figure5_molecular_size")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="infile", required=True)
    ap.add_argument("--outdir", default="paper_figures_v2")
    args = ap.parse_args()

    import os
    os.makedirs(args.outdir, exist_ok=True)

    log(f"loading {args.infile}")
    df = pd.read_parquet(args.infile)
    log(f"{len(df):,} rows loaded")

    raw_counts = {"ord": 2375686, "uspto50k": 50016,
                 "uspto_mit": 479020, "uspto_stereo": 1002967}

    figure1(df, raw_counts, args.outdir)
    figure2(df, args.outdir)
    figure3(args.outdir)
    figure4(args.outdir)
    figure5(df, args.outdir)

    log("done.")


if __name__ == "__main__":
    main()
