#!/usr/bin/env python3
"""
corr_figures.py
================
Run ON TOUBKAL against master_4sources_split_ready.parquet (or any master
parquet with these columns). Produces two-panel scatter+trendline figures in
the "LEVELS r=... / MONTHLY CHANGES r=..." style: colored scatter, dashed
trend line, Pearson r in the panel title.

USAGE (on Toubkal, login node is fine -- pure pandas/matplotlib/scipy, no GPU):
    python corr_figures.py --in master_4sources_split_ready.parquet --outdir figures

Three real correlations computed from your actual master, not invented:
  1. heavy_left vs heavy_product        (does reactant mass predict product mass?)
  2. num_stereocenters vs reaction_center size   (do more stereocenters mean a
     bigger reaction center?)
  3. mapping_confidence vs n_salts_stripped      (does more salt-stripping
     correlate with mapper confidence?)

Each is genuinely two-panel: LEFT = raw values ("LEVELS"), RIGHT = deviation
from each source's own mean ("DEVIATION FROM SOURCE MEAN") -- the closest
faithful analogue to "monthly changes" for cross-sectional reaction data
(there is no time axis in this master; see data card §8). This is stated
explicitly in each figure's right-panel subtitle so it is never confused with
a real time-series differencing.
"""
import argparse
import time

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
from scipy import stats

RED = "#D95335"
NAVY = "#1A237E"
GRAY = "#455A64"
DARKTEXT = "#263238"
GREEN = "#2E7D32"

# palette used for the LEVELS panel's per-source color gradient (like the
# yellow/orange/purple gradient in the reference image)
SOURCE_COLORS = {
    "ord": "#8E24AA",
    "uspto50k": "#F9A825",
    "uspto_mit": "#EF6C00",
    "uspto_stereo": "#5E35B1",
}


def log(m):
    print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


def setup_fonts():
    for f in fm.findSystemFonts(fontpaths=["/usr/share/fonts/truetype/crosextra"]):
        try:
            fm.fontManager.addfont(f)
        except Exception:
            pass
    plt.rcParams["font.family"] = "Carlito"
    plt.rcParams["text.color"] = DARKTEXT
    plt.rcParams["axes.labelcolor"] = DARKTEXT
    plt.rcParams["xtick.color"] = GRAY
    plt.rcParams["ytick.color"] = GRAY


def two_panel_correlation(df, xcol, ycol, xlabel, ylabel, title_stub, outpath,
                          logx=False, logy=False, sample_n=4000):
    """LEFT: levels scatter + trend + Pearson r (colored by source).
    RIGHT: each point's deviation from its OWN SOURCE's mean, same two
    columns -- the cross-sectional analogue of 'changes', since this master
    has no time axis (documented explicitly, not silently substituted)."""
    sub = df[[xcol, ycol, "source"]].dropna()
    if logx:
        sub = sub[sub[xcol] > 0]
    if logy:
        sub = sub[sub[ycol] > 0]
    if len(sub) > sample_n:
        sub = sub.sample(sample_n, random_state=42)

    x = sub[xcol].values.astype(float)
    y = sub[ycol].values.astype(float)
    xp = np.log10(x) if logx else x
    yp = np.log10(y) if logy else y

    r_levels, _ = stats.pearsonr(xp, yp)

    # deviation-from-source-mean (cross-sectional "changes" analogue)
    dev = sub.copy()
    dev["_x"] = xp
    dev["_y"] = yp
    dev["_xdev"] = dev.groupby("source")["_x"].transform(lambda s: s - s.mean())
    dev["_ydev"] = dev.groupby("source")["_y"].transform(lambda s: s - s.mean())
    r_dev, _ = stats.pearsonr(dev["_xdev"], dev["_ydev"])

    fig, axes = plt.subplots(1, 2, figsize=(13, 5.2))

    # ---- LEFT: levels ----
    ax = axes[0]
    for src, sub_src in sub.groupby("source"):
        xs = np.log10(sub_src[xcol]) if logx else sub_src[xcol]
        ys = np.log10(sub_src[ycol]) if logy else sub_src[ycol]
        ax.scatter(sub_src[xcol] if not logx else xs,
                  sub_src[ycol] if not logy else ys,
                  s=14, alpha=0.55, color=SOURCE_COLORS.get(src, GRAY),
                  label=src, linewidths=0)
    z = np.polyfit(xp, yp, 1)
    xs_line = np.linspace(xp.min(), xp.max(), 50)
    ys_line = np.polyval(z, xs_line)
    ax.plot((10**xs_line if logx else xs_line),
           (10**ys_line if logy else ys_line),
           "--", color=NAVY, linewidth=2.2, zorder=5)
    if logx:
        ax.set_xscale("log")
    if logy:
        ax.set_yscale("log")
    ax.set_title(f"LEVELS   r = {r_levels:+.2f}", fontsize=14, fontweight="bold",
                color=GREEN if abs(r_levels) > 0.3 else DARKTEXT, pad=10)
    ax.set_xlabel(xlabel, fontsize=11)
    ax.set_ylabel(ylabel, fontsize=11)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(frameon=False, fontsize=8, loc="best", markerscale=2)

    # ---- RIGHT: deviation from source mean ----
    ax2 = axes[1]
    ax2.axhline(0, color="#BDBDBD", linewidth=0.8, zorder=1)
    ax2.axvline(0, color="#BDBDBD", linewidth=0.8, zorder=1)
    ax2.scatter(dev["_xdev"], dev["_ydev"], s=14, alpha=0.55, color=RED,
               linewidths=0, zorder=3)
    z2 = np.polyfit(dev["_xdev"], dev["_ydev"], 1)
    xs2 = np.linspace(dev["_xdev"].min(), dev["_xdev"].max(), 50)
    ax2.plot(xs2, np.polyval(z2, xs2), "--", color=NAVY, linewidth=2.2, zorder=5)
    ax2.set_title(f"DEVIATION FROM SOURCE MEAN   r = {r_dev:+.2f}", fontsize=14,
                 fontweight="bold", color=RED, pad=10)
    ax2.set_xlabel(f"\u0394 {xlabel} (vs.\\ source mean)", fontsize=11)
    ax2.set_ylabel(f"\u0394 {ylabel} (vs.\\ source mean)", fontsize=11)
    ax2.spines[["top", "right"]].set_visible(False)

    fig.suptitle(title_stub, fontsize=12, color=GRAY, y=1.02)
    fig.tight_layout()
    fig.savefig(outpath, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    log(f"wrote {outpath}  (n={len(sub):,}, r_levels={r_levels:+.3f}, r_dev={r_dev:+.3f})")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="infile", required=True)
    ap.add_argument("--outdir", default="figures")
    args = ap.parse_args()

    import os
    os.makedirs(args.outdir, exist_ok=True)
    setup_fonts()

    log(f"loading {args.infile}")
    df = pd.read_parquet(args.infile)
    log(f"{len(df):,} rows loaded")

    # --- derive reaction_center SIZE (count) from the stored pipe-string ---
    def rc_size(x):
        if x is None or (isinstance(x, float) and pd.isna(x)):
            return np.nan
        if isinstance(x, str):
            if x == "":
                return 0
            return len(x.split("|"))
        return np.nan

    if "reaction_center" in df.columns:
        df["reaction_center_size"] = df["reaction_center"].apply(rc_size)

    # 1. heavy_left vs heavy_product
    if {"heavy_left", "heavy_product"}.issubset(df.columns):
        two_panel_correlation(
            df, "heavy_left", "heavy_product",
            "Heavy atoms, left side (log)", "Heavy atoms, product (log)",
            "Reactant mass vs. product mass, by source",
            f"{args.outdir}/corr_mass_levels_changes.png",
            logx=True, logy=True,
        )

    # 2. num_stereocenters vs reaction_center_size
    if {"num_stereocenters", "reaction_center_size"}.issubset(df.columns):
        d2 = df[df["num_stereocenters"] > 0]  # only reactions with stereo signal
        two_panel_correlation(
            d2, "num_stereocenters", "reaction_center_size",
            "Stereocenters (product)", "Reaction center size (atoms)",
            "Stereocenter count vs. reaction center size, by source",
            f"{args.outdir}/corr_stereo_center_levels_changes.png",
        )

    # 3. mapping_confidence vs n_salts_stripped
    if {"mapping_confidence", "n_salts_stripped"}.issubset(df.columns):
        two_panel_correlation(
            df, "n_salts_stripped", "mapping_confidence",
            "Counterions stripped (count)", "Mapping confidence",
            "Salt-stripping vs. mapper confidence, by source",
            f"{args.outdir}/corr_salts_confidence_levels_changes.png",
        )

    log("done.")


if __name__ == "__main__":
    main()
