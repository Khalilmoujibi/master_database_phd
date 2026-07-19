#!/usr/bin/env python3
"""
add_reagent_stereo_relevant.py
================================
Adds two facts about SPECTATOR molecules (atoms_contributed == 0) to an
existing master parquet: whether they carry a stereocenter, and whether the
reaction as a whole shows a signal of chiral catalysis. Pure RDKit, CPU-only,
no new GPU mapping. Run after add_reaction_center.py / add_stereo_change.py.

WHY THIS EXISTS: a chiral catalyst or auxiliary contributes ZERO atoms to the
product (atoms_contributed == 0 -> classified as a spectator by the master's
core role fact), yet it is the direct CAUSE of the product's stereochemistry
in asymmetric catalysis. If a stereo-aware view strips all spectators
uniformly, it discards exactly the information that explains why a particular
enantiomer formed. This script does not change how atoms_contributed or roles
are computed -- it adds two independent, defensible facts that a downstream
view can use to keep such reagents as context instead of dropping them.

TWO LEVELS, kept separate and NOT merged into one verdict (facts, not a single
computed opinion):

  1. spectator_has_stereocenter (per molecule, objective, convention-free):
     True if RDKit detects >=1 stereocenter (assigned or not) on that
     spectator's own structure. No metal list involved -- this alone already
     flags things like chiral amine auxiliaries, chiral acids used for
     resolution, etc., independent of any catalysis mechanism.

  2. reaction_chiral_catalysis_signal (per REACTION, stricter, sourced):
     True if the reaction's spectator set contains BOTH (a) a transition metal
     from a short, literature-grounded list associated with asymmetric
     catalysis, AND (b) a spectator with a stereocenter (from level 1) --
     whether on the same molecule or two separate spectator molecules (the
     common real case: e.g. "[Pd].[chiral phosphine ligand]" as two separate
     fragments in the same reagent block, as seen in the actual corpus, e.g.
     "[C].[Pd]" alongside deprotection fragments in the ORD-derived rows
     examined while building add_reaction_center.py).

     Metal list (Pd, Rh, Ru, Ir, Cu, Ni) is NOT an arbitrary guess -- it is
     the convergent core list across multiple independent sources checked
     before writing this script: an academic review (Rueda-Becerril et al.,
     PMC6264407, Pd-catalysed asymmetric allylic alkylation), and Cell
     Catalysis 2022 (Asymmetric transformations enabled by synergistic dual
     transition-metal catalysis), which explicitly frames Pd/Rh/Ru/Ir/Cu/Ni as
     the dominant paradigm and traces it to the 2001 Nobel Prize in asymmetric
     catalysis. Broader lists in the same sources also mention Fe, Ti, Pt, Mo,
     W, Os, Ag as usable but less central/consensus metals -- deliberately
     EXCLUDED from this core list to keep the flag conservative and
     defensible rather than over-inclusive. This is a documented convention,
     not a claim of chemical completeness.

VALIDATED before being applied at scale:
  - A bare [Pd] atom: has_metal=True, has_stereocenter=False -- correctly NOT
    flagged as reaction_chiral_catalysis_signal by itself (no chirality
    present at all in that fragment).
  - [Pd] + a genuinely chiral fragment (C[C@H](N)c1ccccc1, four distinct
    substituents, unambiguous textbook stereocenter) as two separate spectator
    fragments: correctly flags the ligand fragment for level 1, and the
    co-occurrence for level 2.
  - An earlier draft test used a symmetric bis-phosphine SMILES that RDKit
    correctly did NOT recognize as chiral (two identical aryl-phosphine arms
    cancel the stereocenter) -- caught by inspecting RDKit's own neighbor list
    before trusting the result, not assumed to be a script bug.

This does NOT redefine or override atoms_contributed or role_source. It adds
context facts alongside them, exactly per the master's fact-not-choice
principle: whether to KEEP a stereo-relevant reagent in a given task view
remains a view-layer decision, made downstream, using these facts.
"""
import argparse
import math
import time
from collections import Counter

import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit import RDLogger
RDLogger.DisableLog("rdApp.*")

# Core list, convergent across independent literature sources (see docstring).
ASYMMETRIC_CATALYSIS_METALS_Z = {
    46,  # Pd
    45,  # Rh
    44,  # Ru
    77,  # Ir
    29,  # Cu
    28,  # Ni
}


def log(m):
    print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


def has_stereocenter(mol):
    if mol is None:
        return False
    centers = Chem.FindMolChiralCenters(mol, includeUnassigned=True,
                                        useLegacyImplementation=False)
    return len(centers) > 0


def has_asymmetric_catalysis_metal(mol):
    if mol is None:
        return False
    return any(a.GetAtomicNum() in ASYMMETRIC_CATALYSIS_METALS_Z
              for a in mol.GetAtoms())


def analyze_spectators(left_smiles_list, atoms_contributed_list):
    """Given the `left` molecule list and the parallel `atoms_contributed`
    list (index-aligned, as stored in the master), return:
      - list of bools, one per LEFT molecule: True if that spectator has a
        stereocenter (non-spectators get False, not evaluated)
      - one bool for the whole reaction: chiral-catalysis co-occurrence signal
    Returns (None, None) if inputs are missing/misaligned -- never guesses."""
    if left_smiles_list is None or atoms_contributed_list is None:
        return None, None
    if len(left_smiles_list) != len(atoms_contributed_list):
        return None, None  # same alignment guard as reaction_center/atoms_contributed

    spectator_stereo_flags = []
    any_metal = False
    any_stereo_spectator = False

    for smi, ac in zip(left_smiles_list, atoms_contributed_list):
        # atoms_contributed comes from a numpy float array (confirmed: e.g.
        # 9.0, 27.0, 0.0) and may contain NaN where no value was recorded --
        # treat NaN as "unknown, not a spectator" rather than crashing on a
        # NaN != 0 comparison (NaN == 0 is always False in Python/numpy, which
        # is actually the safe default here: an unknown atom count should not
        # be silently treated as a spectator).
        is_nan = isinstance(ac, float) and math.isnan(ac)
        is_spectator = (not is_nan) and (ac == 0)
        if not is_spectator:
            spectator_stereo_flags.append(False)  # not a spectator -> not evaluated
            continue
        mol = Chem.MolFromSmiles(smi)
        stereo = has_stereocenter(mol)
        metal = has_asymmetric_catalysis_metal(mol)
        spectator_stereo_flags.append(stereo)
        if metal:
            any_metal = True
        if stereo:
            any_stereo_spectator = True

    reaction_signal = any_metal and any_stereo_spectator
    return spectator_stereo_flags, reaction_signal


def parse_list(x):
    """`left` and `atoms_contributed` are stored as numpy arrays directly in
    this parquet (confirmed by inspecting a real row before writing this:
    numpy.ndarray, e.g. array(['CC(C)N1CCNCC1', ...]) and
    array([9., 27., 0., ...])) -- NOT pipe-joined strings. An earlier version
    of this function assumed the string format used elsewhere in the
    pipeline and silently returned None for every row, because
    isinstance(x, str) is False for a numpy array -- caught by checking
    computed-for-0-rows in the actual run output rather than assuming the
    parsing was correct."""
    if x is None:
        return None
    if isinstance(x, np.ndarray):
        return list(x)
    if isinstance(x, list):
        return x
    if isinstance(x, str):
        if x == "":
            return []
        parts = x.split("|")
        out = []
        for p in parts:
            if p == "None":
                out.append(None)
            else:
                try:
                    out.append(int(p))
                except ValueError:
                    out.append(p)
        return out
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="infile", required=True)
    ap.add_argument("--out", dest="outfile", required=True)
    args = ap.parse_args()

    log(f"loading {args.infile}")
    df = pd.read_parquet(args.infile)
    log(f"{len(df):,} rows loaded")

    for col in ("left", "atoms_contributed"):
        if col not in df.columns:
            raise SystemExit(f"Missing required column '{col}'.")

    log("parsing stored list columns ...")
    left_parsed = df["left"].apply(parse_list)
    ac_parsed = df["atoms_contributed"].apply(parse_list)

    log("analyzing spectators (pure RDKit, CPU) ...")
    t0 = time.time()
    results = [
        analyze_spectators(l, a) for l, a in zip(left_parsed, ac_parsed)
    ]
    log(f"done in {(time.time()-t0)/60:.1f} min")

    spectator_flags = [r[0] for r in results]
    reaction_signal = [r[1] for r in results]

    df["spectator_has_stereocenter"] = [
        "|".join("1" if b else "0" for b in flags) if flags is not None else None
        for flags in spectator_flags
    ]
    df["reaction_chiral_catalysis_signal"] = reaction_signal

    n_valid = sum(1 for x in reaction_signal if x is not None)
    n_flagged = sum(1 for x in reaction_signal if x is True)
    log(f"computed for {n_valid:,}/{len(df):,} rows")
    log(f"reaction_chiral_catalysis_signal = True: {n_flagged:,} "
       f"({100*n_flagged/n_valid:.2f}% of valid rows)")

    n_any_stereo_spectator = sum(
        1 for flags in spectator_flags
        if flags is not None and any(flags)
    )
    log(f"rows with >=1 stereocenter-bearing spectator (level 1, any cause): "
       f"{n_any_stereo_spectator:,}")

    df.to_parquet(args.outfile, index=False)
    log(f"wrote {args.outfile}")


if __name__ == "__main__":
    main()
