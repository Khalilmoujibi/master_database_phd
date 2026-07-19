#!/usr/bin/env python3
"""
add_split_inputs.py
=====================
Adds `scaffold` and `timestamp` -- SPLIT INPUTS, never a split itself -- to
the final merged 4-source master. Per the master spec: a split (train/val/
test) is a task-dependent CHOICE, so it is never stored; only the raw inputs
a split function would need are stored as facts. Splitting is a separate,
deterministic, seeded function run per experiment, not done here.

SCAFFOLD: Bemis-Murcko scaffold of the PRODUCT (not reactants), the standard
cheminformatics grouping unit used to avoid scaffold leakage between train and
test (a model trained on one scaffold's reactions and tested on near-identical
scaffolds overstates generalization -- this is why the master spec explicitly
rules out random splits).

Validated before running at scale:
  - A real product from the corpus (a fluorophenyl-piperazine amide) produces
    a sensible, non-trivial ring-system scaffold.
  - A small acyclic molecule (ethanol) correctly produces an EMPTY scaffold
    string (Bemis-Murcko requires a ring system; acyclic molecules have none)
    -- this is stored as an explicit empty string, distinct from None (parse
    failure), so "no ring system" is never confused with "could not compute."
  - An unparseable SMILES correctly yields mol=None, handled without crashing.

TIMESTAMP: NONE OF THE FOUR SOURCES IN THIS MASTER (ORD, USPTO-50K, USPTO-MIT,
USPTO-STEREO) CARRY A NATIVE PUBLICATION/PATENT DATE in the columns extracted
so far. Rather than fabricate one, this script writes `timestamp = None` for
every row and documents the gap explicitly. This is a known, honest limitation
-- NOT silently worked around -- and must be stated in the data card. A
temporal/OOD split (as called for by the CSD-Retro-Stereo benchmark design) is
NOT achievable with the current master; only scaffold-based splitting is.
If a temporal split becomes necessary, timestamp must be back-filled from a
source that has it (e.g. re-extracting ORD's provenance metadata, or USPTO
patent grant dates from the original patent IDs) -- this is a known, planned
gap-fill, not a rebuild: one new column, derived from data already
identified.
"""
import argparse
import time

import pandas as pd
from rdkit import Chem
from rdkit.Chem.Scaffolds import MurckoScaffold
from rdkit import RDLogger
RDLogger.DisableLog("rdApp.*")


def log(m):
    print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


def get_scaffold(smiles):
    """Bemis-Murcko scaffold SMILES of a molecule.
    Returns: the scaffold SMILES (possibly '' for acyclic molecules -- this
    is a valid, meaningful result, not an error), or None if the input SMILES
    itself could not be parsed at all."""
    if not isinstance(smiles, str) or not smiles:
        return None
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    scaffold = MurckoScaffold.GetScaffoldForMol(mol)
    return Chem.MolToSmiles(scaffold)


def get_product_scaffold(products):
    """`products` is a list (possibly multiple co-products); scaffold is
    computed on the FIRST product, consistent with how stereo_facts and other
    per-reaction facts already treat products[0] elsewhere in this pipeline."""
    if products is None or len(products) == 0:
        return None
    return get_scaffold(products[0])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="infile", required=True)
    ap.add_argument("--out", dest="outfile", required=True)
    args = ap.parse_args()

    log(f"loading {args.infile}")
    df = pd.read_parquet(args.infile)
    log(f"{len(df):,} rows loaded")

    if "products" not in df.columns:
        raise SystemExit("Missing required column 'products'.")

    log("parsing stored 'products' column ...")
    import numpy as np

    def parse_products(x):
        if x is None:
            return None
        if isinstance(x, np.ndarray):
            return list(x)
        if isinstance(x, list):
            return x
        if isinstance(x, str) and x:
            return x.split("|")
        return None

    products_parsed = df["products"].apply(parse_products)

    log("computing product scaffolds (pure RDKit, CPU) ...")
    t0 = time.time()
    df["scaffold"] = products_parsed.apply(get_product_scaffold)
    log(f"done in {(time.time()-t0)/60:.1f} min")

    n_valid = df["scaffold"].notna().sum()
    n_acyclic = (df["scaffold"] == "").sum()
    n_unparseable = len(df) - n_valid
    log(f"scaffold computed for {n_valid:,}/{len(df):,} rows ({100*n_valid/len(df):.1f}%)")
    log(f"  of which acyclic (empty scaffold, valid result): {n_acyclic:,}")
    log(f"  unparseable products (scaffold=None): {n_unparseable:,}")

    n_unique_scaffolds = df["scaffold"].nunique()
    log(f"unique scaffolds: {n_unique_scaffolds:,}")

    # TIMESTAMP: honestly absent from all four sources as currently extracted.
    # Not fabricated. Documented, not silently worked around.
    df["timestamp"] = None
    log("timestamp: set to None for all rows -- NOT available in any of the "
       "four sources as currently extracted. Documented as a known gap in "
       "the data card; scaffold-based splitting is available now, temporal/"
       "OOD splitting is not, until this is back-filled from source metadata.")

    df.to_parquet(args.outfile, index=False)
    log(f"wrote {args.outfile}")


if __name__ == "__main__":
    main()
