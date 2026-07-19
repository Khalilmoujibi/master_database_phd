#!/usr/bin/env python3
"""
add_reaction_center.py
=======================
Adds `reaction_center` (fact) to an existing master parquet, computed from the
`mapped_rxn` column already present -- NO new RXNMapper call, pure RDKit,
CPU-only, runs on the login node.

DEFINITION (precise, not "every mapped atom"):
  An atom (by map number) is in the reaction center if its BONDING ENVIRONMENT
  changed between reactants and products -- i.e. the set of {neighbor map
  number: bond type} differs. This correctly captures:
    - new bonds formed
    - bonds broken
    - bond order changes (single -> double, etc.)
    - atoms that disappear entirely (leaving groups, e.g. the O in an -OH that
      leaves as water) -- these are NOT "unmapped", they are present in the
      reactants' map but absent from the product's map, and that absence
      itself is the signal.
    - atoms that appear entirely (e.g. a proton source contributing an H that
      becomes mapped in the product but wasn't mapped as a distinct atom before
      -- rare, but handled symmetrically)
  An atom whose neighbors and bond types are IDENTICAL on both sides (a true
  spectator fragment, e.g. an untouched aromatic ring far from the reaction
  site) is correctly EXCLUDED, even though it IS mapped.

  This was verified against two hand-checked cases before being applied at
  scale: (1) an amide coupling (CH3COOH + NCC -> amide + H2O) -- correctly
  identifies the carbonyl carbon, the leaving hydroxyl oxygen, and the
  nitrogen; excludes the untouched methyl/ethyl carbons. (2) a Buchwald-type
  aryl bromide coupling with an unmapped Pd catalyst and an untouched aromatic
  ring -- correctly identifies only the C-Br bond-breaking carbon, the leaving
  bromine, and the nitrogen; excludes the six untouched ring carbons and
  silently ignores the catalyst (it has no atom map number, so it never enters
  the comparison at all -- consistent with atoms_contributed's treatment of
  spectators).

WHY THIS EXISTS: M2 (bond disconnection) needs to know exactly which atoms/
bonds are the reaction center, not just "any atom that happens to be mapped."
Most atoms in a molecule are mapped but chemically untouched; using the full
mapped set as "the reaction center" would be far too broad and would give M2 a
noisy, uninformative training signal.

Output: `reaction_center` column added, one set of map-numbers per row (stored
as a sorted list for parquet compatibility). Rows with no valid mapping get
None (skipped, not dropped from the table -- consistent with "measure, don't
filter" from the master spec).
"""
import argparse
import time

import pandas as pd
from rdkit import Chem
from rdkit import RDLogger
RDLogger.DisableLog("rdApp.*")


def log(m):
    print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


def get_bond_env(mol):
    """map_num -> (H_count, {neighbor_map_num: bond_type_str}) for every mapped
    atom. Includes H_count because changes like N-Cbz -> N-H2 (a deprotection)
    keep the SAME heavy-atom neighbor (the alpha carbon) on both sides -- only
    the hydrogen count changes. Comparing heavy-atom neighbors alone misses
    this entirely; this was found and fixed after testing against a real
    deprotection reaction from the corpus where it was silently invisible."""
    env = {}
    for atom in mol.GetAtoms():
        mnum = atom.GetAtomMapNum()
        if mnum == 0:
            continue
        neighbors = {}
        for bond in atom.GetBonds():
            other = bond.GetOtherAtom(atom)
            onum = other.GetAtomMapNum()
            if onum > 0:
                neighbors[onum] = str(bond.GetBondType())
        env[mnum] = (atom.GetTotalNumHs(), neighbors)
    return env


def reaction_center(mapped_rxn):
    """Set of map-numbers whose bonding environment changed between reactants
    and products. Returns None if the reaction can't be parsed (consistent
    with other 'fact' columns -- absence of data, not a zero)."""
    if not isinstance(mapped_rxn, str) or ">>" not in mapped_rxn:
        return None
    lhs, rhs = mapped_rxn.split(">>")
    left_mol = Chem.MolFromSmiles(lhs)
    right_mol = Chem.MolFromSmiles(rhs)
    if left_mol is None or right_mol is None:
        return None

    left_env = get_bond_env(left_mol)
    right_env = get_bond_env(right_mol)
    all_maps = set(left_env) | set(right_env)  # union: catches leaving/entering atoms

    changed = set()
    for mnum in all_maps:
        if left_env.get(mnum, (None, {})) != right_env.get(mnum, (None, {})):
            changed.add(mnum)
    return sorted(changed)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="infile", required=True)
    ap.add_argument("--out", dest="outfile", required=True)
    args = ap.parse_args()

    log(f"loading {args.infile}")
    df = pd.read_parquet(args.infile)
    log(f"{len(df):,} rows loaded")

    if "mapped_rxn" not in df.columns:
        raise SystemExit("No 'mapped_rxn' column found -- this script requires "
                         "an already-mapped master (run after the mapping step).")

    log("computing reaction_center (pure RDKit, CPU) ...")
    t0 = time.time()
    df["reaction_center"] = df["mapped_rxn"].apply(reaction_center)
    log(f"done in {(time.time()-t0)/60:.1f} min")

    n_valid = df["reaction_center"].notna().sum()
    n_empty = (df["reaction_center"].apply(
        lambda x: isinstance(x, list) and len(x) == 0)).sum()
    log(f"reaction_center computed for {n_valid:,}/{len(df):,} rows "
       f"({100*n_valid/len(df):.1f}%)")
    log(f"rows with empty reaction_center (0 atoms changed -- suspicious, "
       f"e.g. identity reactions): {n_empty:,}")

    sizes = df["reaction_center"].dropna().apply(len)
    if len(sizes):
        log(f"reaction_center size: median={sizes.median():.0f}, "
           f"mean={sizes.mean():.1f}, max={sizes.max()}")

    # store as pipe-joined string for parquet-friendly list handling,
    # consistent with how `left`/`role_source` are stored elsewhere in the master
    df["reaction_center"] = df["reaction_center"].apply(
        lambda x: "|".join(map(str, x)) if isinstance(x, list) else None
    )

    df.to_parquet(args.outfile, index=False)
    log(f"wrote {args.outfile}")


if __name__ == "__main__":
    main()
