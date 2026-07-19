#!/usr/bin/env python3
"""
build_master_uspto50k.py
=========================
Adapter for USPTO-50K (raw_train/val/test.csv, format: id,class,reactants>reagents>production).

Key differences from the ORD pipeline:
  - Input is ALREADY atom-mapped (Indigo/RDKit's own mapping, embedded as [C:1] etc.)
    We DELIBERATELY STRIP these maps and re-map with RXNMapper, so "reactant" means
    the SAME thing across every source in the master (one mapper = one definition;
    see build_master_phase0.py header). Trusting USPTO's bundled mapping would
    silently reintroduce a different, less reliable role definition.
  - No source-provided role split (no reactant/reagent/solvent columns) -- unlike
    ORD, there is no `role_source` to extract here. Only `atoms_contributed` (the
    fact) will exist for this source; role labels are derived identically at the
    view layer for both sources.
  - `class` (reaction class, e.g. USPTO-50K's 10 classes) is carried through as a
    provenance fact -- it costs nothing to keep and a future task might want it.

Everything else (stereo facts, quality flags, RXNMapper engine, canonical-SMILES
matching, order/elem_mismatch diagnostics) is IDENTICAL to the ORD script, reused
verbatim so both sources share one core and one set of guarantees.
"""
import argparse, json, re, time
from collections import defaultdict
from pathlib import Path

import pandas as pd
from rdkit import Chem
from rdkit.Chem.MolStandardize import rdMolStandardize
from rdkit import RDLogger
RDLogger.DisableLog("rdApp.*")

MAP_CONF_TAU = 0.5
SEQ_TOKEN_LIMIT = 512
_COUNTERION_Z = {3, 11, 19, 37, 55, 4, 12, 20, 38, 56, 9, 17, 35, 53}
_uncharger = rdMolStandardize.Uncharger()
SMI_REGEX = re.compile(
    r"(\[[^\]]+]|Br?|Cl?|N|O|S|P|F|I|b|c|n|o|s|p|\(|\)|\.|=|#|-|\+|\\|\/|:|~|@|\?|>|\*|\$|\%[0-9]{2}|[0-9])"
)


def log(m):
    print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


def n_tokens(s):
    return len(SMI_REGEX.findall(s))


def is_pure_counterion(mol):
    if mol.GetNumHeavyAtoms() != 1:
        return False
    a = mol.GetAtomWithIdx(0)
    return a.GetFormalCharge() != 0 and a.GetAtomicNum() in _COUNTERION_Z


def canon(mol):
    m = Chem.Mol(mol)
    for a in m.GetAtoms():
        a.SetAtomMapNum(0)
    return Chem.MolToSmiles(m)


def strip_maps_and_canonicalize(smiles):
    """Remove USPTO-50K's bundled atom maps ([C:1] -> C), canonicalize, strip
    salts. Returns list of canonical SMILES (order not meaningful pre-mapping)."""
    m = Chem.MolFromSmiles(smiles)
    if m is None:
        return None
    for a in m.GetAtoms():
        a.SetAtomMapNum(0)
    if is_pure_counterion(m):
        return None
    m = _uncharger.uncharge(m)
    return Chem.MolToSmiles(m)


def parse_uspto_row(rxn_field):
    """'reactants.reagents>>product' with embedded atom maps (may contain a
    second '>' if reagents are present: 'reactants>reagents>product')."""
    parts = rxn_field.split(">")
    if len(parts) == 3:
        react_str, reagent_str, prod_str = parts
    elif len(parts) == 2:
        react_str, prod_str = parts
        reagent_str = ""
    else:
        return None

    left_raw = [s for s in (react_str + "." + reagent_str).split(".") if s.strip()]
    n_salts = 0
    left = []
    for frag in left_raw:
        c = strip_maps_and_canonicalize(frag)
        if c is None:
            n_salts += 1
            continue
        left.append(c)

    products = []
    for frag in prod_str.split("."):
        if not frag.strip():
            continue
        c = strip_maps_and_canonicalize(frag)
        if c is not None:
            products.append(c)

    if not left or not products:
        return None

    rxn = ".".join(left) + ">>" + ".".join(products)
    return dict(rxn=rxn, left=left, products=products, n_salts_stripped=n_salts)


def stereo_facts(smiles):
    m = Chem.MolFromSmiles(smiles)
    if m is None:
        return None
    centers = Chem.FindMolChiralCenters(m, includeUnassigned=True,
                                        useLegacyImplementation=False)
    return dict(num_stereocenters=len([c for c in centers if c[1] != "?"]),
                undefined_stereo=any(c[1] == "?" for c in centers),
                has_stereo_bond=any(b.GetStereo() != Chem.BondStereo.STEREONONE
                                    for b in m.GetBonds()))


def quality_flags(left, products):
    def nh(sl):
        t = 0
        for s in sl:
            m = Chem.MolFromSmiles(s)
            if m is not None:
                t += m.GetNumHeavyAtoms()
        return t
    return dict(is_rdkit_valid=all(Chem.MolFromSmiles(s) is not None
                                   for s in left + products),
                heavy_left=nh(left), heavy_product=nh(products))


def analyse_mapping(mapped_rxn):
    if not mapped_rxn or ">>" not in mapped_rxn:
        return None
    lhs, rhs = mapped_rxn.split(">>")
    prod = Chem.MolFromSmiles(rhs)
    if prod is None:
        return None
    elem_by_map = {a.GetAtomMapNum(): a.GetSymbol()
                   for a in prod.GetAtoms() if a.GetAtomMapNum() > 0}
    per_frag = []
    for frag in lhs.split("."):
        m = Chem.MolFromSmiles(frag)
        if m is None:
            continue
        nc = nm = 0
        for a in m.GetAtoms():
            k = a.GetAtomMapNum()
            if k > 0 and k in elem_by_map:
                if a.GetSymbol() == elem_by_map[k]:
                    nc += 1
                else:
                    nm += 1
        per_frag.append((canon(m), nc, nm))
    return per_frag


def finalize(r):
    per_frag = analyse_mapping(r.get("mapped_rxn"))
    if per_frag is None:
        r.update(atoms_contributed=None, elem_mismatch=None,
                 match_ok=False, order_ok=False)
        return
    r["order_ok"] = [f[0] for f in per_frag] == r["left"]
    pool = defaultdict(list)
    for key, nc, nm in per_frag:
        pool[key].append((nc, nm))
    contrib, mism = [], []
    for s in r["left"]:
        if pool[s]:
            c, m = pool[s].pop(0)
            contrib.append(c); mism.append(m)
        else:
            contrib.append(None); mism.append(None)
    r["atoms_contributed"] = contrib
    r["elem_mismatch"] = mism
    r["match_ok"] = all(c is not None for c in contrib)


class Mapper:
    def __init__(self, batch_size):
        import torch
        from rxnmapper import RXNMapper
        self.batch_size = batch_size
        log("loading RXNMapper ...")
        self.rm = RXNMapper()
        cuda = torch.cuda.is_available()
        try:
            dev = next(self.rm.model.parameters()).device
        except Exception:
            dev = "?"
        log(f"torch.cuda.is_available() = {cuda}")
        log(f"RXNMapper model device    = {dev}")
        if cuda:
            log(f"GPU = {torch.cuda.get_device_name(0)}")
        else:
            log("*** WARNING: NO CUDA -> ~20x slower. ***")

    def map_many(self, rxns):
        n = len(rxns)
        out = [(None, 0.0)] * n
        idx = [i for i in range(n) if n_tokens(rxns[i]) <= SEQ_TOKEN_LIMIT]
        idx.sort(key=lambda i: len(rxns[i]))
        for s in range(0, len(idx), self.batch_size):
            bi = idx[s:s + self.batch_size]
            try:
                res = self.rm.get_attention_guided_atom_maps([rxns[i] for i in bi])
                for i, o in zip(bi, res):
                    out[i] = (o["mapped_rxn"], o["confidence"])
            except Exception:
                for i in bi:
                    try:
                        o = self.rm.get_attention_guided_atom_maps([rxns[i]])[0]
                        out[i] = (o["mapped_rxn"], o["confidence"])
                    except Exception:
                        out[i] = (None, 0.0)
        return out


def process_chunk(df, mapper, split_name):
    recs = []
    for _, row in df.iterrows():
        parsed = parse_uspto_row(row["reactants>reagents>production"])
        if parsed is None:
            continue
        sf = stereo_facts(parsed["products"][0])
        if sf is None:
            continue
        parsed.update(sf)
        parsed.update(quality_flags(parsed["left"], parsed["products"]))
        parsed["id"] = str(row["id"])
        parsed["reaction_class"] = row.get("class", None)  # provenance fact
        parsed["uspto50k_split"] = split_name               # provenance fact,
                                                              # NOT the master split
        recs.append(parsed)
    if not recs:
        return recs
    if mapper is not None:
        for r, (mp, cf) in zip(recs, mapper.map_many([r["rxn"] for r in recs])):
            r["mapped_rxn"] = mp
            r["mapping_confidence"] = cf
            r["low_confidence"] = cf < MAP_CONF_TAU
            r["seq_truncated"] = n_tokens(r["rxn"]) > SEQ_TOKEN_LIMIT
            finalize(r)
    return recs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--indir", required=True,
                    help="dir containing raw_train.csv, raw_val.csv, raw_test.csv")
    ap.add_argument("--outdir", default="master_uspto50k")
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--map", action="store_true")
    args = ap.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    mapper = Mapper(args.batch_size) if args.map else None

    all_recs = []
    for split_name in ("train", "val", "test"):
        f = Path(args.indir) / f"raw_{split_name}.csv"
        df = pd.read_csv(f)
        log(f"{split_name}: {len(df)} rows loaded")
        recs = process_chunk(df, mapper, split_name)
        log(f"{split_name}: kept {len(recs)}/{len(df)}")
        all_recs.extend(recs)

    out = pd.DataFrame(all_recs)
    out["source"] = "uspto50k"
    out.to_parquet(outdir / "master_uspto50k_merged.parquet", index=False)
    log(f"wrote {len(out)} rows -> {outdir}/master_uspto50k_merged.parquet")

    if args.map:
        n = len(out)
        n_order_ok = out["order_ok"].sum()
        n_matched = out["match_ok"].sum()
        n_low = out["low_confidence"].sum()
        n_mm = out["elem_mismatch"].apply(
            lambda em: bool(em) and any(x for x in em if x)).sum()
        print("\n=== DIAGNOSTIC (USPTO-50K) ===")
        print(f"fragment ORDER preserved : {n_order_ok}/{n} ({100*n_order_ok/n:.1f}%)")
        print(f"matched by SMILES        : {n_matched}/{n} ({100*n_matched/n:.1f}%)")
        print(f"low-confidence           : {n_low}/{n} ({100*n_low/n:.1f}%)")
        print(f"element-mismatch         : {n_mm}/{n} ({100*n_mm/n:.1f}%)")
        # No role_source for USPTO -> no (A)/(B) validation here; those are
        # ORD-specific (they validate ORD's OWN role labels against the fact).
        # USPTO-50K only contributes atoms_contributed; roles are derived later
        # identically for both sources at the view layer.


if __name__ == "__main__":
    main()
