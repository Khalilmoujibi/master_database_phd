#!/usr/bin/env python3
"""
build_master_uspto_stereo.py
=============================
Adapter for USPTO_STEREO.csv (DeepChem/MolNet source, ~1M reactions).

Format: single column 'reactions', 'reactants>reagents>products', NO atom maps
(verified: zero ':<digit>' occurrences in the raw file -- confirmed empirically
before writing this, not assumed).

Differences from the other three sources:
  - Simplest input format of the four: no atom maps to strip (unlike ORD/50K/MIT,
    which all needed stripping/re-mapping for consistency). RXNMapper maps from
    scratch here -- there is no pre-existing mapping to distrust or discard.
  - HAS an explicit reagent separator (reactants>reagents>products, 2 '>'s) like
    50K -- unlike MIT which has none. So `left` here still mixes reactants+
    reagents (both blocks go into `left` for atoms_contributed to sort out), but
    at least the source acknowledges a reagent block exists (weak positional
    hint, same caveat as 50K: not trusted as a role verdict, just noted).
  - No natural `id`, `class`, or DOI -- id synthesized as 'stereo_<row_index>'.
  - Known overlap risk: STEREO is also patent-derived (like MIT/50K), so a
    meaningful fraction of its reactions likely duplicate MIT/50K/ORD entries.
    This is NOT handled here -- cross-source dedup happens once, in Phase 2,
    after all sources are merged. Do not dedup within this script.

Everything else (stereo facts, quality flags, RXNMapper engine, canonical-SMILES
matching, order/elem_mismatch diagnostics) is IDENTICAL to the other three
scripts.
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


def canonicalize_no_maps(smiles):
    """No atom maps to strip here (verified empirically: 0 ':<digit>' in the raw
    file) -- just canonicalize and drop pure counterions, same rule as the other
    three sources."""
    m = Chem.MolFromSmiles(smiles)
    if m is None:
        return None
    if is_pure_counterion(m):
        return None
    m = _uncharger.uncharge(m)
    return Chem.MolToSmiles(m)


def parse_stereo_row(rxn_field):
    """'reactants>reagents>products', no atom maps, no trailing edit field.
    Like 50K, has a positional reagent block -- but per the master's fact-only
    rule, this is NOT trusted as a role verdict, only atoms_contributed is."""
    rxn_field = rxn_field.strip()
    if not rxn_field:
        return None
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
        c = canonicalize_no_maps(frag)
        if c is None:
            n_salts += 1
            continue
        left.append(c)

    products = []
    for frag in prod_str.split("."):
        if not frag.strip():
            continue
        c = canonicalize_no_maps(frag)
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


def process_rows(rxn_fields, mapper, start_idx=0):
    recs = []
    for i, field in enumerate(rxn_fields):
        parsed = parse_stereo_row(field)
        if parsed is None:
            continue
        sf = stereo_facts(parsed["products"][0])
        if sf is None:
            continue
        parsed.update(sf)
        parsed.update(quality_flags(parsed["left"], parsed["products"]))
        parsed["id"] = f"stereo_{start_idx + i}"   # synthesized, no native id
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
    ap.add_argument("--csv", required=True, help="path to USPTO_STEREO.csv")
    ap.add_argument("--outdir", default="master_uspto_stereo")
    ap.add_argument("--chunk", type=int, default=20000)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--map", action="store_true")
    ap.add_argument("--limit", type=int, default=0, help="stop after N rows (smoke test)")
    args = ap.parse_args()

    outdir = Path(args.outdir)
    (outdir / "parts").mkdir(parents=True, exist_ok=True)

    log("counting rows ...")
    with open(args.csv) as f:
        total = sum(1 for _ in f) - 1  # minus header
    log(f"{total:,} rows to process")

    mapper = Mapper(args.batch_size) if args.map else None
    done = 0
    t0 = time.time()

    for ci, df in enumerate(pd.read_csv(args.csv, chunksize=args.chunk)):
        part = outdir / "parts" / f"part_{ci:05d}.parquet"
        meta = outdir / "parts" / f"part_{ci:05d}.json"
        if meta.exists():                      # resume: skip finished chunks
            done += len(df)
            continue
        t = time.time()
        recs = process_rows(df["reactions"].tolist(), mapper, start_idx=done)
        if recs:
            out = pd.DataFrame(recs)
            out["source"] = "uspto_stereo"
            tmp = part.with_suffix(".tmp")
            out.to_parquet(tmp, index=False)
            tmp.rename(part)                   # atomic: part exists => complete
        m = dict(n_records=len(recs))
        if args.map and recs:
            m["n_order_ok"] = sum(1 for r in recs if r.get("order_ok"))
            m["n_matched"] = sum(1 for r in recs if r.get("match_ok"))
            m["n_low"] = sum(1 for r in recs if r.get("low_confidence"))
            m["n_elem_mm"] = sum(1 for r in recs
                                 if r.get("elem_mismatch")
                                 and any(x for x in r["elem_mismatch"] if x))
        meta.write_text(json.dumps(m))
        done += len(df)
        dt = time.time() - t
        rate = len(df) / dt if dt else 0
        eta = (total - done) / rate / 3600 if rate else 0
        log(f"chunk {ci:4d} | {done:>9,}/{total:,} | kept {len(recs):5d} | "
            f"{rate:6.1f} rows/s | ETA {eta:5.1f} h")
        if args.limit and done >= args.limit:
            break

    # aggregate diagnostics across all parts
    tot = defaultdict(int)
    for f in sorted((outdir / "parts").glob("*.json")):
        for k, v in json.loads(f.read_text()).items():
            tot[k] += v
    if args.map:
        n = tot.get("n_records", 0)
        print("\n=== DIAGNOSTIC (USPTO-STEREO) ===")
        if n:
            print(f"fragment ORDER preserved : {tot.get('n_order_ok',0)}/{n} "
                  f"({100*tot.get('n_order_ok',0)/n:.1f}%)")
            print(f"matched by SMILES        : {tot.get('n_matched',0)}/{n} "
                  f"({100*tot.get('n_matched',0)/n:.1f}%)")
            print(f"low-confidence            : {tot.get('n_low',0)}/{n} "
                  f"({100*tot.get('n_low',0)/n:.1f}%)")
            print(f"element-mismatch          : {tot.get('n_elem_mm',0)}/{n} "
                  f"({100*tot.get('n_elem_mm',0)/n:.1f}%)")
        # No role_source for STEREO -> no (A)/(B); atoms_contributed only,
        # roles derived identically at the view layer for all 4 sources.

    log(f"done in {(time.time()-t0)/60:.1f} min. "
        f"Merge parts/*.parquet separately (like the ORD script) once ready.")


if __name__ == "__main__":
    main()
