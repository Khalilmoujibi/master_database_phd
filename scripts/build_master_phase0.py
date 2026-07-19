#!/usr/bin/env python3
"""
build_master.py -- ORD -> master facts table.
Streaming + resumable + progress/ETA + device check.
Chemistry logic identical to v3 (validated: order 24.6%, elem_mismatch 0%, A=97.5%).
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


def canonical_components(cell):
    if pd.isna(cell):
        return []
    out = []
    for frag in str(cell).split("."):
        frag = frag.strip()
        if not frag:
            continue
        m = Chem.MolFromSmiles(frag)
        if m is not None:
            out.append(Chem.MolToSmiles(m))
    return out


def recombine(row):
    tagged, n_salts = [], 0
    for block, cell in (("reactant", row["reactants"]),
                        ("catalyst", row["catalysts"]),
                        ("solvent",  row["solvents"])):
        for s in canonical_components(cell):
            m = Chem.MolFromSmiles(s)
            if m is None:
                continue
            if is_pure_counterion(m):
                n_salts += 1
                continue
            m = _uncharger.uncharge(m)
            tagged.append((Chem.MolToSmiles(m), block))
    products = canonical_components(row["products"])
    if not tagged or not products:
        return None
    return dict(rxn=".".join(s for s, _ in tagged) + ">>" + ".".join(products),
                left=[s for s, _ in tagged],
                role_source=[t for _, t in tagged],
                products=products,
                n_salts_stripped=n_salts)


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
            log("*** WARNING: NO CUDA -> ~20x slower. "
                "Is 'module load CUDA/11.1.1-GCC-10.2.0' in the sbatch script? ***")

    def map_many(self, rxns):
        n = len(rxns)
        out = [(None, 0.0)] * n
        # skip >512-token reactions: they get truncated -> garbage anyway (0.04%)
        idx = [i for i in range(n) if n_tokens(rxns[i]) <= SEQ_TOKEN_LIMIT]
        idx.sort(key=lambda i: len(rxns[i]))     # length-sorted -> less padding -> faster
        for s in range(0, len(idx), self.batch_size):
            bi = idx[s:s + self.batch_size]
            try:
                res = self.rm.get_attention_guided_atom_maps([rxns[i] for i in bi])
                for i, o in zip(bi, res):
                    out[i] = (o["mapped_rxn"], o["confidence"])
            except Exception:
                for i in bi:                      # isolate the bad one, keep the rest
                    try:
                        o = self.rm.get_attention_guided_atom_maps([rxns[i]])[0]
                        out[i] = (o["mapped_rxn"], o["confidence"])
                    except Exception:
                        out[i] = (None, 0.0)
        return out


def chunk_metrics(recs, mapped):
    m = defaultdict(int)
    m["n_records"] = len(recs)
    if not mapped:
        return dict(m)
    for r in recs:
        if r.get("mapped_rxn"): m["n_mapped"] += 1
        if r.get("order_ok"):   m["n_order_ok"] += 1
        if r.get("match_ok"):   m["n_matched"] += 1
        if r.get("low_confidence"): m["n_low"] += 1
        em = r.get("elem_mismatch")
        if em and any(x for x in em if x):        # guard: em may be None
            m["n_elem_mm"] += 1
        if not r.get("match_ok"):
            continue
        ac = r["atoms_contributed"]
        rel = not (r["low_confidence"] or r["seq_truncated"])
        for i, tag in enumerate(r["role_source"]):
            if tag == "reactant":
                m["react_all"] += 1
                if ac[i] == 0: m["hidden_all"] += 1
                if rel:
                    m["react_rel"] += 1
                    if ac[i] == 0: m["hidden_rel"] += 1
            else:
                if ac[i] == 0:
                    m["agree_all"] += 1
                    if rel: m["agree_rel"] += 1
                else:
                    m["disagree_all"] += 1
                    if rel: m["disagree_rel"] += 1
    return dict(m)


def process_chunk(df, mapper):
    recs = []
    for _, row in df.iterrows():
        rec = recombine(row)
        if rec is None:
            continue
        sf = stereo_facts(rec["products"][0])
        if sf is None:
            continue
        rec.update(sf)
        rec.update(quality_flags(rec["left"], rec["products"]))
        rec["id"] = str(row["id"])
        rec["temperature_K"] = row["temperature_K"]     # facts: keep them
        rec["yield"] = row["yield"]
        recs.append(rec)
    if not recs:
        return [], chunk_metrics([], mapper is not None)
    if mapper is not None:
        for r, (mp, cf) in zip(recs, mapper.map_many([r["rxn"] for r in recs])):
            r["mapped_rxn"] = mp
            r["mapping_confidence"] = cf
            r["low_confidence"] = cf < MAP_CONF_TAU
            r["seq_truncated"] = n_tokens(r["rxn"]) > SEQ_TOKEN_LIMIT
            finalize(r)
    return recs, chunk_metrics(recs, mapper is not None)


def aggregate(outdir):
    tot = defaultdict(int)
    for f in sorted((outdir / "parts").glob("*.json")):
        for k, v in json.loads(f.read_text()).items():
            tot[k] += v
    t = dict(tot)
    (outdir / "metrics.json").write_text(json.dumps(t, indent=2))
    print("\n=== DIAGNOSTIC ===")
    n, k = t.get("n_mapped", 0), t.get("n_records", 0)
    if n:
        print(f"fragment ORDER preserved  : {t['n_order_ok']}/{n} ({100*t['n_order_ok']/n:.1f}%)")
    if k:
        print(f"matched by SMILES         : {t.get('n_matched',0)}/{k} ({100*t.get('n_matched',0)/k:.1f}%)")
        print(f"low-confidence            : {t.get('n_low',0)}/{k} ({100*t.get('n_low',0)/k:.1f}%)")
        print(f"element-mismatch          : {t.get('n_elem_mm',0)}/{k} ({100*t.get('n_elem_mm',0)/k:.1f}%)")
    for lab, a, d, h, rm in (("ALL", "agree_all", "disagree_all", "hidden_all", "react_all"),
                             ("HIGH-CONF", "agree_rel", "disagree_rel", "hidden_rel", "react_rel")):
        A, D, H, R = t.get(a, 0), t.get(d, 0), t.get(h, 0), t.get(rm, 0)
        print(f"\n=== VALIDATION -- {lab} ===")
        if A + D:
            print(f"(A) role label vs atoms_contributed==0 : {100*A/(A+D):.1f}% agree ({D}/{A+D})")
        if R:
            print(f"(B) spectators hidden in reactants     : {H}/{R} ({100*H/R:.1f}%)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True)
    ap.add_argument("--outdir", default="master_ord")
    ap.add_argument("--chunk", type=int, default=20000)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--map", action="store_true")
    ap.add_argument("--limit", type=int, default=0, help="stop after N rows (smoke test)")
    args = ap.parse_args()

    outdir = Path(args.outdir)
    (outdir / "parts").mkdir(parents=True, exist_ok=True)

    log("counting rows ...")
    with open(args.csv) as f:
        total = sum(1 for _ in f) - 1
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
        recs, m = process_chunk(df, mapper)
        if recs:
            tmp = part.with_suffix(".tmp")
            pd.DataFrame(recs).to_parquet(tmp, index=False)
            tmp.rename(part)                   # atomic: part exists => complete
        meta.write_text(json.dumps(m))
        done += len(df)
        dt = time.time() - t
        rate = len(df) / dt if dt else 0
        eta = (total - done) / rate / 3600 if rate else 0
        log(f"chunk {ci:4d} | {done:>9,}/{total:,} | kept {len(recs):5d} | "
            f"{rate:6.1f} rows/s | ETA {eta:5.1f} h")
        if args.limit and done >= args.limit:
            break

    aggregate(outdir)
    log(f"done in {(time.time()-t0)/60:.1f} min")


if __name__ == "__main__":
    main()