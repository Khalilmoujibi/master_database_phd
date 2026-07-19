[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.21435663.svg)](https://doi.org/10.5281/zenodo.21435663)

## Data Availability

The full master dataset (`master_v1.parquet`, 672.7 MB, 2,523,137 reactions) is archived on Zenodo under CC-BY 4.0:

**DOI: [10.5281/zenodo.21435663](https://doi.org/10.5281/zenodo.21435663)**

This repository contains only the build pipeline (code) and documentation — not the data file itself. See [Reproducing the master dataset](#reproducing-the-master-dataset) to regenerate it from source, or download the archived version directly from Zenodo.
# Master Dataset

Reaction-fact master dataset for **StereoRetro** (stereochemistry-aware retrosynthesis planning), built from four public sources under one shared, validated pipeline.

**2,523,137 unique reactions** — atom-mapped, role-labeled, stereo-annotated, quality-flagged. Built and fully documented as part of a PhD thesis .

---

## Table of Contents

- [What this is](#what-this-is)
- [Key numbers](#key-numbers)
- [Repository structure](#repository-structure)
- [Pipeline](#pipeline)
- [Reproducing the master dataset](#reproducing-the-master-dataset)
- [Design principles](#design-principles)
- [Known limitations](#known-limitations)
- [Figures](#figures)
- [Citation](#citation)
- [License](#license)

---

## What this is

This repository contains the **build pipeline** for a reaction-fact master table — not the data itself (see [Design principles](#design-principles) for why). Every script here is real, tested against actual corpus data, and documented with the bugs that were found and fixed during construction, not just the final state.

The master stores **facts** (atom-mapping results, stereochemistry descriptors, quality measurements) rather than **choices** (role labels, train/test splits). Task-specific views — retrosynthesis targets, condition-prediction labels, stereo-validated subsets — are derived from these facts at query time, never baked into the master itself.

## Key numbers

| Metric | Value |
|---|---|
| Unique reactions in master | 2,523,137 |
| Raw reactions processed (4 sources) | 3,907,689 |
| Sources | ORD, USPTO-50K, USPTO-MIT, USPTO-STEREO |
| Atom-mapping tool | RXNMapper (one instance, all sources) |
| Element-mismatch rate (mapper self-consistency) | 0.0% across all 4 sources |
| Spectators hidden in raw `reactants` columns | 40.9% (validated against ORD's own role labels, 96.9% agreement) |
| Reactions with ≥1 stereocenter | 38.5% (702,969 / 1,823,789, 3-source pass) |

Full breakdown, every threshold, and every disagreement rate: [`docs/data_card.pdf`](docs/data_card.pdf).

## Repository structure

```
master_database_phd/
├── scripts/
│   ├── build_master_phase0.py          # ORD  -> per-reaction facts (source-specific adapter)
│   ├── build_master_uspto50k.py        # USPTO-50K  -> per-reaction facts
│   ├── build_master_uspto_mit.py       # USPTO-MIT  -> per-reaction facts
│   ├── build_master_uspto_stereo.py    # USPTO-STEREO  -> per-reaction facts
│   ├── add_reaction_center.py          # + reaction_center (post-hoc, from mapped_rxn)
│   ├── add_stereo_change.py            # + stereo_change (retained/created/destroyed/inverted/resolved)
│   ├── add_reagent_stereo_relevant.py  # + chiral-catalysis / stereo-relevant spectator flags
│   ├── add_split_inputs.py             # + scaffold (Bemis-Murcko); NOT a split assignment
│   ├── paper_figures_v2.py             # publication-style figures (this repo's figures/)
│   ├── clear_figures.py                # descriptive summary figures
│   └── corr_figures.py                 # correlation diagnostics (levels / deviation-from-mean)
├── docs/
│   └── data_card.pdf / .tex            # full build record: every threshold, every rate, every gap
├── figures/
│   └── figure1-5_*.pdf / .png          # publication figures, 600 DPI + vector PDF
├── requirements.txt
├── LICENSE
└── README.md
```

## Pipeline

Each of the four `build_master_*.py` scripts is a source-specific **adapter** feeding the same shared core:

```
raw source (CSV / TXT)
        │
        ▼
  recombine reaction  ──►  standardize (RDKit)  ──►  strip counterions
        │
        ▼
  RXNMapper (single instance, identical across all 4 sources)
        │
        ▼
  atoms_contributed  (THE role fact — atom-count based, convention-free)
        │
        ▼
  post-hoc fact derivation (no re-mapping):
    add_reaction_center.py  → bonding-environment change per atom
    add_stereo_change.py    → stereocenter fate across the reaction arrow
    add_reagent_stereo_relevant.py → chiral-catalyst / stereo-relevant spectator signal
    add_split_inputs.py     → Bemis-Murcko scaffold (split INPUT, not a split)
        │
        ▼
  merge (4 sources) → deduplicate (canonical rxn key) → master_v1
```

**One mapper, one definition of "reactant," across all four sources** — this was validated, not assumed: an early index-based alignment produced materially wrong numbers (71% vs. the correct 97% role-agreement) before being replaced with canonical-SMILES matching. See `docs/data_card.pdf` §2 for the full account.

## Reproducing the master dataset

Requires: RDKit, RXNMapper, PyTorch (GPU strongly recommended — RXNMapper on CPU is ~20× slower), pandas, pyarrow.

```bash
pip install -r requirements.txt
```

Run each source adapter (GPU job; each writes chunked `.parquet` parts with automatic resume):

```bash
python scripts/build_master_phase0.py       --csv <path_to_ORD_csv>        --outdir master_ord    --map
python scripts/build_master_uspto50k.py     --indir <path_to_uspto50k>     --outdir master_50k     --map
python scripts/build_master_uspto_mit.py    --indir <path_to_uspto_mit>    --outdir master_mit     --map
python scripts/build_master_uspto_stereo.py --csv <path_to_uspto_stereo>   --outdir master_stereo  --map
```

For each source, chain the fact-derivation scripts (pure RDKit, CPU-only, no GPU needed at this stage):

```bash
python scripts/add_reaction_center.py          --in <merged>.parquet     --out <name>_rc.parquet
python scripts/add_stereo_change.py            --in <name>_rc.parquet    --out <name>_rc_sc.parquet
python scripts/add_reagent_stereo_relevant.py  --in <name>_rc_sc.parquet --out <name>_complete.parquet
```

Merge all four sources, deduplicate on the canonical reaction key, then add split inputs:

```python
import pandas as pd
merged = pd.concat([df_ord, df_50k, df_mit, df_stereo], ignore_index=True, sort=False)
merged = merged[~merged.duplicated(subset="rxn", keep="first")]
merged.to_parquet("master_4sources_dedup.parquet", index=False)
```

```bash
python scripts/add_split_inputs.py --in master_4sources_dedup.parquet --out master_v1.parquet
```

Data sources (not redistributed here — see `docs/data_card.pdf` for exact provenance and licensing per source):
- **ORD** — [Open Reaction Database](https://open-reaction-database.org/)
- **USPTO-50K / USPTO-MIT** — [Jin et al., NeurIPS 2017](https://github.com/wengong-jin/nips17-rexgen)
- **USPTO-STEREO** — via [DeepChem/MolNet](https://deepchemdata.s3.us-west-1.amazonaws.com/datasets/USPTO_STEREO.csv)

## Design principles

1. **Facts, not choices.** The master stores atom-mapping results, stereochemistry descriptors, and quality measurements — never role verdicts, quality cutoffs, or splits. A solvent/reagent whitelist correction should cost one line and a view regeneration, never a rebuild.
2. **One shared core, rerun per source.** All four adapters share identical standardization, counterion-stripping, and mapping logic. Source-specific differences (native role fields, atom-map presence, reagent separators) are handled at the adapter boundary, not by forking the core.
3. **No index-based alignment, anywhere.** RXNMapper reorders fragments in 25–65% of reactions depending on source complexity (measured, not assumed). Every fact that depends on fragment identity is matched by canonical SMILES.
4. **Measure quality, never filter silently.** `mapping_confidence`, `is_rdkit_valid`, `low_confidence` are stored as columns. Filtering happens at the view layer, per task, not in the master.
5. **No split assigned in the master.** Only `scaffold` (Bemis-Murcko) is stored, since a split is a task-dependent choice.

## Known limitations

Disclosed in full in `docs/data_card.pdf` §10. Summary:

- No oracle exists for role assignment; all validation is agreement between independent methods.
- RXNMapper's own published reliability ceiling (~84% on independent benchmarks) bounds every derived fact.
- H₂ and other zero-heavy-atom reactants are structurally invisible to atom-count-based role assignment.
- **No source currently carries a native timestamp** — temporal/OOD splitting is not yet possible; scaffold-based splitting is.
- Sources are patent- and database-derived, not exploratory academic chemistry — a known compositional bias, not correctable by cleaning.

## Figures

`figures/` contains five publication-ready figures (vector PDF + 600 DPI PNG), summarizing corpus composition, atom-mapping quality, role-separation validation, and reaction-center/stereo-change statistics. Regenerate with:

```bash
python scripts/paper_figures_v2.py --in master_v1.parquet --outdir figures
```

## Citation

If you use this pipeline or the resulting dataset, please cite:

```bibtex
@misc{moujibi2026stereoretro,
  author = {Moujibi, Khalil},
  title  = {StereoRetro Master Dataset: A Fact-Based Reaction Corpus for Stereochemistry-Aware Retrosynthesis},
  year   = {2026},
  note   = {ISTI Laboratory, Universit\'e Mohammed VI Polytechnique},
  url    = {https://github.com/<your-username>/master_database_phd}
}
```

## License

MIT — see [`LICENSE`](LICENSE). The pipeline code is MIT-licensed; underlying data sources (ORD, USPTO, CSD where applicable) retain their own licenses — see `docs/data_card.pdf` for per-source terms.
