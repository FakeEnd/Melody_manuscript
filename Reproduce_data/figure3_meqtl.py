"""
Generate the source data for Figure 3 (meQTL effect prediction).

Method
------
For each meQTL (a SNP paired with a nearby CpG and an observed effect size), we
build the 10-kb reference sequence centred between the SNP and the CpG, make a
single-nucleotide copy carrying the alternative allele, and run Melody on both.
The *predicted* effect is the change in single-base methylation summed over a
small window around the CpG (averaged over the cell-type tracks that match the
meQTL's tissue), divided by the number of CpGs:

    effect_pred = sum_{j in [cpg_start-margin, cpg_end+margin]} ( P_alt[j] - P_ref[j] ) / n_cpg

We then correlate predicted vs observed effects per meQTL dataset.

Panels produced (written to OUT_DIR, compare against Data/Figure3/ + Data/Supplementary/):
  figure_3C.csv  model x dataset x {related tracks, single track} -> pearson_r
  figure_3D.csv  per dataset: Melody-ST pearson (st_r) vs Melody-MT pearson (mt_r)
  figure_3E.csv  distance-window x model x dataset -> pearson_r (Melody-MT here; baselines appended externally)
  figure_3F.csv  per-point observed vs predicted, dataset = CPG_FG_DATASET_F  (scatter)
  figure_3G.csv  per-point observed vs predicted, dataset = CPG_FG_DATASET_G  (scatter)
  figure_3H.csv  cross-cell matrix: predict dataset X's meQTLs using dataset Y's tracks
  figure_3I.csv  training-step progression: pearson per dataset at several checkpoints

Notes
-----
* Figure 3A/B (ISM logos) are produced by ``Reproduce_figure/Figure3_AB_ism.ipynb``.
* The baseline rows in 3E (CpGenie, iDNA-ABT, INTERACT) come from running those
  baseline models with the same metric definitions; this script emits the Melody
  rows. Plug baseline predictions into ``metrics_from_points`` to extend.
* 3D / 3C "Melody-ST" rows require the 39 single-track checkpoints (ST_CKPT_DIR).
* The meQTL tables are bundled with the Melody code repo under
  ``meqtl/dataset/processed/{GTEX,EPIGEN,Olafur_2024}/``; no BigWig truth needed.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np
import pandas as pd

import _engine as E

# --------------------------------------------------------------------------- #
# meQTL dataset -> matched 39-track indices (tissue/cell-type correspondence).
# "related tracks" averages all listed tracks; "single track" uses the first.
# --------------------------------------------------------------------------- #
EQTL_TO_TRACK_INDICES: Dict[str, List[int]] = {
    "GTEX_WholeBlood": [21, 22, 23, 24, 25, 26],
    "EPIC": [21, 22, 23, 24, 25, 26],
    "CPG_units": [21, 22, 23, 24, 25, 26],
    "MDSs": [21, 22, 23, 24, 25, 26],
    "skin": [27, 5],
    "GTEX_BreastMammaryTissue": [32, 33, 0],
    "GTEX_ColonTransverse": [37, 3, 38],
    "GTEX_KidneyCortex": [2],
    "GTEX_Lung": [29, 34],
    "GTEX_MuscleSkeletal": [6, 7],
    "GTEX_Ovary": [20, 19],
    "GTEX_Prostate": [30],
    "GTEX_Testis": [20, 19, 21, 22, 23, 24, 25, 26],
}

HALF_LEN = 5000          # half of the 10-kb model window
MAIN_MARGIN = 1          # +/- bp added around the CpG when summing the effect
BATCH_SIZE = 128

# Datasets shown as per-point scatters in 3F / 3G (n=524 and n=1334 in the paper).
CPG_FG_DATASET_F = os.environ.get("FIG3F_DATASET", "MDSs")
CPG_FG_DATASET_G = os.environ.get("FIG3G_DATASET", "GTEX_WholeBlood")

# Distance windows for 3E (SNP-CpG distance upper bound, bp).
FIG3E_WINDOWS = [20, 40, 75, 100, 150, 200, 500, 1000, 2000]

# Checkpoints for the 3I training-step progression: {step: checkpoint_path}.
# Fill these in with your own checkpoints; defaults are placeholders.
FIG3I_STEPS: Dict[int, str] = {
    200000: os.environ.get("CKPT_200K", "/path/to/checkpoint_step_200000.pth"),
    250000: os.environ.get("CKPT_250K", "/path/to/checkpoint_step_250000.pth"),
    300000: os.environ.get("CKPT_300K", "/path/to/checkpoint_step_300000.pth"),
    500000: os.environ.get("CKPT_500K", "/path/to/checkpoint_step_500000.pth"),
    750000: os.environ.get("CKPT_750K", "/path/to/checkpoint_step_750000.pth"),
    1000000: os.environ.get("CHECKPOINT", E.CONFIG["CHECKPOINT"]),
}


# --------------------------------------------------------------------------- #
# meQTL tables (bundled with the Melody code repo)
# --------------------------------------------------------------------------- #
def load_meqtl_tables() -> Dict[str, pd.DataFrame]:
    base = Path(E.CONFIG["MELODY_CODE"]) / "meqtl" / "dataset" / "processed"
    df_dict: Dict[str, pd.DataFrame] = {}
    for sub in ("GTEX", "EPIGEN", "Olafur_2024"):
        d = base / sub
        if not d.is_dir():
            continue
        for fn in sorted(os.listdir(d)):
            if fn.endswith(".csv"):
                df_dict[fn[:-4]] = pd.read_csv(d / fn, header=0, dtype=str)
    if not df_dict:
        raise FileNotFoundError(f"No meQTL tables found under {base}")
    return df_dict


_NUC = {"A": [1, 0, 0, 0], "C": [0, 1, 0, 0], "G": [0, 0, 1, 0], "T": [0, 0, 0, 1]}
_REQUIRED = ["chrom", "SNP_region_start", "SNP_region_end", "SNP_ref", "SNP_alt",
             "CPG_region_start", "CPG_region_end", "effect_size", "cpg_number"]


def _build_examples(df: pd.DataFrame):
    """Yield per-meQTL (ref_onehot, alt_onehot, obs_effect, cpg_rel, n_cpg, dist)."""
    for col in _REQUIRED:
        if col not in df.columns:
            if col == "cpg_number":
                df = df.assign(cpg_number="1")
            else:
                raise ValueError(f"meQTL table missing column: {col}")
    return df


def _predict_effects(model, device, genome, df: pd.DataFrame, track_indices: Sequence[int],
                     margin: int = MAIN_MARGIN, batch_size: int = BATCH_SIZE):
    """Return arrays (observed, predicted, distance) over all usable meQTLs in df."""
    import torch
    from stateless import sigmoid_first

    df = _build_examples(df)
    ti = torch.tensor(list(track_indices), device=device)

    obs_all: List[float] = []
    pred_all: List[float] = []
    dist_all: List[float] = []

    refs, alts, metas = [], [], []

    def flush():
        if not refs:
            return
        ref_b = torch.from_numpy(np.stack(refs)).float().to(device).transpose(1, 2)
        alt_b = torch.from_numpy(np.stack(alts)).float().to(device).transpose(1, 2)
        with torch.no_grad():
            out_ref = sigmoid_first(model(ref_b))
            out_alt = sigmoid_first(model(alt_b))
        if isinstance(out_ref, (list, tuple)):
            out_ref = out_ref[0]
        if isinstance(out_alt, (list, tuple)):
            out_alt = out_alt[0]
        diff = (out_alt.index_select(1, ti).mean(1) - out_ref.index_select(1, ti).mean(1))
        diff = diff.cpu().numpy()
        L = diff.shape[1]
        for k, (cst, ced, ncpg, obs, dist) in enumerate(metas):
            st, ed = cst - margin, ced + margin
            if not (0 <= st < L and 0 < ed <= L and st < ed):
                continue
            eff = float(diff[k, st:ed].sum()) / max(ncpg, 1.0)
            obs_all.append(obs); pred_all.append(eff); dist_all.append(dist)
        refs.clear(); alts.clear(); metas.clear()

    for _, row in df.iterrows():
        try:
            sv_s = int(float(row["SNP_region_start"]))
            sv_e = int(float(row["SNP_region_end"]))
            cpg_s = int(float(row["CPG_region_start"]))
            cpg_e = int(float(row["CPG_region_end"]))
            if len(str(row["SNP_alt"])) != 1 or sv_s != sv_e:
                continue
            center = (cpg_s + sv_s) // 2
            fs, fe = max(0, center - HALF_LEN), center + HALF_LEN
            seq = genome.get(row["chrom"], fs, fe)
            if seq is None or np.asarray(seq).shape[0] == 0:
                continue
            seq = np.asarray(seq, dtype=np.float32)
            mut = sv_s - fs - 1
            if not (0 <= mut < seq.shape[0]):
                continue
            ref = seq.copy(); alt = seq.copy()
            ref[mut] = _NUC[str(row["SNP_ref"])]
            alt[mut] = _NUC[str(row["SNP_alt"])]
            refs.append(ref); alts.append(alt)
            metas.append((cpg_s - fs, cpg_e - fs, int(float(row["cpg_number"])),
                          float(row["effect_size"]), abs(cpg_s - sv_s)))
            if len(refs) >= batch_size:
                flush()
        except Exception:
            continue
    flush()
    return np.array(obs_all), np.array(pred_all), np.array(dist_all)


# --------------------------------------------------------------------------- #
# Panel assembly
# --------------------------------------------------------------------------- #
def gen_3C_3D(model, device, genome, df_dict, st_models: Dict[str, object] | None = None):
    """3C (related vs single, MT[+ST]) and 3D (st_r vs mt_r)."""
    rows_3c = []
    mt_r: Dict[str, float] = {}
    st_r: Dict[str, float] = {}

    for ds, ti in EQTL_TO_TRACK_INDICES.items():
        if ds not in df_dict:
            continue
        df = df_dict[ds]
        # Melody-MT, related tracks (average of matched tracks)
        obs, pred, dist = _predict_effects(model, device, genome, df, ti)
        r_rel, _ = E.pearson(obs, pred)
        rows_3c.append(("Melody-MT", ds, "related tracks", r_rel))
        mt_r[ds] = r_rel
        # Melody-MT, single track (first matched track only)
        obs1, pred1, _ = _predict_effects(model, device, genome, df, ti[:1])
        r_single, _ = E.pearson(obs1, pred1)
        rows_3c.append(("Melody-MT", ds, "single track", r_single))

        # Melody-ST (single-track model for the representative cell type), if available
        if st_models:
            st_model = st_models.get(ds)
            if st_model is not None:
                obs_s, pred_s, _ = _predict_effects(st_model, device, genome, df, [0])
                r_st_rel, _ = E.pearson(obs_s, pred_s)
                rows_3c.append(("Melody-ST", ds, "related tracks", r_st_rel))
                obs_s1, pred_s1, _ = _predict_effects(st_model, device, genome, df, [0])
                rows_3c.append(("Melody-ST", ds, "single track", E.pearson(obs_s1, pred_s1)[0]))
                st_r[ds] = r_st_rel

    df_3c = pd.DataFrame(rows_3c, columns=["model", "meQTL_Dataset", "method", "pearson_r"])
    df_3d = pd.DataFrame(
        [(ds, st_r.get(ds, float("nan")), mt_r[ds]) for ds in mt_r],
        columns=["df_name", "st_r", "mt_r"],
    )
    return df_3c, df_3d


def gen_3E(model, device, genome, df_dict):
    """Pearson within cumulative SNP-CpG distance windows (dist <= w), Melody-MT."""
    rows = []
    for ds, ti in EQTL_TO_TRACK_INDICES.items():
        if ds not in df_dict:
            continue
        obs, pred, dist = _predict_effects(model, device, genome, df_dict[ds], ti)
        for w in FIG3E_WINDOWS:
            sel = dist <= w
            if sel.sum() < 2:
                continue
            r, _ = E.pearson(obs[sel], pred[sel])
            rows.append((w, "Melody-MT", ds, r))
    return pd.DataFrame(rows, columns=["distance_window", "model", "meQTL_dataset", "score"])


def gen_3F_3G(model, device, genome, df_dict):
    """Per-point observed-vs-predicted scatters for two datasets."""
    out = {}
    for tag, ds in (("3F", CPG_FG_DATASET_F), ("3G", CPG_FG_DATASET_G)):
        if ds not in df_dict:
            continue
        ti = EQTL_TO_TRACK_INDICES[ds]
        obs, pred, _ = _predict_effects(model, device, genome, df_dict[ds], ti)
        out[tag] = pd.DataFrame({"observed": obs, "predicted": pred})
    return out.get("3F"), out.get("3G")


def gen_3H(model, device, genome, df_dict):
    """Cross-cell matrix: evaluate tgt_dataset's meQTLs using src_dataset's track set.

    n_points therefore follows the tgt dataset (the meQTLs being scored).
    """
    gtex = [d for d in EQTL_TO_TRACK_INDICES if d.startswith("GTEX_") and d in df_dict]
    rows = []
    for src in gtex:                      # whose tracks are used
        for tgt in gtex:                  # the meQTL dataset being scored
            obs, pred, _ = _predict_effects(model, device, genome,
                                            df_dict[tgt], EQTL_TO_TRACK_INDICES[src])
            r, n = E.pearson(obs, pred)
            rows.append((src, tgt, n, r))
    return pd.DataFrame(rows, columns=["src_dataset", "tgt_dataset", "n_points", "pearson_r"])


def gen_3I(device, genome, df_dict):
    """Pearson per dataset across training checkpoints (uses matched tracks)."""
    rows = []
    for step, ckpt in sorted(FIG3I_STEPS.items()):
        if not Path(ckpt).is_file():
            print(f"[3I] skip step {step}: checkpoint not found ({ckpt})")
            continue
        model, _ = E.load_model(checkpoint=ckpt, device=device)
        for ds, ti in EQTL_TO_TRACK_INDICES.items():
            if ds not in df_dict:
                continue
            obs, pred, _ = _predict_effects(model, device, genome, df_dict[ds], ti)
            r, n = E.pearson(obs, pred)
            rows.append((step, ds, n, r))
        del model
    return pd.DataFrame(rows, columns=["ckpt_step", "meQTL_Dataset", "n_points", "pearson_r"])


# --------------------------------------------------------------------------- #
def main():
    out = E.out_dir()
    genome = E.load_genome()
    df_dict = load_meqtl_tables()
    print(f"Loaded {len(df_dict)} meQTL datasets: {sorted(df_dict)}")

    model, device = E.load_model()

    df_3c, df_3d = gen_3C_3D(model, device, genome, df_dict)
    df_3c.to_csv(out / "figure_3C.csv", index=False)
    df_3d.to_csv(out / "figure_3D.csv", index=False)

    gen_3E(model, device, genome, df_dict).to_csv(out / "figure_3E.csv", index=False)

    f, g = gen_3F_3G(model, device, genome, df_dict)
    if f is not None:
        f.to_csv(out / "figure_3F.csv", index=False)
    if g is not None:
        g.to_csv(out / "figure_3G.csv", index=False)

    gen_3H(model, device, genome, df_dict).to_csv(out / "figure_3H.csv", index=False)

    del model
    gen_3I(device, genome, df_dict).to_csv(out / "figure_3I.csv", index=False)

    print(f"Wrote Figure 3 source data to {out}")


if __name__ == "__main__":
    main()
