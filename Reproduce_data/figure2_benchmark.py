"""
Generate the per-track benchmark source data for Figure 2.

For one Melody checkpoint this evaluates, per cell-type track and per split, the
CpG-level agreement between predicted and observed methylation:

    Spearman, Pearson, MAE, AUC (truth binarized at 0.5), Accuracy

over the five splits used in the paper:

    SAMPLE_VALID / SAMPLE_TEST : random 10-kb regions drawn from the
                                 validation / test chromosomes
    VALID                      : non-overlapping 10-kb tiling of chr10
    TEST                       : non-overlapping 10-kb tiling of chr8 + chr9
    CST                        : cell-type-specific regions (cell_zones/test_data_dict.json),
                                 each cell type scored only on its own track

Metrics are computed only over CpG positions whose ground-truth methylation lies
in [0, 1] (the 'QC' convention).

Panels produced (written to OUT_DIR; compare against Data/):
  figure_2AB.csv  -> model,track,split,metric,value   (metric in {Spearman, AUC})
  figure_2G.csv   -> model,track,split,AUC,count       (7 selected tracks, VALID/TEST)

Run once per model. ``MODEL_NAME`` labels the rows (e.g. Melody-MT / Melody-ST).
Baseline rows (INTERACT, CpGenie, iDNA-ABT) are produced by the respective
baseline models with the same metric definitions and concatenated afterwards.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

import _engine as E

MODEL_NAME = os.environ.get("MODEL_NAME", "Melody-MT")
WINDOW = 10000
BATCH = 64
AUC_BINS = 20480
SPEARMAN_CAP = 300_000

VALID_CHROMS = ["chr10"]
TEST_CHROMS = ["chr8", "chr9"]

# The seven tracks shown in Figure 2G.
FIG2G_TRACKS = [
    "Aorta-Smooth-Muscle", "Bone-Osteoblasts", "Colon-Right-Epithelial",
    "Cortex-Neuron", "Heart-Cardiomyocyte", "Heart-Fibroblasts", "Ovary-Epithelial",
]


# --------------------------------------------------------------------------- #
# Streaming CpG metrics (exact Pearson/MAE/Accuracy; histogram AUC; reservoir Spearman)
# --------------------------------------------------------------------------- #
class TrackMetrics:
    def __init__(self, seed: int, qc: str = "clip01"):
        # qc: 'clip01' keeps truth in [0, 1]; 'nonneg' only excludes truth < 0
        # (the CST evaluation uses 'nonneg', matching the original pipeline).
        self.qc = qc
        self.n = 0
        self.sp = self.st = self.spp = self.stt = self.spt = 0.0
        self.abs_err = 0.0
        self.acc = 0
        self.pos = np.zeros(AUC_BINS, dtype=np.int64)
        self.neg = np.zeros(AUC_BINS, dtype=np.int64)
        self._rng = np.random.default_rng(seed)
        self._keys = np.empty(0, np.float32)
        self._p = np.empty(0, np.float32)
        self._t = np.empty(0, np.float32)

    def update(self, pred: np.ndarray, truth: np.ndarray):
        p = np.asarray(pred, np.float64).ravel()
        t = np.asarray(truth, np.float64).ravel()
        keep = (t >= 0.0) if self.qc == "nonneg" else ((t >= 0.0) & (t <= 1.0))
        p, t = p[keep], t[keep]
        if p.size == 0:
            return
        self.n += p.size
        self.sp += p.sum(); self.st += t.sum()
        self.spp += float(p @ p); self.stt += float(t @ t); self.spt += float(p @ t)
        self.abs_err += float(np.abs(p - t).sum())
        self.acc += int(np.count_nonzero((p > 0.5) == (t > 0.5)))
        pc = np.clip(p, 0, 1)
        b = np.minimum((pc * (AUC_BINS - 1)).astype(np.int32), AUC_BINS - 1)
        tb = (t > 0.5)
        self.pos += np.bincount(b[tb], minlength=AUC_BINS)
        self.neg += np.bincount(b[~tb], minlength=AUC_BINS)
        # reservoir for Spearman
        k = self._rng.random(p.size, dtype=np.float32)
        self._keys = np.concatenate([self._keys, k])
        self._p = np.concatenate([self._p, p.astype(np.float32)])
        self._t = np.concatenate([self._t, t.astype(np.float32)])
        if self._keys.size > SPEARMAN_CAP:
            idx = np.argpartition(self._keys, SPEARMAN_CAP - 1)[:SPEARMAN_CAP]
            self._keys, self._p, self._t = self._keys[idx], self._p[idx], self._t[idx]

    def pearson(self) -> float:
        if self.n < 2:
            return float("nan")
        num = self.n * self.spt - self.sp * self.st
        den = (self.n * self.spp - self.sp ** 2) * (self.n * self.stt - self.st ** 2)
        return float(num / np.sqrt(den)) if den > 0 else float("nan")

    def spearman(self) -> float:
        return E.spearman(self._p, self._t)[0]

    def mae(self) -> float:
        return float(self.abs_err / self.n) if self.n else float("nan")

    def accuracy(self) -> float:
        return float(self.acc / self.n) if self.n else float("nan")

    def auc(self) -> float:
        tp, tn = int(self.pos.sum()), int(self.neg.sum())
        if tp == 0 or tn == 0:
            return float("nan")
        cum_neg, conc = 0, 0.0
        for pc, nc in zip(self.pos, self.neg):
            conc += float(pc) * (cum_neg + 0.5 * float(nc))
            cum_neg += int(nc)
        return float(conc / (tp * tn))

    def as_dict(self) -> Dict[str, float]:
        return {"Spearman": self.spearman(), "Pearson": self.pearson(), "MAE": self.mae(),
                "AUC": self.auc(), "Accuracy": self.accuracy()}


# --------------------------------------------------------------------------- #
# Prediction helpers
# --------------------------------------------------------------------------- #
def _predict(model, device, seq_bchw):
    import torch
    from stateless import sigmoid_first
    with torch.no_grad():
        out = sigmoid_first(model(seq_bchw))
    if isinstance(out, (list, tuple)):
        out = out[0]
    return out.detach().cpu().numpy()      # (B, n_track, L) methylation in [0,1]


def _eval_windows(model, device, genome, methy, track_names, windows: List[Tuple[str, int]]):
    """Score a list of (chrom, start) 10-kb windows; returns {track: TrackMetrics}."""
    import torch
    acc = {t: TrackMetrics(seed=1000 + i) for i, t in enumerate(track_names)}
    buf_seq, buf_meta = [], []

    def flush():
        if not buf_seq:
            return
        seq_np = np.stack(buf_seq)                                  # (B, L, 4)
        seq_t = torch.from_numpy(seq_np).float().permute(0, 2, 1).to(device)
        pred = _predict(model, device, seq_t)                       # (B, T, L)
        for bi, (chrom, start) in enumerate(buf_meta):
            cpg = E.cpg_mask(seq_np[bi])
            if not cpg.any():
                continue
            truth = methy.get(chrom, start, start + WINDOW)         # (T, L)
            for ti, tname in enumerate(track_names):
                acc[tname].update(pred[bi, ti, cpg], truth[ti, cpg])
        buf_seq.clear(); buf_meta.clear()

    for chrom, start in windows:
        seq = genome.get(chrom, start, start + WINDOW)
        if seq is None or np.asarray(seq).shape[0] != WINDOW:
            continue
        seq = np.asarray(seq, np.float32)
        if np.all(seq == 0.25):                                     # all-N window
            continue
        buf_seq.append(seq); buf_meta.append((chrom, start))
        if len(buf_seq) >= BATCH:
            flush()
    flush()
    return acc


def _tile(genome, chroms: List[str]) -> List[Tuple[str, int]]:
    lens = dict(genome.get_chr_lens())
    out = []
    for c in chroms:
        for s in range(0, lens.get(c, 0) - WINDOW + 1, WINDOW):
            out.append((c, s))
    return out


def _sample_runs(split: str) -> List[Tuple[str, int]]:
    from check_utils import get_all_run_splits_var_blocks
    _, valid_runs, test_runs = get_all_run_splits_var_blocks(window_size=WINDOW)
    return valid_runs if split == "SAMPLE_VALID" else test_runs


def _normalize_region(r):
    if not isinstance(r, (list, tuple)) or len(r) < 3:
        return None
    ch, s, e = str(r[0]), int(r[1]), int(r[2])
    if s == e:
        return None
    if s > e:
        s, e = e, s
    return ch, s, e


def _cst_window(rs, re, ws, cl):
    """Centre the CST region in a model window; return (fetch_start, fetch_end, rel_start, rel_end)."""
    if cl < ws:
        return None
    c = (rs + re) // 2
    fs = c - ws // 2
    fe = fs + ws
    if fs < 0:
        fe -= fs
        fs = 0
    if fe > cl:
        fs -= fe - cl
        fe = cl
    if fs < 0:
        fs, fe = 0, ws
    if fe - fs != ws:
        return None
    ts, te = max(rs, fs), min(re, fe)
    if te <= ts:
        return None
    return fs, fe, ts - fs, te - fs


def _match_track(cell_type, track_names):
    exact = [i for i, n in enumerate(track_names) if n == cell_type]
    if len(exact) == 1:
        return exact[0]
    fuzzy = [i for i, n in enumerate(track_names) if cell_type in n or n in cell_type]
    if len(fuzzy) == 1:
        return fuzzy[0]
    return None


def _eval_cst(model, device, genome, methy, track_names):
    """Cell-type-specific regions: each cell type scored only on its own track.

    Window handling and QC follow the original CST evaluation: the region is
    centred in the model window (``_cst_window``), and truth QC only excludes
    negative labels (qc='nonneg').
    """
    import torch
    cst_path = Path(E.CONFIG["MELODY_CODE"]) / "cell_zones" / "test_data_dict.json"
    cst = json.loads(cst_path.read_text())
    lens = dict(genome.get_chr_lens())
    acc: Dict[str, TrackMetrics] = {}

    items = []
    for cell, regions in cst.items():
        ti = _match_track(cell, track_names)
        if ti is None:
            continue
        acc[cell] = TrackMetrics(seed=hash(cell) % 99999, qc="nonneg")
        for r in regions:
            nr = _normalize_region(r)
            if nr is None:
                continue
            ch, rs, re = nr
            w = _cst_window(rs, re, WINDOW, lens.get(ch, 0))
            if w is None:
                continue
            fs, fe, rel_s, rel_e = w
            items.append((cell, ti, ch, fs, rel_s, rel_e))

    for i in range(0, len(items), BATCH):
        batch = items[i:i + BATCH]
        seqs = [np.asarray(genome.get(ch, fs, fs + WINDOW), np.float32) for _, _, ch, fs, _, _ in batch]
        seq_t = torch.from_numpy(np.stack(seqs)).float().permute(0, 2, 1).to(device)
        pred = _predict(model, device, seq_t)
        for bi, (cell, ti, ch, fs, rel_s, rel_e) in enumerate(batch):
            sub_seq = seqs[bi][rel_s:rel_e]
            cpg = E.cpg_mask(sub_seq)
            if not cpg.any():
                continue
            truth = methy.get(ch, fs, fs + WINDOW)[ti, rel_s:rel_e]
            acc[cell].update(pred[bi, ti, rel_s:rel_e][cpg], truth[cpg])
    return acc


ALL_SPLITS = ["SAMPLE_VALID", "SAMPLE_TEST", "VALID", "TEST", "CST"]


def evaluate_all_splits(model, device, genome, methy, track_names,
                        splits: List[str] = None) -> Dict[str, Dict[str, TrackMetrics]]:
    """Per-track CpG metrics for the requested splits. Reused by the S2/S3/S4 scripts."""
    splits = splits or ALL_SPLITS
    per_split: Dict[str, Dict[str, TrackMetrics]] = {}
    for split in splits:
        print(f"{split} ...")
        if split in ("SAMPLE_VALID", "SAMPLE_TEST"):
            per_split[split] = _eval_windows(model, device, genome, methy, track_names, _sample_runs(split))
        elif split == "VALID":
            per_split[split] = _eval_windows(model, device, genome, methy, track_names, _tile(genome, VALID_CHROMS))
        elif split == "TEST":
            per_split[split] = _eval_windows(model, device, genome, methy, track_names, _tile(genome, TEST_CHROMS))
        elif split == "CST":
            per_split[split] = _eval_cst(model, device, genome, methy, track_names)
    return per_split


# --------------------------------------------------------------------------- #
def main():
    out = E.out_dir()
    genome = E.load_genome()
    methy, track_names = E.load_truth()
    model, device = E.load_model()
    print(f"[{MODEL_NAME}] {len(track_names)} tracks on {device}")

    per_split = evaluate_all_splits(model, device, genome, methy, track_names)

    # ---- assemble long-format rows ----
    # Track-name convention matching the published source data: the whole-genome
    # / sampling splits keep the full BigWig-derived names; the CST split and
    # Figure 2G use the short cell-type labels.
    ab_rows, g_rows = [], []
    for split, accs in per_split.items():
        is_cst = split == "CST"
        for track, m in accs.items():
            short = E.short_track_name(track)
            name = short if is_cst else track
            d = m.as_dict()
            for metric, value in d.items():
                if metric in ("Spearman", "AUC"):
                    ab_rows.append((MODEL_NAME, name, split, metric, value))
            if split in ("VALID", "TEST") and short in FIG2G_TRACKS:
                g_rows.append((MODEL_NAME, short, split, d["AUC"], m.n))

    import pandas as pd
    pd.DataFrame(ab_rows, columns=["model", "track", "split", "metric", "value"]).to_csv(out / "figure_2AB.csv", index=False)
    pd.DataFrame(g_rows, columns=["model", "track", "split", "AUC", "count"]).to_csv(out / "figure_2G.csv", index=False)
    print(f"Wrote figure_2AB.csv / figure_2G.csv to {out} (model={MODEL_NAME})")
    print("Re-run with MODEL_NAME=Melody-ST + an ST checkpoint, then concatenate the per-model CSVs.")


if __name__ == "__main__":
    main()
