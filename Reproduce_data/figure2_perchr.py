"""
Generate per-chromosome / genome-pooled source data for Figure 2 C, D, E, F.

This runs whole-genome inference (chr1-22 + chrX, chrY excluded) for every cell
type and compares predicted vs observed methylation at CpG positions whose truth
lies in [0, 1] ('QC'). It is the heaviest script here: 23 chromosomes x 39 cell
types of dense prediction. Expect it to run for hours on a single GPU.

Panels produced (written to OUT_DIR; compare against Data/Figure2/):
  figure_2C.csv  -> chr,model,cell,corr,mse,count     (per chromosome, per cell)
  figure_2D.csv  -> model,fpr,tpr                      (ROC, pooled over all cells+chroms)
  figure_2E.csv  -> model,recall,precision             (PR,  pooled over all cells+chroms)
  figure_2F.csv  -> bin_mid,model,mse,count            (MSE within 0.1-wide truth bins)

Truth is binarized at 0.5 for the ROC/PR curves. Baseline rows (INTERACT,
CpGenie, iDNA-ABT) come from the baseline models; this script emits MODEL_NAME.
"""

from __future__ import annotations

import os
from typing import Dict, List, Tuple

import numpy as np

import _engine as E

MODEL_NAME = os.environ.get("MODEL_NAME", "Melody-MT")
WINDOW = 10000
BATCH = 64
# chrY excluded. Override with CHROMS env (comma-separated) for partial runs.
CHROMS = (os.environ["CHROMS"].split(",") if os.environ.get("CHROMS")
          else [f"chr{i}" for i in range(1, 23)] + ["chrX"])
CURVE_POINTS = 10000          # points sampled per ROC / PR curve (matches the paper)
SCORE_BINS = 20000            # histogram resolution for pooled ROC/PR
TRUTH_BIN_EDGES = np.arange(0.0, 1.1 + 1e-9, 0.1)       # 11 bins -> mids 0.05..1.05


class CorrMSE:
    """Streaming Pearson + MSE for one (chromosome, cell)."""
    __slots__ = ("n", "sp", "st", "spp", "stt", "spt", "sse")

    def __init__(self):
        self.n = 0
        self.sp = self.st = self.spp = self.stt = self.spt = self.sse = 0.0

    def update(self, p: np.ndarray, t: np.ndarray):
        self.n += p.size
        self.sp += p.sum(); self.st += t.sum()
        self.spp += float(p @ p); self.stt += float(t @ t); self.spt += float(p @ t)
        self.sse += float(((p - t) ** 2).sum())

    def corr(self) -> float:
        if self.n < 2:
            return float("nan")
        num = self.n * self.spt - self.sp * self.st
        den = (self.n * self.spp - self.sp ** 2) * (self.n * self.stt - self.st ** 2)
        return float(num / np.sqrt(den)) if den > 0 else float("nan")

    def mse(self) -> float:
        return float(self.sse / self.n) if self.n else float("nan")


def _predict(model, device, seq_bchw):
    import torch
    from stateless import sigmoid_first
    with torch.no_grad():
        out = sigmoid_first(model(seq_bchw))
    if isinstance(out, (list, tuple)):
        out = out[0]
    return out.detach().cpu().numpy()


def main():
    import torch
    import pandas as pd

    out = E.out_dir()
    genome = E.load_genome()
    methy, track_names = E.load_truth()
    model, device = E.load_model()
    lens = dict(genome.get_chr_lens())

    per_chr_cell: Dict[Tuple[str, str], CorrMSE] = {}
    pos_hist = np.zeros(SCORE_BINS, np.int64)        # pooled ROC/PR: truth>0.5
    neg_hist = np.zeros(SCORE_BINS, np.int64)
    bin_sse = np.zeros(len(TRUTH_BIN_EDGES) - 1, np.float64)   # 2F per truth bin
    bin_cnt = np.zeros(len(TRUTH_BIN_EDGES) - 1, np.int64)

    for chrom in CHROMS:
        clen = lens.get(chrom, 0)
        starts = list(range(0, clen - WINDOW + 1, WINDOW))
        print(f"{chrom}: {len(starts)} windows")
        buf_seq, buf_start = [], []

        def flush():
            if not buf_seq:
                return
            seq_np = np.stack(buf_seq)
            seq_t = torch.from_numpy(seq_np).float().permute(0, 2, 1).to(device)
            pred = _predict(model, device, seq_t)
            for bi, start in enumerate(buf_start):
                cpg = E.cpg_mask(seq_np[bi])
                if not cpg.any():
                    continue
                truth_all = methy.get(chrom, start, start + WINDOW)   # (T, L)
                for ti, cell in enumerate(track_names):
                    t = truth_all[ti, cpg].astype(np.float64)
                    p = pred[bi, ti, cpg].astype(np.float64)
                    keep = (t >= 0.0) & (t <= 1.0)
                    p, t = p[keep], t[keep]
                    if p.size == 0:
                        continue
                    per_chr_cell.setdefault((chrom, cell), CorrMSE()).update(p, t)
                    # pooled ROC/PR histogram
                    b = np.minimum((np.clip(p, 0, 1) * (SCORE_BINS - 1)).astype(np.int32), SCORE_BINS - 1)
                    tb = t > 0.5
                    pos_hist[:] += np.bincount(b[tb], minlength=SCORE_BINS)
                    neg_hist[:] += np.bincount(b[~tb], minlength=SCORE_BINS)
                    # 2F truth bins
                    tbin = np.minimum(np.digitize(t, TRUTH_BIN_EDGES) - 1, len(bin_cnt) - 1)
                    se = (p - t) ** 2
                    np.add.at(bin_sse, tbin, se)
                    np.add.at(bin_cnt, tbin, 1)
            buf_seq.clear(); buf_start.clear()

        for start in starts:
            seq = genome.get(chrom, start, start + WINDOW)
            if seq is None or np.asarray(seq).shape[0] != WINDOW:
                continue
            seq = np.asarray(seq, np.float32)
            if np.all(seq == 0.25):
                continue
            buf_seq.append(seq); buf_start.append(start)
            if len(buf_seq) >= BATCH:
                flush()
        flush()

    # ---- 2C ----
    rows_2c = [(chrom, MODEL_NAME, cell, m.corr(), m.mse(), m.n)
               for (chrom, cell), m in sorted(per_chr_cell.items())]
    pd.DataFrame(rows_2c, columns=["chr", "model", "cell", "corr", "mse", "count"]).to_csv(
        out / "figure_2C.csv", index=False)

    # ---- 2D / 2E : ROC + PR from the pooled histogram (high score -> low) ----
    P, N = int(pos_hist.sum()), int(neg_hist.sum())
    cum_tp = np.cumsum(pos_hist[::-1]).astype(np.float64)
    cum_fp = np.cumsum(neg_hist[::-1]).astype(np.float64)
    tpr = cum_tp / max(P, 1)
    fpr = cum_fp / max(N, 1)
    recall = cum_tp / max(P, 1)
    precision = np.divide(cum_tp, cum_tp + cum_fp, out=np.ones_like(cum_tp), where=(cum_tp + cum_fp) > 0)
    sel = np.linspace(0, SCORE_BINS - 1, CURVE_POINTS).astype(int)
    pd.DataFrame({"model": MODEL_NAME, "fpr": fpr[sel], "tpr": tpr[sel]}).to_csv(
        out / "figure_2D.csv", index=False)
    pd.DataFrame({"model": MODEL_NAME, "recall": recall[sel], "precision": precision[sel]}).to_csv(
        out / "figure_2E.csv", index=False)

    # ---- 2F ----
    mids = (TRUTH_BIN_EDGES[:-1] + TRUTH_BIN_EDGES[1:]) / 2
    rows_2f = [(round(float(mids[i]), 2), MODEL_NAME,
                float(bin_sse[i] / bin_cnt[i]) if bin_cnt[i] else float("nan"), int(bin_cnt[i]))
               for i in range(len(mids))]
    pd.DataFrame(rows_2f, columns=["bin_mid", "model", "mse", "count"]).to_csv(
        out / "figure_2F.csv", index=False)

    print(f"Wrote figure_2C/2D/2E/2F.csv to {out} (model={MODEL_NAME})")


if __name__ == "__main__":
    main()
