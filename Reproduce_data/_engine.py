"""
Shared engine for the Melody figure-data generation scripts.

Each ``figure*.py`` script in this directory imports the
helpers below to load the Melody model, the GRCh38 genome, and the cell-type
methylation ground truth, and to compute the per-CpG metrics that the manuscript
figures are built from.

------------------------------------------------------------------------------
What you need to run these scripts
------------------------------------------------------------------------------
These scripts re-derive the source-data CSVs (the files under ``Data/``) from the
*model* and the *raw methylation data*. They therefore need a GPU machine with:

  1. The Melody model code        -> set ``MELODY_CODE``  (the public Melody repo)
  2. A trained checkpoint (.pth)  -> set ``CHECKPOINT``   (e.g. the 39-track Melody-MT)
  3. The GRCh38 primary assembly  -> set ``GENOME_FA``
  4. The 39 cell-type BigWig tracks -> set ``BIGWIG_DIR`` (methylation ground truth;
                                       NOT needed by the meQTL script, which uses
                                       only the bundled meQTL tables)

Set them as environment variables, e.g.::

    export MELODY_CODE=/path/to/Melody
    export CHECKPOINT=/path/to/Melody-MT-39.pth
    export GENOME_FA=/path/to/GRCh38.primary_assembly.fa
    export BIGWIG_DIR=/path/to/bigwigs
    export OUT_DIR=./generated_csv      # where regenerated CSVs are written

or edit the defaults in the ``CONFIG`` block below. Regenerated CSVs are written
to ``OUT_DIR`` (default ``./generated_csv``); they are meant to be compared
against the committed files in ``Data/`` rather than to overwrite them.
"""

from __future__ import annotations

import os
import re
import sys
from pathlib import Path
from typing import List, Tuple

import numpy as np

# --------------------------------------------------------------------------- #
# CONFIG -- override via environment variables or edit these placeholders.
# --------------------------------------------------------------------------- #
CONFIG = {
    "MELODY_CODE": os.environ.get("MELODY_CODE", "/path/to/Melody"),
    "CHECKPOINT":  os.environ.get("CHECKPOINT",  "/path/to/Melody-MT-39.pth"),
    "GENOME_FA":   os.environ.get("GENOME_FA",   "/path/to/GRCh38.primary_assembly.fa"),
    "BIGWIG_DIR":  os.environ.get("BIGWIG_DIR",  "/path/to/bigwigs"),
    # Directory of 39 single-track (Melody-ST) checkpoints, one per cell type,
    # named ``<cell>.pth`` (only needed by scripts that compare ST vs MT).
    "ST_CKPT_DIR": os.environ.get("ST_CKPT_DIR", "/path/to/melody_st_checkpoints"),
    "OUT_DIR":     os.environ.get("OUT_DIR",     "./generated_csv"),
    "DEVICE":      os.environ.get("DEVICE",      "cuda"),
    "N_TRACK":     int(os.environ.get("N_TRACK", "39")),
    "WINDOW":      int(os.environ.get("WINDOW",  "10000")),
}

# Blood cell types are tracks 21-26 in the 39-track ordering; several panels
# average them. (Index order is fixed by track_39_bigwig_file_names.)
BLOOD_TRACKS = [21, 22, 23, 24, 25, 26]


# --------------------------------------------------------------------------- #
# Repo bootstrap
# --------------------------------------------------------------------------- #
def setup_repo(code_dir: str | None = None) -> None:
    """Put the Melody model code on ``sys.path`` so its modules import cleanly."""
    code_dir = code_dir or CONFIG["MELODY_CODE"]
    if not Path(code_dir).is_dir():
        raise FileNotFoundError(
            f"MELODY_CODE does not exist: {code_dir}\n"
            "Point it at a checkout of the public Melody model repository."
        )
    if code_dir not in sys.path:
        sys.path.insert(0, code_dir)


def out_dir() -> Path:
    d = Path(CONFIG["OUT_DIR"])
    d.mkdir(parents=True, exist_ok=True)
    return d


# --------------------------------------------------------------------------- #
# Model / genome / truth loaders (use the public Melody API)
# --------------------------------------------------------------------------- #
def load_model(checkpoint: str | None = None, n_track: int | None = None,
               device: str | None = None):
    """Build a Melody model and load a checkpoint. Returns ``model.eval()``."""
    setup_repo()
    import torch
    from models import Melody
    from stateless import load_ckpt

    checkpoint = checkpoint or CONFIG["CHECKPOINT"]
    n_track = n_track or CONFIG["N_TRACK"]
    device = device or CONFIG["DEVICE"]
    if device == "cuda" and not torch.cuda.is_available():
        device = "cpu"

    model = Melody(None, n_track=n_track)
    load_ckpt(model, checkpoint, mute=True)
    model = model.to(device).eval()
    return model, device


def load_genome(genome_fa: str | None = None, cuda: bool = False):
    """Load the GRCh38 genome via selene-sdk (the public Melody dependency)."""
    setup_repo()
    from selene_sdk.sequences import Genome

    genome = Genome(input_path=genome_fa or CONFIG["GENOME_FA"], blacklist_regions="hg38")
    # Match training-time accessor (returns an (L, 4) one-hot in A,C,G,T order).
    genome.get = genome.get_encoding_from_coords
    return genome


def load_truth(bigwig_dir: str | None = None, window: int | None = None):
    """Methylation ground truth as a selene GenomicSignalFeatures over 39 BigWigs.

    Returns ``(methy_data, track_names)`` where ``methy_data.get(chrom, s, e)``
    yields a ``(n_track, length)`` array of per-base methylation.
    """
    setup_repo()
    from stateless import get_bigwig_filepaths
    from selene_util import GenomicSignalFeatures
    from global_constants import track_39_bigwig_file_names

    bigwig_files, track_names = get_bigwig_filepaths(
        dir_=bigwig_dir or CONFIG["BIGWIG_DIR"], filenames=track_39_bigwig_file_names
    )
    track_names = [n.replace(".hg38.bigwig", "") for n in track_names]
    methy_data = GenomicSignalFeatures(
        bigwig_files, features=bigwig_files, shape=(window or CONFIG["WINDOW"],)
    )
    return methy_data, track_names


def track_names() -> List[str]:
    setup_repo()
    from global_constants import track_39_names
    return list(track_39_names)


_GSM_PREFIX = re.compile(r"^GSM\d+_")
_Z_SUFFIX = re.compile(r"-Z\w+$")


def short_track_name(name: str) -> str:
    """Map a full track name to its short label.

    ``GSM5652176_Adipocytes-Z000000T`` -> ``Adipocytes``. The CST split and
    panel 2G use the short label; the whole-genome / sampling splits keep the
    full BigWig-derived name. Names that don't match are returned unchanged.
    """
    return _Z_SUFFIX.sub("", _GSM_PREFIX.sub("", name))


def cpg_mask(seq_onehot: np.ndarray) -> np.ndarray:
    """Boolean CpG mask for a single (L, 4) one-hot sequence (A,C,G,T order)."""
    setup_repo()
    from check_utils import get_one_hot_embedding_mask
    return np.asarray(get_one_hot_embedding_mask(seq_onehot, mask_type="CpG")).ravel().astype(bool)


# --------------------------------------------------------------------------- #
# Metric helpers (single-base methylation in [0, 1])
# --------------------------------------------------------------------------- #
def pearson(x: np.ndarray, y: np.ndarray) -> Tuple[float, int]:
    x = np.asarray(x, dtype=np.float64).ravel()
    y = np.asarray(y, dtype=np.float64).ravel()
    m = ~(np.isnan(x) | np.isnan(y))
    x, y = x[m], y[m]
    if x.size < 2 or x.std() < 1e-8 or y.std() < 1e-8:
        return float("nan"), int(x.size)
    return float(np.corrcoef(x, y)[0, 1]), int(x.size)


def spearman(x: np.ndarray, y: np.ndarray) -> Tuple[float, int]:
    from scipy.stats import spearmanr
    x = np.asarray(x, dtype=np.float64).ravel()
    y = np.asarray(y, dtype=np.float64).ravel()
    m = ~(np.isnan(x) | np.isnan(y))
    x, y = x[m], y[m]
    if x.size < 2:
        return float("nan"), int(x.size)
    r = spearmanr(x, y)
    c = r.correlation if hasattr(r, "correlation") else r[0]
    return (float(c) if c is not None else float("nan")), int(x.size)


def mae(x: np.ndarray, y: np.ndarray) -> Tuple[float, int]:
    x = np.asarray(x, dtype=np.float64).ravel()
    y = np.asarray(y, dtype=np.float64).ravel()
    m = ~(np.isnan(x) | np.isnan(y))
    x, y = x[m], y[m]
    if x.size == 0:
        return float("nan"), 0
    return float(np.abs(x - y).mean()), int(x.size)


def auroc_binary(pred: np.ndarray, truth: np.ndarray, thresh: float = 0.5) -> float:
    """AUROC of continuous ``pred`` against truth binarized at ``thresh``."""
    from sklearn.metrics import roc_auc_score
    pred = np.asarray(pred, dtype=np.float64).ravel()
    yb = (np.asarray(truth, dtype=np.float64).ravel() > thresh).astype(np.int8)
    if yb.min() == yb.max():
        return float("nan")
    return float(roc_auc_score(yb, pred))


def truth_qc(truth: np.ndarray) -> np.ndarray:
    """Keep only positions whose ground-truth methylation lies in [0, 1].

    BigWig tracks can carry out-of-range sentinel/noise values; the manuscript
    metrics ('QC' variants) restrict to valid [0, 1] truth.
    """
    t = np.asarray(truth, dtype=np.float64)
    return (t >= 0.0) & (t <= 1.0)
