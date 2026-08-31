#!/usr/bin/env python3
"""Re-evaluate Figure 5B on the 34 seen methylation tracks.

This script preserves the sampled-variable-block and chromosome-wide split
definitions, exact CpG-level AUROC accumulation, model/checkpoint loading, and
long source-data schema from the unified benchmark.
"""

from __future__ import annotations

import argparse
import csv
import gc
import importlib
import json
import math
import os
import random
import sys
import types
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Sequence, Tuple

import numpy as np


PANEL = "5B"
MODEL_KINDS = ("g1", "g2")
MODEL_NAMES = {
    "g1": "Melody-G1",
    "g2": "Melody-G2",
}
UNSEEN_CELL_TYPES = frozenset(
    (
        "Pancreas-Delta",
        "Blood-Granulocytes",
        "Blood-Monocytes",
        "Aorta-Endothel",
        "Cortex-Neuron",
    )
)
FIGURE5B_SPLIT_ORDER = ("SAMPLE_VALID", "SAMPLE_TEST", "VALID", "TEST")
VALID_CHROMS = ("chr10",)
TEST_CHROMS = ("chr8", "chr9")
FIGURE5B_SEEN_TRACK_NAMES = (
    "GSM5652176_Adipocytes-Z000000T7",
    "GSM5652189_Kidney-Tubular-Endothel-Z0000042R",
    "GSM5652198_Colon-Fibroblasts-Z0000042A",
    "GSM5652200_Heart-Fibroblasts-Z0000043R",
    "GSM5652204_Dermal-Fibroblasts-Z00000423",
    "GSM5652205_Skeletal-Muscle-Z00000427",
    "GSM5652207_Aorta-Smooth-Muscle-Z0000041U",
    "GSM5652215_Heart-Cardiomyocyte-Z0000044P",
    "GSM5652218_Bone-Osteoblasts-Z0000042Z",
    "GSM5652219_Oligodendrocytes-Z000000TK",
    "GSM5652233_Liver-Hepatocytes-Z000000R3",
    "GSM5652239_Pancreas-Duct-Z0000043T",
    "GSM5652243_Pancreas-Acinar-Z000000QX",
    "GSM5652250_Pancreas-Beta-Z00000452",
    "GSM5652253_Pancreas-Alpha-Z00000453",
    "GSM5652264_Thyroid-Epithelial-Z0000042S",
    "GSM5652267_Fallopian-Epithelial-Z000000Q7",
    "GSM5652270_Ovary-Epithelial-Z000000QT",
    "GSM5652274_Bone_marrow-Erythrocyte_progenitors-Z000000RF",
    "GSM5652277_Blood-T-CD3-Z000000TV",
    "GSM5652299_Blood-NK-Z000000TM",
    "GSM5652317_Blood-B-Z000000UB",
    "GSM5652321_Epidermal-Keratinocytes-Z00000424",
    "GSM5652322_Tonsil-Palatine-Epithelial-Z000000QF",
    "GSM5652335_Lung-Bronchus-Epithelial-Z000000QD",
    "GSM5652338_Prostate-Epithelial-Z000000RV",
    "GSM5652342_Bladder-Epithelial-Z000000QM",
    "GSM5652347_Breast-Luminal-Epithelial-Z000000V2",
    "GSM5652350_Breast-Basal-Epithelial-Z000000V6",
    "GSM5652354_Lung-Alveolar-Epithelial-Z000000T1",
    "GSM5652358_Gallbladder-Epithelial-Z00000432",
    "GSM5652359_Gastric-fundus-Epithelial-Z000000RX",
    "GSM5652370_Colon-Right-Epithelial-Z000000V0",
    "GSM5652378_Small-int-Epithelial-Z0000042V",
)
CELL_EMBEDDING_DIM = 512
FIGURE5B_CSV_FIELDS = ["model", "track", "split", "metric", "value"]
SCRIPT_DIR = Path(__file__).resolve().parent
SUBMIT_DIR = SCRIPT_DIR.parent
DEFAULT_OUTPUT_CSV = SUBMIT_DIR / "generated_csv" / "figure_5B_G1_G2.csv"


@dataclass(frozen=True)
class TrackInput:
    name: str
    cell_type: str
    bigwig_path: Path
    embedding_path: Path


@dataclass
class ExactAUCAccumulator:
    """Exact per-window CpG score accumulator used only by Figure 5B."""

    pred_parts: List[np.ndarray]
    label_parts: List[np.ndarray]
    windows: int = 0
    cpg_count: int = 0

    @classmethod
    def create(cls) -> "ExactAUCAccumulator":
        return cls(pred_parts=[], label_parts=[])

    def update_batch(
        self,
        predictions: np.ndarray,
        truths: np.ndarray,
        sequences: np.ndarray,
    ) -> None:
        predictions = np.asarray(predictions)
        truths = np.asarray(truths)
        sequences = np.asarray(sequences)
        if predictions.ndim != 2 or truths.ndim != 2 or sequences.ndim != 3:
            raise ValueError(
                "Expected predictions/truths (B,L) and sequences (B,L,4), got "
                f"{predictions.shape}, {truths.shape}, {sequences.shape}"
            )
        if (
            predictions.shape != truths.shape
            or predictions.shape != sequences.shape[:2]
        ):
            raise ValueError(
                "Batch dimensions differ: "
                f"pred={predictions.shape}, truth={truths.shape}, seq={sequences.shape}"
            )

        for prediction, truth, sequence in zip(predictions, truths, sequences):
            keep = cpg_mask(sequence) & np.isfinite(truth) & (truth >= 0.0)
            self.windows += 1
            if not np.any(keep):
                continue
            selected_prediction = np.asarray(prediction[keep], dtype=np.float32)
            # Figure 5B first applied methy2bin: raw values >= 0.5 are positive.
            selected_label = np.asarray(truth[keep] >= 0.5, dtype=np.bool_)
            self.pred_parts.append(selected_prediction)
            self.label_parts.append(selected_label)
            self.cpg_count += int(selected_prediction.size)

    def finalize(self) -> Dict[str, float | int]:
        from sklearn.metrics import roc_auc_score

        if not self.pred_parts:
            return {
                "auc_CpG": float("nan"),
                "count_CpG": 0,
                "positive_CpG": 0,
                "negative_CpG": 0,
                "windows": self.windows,
            }

        predictions = np.concatenate(self.pred_parts)
        labels = np.concatenate(self.label_parts)
        positives = int(np.count_nonzero(labels))
        negatives = int(labels.size - positives)
        auc = (
            float(roc_auc_score(labels, predictions))
            if positives > 0 and negatives > 0
            else float("nan")
        )
        result: Dict[str, float | int] = {
            "auc_CpG": auc,
            "count_CpG": int(labels.size),
            "positive_CpG": positives,
            "negative_CpG": negatives,
            "windows": self.windows,
        }
        self.pred_parts.clear()
        self.label_parts.clear()
        return result


def _require_file(path: Path, label: str) -> Path:
    path = path.expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"{label} does not exist or is not a file: {path}")
    return path


def _require_dir(path: Path, label: str) -> Path:
    path = path.expanduser().resolve()
    if not path.is_dir():
        raise FileNotFoundError(f"{label} does not exist or is not a directory: {path}")
    return path


def strip_bigwig_suffix(filename: str) -> str:
    for suffix in (".hg38.bigwig", ".bigwig", ".bw"):
        if filename.endswith(suffix):
            return filename[: -len(suffix)]
    return Path(filename).stem


def tissue_name_from_track(track_name: str) -> str | None:
    name = strip_bigwig_suffix(Path(track_name).name)
    first_underscore = name.find("_")
    last_dash = name.rfind("-")
    if first_underscore != -1 and last_dash > first_underscore:
        return name[first_underscore + 1 : last_dash]
    return name if name in UNSEEN_CELL_TYPES else None


def validate_embedding_file(path: Path) -> None:
    embedding = np.load(path, mmap_mode="r")
    if embedding.ndim < 1 or embedding.shape[-1] != CELL_EMBEDDING_DIM:
        raise ValueError(
            f"Embedding must end in dimension {CELL_EMBEDDING_DIM}, got "
            f"{embedding.shape} in {path}"
        )
    if not np.all(np.isfinite(embedding)):
        raise ValueError(f"Embedding contains non-finite values: {path}")


def discover_figure5b_tracks(
    bigwig_dir: Path,
    pattern: str,
    embedding_dir: Path,
) -> List[TrackInput]:
    tracks: List[TrackInput] = []
    seen_names: set[str] = set()
    for path in sorted(bigwig_dir.glob(pattern)):
        if not path.is_file():
            continue
        name = strip_bigwig_suffix(path.name)
        cell_type = tissue_name_from_track(name)
        if cell_type is None:
            continue
        if name in seen_names:
            raise ValueError(f"Duplicate Figure 5B track name discovered: {name}")
        seen_names.add(name)
        tracks.append(
            TrackInput(
                name=name,
                cell_type=cell_type,
                bigwig_path=path.resolve(),
                embedding_path=(embedding_dir / f"{cell_type}.npy").resolve(),
            )
        )
    return tracks


def select_figure5b_tracks(
    discovered: Sequence[TrackInput],
    requested: Sequence[str] | None,
) -> List[TrackInput]:
    seen = [track for track in discovered if track.cell_type not in UNSEEN_CELL_TYPES]
    by_name = {track.name: track for track in seen}
    expected = set(FIGURE5B_SEEN_TRACK_NAMES)
    available = set(by_name)
    if requested is None:
        missing = sorted(expected - available)
        extra = sorted(available - expected)
        if missing or extra:
            raise ValueError(
                "BigWig tracks do not exactly match the 34 Figure 5B seen tracks: "
                f"missing={missing}, extra={extra}"
            )
        return [by_name[name] for name in FIGURE5B_SEEN_TRACK_NAMES]

    lookup: Dict[str, TrackInput] = {}
    for name in FIGURE5B_SEEN_TRACK_NAMES:
        if name not in by_name:
            continue
        track = by_name[name]
        for key in (track.name, track.cell_type, track.bigwig_path.name):
            if key in lookup and lookup[key] != track:
                raise ValueError(f"Ambiguous Figure 5B track selector {key!r}")
            lookup[key] = track

    selected: List[TrackInput] = []
    for name in requested:
        if name not in lookup:
            if name in UNSEEN_CELL_TYPES:
                raise ValueError(f"{name} belongs to Figure 5A, not Figure 5B")
            raise ValueError(f"Requested Figure 5B track was not found: {name}")
        if lookup[name] not in selected:
            selected.append(lookup[name])
    if not selected:
        raise ValueError("--tracks selected no Figure 5B tracks")
    return selected


def validate_variable_blocks_header(path: Path) -> None:
    with path.open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        required = {"chr_hg38", "start_hg38", "end_hg38"}
        missing = required - set(reader.fieldnames or [])
        if missing:
            raise ValueError(
                f"Variable-block CSV is missing columns: {sorted(missing)}"
            )
        try:
            first = next(reader)
        except StopIteration as error:
            raise ValueError(f"Variable-block CSV is empty: {path}") from error
        int(first["start_hg38"])
        int(first["end_hg38"])


def setup_model_repo(code_dir: Path) -> None:
    code_dir_str = str(code_dir)
    while code_dir_str in sys.path:
        sys.path.remove(code_dir_str)
    sys.path.insert(0, code_dir_str)


def seed_everything(seed: int, panel: str) -> None:
    random.seed(seed)
    np.random.seed(seed)
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

    import torch

    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        # Figure 5A and the scientifically consistent 5C/5D/5E evaluations
        # disable cuDNN; the historical high-throughput Figure 5B keeps it on.
        torch.backends.cudnn.enabled = panel not in ("5A", "5C", "5D", "5E")


def resolve_device(requested: str):
    import torch

    if requested.startswith("cuda") and not torch.cuda.is_available():
        print("CUDA is unavailable; falling back to CPU.")
        requested = "cpu"
    return torch.device(requested)


def _strip_checkpoint_prefix(key: str) -> str:
    changed = True
    while changed:
        changed = False
        for prefix in ("module.", "_orig_mod."):
            if key.startswith(prefix):
                key = key[len(prefix) :]
                changed = True
    return key


def load_checkpoint(
    model,
    checkpoint_path: Path,
    model_kind: str,
    allow_partial: bool,
) -> None:
    import torch

    try:
        payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    except TypeError:
        payload = torch.load(checkpoint_path, map_location="cpu")
    state = (
        payload.get("model_state_dict", payload)
        if isinstance(payload, dict)
        else payload
    )
    if not isinstance(state, Mapping):
        raise TypeError(f"Checkpoint does not contain a state dict: {checkpoint_path}")
    state = {_strip_checkpoint_prefix(str(key)): value for key, value in state.items()}

    model_state = model.state_dict()
    compatible = {
        key: value
        for key, value in state.items()
        if key in model_state
        and getattr(value, "shape", None) == model_state[key].shape
    }
    missing = sorted(set(model_state) - set(compatible))
    unexpected = sorted(set(state) - set(model_state))
    mismatched = sorted(
        key
        for key in set(state) & set(model_state)
        if getattr(state[key], "shape", None) != model_state[key].shape
    )

    mt_inactive_prefixes = ("cell_mod.", "cell_mlp.", "dynamic_layer_generator.")
    permitted_missing = (
        [key for key in missing if key.startswith(mt_inactive_prefixes)]
        if model_kind == "mt-mean"
        else []
    )
    active_missing = [key for key in missing if key not in permitted_missing]

    if (active_missing or unexpected or mismatched) and not allow_partial:
        raise RuntimeError(
            "Checkpoint is not an exact architecture match: "
            f"active_missing={len(active_missing)}, unexpected={len(unexpected)}, "
            f"shape_mismatch={len(mismatched)}. Use the correct --model-code-dir "
            "or explicitly pass --allow-partial-checkpoint."
        )

    model_state.update(compatible)
    model.load_state_dict(model_state)
    print(
        f"Loaded {len(compatible)}/{len(model_state)} model tensors "
        f"from {checkpoint_path.name}"
    )
    if permitted_missing:
        print(
            f"Accepted {len(permitted_missing)} missing MT-mean tensors from "
            "inactive forward branches."
        )
    if active_missing or unexpected or mismatched:
        print(
            "Partial checkpoint load: "
            f"active_missing={len(active_missing)}, unexpected={len(unexpected)}, "
            f"shape_mismatch={len(mismatched)}"
        )
    del payload, state, compatible, model_state


def load_model(args: argparse.Namespace, device):
    import torch

    previous_common = sys.modules.get("common")
    if args.model_kind == "mt-mean":
        common_stub = types.ModuleType("common")
        common_stub.track_39_names = []
        sys.modules["common"] = common_stub
    try:
        module = importlib.import_module("cell_embedding")
    finally:
        if args.model_kind == "mt-mean":
            if previous_common is None:
                sys.modules.pop("common", None)
            else:
                sys.modules["common"] = previous_common

    module_path = Path(module.__file__).resolve()
    if module_path.parent != args.model_code_dir:
        raise ImportError(
            f"Imported cell_embedding from {module_path}, not {args.model_code_dir}. "
            "Evaluate one model per Python process."
        )
    model_class = getattr(module, "PuffinDWithCellEmbedding")
    n_track = 34 if args.model_kind == "mt-mean" else 1
    model = model_class(args, n_track=n_track)
    load_checkpoint(
        model,
        args.checkpoint,
        args.model_kind,
        args.allow_partial_checkpoint,
    )
    model = model.to(device).eval()
    if args.compile:
        if hasattr(torch, "compile"):
            model = torch.compile(model)
            print("Enabled torch.compile.")
        else:
            print(
                "WARNING: this PyTorch version has no torch.compile; continuing uncompiled."
            )
    return model


def load_data_backend(
    args: argparse.Namespace,
    device,
    expected_code_dir: Path | None = None,
):
    try:
        module = importlib.import_module("selene_mini")
    except ModuleNotFoundError as error:
        raise ModuleNotFoundError(
            "Could not import the repository's selene_mini backend. Run in the "
            "Melody environment with pyfaidx, pyBigWig, h5py, h5sparse and tabix."
        ) from error
    if expected_code_dir is not None:
        imported_dir = Path(module.__file__).resolve().parent
        expected_dir = (expected_code_dir / "selene_mini").resolve()
        if imported_dir != expected_dir:
            raise ImportError(
                f"Imported selene_mini from {imported_dir}, expected {expected_dir}"
            )
    genome_class = getattr(module, "Genome")
    dataset_class = getattr(module, "GenomicDataset")
    genome = genome_class(input_path=str(args.genome_fa), cuda=device.type == "cuda")
    return genome, dataset_class


def load_mean_embedding(path: Path) -> np.ndarray:
    embedding = np.asarray(np.load(path), dtype=np.float32)
    if embedding.ndim == 1:
        mean = embedding
    elif embedding.ndim >= 2:
        mean = embedding.mean(axis=0, dtype=np.float32)
    else:
        raise ValueError(f"Invalid embedding shape in {path}: {embedding.shape}")
    if mean.ndim != 1 or mean.size != CELL_EMBEDDING_DIM:
        raise ValueError(
            f"Mean embedding must have shape ({CELL_EMBEDDING_DIM},), got "
            f"{mean.shape} in {path}"
        )
    if not np.all(np.isfinite(mean)):
        raise ValueError(f"Mean embedding contains non-finite values: {path}")
    return np.asarray(mean, dtype=np.float32)


def cpg_mask(sequence: np.ndarray) -> np.ndarray:
    """Mark both C and G in each CpG pair."""
    sequence = np.asarray(sequence)
    if sequence.ndim != 2 or sequence.shape[1] != 4:
        raise ValueError(
            f"Expected one-hot sequence shaped (L,4), got {sequence.shape}"
        )
    mask = np.zeros(sequence.shape[0], dtype=bool)
    if sequence.shape[0] < 2:
        return mask
    pairs = (sequence[:-1, 1] == 1) & (sequence[1:, 2] == 1)
    mask[:-1] |= pairs
    mask[1:] |= pairs
    return mask


def chunks(items: Sequence, size: int) -> Iterable[Sequence]:
    for start in range(0, len(items), size):
        yield items[start : start + size]


def load_variable_block_windows(
    path: Path,
    window_size: int,
    chromosome_lengths: Mapping[str, int],
) -> Tuple[List[Tuple[str, int]], List[Tuple[str, int]]]:
    valid: List[Tuple[str, int]] = []
    test: List[Tuple[str, int]] = []
    with path.open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        for row_number, row in enumerate(reader, start=2):
            try:
                chrom = str(row["chr_hg38"])
                source_start = int(row["start_hg38"])
                source_end = int(row["end_hg38"])
            except (KeyError, TypeError, ValueError) as error:
                raise ValueError(
                    f"Invalid variable block at CSV row {row_number}"
                ) from error
            centre = (source_start + source_end) // 2
            start = centre - window_size // 2
            end = start + window_size
            if (
                start < 0
                or chrom not in chromosome_lengths
                or end > chromosome_lengths[chrom]
            ):
                continue
            if chrom in VALID_CHROMS:
                valid.append((chrom, start))
            elif chrom in TEST_CHROMS:
                test.append((chrom, start))
    return valid, test


def tiled_windows(
    chromosome_lengths: Mapping[str, int],
    chromosomes: Sequence[str],
    window_size: int,
) -> List[Tuple[str, int]]:
    windows: List[Tuple[str, int]] = []
    for chrom in chromosomes:
        if chrom not in chromosome_lengths:
            raise ValueError(f"Chromosome {chrom} is absent from the supplied genome")
        windows.extend(
            (chrom, start)
            for start in range(
                0,
                chromosome_lengths[chrom] - window_size + 1,
                window_size,
            )
        )
    return windows


def build_figure5b_split_windows(
    args: argparse.Namespace,
    genome,
) -> Dict[str, List[Tuple[str, int]]]:
    chromosome_lengths = dict(genome.get_chr_lens())
    sample_valid, sample_test = load_variable_block_windows(
        args.variable_blocks_csv,
        args.window_size,
        chromosome_lengths,
    )
    all_windows = {
        "SAMPLE_VALID": sample_valid,
        "SAMPLE_TEST": sample_test,
        "VALID": tiled_windows(chromosome_lengths, VALID_CHROMS, args.window_size),
        "TEST": tiled_windows(chromosome_lengths, TEST_CHROMS, args.window_size),
    }
    selected: Dict[str, List[Tuple[str, int]]] = {}
    for split in args.splits:
        windows = all_windows[split]
        if args.max_windows_per_split is not None:
            windows = windows[: args.max_windows_per_split]
        if not windows:
            raise ValueError(f"No windows were generated for {split}")
        selected[split] = windows
        print(f"{split:12s}: {len(windows):6d} windows")
    return selected


def fetch_sequence_batch(
    genome,
    windows: Sequence[Tuple[str, int]],
    window_size: int,
) -> np.ndarray:
    sequences: List[np.ndarray] = []
    for chrom, start in windows:
        sequence = np.asarray(
            genome.get(chrom, start, start + window_size),
            dtype=np.float32,
        )
        if sequence.shape != (window_size, 4):
            raise ValueError(
                f"Genome returned {sequence.shape} for "
                f"{chrom}:{start}-{start + window_size}; expected "
                f"({window_size}, 4)"
            )
        sequences.append(sequence)
    return np.stack(sequences)


def fetch_truth_batch(
    methylation,
    windows: Sequence[Tuple[str, int]],
    window_size: int,
) -> np.ndarray:
    truths: List[np.ndarray] = []
    for chrom, start in windows:
        truth = np.asarray(
            methylation.get(chrom, start, start + window_size),
            dtype=np.float32,
        )
        if truth.shape != (1, window_size):
            raise ValueError(
                f"BigWig backend returned {truth.shape} for "
                f"{chrom}:{start}-{start + window_size}; expected "
                f"(1, {window_size})"
            )
        truths.append(truth[0])
    return np.stack(truths)


def predict_figure5b_batch(
    model,
    sequence_tensor,
    mean_embedding_tensor,
) -> np.ndarray:
    import torch

    embedding = (
        mean_embedding_tensor.unsqueeze(0)
        .expand(sequence_tensor.shape[0], -1)
        .contiguous()
    )
    with torch.inference_mode():
        output = model(sequence_tensor, embedding)
        logits = output[0] if isinstance(output, (tuple, list)) else output
    if logits.ndim != 3 or logits.shape[1] != 1:
        raise ValueError(
            f"Expected Figure 5B model output (B,1,L), got {tuple(logits.shape)}"
        )
    # AUROC is invariant under sigmoid, so preserve the historical raw logits.
    return logits[:, 0, :].detach().cpu().numpy().astype(np.float32, copy=False)


def evaluate_figure5b(
    args: argparse.Namespace,
    tracks: Sequence[TrackInput],
) -> Dict[str, Dict[str, Dict[str, float | int]]]:
    import torch

    setup_model_repo(args.model_code_dir)
    seed_everything(args.seed, args.panel)
    device = resolve_device(args.device)
    genome, dataset_class = load_data_backend(args, device)
    split_windows = build_figure5b_split_windows(args, genome)
    model = load_model(args, device)

    methylation = [
        dataset_class([str(track.bigwig_path)], genome, storage="BigWig")
        for track in tracks
    ]
    embeddings = [
        torch.from_numpy(load_mean_embedding(track.embedding_path)).to(
            device=device,
            dtype=torch.float32,
        )
        for track in tracks
    ]
    results: Dict[str, Dict[str, Dict[str, float | int]]] = {
        track.name: {} for track in tracks
    }

    print(f"[{args.model_name}] evaluating {len(tracks)} seen tracks on {device}")
    try:
        for split, windows in split_windows.items():
            accumulators = {
                track.name: ExactAUCAccumulator.create() for track in tracks
            }
            total_batches = math.ceil(len(windows) / args.batch_size)
            for batch_index, window_batch_sequence in enumerate(
                chunks(windows, args.batch_size),
                start=1,
            ):
                # chunks() returns a generic Sequence; make a list because the
                # chromosome-wide branch may filter individual windows below.
                window_batch = list(window_batch_sequence)
                sequence_batch = fetch_sequence_batch(
                    genome,
                    window_batch,
                    args.window_size,
                )
                # The original sequential VALID/TEST branch dropped windows
                # entirely encoded as [0.25, 0.25, 0.25, 0.25].
                if split in ("VALID", "TEST"):
                    keep_windows = ~np.all(sequence_batch == 0.25, axis=(1, 2))
                    if not np.any(keep_windows):
                        continue
                    if not np.all(keep_windows):
                        sequence_batch = sequence_batch[keep_windows]
                        window_batch = [
                            window
                            for window, keep in zip(window_batch, keep_windows)
                            if keep
                        ]

                sequence_tensor = (
                    torch.from_numpy(sequence_batch)
                    .to(device=device, dtype=torch.float32)
                    .permute(0, 2, 1)
                    .contiguous()
                )
                for track, methylation_data, embedding in zip(
                    tracks,
                    methylation,
                    embeddings,
                ):
                    truth_batch = fetch_truth_batch(
                        methylation_data,
                        window_batch,
                        args.window_size,
                    )
                    prediction_batch = predict_figure5b_batch(
                        model,
                        sequence_tensor,
                        embedding,
                    )
                    accumulators[track.name].update_batch(
                        prediction_batch,
                        truth_batch,
                        sequence_batch,
                    )

                if (
                    batch_index == 1
                    or batch_index == total_batches
                    or batch_index % 25 == 0
                ):
                    print(f"  {split:12s} batch {batch_index:5d}/{total_batches}")
                del sequence_tensor, sequence_batch

            print(f"[{split}] exact CpG AUROC")
            for track in tracks:
                metrics = accumulators[track.name].finalize()
                results[track.name][split] = metrics
                print(
                    f"  {track.name:58s} AUC={metrics['auc_CpG']:.9f} "
                    f"n={metrics['count_CpG']}"
                )
            del accumulators
            gc.collect()
    finally:
        del model
        for dataset in methylation:
            if hasattr(dataset, "uninitialize"):
                dataset.uninitialize()
        if hasattr(genome, "uninitialize"):
            genome.uninitialize()
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    return results


def read_figure5b_rows(path: Path) -> List[Dict[str, str]]:
    if not path.is_file():
        return []
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames != FIGURE5B_CSV_FIELDS:
            raise ValueError(
                f"Existing Figure 5B CSV has columns {reader.fieldnames}; "
                f"expected {FIGURE5B_CSV_FIELDS}"
            )
        rows = list(reader)
    for row in rows:
        if row["metric"] != "AUC":
            raise ValueError(
                "Existing Figure 5B G1/G2 CSV contains non-AUC metric "
                f"{row['metric']!r}"
            )
    return rows


def figure5b_result_rows(
    model_name: str,
    tracks: Sequence[TrackInput],
    results: Mapping[str, Mapping[str, Mapping[str, float | int]]],
) -> List[Dict[str, str]]:
    rows: List[Dict[str, str]] = []
    for track in tracks:
        for split in FIGURE5B_SPLIT_ORDER:
            if split not in results[track.name]:
                continue
            auc = float(results[track.name][split]["auc_CpG"])
            rows.append(
                {
                    "model": model_name,
                    "track": track.name,
                    "split": split,
                    "metric": "AUC",
                    "value": repr(round(auc, 9)),
                }
            )
    return rows


def write_figure5b_csv(
    path: Path,
    model_name: str,
    tracks: Sequence[TrackInput],
    results: Mapping[str, Mapping[str, Mapping[str, float | int]]],
    append: bool,
) -> None:
    existing = read_figure5b_rows(path) if append else []
    rows = [row for row in existing if row["model"] != model_name]
    rows.extend(figure5b_result_rows(model_name, tracks, results))

    preferred_models = {"Melody-G1": 0, "Melody-G2": 1}
    split_rank = {split: index for index, split in enumerate(FIGURE5B_SPLIT_ORDER)}
    fallback_models = {
        name: index + len(preferred_models)
        for index, name in enumerate(dict.fromkeys(row["model"] for row in rows))
        if name not in preferred_models
    }
    rows.sort(
        key=lambda row: (
            preferred_models.get(
                row["model"],
                fallback_models.get(row["model"], 999),
            ),
            row["track"],
            split_rank.get(row["split"], 999),
        )
    )

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=FIGURE5B_CSV_FIELDS,
            lineterminator="\n",
        )
        writer.writeheader()
        writer.writerows(rows)


def write_figure5b_json(
    path: Path,
    args: argparse.Namespace,
    tracks: Sequence[TrackInput],
    results: Mapping[str, Mapping[str, Mapping[str, float | int]]],
) -> None:
    payload = {
        "panel": args.panel,
        "model": args.model_name,
        "model_kind": args.model_kind,
        "checkpoint": args.checkpoint.name,
        "model_code_dir": args.model_code_dir.name,
        "genome_fa": args.genome_fa.name,
        "variable_blocks_csv": args.variable_blocks_csv.name,
        "window_size": args.window_size,
        "batch_size": args.batch_size,
        "torch_compile": args.compile,
        "splits": args.splits,
        "excluded_unseen_cell_types": sorted(UNSEEN_CELL_TYPES),
        "tracks": [track.name for track in tracks],
        "results": results,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, allow_nan=True)

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Re-evaluate a Melody-G1 or Melody-G2 Figure 5B checkpoint.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--model-kind", required=True, choices=MODEL_KINDS)
    parser.add_argument("--model-name")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--model-code-dir", type=Path, required=True)
    parser.add_argument("--genome-fa", type=Path, required=True)
    parser.add_argument("--bigwig-dir", type=Path, required=True)
    parser.add_argument("--embedding-dir", type=Path, required=True)
    parser.add_argument("--variable-blocks-csv", type=Path, required=True)
    parser.add_argument("--output-csv", type=Path, default=DEFAULT_OUTPUT_CSV)
    parser.add_argument("--output-json", type=Path)
    parser.add_argument("--append", action="store_true")
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--compile",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--window-size", type=int, default=10_000)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--seed", type=int, default=5)
    parser.add_argument("--bigwig-glob", default="*.bigwig")
    parser.add_argument("--allow-partial-checkpoint", action="store_true")
    parser.add_argument("--validate-only", action="store_true")
    parser.add_argument(
        "--splits",
        nargs="+",
        choices=FIGURE5B_SPLIT_ORDER,
    )
    parser.add_argument("--tracks", nargs="+")
    parser.add_argument("--max-windows-per-split", type=int)
    args = parser.parse_args()
    args.panel = PANEL
    args.model_name = args.model_name or MODEL_NAMES[args.model_kind]
    requested = set(args.splits or FIGURE5B_SPLIT_ORDER)
    args.splits = [
        split for split in FIGURE5B_SPLIT_ORDER if split in requested
    ]
    return args


def validate_args(args: argparse.Namespace) -> List[TrackInput]:
    if args.window_size <= 0:
        raise ValueError("--window-size must be positive")
    if args.batch_size <= 0:
        raise ValueError("--batch-size must be positive")
    if args.max_windows_per_split is not None and args.max_windows_per_split <= 0:
        raise ValueError("--max-windows-per-split must be positive")
    expected_model_name = MODEL_NAMES[args.model_kind]
    if args.model_name != expected_model_name:
        raise ValueError(
            f"--model-kind {args.model_kind!r} requires model name "
            f"{expected_model_name!r}"
        )

    args.checkpoint = _require_file(args.checkpoint, "checkpoint")
    args.model_code_dir = _require_dir(args.model_code_dir, "model code directory")
    _require_file(args.model_code_dir / "cell_embedding.py", "model definition")
    _require_dir(args.model_code_dir / "selene_mini", "selene_mini package")
    args.genome_fa = _require_file(args.genome_fa, "genome FASTA")
    args.bigwig_dir = _require_dir(args.bigwig_dir, "BigWig directory")
    args.embedding_dir = _require_dir(args.embedding_dir, "embedding directory")
    args.variable_blocks_csv = _require_file(
        args.variable_blocks_csv,
        "variable-block CSV",
    )
    validate_variable_blocks_header(args.variable_blocks_csv)
    args.output_csv = args.output_csv.expanduser().resolve()
    if args.output_json is not None:
        args.output_json = args.output_json.expanduser().resolve()

    discovered = discover_figure5b_tracks(
        args.bigwig_dir,
        args.bigwig_glob,
        args.embedding_dir,
    )
    tracks = select_figure5b_tracks(discovered, args.tracks)
    for track in tracks:
        _require_file(track.embedding_path, f"embedding for {track.cell_type}")
        validate_embedding_file(track.embedding_path)

    partial = (
        args.tracks is not None
        or args.max_windows_per_split is not None
        or set(args.splits) != set(FIGURE5B_SPLIT_ORDER)
    )
    if args.append and args.output_csv.is_file() and partial:
        raise ValueError(
            "Refusing to append a Figure 5B debug subset to an existing CSV"
        )

    print(f"panel            : {args.panel}")
    print(f"model            : {args.model_kind} ({args.model_name})")
    print(f"checkpoint       : {args.checkpoint.name}")
    print(f"variable blocks  : {args.variable_blocks_csv.name}")
    print(f"splits           : {', '.join(args.splits)}")
    print(f"Figure 5B tracks : {len(tracks)}")
    return tracks


def main() -> None:
    args = parse_args()
    tracks = validate_args(args)
    if args.validate_only:
        print("Validation successful; model evaluation was not run.")
        return

    results = evaluate_figure5b(args, tracks)
    write_figure5b_csv(
        args.output_csv,
        args.model_name,
        tracks,
        results,
        args.append,
    )
    print(f"Wrote {args.model_name} rows: {args.output_csv.name}")
    if args.output_json is not None:
        write_figure5b_json(args.output_json, args, tracks, results)
        print(f"Wrote detailed metrics: {args.output_json.name}")
    if args.max_windows_per_split is not None or args.tracks is not None:
        print(
            "WARNING: a debug/subset option was set; these are not complete "
            "manuscript values."
        )


if __name__ == "__main__":
    main()
