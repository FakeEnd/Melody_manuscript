#!/usr/bin/env python3
"""Re-evaluate Figure 5A on the five held-out cell types.

Figure 5F-H use the same Figure 5A evaluation logic: the same held-out cell
types, cell-type-specific regions, exact concatenated-region CpG AUROC, and
wide source-data schema.
"""

from __future__ import annotations

import argparse
import csv
import gc
import importlib
import json
import os
import random
import sys
import types
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Sequence, Tuple

import numpy as np


MODEL_KINDS = ("g1", "g2", "mt-mean")
MODEL_NAMES = {
    "g1": "Melody-G1",
    "g2": "Melody-G2",
    "mt-mean": "MT-mean",
}
FIGURE5A_CELL_TYPES = (
    "Pancreas-Delta",
    "Blood-Granulocytes",
    "Blood-Monocytes",
    "Aorta-Endothel",
    "Cortex-Neuron",
)
UNSEEN_CELL_TYPES = frozenset(FIGURE5A_CELL_TYPES)
CELL_EMBEDDING_DIM = 512
SCRIPT_DIR = Path(__file__).resolve().parent
SUBMIT_DIR = SCRIPT_DIR.parent
DEFAULT_OUTPUT_CSV = SUBMIT_DIR / "generated_csv" / "figure_5A.csv"


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


def discover_figure5a_bigwigs(bigwig_dir: Path, pattern: str) -> Dict[str, Path]:
    matches: Dict[str, List[Path]] = {}
    for path in sorted(bigwig_dir.glob(pattern)):
        if not path.is_file():
            continue
        cell_type = tissue_name_from_track(path.name)
        if cell_type in UNSEEN_CELL_TYPES:
            matches.setdefault(cell_type, []).append(path.resolve())
    ambiguous = {cell: paths for cell, paths in matches.items() if len(paths) > 1}
    if ambiguous:
        details = "; ".join(
            f"{cell}: {[str(path) for path in paths]}"
            for cell, paths in sorted(ambiguous.items())
        )
        raise ValueError(f"Ambiguous Figure 5A BigWig matches: {details}")
    return {cell: paths[0] for cell, paths in matches.items()}


def validate_regions_json(path: Path) -> None:
    with path.open(encoding="utf-8") as handle:
        regions = json.load(handle)
    for cell_type in FIGURE5A_CELL_TYPES:
        if cell_type not in regions or not regions[cell_type]:
            raise ValueError(f"regions JSON has no regions for {cell_type}")


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
        torch.backends.cudnn.enabled = False


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


def normalize_region(region: Sequence[object]) -> Tuple[str, int, int]:
    if not isinstance(region, (list, tuple)) or len(region) < 3:
        raise ValueError(f"Invalid region: {region!r}")
    chrom, start, end = str(region[0]), int(region[1]), int(region[2])
    if start > end:
        start, end = end, start
    if start == end:
        raise ValueError(f"Zero-length region: {region!r}")
    return chrom, start, end


def calculate_figure5a_metrics(
    pred_parts: Sequence[np.ndarray],
    truth_parts: Sequence[np.ndarray],
    sequence_parts: Sequence[np.ndarray],
) -> Dict[str, float | int]:
    """Calculate the exact concatenated-region metrics used for Figure 5A."""
    from sklearn.metrics import roc_auc_score

    prediction = np.concatenate(pred_parts).reshape(-1)
    truth = np.concatenate(truth_parts).reshape(-1)
    sequence = np.concatenate(sequence_parts, axis=0)
    if not (prediction.size == truth.size == sequence.shape[0]):
        raise ValueError(
            "Figure 5A metric input lengths differ: "
            f"pred={prediction.size}, truth={truth.size}, "
            f"sequence={sequence.shape[0]}"
        )

    # Preserve test_regions_cat.py exactly: concatenate cropped regions first,
    # then compute the CpG mask once over the concatenated sequence.
    keep = cpg_mask(sequence) & np.isfinite(truth) & (truth >= 0.0)
    prediction = prediction[keep]
    truth = truth[keep]
    if prediction.size == 0:
        return {
            "auc_CpG": float("nan"),
            "acc_CpG": float("nan"),
            "count_CpG": 0,
        }

    # Figure 5A uses a strict threshold: a raw methylation value of exactly
    # 0.5 is negative, matching the historical evaluator.
    binary_truth = np.asarray(truth > 0.5, dtype=np.int8)
    auc = (
        float(roc_auc_score(binary_truth, prediction))
        if np.unique(binary_truth).size > 1
        else float("nan")
    )
    accuracy = float(np.mean((prediction > 0.5) == binary_truth))
    return {
        "auc_CpG": auc,
        "acc_CpG": accuracy,
        "count_CpG": int(prediction.size),
    }


def predict_figure5a_batch(
    model,
    model_kind: str,
    sequence_batch: np.ndarray,
    embedding: np.ndarray | None,
    device,
) -> np.ndarray:
    import torch

    sequence_tensor = (
        torch.from_numpy(sequence_batch)
        .to(device=device, dtype=torch.float32)
        .permute(0, 2, 1)
        .contiguous()
    )
    with torch.inference_mode():
        if model_kind in ("g1", "g2"):
            if embedding is None:
                raise ValueError(f"An embedding is required for {model_kind}")
            embedding_tensor = torch.from_numpy(embedding).to(
                device=device,
                dtype=torch.float32,
            )
            embedding_tensor = embedding_tensor.unsqueeze(0).expand(
                sequence_tensor.shape[0], -1
            )
            output = model(sequence_tensor, cell_embedding=embedding_tensor)
        else:
            output = model(sequence_tensor)

        logits = output[0] if isinstance(output, (tuple, list)) else output
        if model_kind == "mt-mean":
            # The baseline averages its 34 seen-track logits before sigmoid.
            if logits.ndim != 3 or logits.shape[1] != 34:
                raise ValueError(
                    f"Expected MT-mean model output (B,34,L), got {tuple(logits.shape)}"
                )
            logits = logits.mean(dim=1, keepdim=True)
        if logits.ndim != 3 or logits.shape[1] != 1:
            raise ValueError(
                f"Expected Figure 5A model output (B,1,L), got {tuple(logits.shape)}"
            )
        prediction = torch.sigmoid(logits)
    return prediction.detach().cpu().numpy()


def evaluate_figure5a_cell_type(
    *,
    model,
    model_kind: str,
    device,
    genome,
    dataset_class,
    bigwig_path: Path,
    regions: Sequence[Sequence[object]],
    embedding_path: Path | None,
    window_size: int,
    batch_size: int,
) -> Dict[str, float | int]:
    methylation = dataset_class([str(bigwig_path)], genome, storage="BigWig")
    embedding = load_mean_embedding(embedding_path) if embedding_path else None
    pred_parts: List[np.ndarray] = []
    truth_parts: List[np.ndarray] = []
    sequence_parts: List[np.ndarray] = []

    try:
        for region_batch in chunks(regions, batch_size):
            sequences: List[np.ndarray] = []
            truths: List[np.ndarray] = []
            crop_slices: List[Tuple[int, int]] = []

            for region in region_batch:
                chrom, region_start, region_end = normalize_region(region)
                region_length = region_end - region_start
                if region_length > window_size:
                    raise ValueError(
                        f"Region {chrom}:{region_start}-{region_end} is longer than "
                        f"the {window_size}-bp model window"
                    )

                centre = (region_start + region_end) // 2
                fetch_start = centre - window_size // 2
                fetch_end = fetch_start + window_size
                if fetch_start < 0:
                    raise ValueError(
                        f"Region {chrom}:{region_start}-{region_end} produces "
                        "a negative fetch start"
                    )

                sequence = np.asarray(
                    genome.get(chrom, fetch_start, fetch_end),
                    dtype=np.float32,
                )
                truth = np.asarray(
                    methylation.get(chrom, fetch_start, fetch_end),
                    dtype=np.float32,
                )
                if sequence.shape != (window_size, 4):
                    raise ValueError(
                        f"Genome returned {sequence.shape} for "
                        f"{chrom}:{fetch_start}-{fetch_end}; expected "
                        f"({window_size}, 4)"
                    )
                if truth.shape != (1, window_size):
                    raise ValueError(
                        f"BigWig returned {truth.shape} for "
                        f"{chrom}:{fetch_start}-{fetch_end}; expected "
                        f"(1, {window_size})"
                    )

                # Preserve the original crop convention. For odd region lengths
                # this deliberately retains one fewer base than region_length.
                crop_start = window_size // 2 - region_length // 2
                crop_end = window_size // 2 + region_length // 2
                sequences.append(sequence)
                truths.append(truth)
                crop_slices.append((crop_start, crop_end))

            sequence_batch = np.stack(sequences)
            prediction_batch = predict_figure5a_batch(
                model,
                model_kind,
                sequence_batch,
                embedding,
                device,
            )
            for index, (crop_start, crop_end) in enumerate(crop_slices):
                pred_parts.append(prediction_batch[index, 0, crop_start:crop_end])
                truth_parts.append(truths[index][0, crop_start:crop_end])
                sequence_parts.append(sequences[index][crop_start:crop_end])
    finally:
        if hasattr(methylation, "uninitialize"):
            methylation.uninitialize()

    if not pred_parts:
        raise ValueError(f"No Figure 5A regions were evaluated for {bigwig_path}")
    return calculate_figure5a_metrics(pred_parts, truth_parts, sequence_parts)


def evaluate_figure5a(
    args: argparse.Namespace,
    bigwigs: Mapping[str, Path],
) -> Dict[str, Dict[str, float | int]]:
    setup_model_repo(args.model_code_dir)
    seed_everything(args.seed, args.panel)
    device = resolve_device(args.device)
    genome, dataset_class = load_data_backend(args, device)
    model = load_model(args, device)

    with args.regions_json.open(encoding="utf-8") as handle:
        region_map = json.load(handle)

    results: Dict[str, Dict[str, float | int]] = {}
    try:
        print(
            f"[{args.model_name}] evaluating {len(FIGURE5A_CELL_TYPES)} "
            f"unseen cell types on {device}"
        )
        for cell_type in FIGURE5A_CELL_TYPES:
            regions = region_map[cell_type]
            if args.max_regions_per_cell is not None:
                regions = regions[: args.max_regions_per_cell]
            embedding_path = (
                args.embedding_dir / f"{cell_type}.npy"
                if args.model_kind in ("g1", "g2")
                else None
            )
            metrics = evaluate_figure5a_cell_type(
                model=model,
                model_kind=args.model_kind,
                device=device,
                genome=genome,
                dataset_class=dataset_class,
                bigwig_path=bigwigs[cell_type],
                regions=regions,
                embedding_path=embedding_path,
                window_size=args.window_size,
                batch_size=args.batch_size,
            )
            results[cell_type] = metrics
            print(
                f"  {cell_type:20s} AUC={metrics['auc_CpG']:.9f} "
                f"n={metrics['count_CpG']}"
            )
    finally:
        del model
        gc.collect()
        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except ImportError:
            pass
        if hasattr(genome, "uninitialize"):
            genome.uninitialize()
    return results


def read_figure5a_rows(path: Path) -> Dict[str, Dict[str, str]]:
    if not path.is_file():
        return {}
    expected_fields = ["", *FIGURE5A_CELL_TYPES]
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames != expected_fields:
            raise ValueError(
                f"Existing Figure 5A CSV has columns {reader.fieldnames}; "
                f"expected {expected_fields}"
            )
        return {row[""]: row for row in reader}


def write_figure5a_csv(
    path: Path,
    model_name: str,
    results: Mapping[str, Mapping[str, float | int]],
    append: bool,
) -> None:
    rows = read_figure5a_rows(path) if append else {}
    rows[model_name] = {
        "": model_name,
        **{
            cell_type: repr(round(float(results[cell_type]["auc_CpG"]), 9))
            for cell_type in FIGURE5A_CELL_TYPES
        },
    }
    preferred_order = ["Melody-G1", "Melody-G2", "MT-mean"]
    ordered_names = [name for name in preferred_order if name in rows]
    ordered_names.extend(name for name in rows if name not in ordered_names)

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["", *FIGURE5A_CELL_TYPES],
            lineterminator="\n",
        )
        writer.writeheader()
        for name in ordered_names:
            writer.writerow(rows[name])


def write_figure5a_json(
    path: Path,
    args: argparse.Namespace,
    results: Mapping[str, Mapping[str, float | int]],
) -> None:
    payload = {
        "panel": args.panel,
        "model": args.model_name,
        "model_kind": args.model_kind,
        "checkpoint": args.checkpoint.name,
        "model_code_dir": args.model_code_dir.name,
        "genome_fa": args.genome_fa.name,
        "regions_json": args.regions_json.name,
        "window_size": args.window_size,
        "batch_size": args.batch_size,
        "torch_compile": args.compile,
        "cell_types": results,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, allow_nan=True)

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Re-evaluate one Figure 5A-style unseen-cell benchmark row.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--model-kind", required=True, choices=MODEL_KINDS)
    parser.add_argument(
        "--model-name",
        help=(
            "Output row label; omitted uses the Figure 5A label for model kind."
        ),
    )
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--model-code-dir", type=Path, required=True)
    parser.add_argument("--genome-fa", type=Path, required=True)
    parser.add_argument("--bigwig-dir", type=Path, required=True)
    parser.add_argument("--embedding-dir", type=Path)
    parser.add_argument("--regions-json", type=Path, required=True)
    parser.add_argument("--output-csv", type=Path, default=DEFAULT_OUTPUT_CSV)
    parser.add_argument("--output-json", type=Path)
    parser.add_argument("--append", action="store_true")
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--compile",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    parser.add_argument("--window-size", type=int, default=10_000)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--seed", type=int, default=5)
    parser.add_argument("--bigwig-glob", default="*.bigwig")
    parser.add_argument("--allow-partial-checkpoint", action="store_true")
    parser.add_argument("--validate-only", action="store_true")
    parser.add_argument("--max-regions-per-cell", type=int)
    args = parser.parse_args()
    args.panel = "5A"
    args.model_name = args.model_name or MODEL_NAMES[args.model_kind]
    return args


def validate_args(args: argparse.Namespace) -> Dict[str, Path]:
    if args.window_size <= 0 or args.window_size % 2:
        raise ValueError("--window-size must be a positive even integer")
    if args.batch_size <= 0:
        raise ValueError("--batch-size must be positive")
    if args.max_regions_per_cell is not None and args.max_regions_per_cell <= 0:
        raise ValueError("--max-regions-per-cell must be positive")

    if not args.model_name.strip():
        raise ValueError("--model-name must not be empty")

    args.checkpoint = _require_file(args.checkpoint, "checkpoint")
    args.model_code_dir = _require_dir(args.model_code_dir, "model code directory")
    _require_file(args.model_code_dir / "cell_embedding.py", "model definition")
    _require_dir(args.model_code_dir / "selene_mini", "selene_mini package")
    args.genome_fa = _require_file(args.genome_fa, "genome FASTA")
    args.bigwig_dir = _require_dir(args.bigwig_dir, "BigWig directory")
    args.regions_json = _require_file(args.regions_json, "regions JSON")
    validate_regions_json(args.regions_json)
    args.output_csv = args.output_csv.expanduser().resolve()
    if args.output_json is not None:
        args.output_json = args.output_json.expanduser().resolve()

    if args.embedding_dir is not None:
        args.embedding_dir = _require_dir(args.embedding_dir, "embedding directory")
    if args.model_kind in ("g1", "g2"):
        if args.embedding_dir is None:
            raise ValueError(
                "--embedding-dir is required for embedding-conditioned models"
            )
        for cell_type in FIGURE5A_CELL_TYPES:
            embedding_path = _require_file(
                args.embedding_dir / f"{cell_type}.npy",
                f"embedding for {cell_type}",
            )
            validate_embedding_file(embedding_path)

    bigwigs = discover_figure5a_bigwigs(args.bigwig_dir, args.bigwig_glob)
    missing = [cell for cell in FIGURE5A_CELL_TYPES if cell not in bigwigs]
    if missing:
        raise FileNotFoundError(
            f"Missing Figure 5A-style unseen-cell BigWig tracks: {missing}"
        )
    if (
        args.append
        and args.output_csv.is_file()
        and args.max_regions_per_cell is not None
    ):
        raise ValueError(
            "Refusing to append a debug subset to an existing panel CSV"
        )

    print("benchmark        : Figure 5A logic (also used by Figure 5F-H)")
    print(f"row              : {args.model_name}")
    print(f"model kind       : {args.model_kind}")
    print(f"checkpoint       : {args.checkpoint.name}")
    print(f"regions          : {args.regions_json.name}")
    print("unseen BigWigs   : all five tracks found")
    return bigwigs


def main() -> None:
    args = parse_args()
    bigwigs = validate_args(args)
    if args.validate_only:
        print("Validation successful; model evaluation was not run.")
        return

    results = evaluate_figure5a(args, bigwigs)
    write_figure5a_csv(
        args.output_csv,
        args.model_name,
        results,
        args.append,
    )
    print(f"Wrote row {args.model_name}: {args.output_csv.name}")
    if args.output_json is not None:
        write_figure5a_json(args.output_json, args, results)
        print(f"Wrote detailed metrics: {args.output_json.name}")
    if args.max_regions_per_cell is not None:
        print(
            "WARNING: --max-regions-per-cell was set; these are debug "
            "values, not manuscript values."
        )


if __name__ == "__main__":
    main()
