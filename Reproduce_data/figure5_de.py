#!/usr/bin/env python3
"""Recompute or post-process the source data for Figure 5D and Figure 5E.

The recompute path evaluates one embedding-conditioned Melody checkpoint with a same-run
baseline embedding and 1,200 in silico knockout embeddings.  The directional
score is

    mean_prediction(KO) - mean_prediction(baseline).

The direct-score path starts from a completed score CSV.  Figure 5D selects
representative genes from the four highest-ranked FDR-significant pathways in
a frozen full-panel Enrichr report.  Figure 5E writes all 1,200 signed
piecewise-scaled scores and verifies the central-98% plotting selection.
"""

from __future__ import annotations

import argparse
import gc
import importlib
import json
import os
import random
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np
import pandas as pd
from tqdm.auto import tqdm


PANELS = ("5D", "5E")
CELL_TYPE = "Pancreas-Delta"
EXPECTED_GENES = 1_200
EXPECTED_REGIONS = 100
EXPECTED_CELLS = 3_383
EMBEDDING_DIM = 512
WINDOW_SIZE = 10_000
EXPECTED_KEGG_TERMS = 290
FDR_THRESHOLD = 0.05
PATHWAY_COUNT = 4
GENES_PER_PATHWAY = 4
OTHER_GENES = 4
TAIL_QUANTILE = 0.01

SCRIPT_DIR = Path(__file__).resolve().parent
SUBMIT_DIR = SCRIPT_DIR.parent
DEFAULT_OUTPUTS = {
    "5D": SUBMIT_DIR / "generated_csv" / "figure_5D.csv",
    "5E": SUBMIT_DIR / "generated_csv" / "figure_5E.csv",
}


@dataclass(frozen=True)
class Interval:
    chrom: str
    start: int
    end: int

    @property
    def length(self) -> int:
        return self.end - self.start


@dataclass(frozen=True)
class EmbeddingInputs:
    genes: tuple[str, ...]
    baseline: Path
    noop: Path
    knockouts: tuple[Path, ...]
    shape: tuple[int, int]


def require_file(path: Path, label: str) -> Path:
    path = path.expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"{label} does not exist or is not a file: {path}")
    return path


def require_dir(path: Path, label: str) -> Path:
    path = path.expanduser().resolve()
    if not path.is_dir():
        raise NotADirectoryError(f"{label} is not a directory: {path}")
    return path


def atomic_write_csv(frame: pd.DataFrame, path: Path, force: bool) -> None:
    path = path.expanduser().resolve()
    if path.exists() and not force:
        raise FileExistsError(f"Output already exists; use --force to replace it: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            newline="",
            encoding="utf-8",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as handle:
            temporary = Path(handle.name)
            frame.to_csv(handle, index=False, lineterminator="\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def piecewise_scale(scores: np.ndarray) -> np.ndarray:
    """Min-max scale positive and negative magnitudes separately."""
    scores = np.asarray(scores, dtype=np.float64)
    if scores.ndim != 1 or not np.all(np.isfinite(scores)):
        raise ValueError("Directional scores must be a finite one-dimensional array")
    scaled = np.zeros_like(scores)
    epsilon = 1e-6
    for sign in (1.0, -1.0):
        mask = scores * sign > 0.0
        if not np.any(mask):
            continue
        magnitudes = np.abs(scores[mask])
        low = float(magnitudes.min())
        high = float(magnitudes.max())
        if high == low:
            transformed = np.ones_like(magnitudes)
        else:
            transformed = epsilon + (1.0 - epsilon) * (
                (magnitudes - low) / (high - low)
            )
        scaled[mask] = sign * transformed
    return scaled


def effect_direction(score: float) -> str:
    if score > 0.0:
        return "knockout_increases_methylation"
    if score < 0.0:
        return "knockout_decreases_methylation"
    return "no_change"


def load_scores(path: Path) -> pd.DataFrame:
    """Load either the full recompute table or a two-column source table."""
    frame = pd.read_csv(require_file(path, "score CSV"))
    if "gene" not in frame:
        raise ValueError("Score CSV must contain a 'gene' column")
    if len(frame) != EXPECTED_GENES:
        raise ValueError(f"Expected {EXPECTED_GENES} genes, found {len(frame)}")
    genes = frame["gene"]
    if genes.isna().any() or genes.astype(str).str.len().eq(0).any():
        raise ValueError("Score CSV contains an empty gene name")
    if genes.astype(str).duplicated().any():
        raise ValueError("Score CSV contains duplicate genes")
    frame = frame.copy()
    frame["gene"] = frame["gene"].astype(str)

    if "piecewise_scaled_score" in frame:
        piecewise_column = "piecewise_scaled_score"
    elif "score" in frame:
        piecewise_column = "score"
    elif "directional_diff_score" in frame:
        directional = pd.to_numeric(
            frame["directional_diff_score"], errors="raise"
        ).to_numpy(dtype=np.float64)
        frame["piecewise_scaled_score"] = piecewise_scale(directional)
        piecewise_column = "piecewise_scaled_score"
    else:
        raise ValueError(
            "Score CSV needs 'piecewise_scaled_score', 'score', or "
            "'directional_diff_score'"
        )

    piecewise = pd.to_numeric(frame[piecewise_column], errors="raise").to_numpy(
        dtype=np.float64
    )
    if not np.all(np.isfinite(piecewise)):
        raise ValueError("Piecewise scores contain a non-finite value")
    if np.any(np.abs(piecewise) > 1.0 + 1e-12):
        raise ValueError("Piecewise scores must lie in [-1, 1]")
    frame["piecewise_scaled_score"] = piecewise

    if "directional_diff_score" in frame:
        directional = pd.to_numeric(
            frame["directional_diff_score"], errors="raise"
        ).to_numpy(dtype=np.float64)
        if not np.all(np.isfinite(directional)):
            raise ValueError("Directional scores contain a non-finite value")
        expected_piecewise = piecewise_scale(directional)
        if not np.allclose(piecewise, expected_piecewise, rtol=0.0, atol=1e-12):
            raise ValueError(
                "piecewise_scaled_score is inconsistent with directional_diff_score"
            )
        frame["directional_diff_score"] = directional

    if "absolute_rank" in frame:
        ranks = pd.to_numeric(frame["absolute_rank"], errors="raise").astype(int)
        if set(ranks) != set(range(1, EXPECTED_GENES + 1)):
            raise ValueError("absolute_rank must be a permutation of 1..1200")
        frame["_source_order"] = ranks
    else:
        frame["_source_order"] = np.arange(1, len(frame) + 1)

    frame["_absolute_piecewise"] = np.abs(piecewise)
    piecewise_order = frame.sort_values(
        ["_absolute_piecewise", "gene"],
        ascending=[False, True],
        kind="stable",
    ).index
    piecewise_ranks = pd.Series(
        np.arange(1, EXPECTED_GENES + 1), index=piecewise_order
    )
    frame["_piecewise_rank"] = piecewise_ranks.reindex(frame.index).to_numpy()
    return frame


def load_query_panel(path: Path, column: str, expected: set[str]) -> None:
    frame = pd.read_csv(require_file(path, "query-gene CSV"))
    if column not in frame:
        raise ValueError(f"Query-gene CSV has no column {column!r}")
    values = frame[column]
    if values.isna().any() or values.astype(str).duplicated().any():
        raise ValueError("Query-gene CSV contains empty or duplicate genes")
    observed = set(values.astype(str))
    if observed != expected:
        raise ValueError("Query-gene panel does not match the 1,200 scored genes")


def load_kegg_report(
    path: Path,
    evaluated_genes: set[str],
    fdr_threshold: float,
) -> tuple[pd.DataFrame, dict[str, frozenset[str]]]:
    report = pd.read_csv(require_file(path, "KEGG Enrichr report"), sep="\t")
    required = {
        "Gene_set",
        "Term",
        "Overlap",
        "P-value",
        "Adjusted P-value",
        "Odds Ratio",
        "Combined Score",
        "Genes",
    }
    missing = sorted(required - set(report.columns))
    if missing:
        raise ValueError(f"KEGG report is missing columns: {missing}")
    report = report[report["Gene_set"] == "KEGG_2021_Human"].copy()
    if len(report) != EXPECTED_KEGG_TERMS:
        raise ValueError(
            f"Expected {EXPECTED_KEGG_TERMS} KEGG terms, found {len(report)}"
        )
    if report["Term"].isna().any() or report["Term"].astype(str).duplicated().any():
        raise ValueError("KEGG report contains an empty or duplicate term")

    for column in ("P-value", "Adjusted P-value", "Odds Ratio", "Combined Score"):
        report[column] = pd.to_numeric(report[column], errors="raise")
        if not np.all(np.isfinite(report[column].to_numpy(dtype=np.float64))):
            raise ValueError(f"KEGG report contains a non-finite {column}")

    memberships: dict[str, frozenset[str]] = {}
    for _, row in report.iterrows():
        term = str(row["Term"])
        members = frozenset(
            gene.strip() for gene in str(row["Genes"]).split(";") if gene.strip()
        )
        outside = members - evaluated_genes
        if outside:
            raise ValueError(
                f"KEGG term {term!r} contains genes outside the scored panel: "
                f"{sorted(outside)[:5]}"
            )
        overlap_fields = str(row["Overlap"]).split("/")
        if len(overlap_fields) != 2 or int(overlap_fields[0]) != len(members):
            raise ValueError(f"Invalid Overlap/Genes entry for KEGG term {term!r}")
        memberships[term] = members

    report = report.sort_values(
        ["Combined Score", "Adjusted P-value", "P-value", "Term"],
        ascending=[False, True, True, True],
        kind="stable",
    ).reset_index(drop=True)
    significant = report[report["Adjusted P-value"] < fdr_threshold]
    if len(significant) < PATHWAY_COUNT:
        raise ValueError(
            f"Only {len(significant)} pathways pass adjusted P < {fdr_threshold}"
        )
    return significant.head(PATHWAY_COUNT).reset_index(drop=True), memberships


def build_figure5d(
    scores: pd.DataFrame,
    kegg_report: Path,
    query_gene_csv: Path,
    query_gene_column: str,
    fdr_threshold: float,
) -> tuple[pd.DataFrame, list[str]]:
    evaluated = set(scores["gene"])
    load_query_panel(query_gene_csv, query_gene_column, evaluated)
    pathways, memberships = load_kegg_report(
        kegg_report, evaluated, fdr_threshold
    )
    ranked = scores.sort_values(
        ["_absolute_piecewise", "gene"],
        ascending=[False, True],
        kind="stable",
    )
    score_by_gene = ranked.set_index("gene")["piecewise_scaled_score"]

    selected: list[dict[str, object]] = []
    used: set[str] = set()
    pathway_terms = pathways["Term"].astype(str).tolist()
    for term in pathway_terms:
        candidates = [
            gene
            for gene in ranked["gene"]
            if gene in memberships[term] and gene not in used
        ]
        if len(candidates) < GENES_PER_PATHWAY:
            raise ValueError(
                f"KEGG pathway {term!r} has fewer than {GENES_PER_PATHWAY} "
                "unassigned scored genes"
            )
        for gene in candidates[:GENES_PER_PATHWAY]:
            used.add(gene)
            selected.append(
                {
                    "gene": gene,
                    "scores": float(score_by_gene.loc[gene]),
                    "KEGG Pathway": term,
                }
            )

    membership_union = frozenset().union(
        *(memberships[term] for term in pathway_terms)
    )
    others = [gene for gene in ranked["gene"] if gene not in membership_union]
    if len(others) < OTHER_GENES:
        raise ValueError("Fewer than four genes lie outside all selected pathways")
    for gene in others[:OTHER_GENES]:
        selected.append(
            {
                "gene": gene,
                "scores": float(score_by_gene.loc[gene]),
                "KEGG Pathway": "Others",
            }
        )

    output = pd.DataFrame(selected)
    if len(output) != PATHWAY_COUNT * GENES_PER_PATHWAY + OTHER_GENES:
        raise RuntimeError("Figure 5D gene selection is incomplete")
    output["_absolute"] = output["scores"].abs()
    output = output.sort_values(
        ["_absolute", "gene"], ascending=[False, True], kind="stable"
    ).drop(columns="_absolute")
    return output.reset_index(drop=True), pathway_terms


def build_figure5e(
    scores: pd.DataFrame,
    tail_quantile: float,
) -> tuple[pd.DataFrame, np.ndarray, tuple[float, float]]:
    if not 0.0 <= tail_quantile < 0.5:
        raise ValueError("--tail-trim-quantile must lie in [0, 0.5)")
    ordered = scores.sort_values("_source_order", kind="stable")
    values = ordered["piecewise_scaled_score"].to_numpy(dtype=np.float64)
    lower, upper = (
        float(value)
        for value in np.quantile(
            values,
            [tail_quantile, 1.0 - tail_quantile],
            method="linear",
        )
    )
    retained = (values >= lower) & (values <= upper)
    if int(np.count_nonzero(retained)) < 2:
        raise ValueError("Central score selection retained fewer than two genes")
    output = ordered[["gene", "piecewise_scaled_score"]].rename(
        columns={"piecewise_scaled_score": "score"}
    )
    return output.reset_index(drop=True), retained, (lower, upper)


def load_embedding_inputs(args: argparse.Namespace) -> EmbeddingInputs:
    directory = require_dir(args.gene_embedding_dir, "embedding directory")
    genes_path = require_file(
        args.genes_json or directory / "hvg_genes.json", "ordered gene JSON"
    )
    with genes_path.open(encoding="utf-8") as handle:
        genes = json.load(handle)
    if (
        not isinstance(genes, list)
        or len(genes) != EXPECTED_GENES
        or not all(isinstance(gene, str) and gene for gene in genes)
        or len(set(genes)) != EXPECTED_GENES
    ):
        raise ValueError("Gene JSON must contain exactly 1,200 unique gene names")
    if any(Path(gene).name != gene for gene in genes):
        raise ValueError("Gene names must not contain path separators")

    baseline = require_file(directory / "origin.npy", "baseline embedding")
    noop = require_file(directory / "noop.npy", "independent no-op embedding")
    knockouts = tuple(
        require_file(directory / f"{gene}_embedding.npy", f"KO embedding for {gene}")
        for gene in genes
    )

    expected_shape = (args.expected_cells, EMBEDDING_DIM)
    iterator = tqdm(
        (baseline, noop, *knockouts),
        desc="Validating embeddings",
        unit="file",
        disable=not args.progress,
        dynamic_ncols=True,
    )
    for path in iterator:
        array = np.load(path, mmap_mode="r", allow_pickle=False)
        if tuple(array.shape) != expected_shape:
            raise ValueError(
                f"{path.name} has shape {array.shape}; expected {expected_shape}"
            )
        if array.dtype != np.float32:
            raise ValueError(f"{path.name} must have dtype float32")

    baseline_array = np.load(baseline, allow_pickle=False)
    noop_array = np.load(noop, allow_pickle=False)
    if not np.all(np.isfinite(baseline_array)) or not np.all(np.isfinite(noop_array)):
        raise ValueError("Baseline or no-op embedding contains non-finite values")
    if not np.allclose(baseline_array, noop_array, rtol=1e-5, atol=1e-6):
        maximum = float(np.max(np.abs(baseline_array - noop_array)))
        raise ValueError(
            "Baseline and independent no-op embeddings are inconsistent: "
            f"maximum absolute difference={maximum:.9g}"
        )
    return EmbeddingInputs(
        genes=tuple(genes),
        baseline=baseline,
        noop=noop,
        knockouts=knockouts,
        shape=expected_shape,
    )


def load_regions(path: Path) -> tuple[Interval, ...]:
    with require_file(path, "regions JSON").open(encoding="utf-8") as handle:
        region_map = json.load(handle)
    raw_regions = region_map.get(CELL_TYPE)
    if not isinstance(raw_regions, list) or len(raw_regions) != EXPECTED_REGIONS:
        count = len(raw_regions) if isinstance(raw_regions, list) else 0
        raise ValueError(
            f"Expected {EXPECTED_REGIONS} {CELL_TYPE} regions, found {count}"
        )
    regions: list[Interval] = []
    for raw in raw_regions:
        if not isinstance(raw, (list, tuple)) or len(raw) < 3:
            raise ValueError(f"Invalid genomic region: {raw!r}")
        chrom, start, end = str(raw[0]), int(raw[1]), int(raw[2])
        if start > end:
            start, end = end, start
        if start == end:
            raise ValueError(f"Zero-length genomic region: {raw!r}")
        regions.append(Interval(chrom, start, end))
    return tuple(regions)


def seed_everything(seed: int) -> None:
    import torch

    random.seed(seed)
    np.random.seed(seed)
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.enabled = False


def resolve_device(requested: str):
    import torch

    if requested.startswith("cuda") and not torch.cuda.is_available():
        print("CUDA is unavailable; falling back to CPU.")
        requested = "cpu"
    return torch.device(requested)


def setup_model_code(directory: Path) -> Path:
    directory = require_dir(directory, "model code directory")
    require_file(directory / "cell_embedding.py", "model definition")
    require_dir(directory / "selene_mini", "selene_mini package")
    text = str(directory)
    while text in sys.path:
        sys.path.remove(text)
    sys.path.insert(0, text)
    return directory


def strip_checkpoint_prefix(key: str) -> str:
    changed = True
    while changed:
        changed = False
        for prefix in ("module.", "_orig_mod."):
            if key.startswith(prefix):
                key = key[len(prefix) :]
                changed = True
    return key


def load_model(args: argparse.Namespace, device):
    import torch

    module = importlib.import_module("cell_embedding")
    module_path = Path(module.__file__).resolve()
    if module_path.parent != args.model_code_dir:
        raise ImportError(
            f"Imported cell_embedding from {module_path}, not {args.model_code_dir}"
        )
    model = module.PuffinDWithCellEmbedding(args, n_track=1)
    try:
        payload = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    except TypeError:
        payload = torch.load(args.checkpoint, map_location="cpu")
    state = payload.get("model_state_dict", payload) if isinstance(payload, dict) else payload
    if not isinstance(state, Mapping):
        raise TypeError("Checkpoint does not contain a state dictionary")
    state = {strip_checkpoint_prefix(str(key)): value for key, value in state.items()}
    model_state = model.state_dict()
    missing = sorted(set(model_state) - set(state))
    unexpected = sorted(set(state) - set(model_state))
    mismatched = sorted(
        key
        for key in set(state) & set(model_state)
        if getattr(state[key], "shape", None) != model_state[key].shape
    )
    if missing or unexpected or mismatched:
        raise RuntimeError(
            "Checkpoint is not an exact embedding-conditioned model match: "
            f"missing={len(missing)}, unexpected={len(unexpected)}, "
            f"shape_mismatch={len(mismatched)}"
        )
    model.load_state_dict(state, strict=True)
    model = model.to(device).eval()
    if not hasattr(model, "cell_mod") or hasattr(model, "dynamic_layer_generator"):
        raise ValueError("Model is not the expected FiLM cell_mod architecture")
    print(f"Loaded {len(state)} checkpoint tensors from {args.checkpoint.name}")
    return model


def load_genome(args: argparse.Namespace, device):
    module = importlib.import_module("selene_mini")
    imported = Path(module.__file__).resolve().parent
    expected = (args.model_code_dir / "selene_mini").resolve()
    if imported != expected:
        raise ImportError(f"Imported selene_mini from {imported}, expected {expected}")
    return module.Genome(
        input_path=str(args.genome_fa), cuda=device.type == "cuda"
    )


def cpg_mask(sequence: np.ndarray) -> np.ndarray:
    if sequence.ndim != 2 or sequence.shape[1] != 4:
        raise ValueError(f"Expected one-hot sequence shaped (L,4), got {sequence.shape}")
    mask = np.zeros(sequence.shape[0], dtype=bool)
    pairs = (sequence[:-1, 1] == 1) & (sequence[1:, 2] == 1)
    mask[:-1] |= pairs
    mask[1:] |= pairs
    return mask


def prepare_sequences(
    genome,
    regions: Sequence[Interval],
) -> tuple[np.ndarray, np.ndarray]:
    chromosome_lengths = dict(genome.get_chr_lens())
    sequences = np.empty((len(regions), WINDOW_SIZE, 4), dtype=np.float32)
    target_masks = np.zeros((len(regions), WINDOW_SIZE), dtype=bool)
    for index, region in enumerate(regions):
        if region.chrom not in chromosome_lengths:
            raise ValueError(f"Chromosome {region.chrom} is absent from the genome")
        centre = (region.start + region.end) // 2
        context_start = centre - WINDOW_SIZE // 2
        context_end = context_start + WINDOW_SIZE
        if context_start < 0 or context_end > chromosome_lengths[region.chrom]:
            raise ValueError(f"10-kb context for {region} lies outside the genome")
        crop_start = region.start - context_start
        crop_end = region.end - context_start
        if crop_start < 0 or crop_end > WINDOW_SIZE or crop_end <= crop_start:
            raise ValueError(f"Region {region} is outside its 10-kb context")
        sequence = np.asarray(
            genome.get(region.chrom, context_start, context_end), dtype=np.float32
        )
        if sequence.shape != (WINDOW_SIZE, 4):
            raise ValueError(
                f"Genome returned {sequence.shape}; expected ({WINDOW_SIZE}, 4)"
            )
        target_masks[index, crop_start:crop_end] = cpg_mask(sequence)[
            crop_start:crop_end
        ]
        sequences[index] = sequence
    if not np.any(target_masks):
        raise ValueError("Figure 5D/E regions contain no CpG positions")
    return sequences, target_masks


def mean_embedding(path: Path) -> np.ndarray:
    array = np.load(path, allow_pickle=False)
    if array.ndim != 2 or array.shape[1] != EMBEDDING_DIM:
        raise ValueError(f"Invalid embedding shape in {path}: {array.shape}")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"Embedding contains a non-finite value: {path}")
    return np.asarray(array.mean(axis=0, dtype=np.float32), dtype=np.float32)


def mean_prediction(
    model,
    sequences: np.ndarray,
    masks: np.ndarray,
    embedding: np.ndarray,
    batch_size: int,
    device,
) -> float:
    import torch

    total = 0.0
    count = 0
    for start in range(0, len(sequences), batch_size):
        end = min(start + batch_size, len(sequences))
        sequence_tensor = (
            torch.from_numpy(sequences[start:end])
            .to(device=device, dtype=torch.float32)
            .permute(0, 2, 1)
            .contiguous()
        )
        embedding_tensor = torch.from_numpy(embedding).to(
            device=device, dtype=torch.float32
        )
        embedding_tensor = embedding_tensor.unsqueeze(0).expand(end - start, -1)
        with torch.inference_mode():
            output = model(sequence_tensor, cell_embedding=embedding_tensor)
            logits = output[0] if isinstance(output, (tuple, list)) else output
            if logits.ndim != 3 or logits.shape[1] != 1:
                raise ValueError(f"Expected model output (B,1,L), got {tuple(logits.shape)}")
            predictions = torch.sigmoid(logits).detach().cpu().numpy()[:, 0, :]
        selected = np.asarray(predictions[masks[start:end]], dtype=np.float64)
        if not np.all(np.isfinite(selected)):
            raise ValueError("Model produced a non-finite CpG probability")
        total += float(selected.sum(dtype=np.float64))
        count += int(selected.size)
    expected = int(np.count_nonzero(masks))
    if count != expected or count == 0:
        raise RuntimeError(f"CpG aggregation count changed: {count} versus {expected}")
    return total / count


def recompute_scores(
    args: argparse.Namespace,
    embeddings: EmbeddingInputs,
    regions: Sequence[Interval],
) -> pd.DataFrame:
    import torch

    seed_everything(args.seed)
    device = resolve_device(args.device)
    genome = load_genome(args, device)
    model = None
    try:
        sequences, masks = prepare_sequences(genome, regions)
        model = load_model(args, device)
        baseline_mean = mean_prediction(
            model,
            sequences,
            masks,
            mean_embedding(embeddings.baseline),
            args.batch_size,
            device,
        )
        knockout_means = np.empty(EXPECTED_GENES, dtype=np.float64)
        iterator = tqdm(
            zip(embeddings.genes, embeddings.knockouts),
            total=EXPECTED_GENES,
            desc="Melody KO inference",
            unit="gene",
            disable=not args.progress,
            dynamic_ncols=True,
        )
        for index, (gene, path) in enumerate(iterator):
            iterator.set_postfix_str(gene, refresh=False)
            knockout_means[index] = mean_prediction(
                model,
                sequences,
                masks,
                mean_embedding(path),
                args.batch_size,
                device,
            )
    finally:
        if model is not None:
            del model
        if hasattr(genome, "uninitialize"):
            genome.uninitialize()
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    directional = knockout_means - baseline_mean
    max_absolute = float(np.max(np.abs(directional)))
    normalized = (
        directional / max_absolute
        if max_absolute > 0.0
        else np.zeros_like(directional)
    )
    piecewise = piecewise_scale(directional)
    ranking = sorted(
        range(EXPECTED_GENES),
        key=lambda index: (-abs(float(directional[index])), embeddings.genes[index]),
    )
    rank_by_index = {index: rank for rank, index in enumerate(ranking, start=1)}
    rows = []
    for index in ranking:
        score = float(directional[index])
        rows.append(
            {
                "absolute_rank": rank_by_index[index],
                "gene": embeddings.genes[index],
                "baseline_mean_cpg_methylation": baseline_mean,
                "knockout_mean_cpg_methylation": float(knockout_means[index]),
                "directional_diff_score": score,
                "normalized_directional_score": float(normalized[index]),
                "piecewise_scaled_score": float(piecewise[index]),
                "absolute_directional_diff": abs(score),
                "effect_direction": effect_direction(score),
            }
        )
    print(
        "Checkpoint/embedding training-coordinate compatibility was not verified."
    )
    return pd.DataFrame(rows)


def validate_recompute_args(args: argparse.Namespace) -> tuple[EmbeddingInputs, tuple[Interval, ...]]:
    required = {
        "--gene-embedding-dir": args.gene_embedding_dir,
        "--checkpoint": args.checkpoint,
        "--model-code-dir": args.model_code_dir,
        "--genome-fa": args.genome_fa,
        "--regions-json": args.regions_json,
        "--recomputed-score-dir": args.recomputed_score_dir,
    }
    missing = [option for option, value in required.items() if value is None]
    if missing:
        raise ValueError(
            "Recompute mode requires: " + ", ".join(missing)
        )
    if args.batch_size <= 0:
        raise ValueError("--batch-size must be positive")
    if args.expected_cells <= 0:
        raise ValueError("--expected-cells must be positive")
    args.checkpoint = require_file(args.checkpoint, "Melody checkpoint")
    args.model_code_dir = setup_model_code(args.model_code_dir)
    args.genome_fa = require_file(args.genome_fa, "genome FASTA")
    require_file(Path(f"{args.genome_fa}.fai"), "genome FASTA index")
    args.regions_json = require_file(args.regions_json, "regions JSON")
    args.recomputed_score_dir = args.recomputed_score_dir.expanduser().resolve()
    score_path = args.recomputed_score_dir / "methylation_diff_scores.csv"
    if score_path.exists() and not args.force:
        raise FileExistsError(
            f"Recomputed score output already exists: {score_path}"
        )
    embeddings = load_embedding_inputs(args)
    regions = load_regions(args.regions_json)
    return embeddings, regions


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Recompute or post-process Figure 5D/5E source data.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--panel", required=True, type=str.upper, choices=PANELS)
    parser.add_argument("--scores-csv", type=Path)
    parser.add_argument("--output-csv", type=Path)
    parser.add_argument("--validate-only", action="store_true")
    parser.add_argument("--force", action="store_true")

    parser.add_argument("--kegg-report", type=Path)
    parser.add_argument("--query-gene-list-csv", type=Path)
    parser.add_argument("--query-gene-column", default="gene")
    parser.add_argument("--fdr-threshold", type=float, default=FDR_THRESHOLD)
    parser.add_argument(
        "--tail-trim-quantile", type=float, default=TAIL_QUANTILE
    )

    parser.add_argument("--gene-embedding-dir", type=Path)
    parser.add_argument("--genes-json", type=Path)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--model-code-dir", type=Path)
    parser.add_argument("--genome-fa", type=Path)
    parser.add_argument("--regions-json", type=Path)
    parser.add_argument("--recomputed-score-dir", type=Path)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--seed", type=int, default=5)
    parser.add_argument("--expected-cells", type=int, default=EXPECTED_CELLS)
    parser.add_argument(
        "--progress", action=argparse.BooleanOptionalAction, default=True
    )
    args = parser.parse_args()
    args.output_csv = args.output_csv or DEFAULT_OUTPUTS[args.panel]
    args.allow_partial_checkpoint = False
    args.compile = False
    args.window_size = WINDOW_SIZE
    return args


def main() -> None:
    args = parse_args()
    if args.panel == "5D":
        if args.kegg_report is None or args.query_gene_list_csv is None:
            raise ValueError(
                "Figure 5D requires --kegg-report and --query-gene-list-csv"
            )
        if not 0.0 < args.fdr_threshold <= 1.0:
            raise ValueError("--fdr-threshold must lie in (0, 1]")

    if args.scores_csv is not None:
        scores = load_scores(args.scores_csv)
    else:
        embeddings, regions = validate_recompute_args(args)
        if args.validate_only:
            if args.panel == "5D":
                evaluated = set(embeddings.genes)
                load_query_panel(
                    args.query_gene_list_csv,
                    args.query_gene_column,
                    evaluated,
                )
                load_kegg_report(
                    args.kegg_report,
                    evaluated,
                    args.fdr_threshold,
                )
            print("Validation successful; inference was not run.")
            return
        scores = recompute_scores(args, embeddings, regions)
        score_path = args.recomputed_score_dir / "methylation_diff_scores.csv"
        atomic_write_csv(scores, score_path, args.force)
        print(f"Wrote recomputed scores: {score_path}")
        scores = load_scores(score_path)

    if args.panel == "5D":
        output, pathways = build_figure5d(
            scores,
            args.kegg_report,
            args.query_gene_list_csv,
            args.query_gene_column,
            args.fdr_threshold,
        )
        print("Selected FDR-significant pathways: " + "; ".join(pathways))
    else:
        output, retained, thresholds = build_figure5e(
            scores, args.tail_trim_quantile
        )
        excluded = len(retained) - int(np.count_nonzero(retained))
        print(
            f"Central selection: {int(np.count_nonzero(retained))} retained, "
            f"{excluded} excluded; thresholds={thresholds[0]:.9g},"
            f"{thresholds[1]:.9g}"
        )

    if args.validate_only:
        print("Validation successful; output was not written.")
        return
    atomic_write_csv(output, args.output_csv, args.force)
    print(f"Wrote Figure {args.panel} source data: {args.output_csv}")


if __name__ == "__main__":
    main()
