"""
ISM (in-silico saturation mutagenesis) sequence-logo plotting for Figure 3A/B.

For a single meQTL (a SNP paired with a nearby CpG), this produces a four-panel
figure: the model's predicted per-base methylation landscape around the SNP for
the reference and alternative alleles, and the saturation-mutagenesis sequence
logos (attribution of every possible single-base substitution) for both alleles.

The Figure3_AB_ism.ipynb notebook builds the model and genome and passes them to
:func:`plot_meqtl_plus_ism`; this file holds only the plotting logic. It imports a
few helpers from the Melody model repository (which the notebook puts on
``sys.path``): ``stateless.sigmoid_first``, ``global_constants.track_39_bigwig_file_names``,
and the bundled meQTL tables ``eqtl_pure_util.df_dict``. Saturation mutagenesis and
logo drawing use the third-party ``tangermeme`` package (``pip install tangermeme``).
"""

from dataclasses import dataclass
from typing import Optional, List, Tuple, Union, Sequence, Callable

import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

# --- helpers from the Melody model repository (put on sys.path by the notebook) ---
from selene_sdk.sequences import Genome            # used only for type hints
from stateless import sigmoid_first
from global_constants import track_39_bigwig_file_names
from eqtl_pure_util import df_dict

# --- third-party: saturation mutagenesis + logo plotting ---
from tangermeme.saturation_mutagenesis import saturation_mutagenesis
from tangermeme.plot import plot_logo


# ---------------------------------------------------------------------------
# 0) Track handling & model forward
# ---------------------------------------------------------------------------
def _ensure_track_indices(track_indices: Optional[List[int]], default_n_tracks: int = 1) -> List[int]:
    if not track_indices:
        return list(range(default_n_tracks)) if default_n_tracks > 1 else [0]
    return list(track_indices)


def _select_and_reduce_tracks(t: torch.Tensor, track_indices: List[int]) -> torch.Tensor:
    if t.dim() == 2:
        return t  # (B, L)
    if t.dim() == 3:
        if t.size(1) == 1:
            return t.squeeze(1)  # (B, L)
        idx = torch.tensor(track_indices, device=t.device, dtype=torch.long)
        t_sel = torch.index_select(t, 1, idx)  # (B, len(idx), L)
        return t_sel.mean(dim=1)  # (B, L)
    raise ValueError(f"Unexpected tensor shape for track reduce: {tuple(t.shape)}")


def _forward_model_get_prob_map(
    model: nn.Module,
    x_batch: torch.Tensor,
    track_indices: Optional[List[int]] = None,
) -> torch.Tensor:
    device = next(model.parameters()).device if any(p.requires_grad for p in model.parameters()) else (
        torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    )
    x_batch = x_batch.to(device)
    model.to(device)
    model.eval()
    with torch.no_grad():
        out = model(x_batch)
    prob = sigmoid_first(out)  # (B, T, L) or (B, L)
    track_indices = _ensure_track_indices(track_indices or [0])
    prob = _select_and_reduce_tracks(prob, track_indices)  # (B, L)
    return prob


class TrackSliceReduceWrapper(nn.Module):
    """Wrap a Melody model so it exposes a single reduced track over a slice,
    as required by tangermeme's saturation_mutagenesis (which expects a model
    mapping a one-hot sequence to a scalar/1-D output per position)."""

    def __init__(
        self,
        base_model: nn.Module,
        tracks_idx: Optional[Sequence[int]] = None,
        predict_slice: Optional[Tuple[int, int]] = None,
        postproc: Union[str, Callable[[torch.Tensor], torch.Tensor], None] = "sigmoid",
        channel_reduce: str = "mean",
    ):
        super().__init__()
        self.base_model = base_model
        self.tracks_idx = None if tracks_idx is None else [int(i) for i in tracks_idx]
        self.predict_slice = predict_slice  # inclusive tuple (s, e)

        if postproc is None:
            self.postproc_fn = None
        elif isinstance(postproc, str):
            if postproc.lower() == "sigmoid":
                self.postproc_fn = torch.sigmoid
            elif postproc.lower() in ("id", "identity", "none"):
                self.postproc_fn = None
            else:
                raise ValueError("postproc must be 'sigmoid' or 'identity'")
        elif callable(postproc):
            self.postproc_fn = postproc
        else:
            raise ValueError("invalid postproc argument")

        if channel_reduce not in ("sum", "mean"):
            raise ValueError("channel_reduce must be 'sum' or 'mean'")
        self.channel_reduce = channel_reduce

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out_final, _, _ = self.base_model(x)   # Melody -> (out_final[B,T,L], pred_cg, pred_avg)
        y = out_final

        if self.postproc_fn is not None:
            y = self.postproc_fn(y)

        if self.tracks_idx is not None:
            y = y[:, self.tracks_idx, :]

        if self.channel_reduce == "sum":
            y = y.sum(dim=1, keepdim=True)
        else:
            y = y.mean(dim=1, keepdim=True)

        if self.predict_slice is not None:
            s, e = self.predict_slice
            y = y[:, :, s:e + 1]
        return y


# ---------------------------------------------------------------------------
# 1) Extract a single meQTL record & predict the allelic effect
# ---------------------------------------------------------------------------
@dataclass
class MeQTLInstance:
    dataset: str
    index: int
    chrom: str
    snp_start: int
    snp_ref: str
    snp_alt: str
    cpg_start: int
    cpg_end: int
    fetch_start: int
    fetch_end: int
    observed_effect: float


def _get_instance(df, idx: int) -> Tuple[int, MeQTLInstance]:
    row = df.iloc[idx]
    row_idx = idx

    chrom = str(row['chrom'])
    snp_start = int(float(row['SNP_region_start']))
    snp_ref = str(row['SNP_ref'])
    snp_alt = str(row['SNP_alt'])
    cpg_start = int(float(row['CPG_region_start']))
    cpg_end = int(float(row['CPG_region_end']))
    obs = float(row['effect_size'])

    center = (cpg_start + snp_start) // 2
    fetch_start = max(0, center - 5000)
    fetch_end = center + 5000

    inst = MeQTLInstance(
        dataset="", index=row_idx, chrom=chrom,
        snp_start=snp_start, snp_ref=snp_ref, snp_alt=snp_alt,
        cpg_start=cpg_start, cpg_end=cpg_end,
        fetch_start=fetch_start, fetch_end=fetch_end,
        observed_effect=obs,
    )
    return row_idx, inst


def _onehot_from_base(base: str):
    base = base.upper()
    if base == 'A': return [1, 0, 0, 0]
    if base == 'C': return [0, 1, 0, 0]
    if base == 'G': return [0, 0, 1, 0]
    if base == 'T': return [0, 0, 0, 1]
    return [0, 0, 0, 0]


def predict_change_for_instance(
    model: nn.Module,
    genome: Genome,
    df,
    dataset_name: str,
    idx: int,
    half_model_input_len: int = 5000,
    margin: int = 0,
    track_indices: Optional[List[int]] = None,
):
    row_idx, inst = _get_instance(df, idx)
    inst.dataset = dataset_name

    center = (inst.cpg_start + inst.snp_start) // 2
    fetch_start = max(0, center - half_model_input_len)
    fetch_end = center + half_model_input_len

    seq = genome.get(inst.chrom, fetch_start, fetch_end)
    if not isinstance(seq, np.ndarray):
        seq = np.array(seq, dtype=np.float32)
    seq = seq.astype(np.float32)
    seq_mut = seq.copy()

    snp_rel = inst.snp_start - fetch_start - 1
    if not (0 <= snp_rel < seq.shape[0]):
        raise RuntimeError(f"SNP relative coordinate out of range: rel={snp_rel}, L={seq.shape[0]}")

    seq[snp_rel:snp_rel + 1, :] = np.array(_onehot_from_base(inst.snp_ref), dtype=np.float32)
    seq_mut[snp_rel:snp_rel + 1, :] = np.array(_onehot_from_base(inst.snp_alt), dtype=np.float32)

    x_ref = torch.tensor(seq, dtype=torch.float32).unsqueeze(0).transpose(1, 2)
    x_alt = torch.tensor(seq_mut, dtype=torch.float32).unsqueeze(0).transpose(1, 2)

    prob_ref = _forward_model_get_prob_map(model, x_ref, track_indices=track_indices)[0].cpu().numpy()
    prob_alt = _forward_model_get_prob_map(model, x_alt, track_indices=track_indices)[0].cpu().numpy()

    cpg_rel_start = inst.cpg_start - fetch_start
    cpg_rel_end = inst.cpg_end - fetch_start

    check_start = max(0, cpg_rel_start - margin)
    check_end = min(len(prob_ref), cpg_rel_end + margin)

    pred_effect = float(np.sum(prob_alt[check_start:check_end] - prob_ref[check_start:check_end]))
    observed_effect = float(inst.observed_effect)

    extra = dict(
        fetch_start=fetch_start, fetch_end=fetch_end, chrom=inst.chrom,
        prob_ref=prob_ref, prob_alt=prob_alt,
        snp_rel=snp_rel, snp_abs=inst.snp_start,
        cpg_rel_start=cpg_rel_start, cpg_rel_end=cpg_rel_end,
        cpg_abs_start=inst.cpg_start, cpg_abs_end=inst.cpg_end,
        pred_effect=pred_effect, observed_effect=observed_effect,
        margin=margin, index=row_idx, snp_ref=inst.snp_ref, snp_alt=inst.snp_alt,
    )
    return pred_effect, observed_effect, extra, inst


# ---------------------------------------------------------------------------
# 2) ISM helpers
# ---------------------------------------------------------------------------
def _as_model_input(x) -> torch.Tensor:
    if isinstance(x, np.ndarray):
        x = torch.from_numpy(x)
    x = x.float()
    if x.ndim == 2:
        x = x.unsqueeze(0).transpose(1, 2)  # [L, 4] -> [1, 4, L]
    elif x.ndim == 3:
        if x.shape[1] != 4 and x.shape[-1] == 4:
            x = x.transpose(1, 2)
    else:
        raise ValueError("seq_onehot must be [L, 4] or [1, 4, L]")
    return x


def _get_chrom_len(genome: Genome, chrom: str) -> Optional[int]:
    cand_attrs = ["len_chrs", "chrom_lengths", "chrom_lens", "chrom_sizes", "chrom_sizes_bp", "lengths", "lens"]
    for attr in cand_attrs:
        if hasattr(genome, attr):
            d = getattr(genome, attr)
            try:
                return int(d[chrom])
            except Exception:
                pass
    return None


def _fetch_window_from_genome(
    genome: Genome,
    chrom: str,
    start: int,
    end: int,
    required_len: Optional[int],
    align: str = "center",
) -> Tuple[torch.Tensor, int, int]:
    if required_len is None:
        final_start = int(start)
        final_end = int(end)
        seq = genome.get(chrom, final_start, final_end)
        X = _as_model_input(seq)
        return X, final_start, final_end

    req_len = int(required_len)
    if align not in ("center", "left", "right"):
        raise ValueError("align must be 'center'/'left'/'right'")

    start = int(start)
    end = int(end)
    if end <= start:
        raise ValueError(f"invalid interval: start={start}, end={end}")

    if align == "center":
        mid = (start + end) // 2
        final_start = mid - req_len // 2
        final_end = final_start + req_len
    elif align == "left":
        final_start = start
        final_end = start + req_len
    else:
        final_end = end
        final_start = final_end - req_len

    chrom_len = _get_chrom_len(genome, chrom)
    if final_start < 0:
        final_start = 0
        final_end = final_start + req_len
    if chrom_len is not None and final_end > chrom_len:
        final_end = chrom_len
        final_start = final_end - req_len
        if final_start < 0:
            raise ValueError(f"need {req_len}bp but {chrom} is only {chrom_len}bp long")

    seq = genome.get(chrom, final_start, final_end)
    X = _as_model_input(seq)
    if X.shape[-1] != req_len:
        raise RuntimeError(f"genome.get length {X.shape[-1]} != expected {req_len}bp.")
    return X, final_start, final_end


def _shorten_track_name(fn: str) -> str:
    import os
    base = os.path.basename(fn)
    if "_" in base:
        base = base.split("_", 1)[1]
    base = base.replace(".hg38.bigwig", "")
    if "-Z" in base:
        base = base.split("-Z", 1)[0]
    return base


def _get_track_label_short(track_indices: List[int]) -> str:
    short_names = []
    for ti in track_indices:
        if 0 <= ti < len(track_39_bigwig_file_names):
            short_names.append(_shorten_track_name(track_39_bigwig_file_names[ti]))
        else:
            short_names.append(f"track{ti}")
    return ",".join(short_names)


def _apply_allele_inplace(x14L: torch.Tensor, pos: int, base: str):
    vec = torch.tensor(_onehot_from_base(base), dtype=x14L.dtype, device=x14L.device)
    x14L[0, :, pos] = vec


def _prepare_ref_alt_sequences_for_snp(
    genome: Genome,
    chrom: str,
    snp_pos_1based: int,
    ref_base: str,
    alt_base: str,
    input_len: int = 10_000,
    ism_flank: int = 75,
):
    region_start = int(snp_pos_1based - ism_flank)
    region_end = int(snp_pos_1based + ism_flank)

    X_ref, final_start, final_end = _fetch_window_from_genome(
        genome, chrom=chrom, start=region_start, end=region_end,
        required_len=input_len, align="center",
    )
    X_alt = X_ref.clone()

    snp_rel = snp_pos_1based - final_start - 1
    if snp_rel < 0 or snp_rel >= X_ref.shape[-1]:
        raise RuntimeError(
            f"SNP relative coordinate out of range: snp_rel={snp_rel}, L={X_ref.shape[-1]}, "
            f"snp_pos={snp_pos_1based}, final_start={final_start}"
        )

    _apply_allele_inplace(X_ref, snp_rel, ref_base)
    _apply_allele_inplace(X_alt, snp_rel, alt_base)
    return X_ref, X_alt, snp_rel, final_start, final_end


def _run_ism_and_plot_logo(
    model: nn.Module,
    X_input: torch.Tensor,
    snp_rel: int,
    ism_flank: int,
    tracks_idx: Optional[List[int]],
    track_label_short: str,
    ax: plt.Axes,
    batch_size: int = 128,
    allele_label: str = "REF",
    chrom: str = "",
    snp_abs_1based: int = 0,
    ref_base: str = "",
    alt_base: str = "",
):
    L_in = X_input.shape[-1]
    mut_start = max(0, snp_rel - ism_flank)
    mut_end = min(L_in - 1, snp_rel + ism_flank)
    mut_end_excl = mut_end + 1

    # tangermeme's saturation_mutagenesis runs on CPU here.
    wrapper = TrackSliceReduceWrapper(
        base_model=model,
        tracks_idx=tracks_idx,
        predict_slice=(mut_start, mut_end),
        postproc="sigmoid",
        channel_reduce="mean",
    ).to('cpu')
    wrapper.eval()

    with torch.no_grad():
        y_attr = saturation_mutagenesis(
            wrapper,
            X_input.cpu(),
            batch_size=batch_size,
            start=torch.tensor(mut_start, device='cpu'),
            end=torch.tensor(mut_end_excl, device='cpu'),
        )

    mat = y_attr[0].detach().cpu().numpy()

    snp_local = snp_rel - mut_start
    snp_local_f = float(snp_local)

    plot_logo(mat, ax=ax)

    ax.axvspan(snp_local_f - 0.5, snp_local_f + 0.5, color='red', alpha=0.15, linewidth=0)
    ax.axvline(snp_local_f, color='red', linestyle='--', linewidth=0.8)

    ymin, ymax = ax.get_ylim()
    snp_change_label = f"{ref_base}→{alt_base}" if ref_base and alt_base else "SNP"
    ax.text(snp_local_f, ymax * 1.05, snp_change_label, color='red',
            ha='center', va='bottom', fontsize=8, clip_on=False)
    ax.set_ylim(ymin, ymax * 1.15)
    ax.set_xlabel(f"Position around SNP ±{ism_flank}bp")
    ax.set_ylabel("Δpred (ISM)")


# ---------------------------------------------------------------------------
# 3) Combined four-panel figure
# ---------------------------------------------------------------------------
def plot_meqtl_plus_ism(
    dataset_name: str,
    idx: int,
    model: nn.Module,
    genome: Genome,
    track_indices: Optional[List[int]] = None,
    half_model_input_len: int = 5000,
    margin: int = 0,
    view_bp: int = 75,
    ism_flank: int = 75,
    batch_size: int = 128,
    save_path: Optional[str] = None,
    figsize: Tuple[float, float] = (12, 6),
    rank: Optional[int] = None,
    quadrant: Optional[str] = None,
    score: Optional[float] = None,
    landscape_half_len: int = 1500,
):
    """Four-panel figure for one meQTL:
        1) predicted methylation landscape, reference allele
        2) predicted methylation landscape, alternative allele
        3) ISM saturation-mutagenesis logo (reference)
        4) ISM saturation-mutagenesis logo (alternative)

    Rows 1-2 show only the model prediction over a window of
    ``±landscape_half_len`` around the SNP; rows 3-4 show the ISM attribution
    over ``±ism_flank`` around the SNP.
    """
    df = df_dict[dataset_name]
    pred, obs, extra, inst = predict_change_for_instance(
        model=model, genome=genome, df=df, dataset_name=dataset_name, idx=idx,
        half_model_input_len=half_model_input_len, margin=margin, track_indices=track_indices,
    )

    chrom = extra['chrom']
    fetch_start = extra['fetch_start']
    fetch_end = extra['fetch_end']
    prob_ref = extra['prob_ref']
    prob_alt = extra['prob_alt']
    snp_rel = extra['snp_rel']
    snp_abs = extra['snp_abs']
    cpg_rel_start = extra['cpg_rel_start']
    cpg_rel_end = extra['cpg_rel_end']

    L = len(prob_ref)
    win_lo = max(0, snp_rel - landscape_half_len)
    win_hi = min(L, snp_rel + landscape_half_len)

    sel_tracks = _ensure_track_indices(track_indices or [0])
    track_label_short = _get_track_label_short(sel_tracks)

    fig = plt.figure(figsize=figsize)
    gs = fig.add_gridspec(nrows=4, ncols=1, height_ratios=[1.2, 1.2, 1.6, 1.6], hspace=0.6)
    ax_pred_ref = fig.add_subplot(gs[0, 0])
    ax_pred_alt = fig.add_subplot(gs[1, 0])
    ax_ism_ref = fig.add_subplot(gs[2, 0])
    ax_ism_alt = fig.add_subplot(gs[3, 0])

    def _draw_pred(ax: plt.Axes, y: np.ndarray, label: str):
        x = np.arange(win_lo, win_hi)
        ax.plot(x, y[win_lo:win_hi], lw=1.2, label=label, alpha=0.9)
        ax.axvline(snp_rel, ls="--", alpha=0.6, color='k')
        ax.axvspan(cpg_rel_start, cpg_rel_end, alpha=0.15, color='gray')
        ax.set_xlim(win_lo, win_hi - 1)
        ax.set_ylim(-0.05, 1.05)
        ax.set_xlabel(f"Relative position in 10kb window (±{landscape_half_len}bp view)")
        ax.legend(frameon=False, fontsize=8)
        ax.grid(alpha=0.25)

    _draw_pred(ax_pred_ref, prob_ref, label="Pred-Before")
    _draw_pred(ax_pred_alt, prob_alt, label="Pred-After")

    X_ref, X_alt, snp_rel_snpwin, _fs, _fe = _prepare_ref_alt_sequences_for_snp(
        genome=genome, chrom=inst.chrom, snp_pos_1based=inst.snp_start,
        ref_base=inst.snp_ref, alt_base=inst.snp_alt,
        input_len=half_model_input_len * 2, ism_flank=ism_flank,
    )

    _run_ism_and_plot_logo(
        model=model, X_input=X_ref, snp_rel=snp_rel_snpwin, ism_flank=ism_flank,
        tracks_idx=sel_tracks, track_label_short=track_label_short, ax=ax_ism_ref,
        batch_size=batch_size, allele_label=f"REF={inst.snp_ref}", chrom=inst.chrom,
        snp_abs_1based=inst.snp_start, ref_base=inst.snp_ref, alt_base=inst.snp_alt,
    )
    _run_ism_and_plot_logo(
        model=model, X_input=X_alt, snp_rel=snp_rel_snpwin, ism_flank=ism_flank,
        tracks_idx=sel_tracks, track_label_short=track_label_short, ax=ax_ism_alt,
        batch_size=batch_size, allele_label=f"ALT={inst.snp_alt}", chrom=inst.chrom,
        snp_abs_1based=inst.snp_start, ref_base=inst.snp_ref, alt_base=inst.snp_alt,
    )

    title_parts = [f"{dataset_name}", f"index={inst.index}"]
    if quadrant is not None:
        title_parts.append(f"quadrant={quadrant}")
    if rank is not None:
        title_parts.append(f"rank={rank}")
    title_parts.append(f"tracks={track_label_short}")

    line1 = " | ".join(title_parts)
    line2 = (
        f"{chrom}:{fetch_start}-{fetch_end} | "
        f"SNP={chrom}:{snp_abs} ({extra['snp_ref']}→{extra['snp_alt']}); "
        f"CpG={chrom}:{extra['cpg_abs_start']}-{extra['cpg_abs_end']}"
    )
    line3 = (
        f"predΔ={pred:.4f} | obsΔ={obs:.4f}"
        + (f" | score={score:.4f}" if score is not None else "")
        + f" | margin={extra['margin']} | ISM flank=±{ism_flank}bp (REF/ALT)"
        + f" | landscape=±{landscape_half_len}bp (length {min(L, win_hi) - win_lo}bp)"
    )
    fig.suptitle("\n".join([line1, line2, line3]), fontsize=10)
    fig.tight_layout(rect=[0, 0, 1, 0.9])

    if save_path is not None:
        fig.savefig(save_path, dpi=300)
    return fig
