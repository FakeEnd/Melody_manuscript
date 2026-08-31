#!/usr/bin/env python
# -*- coding: utf-8 -*-

import argparse
import os
import numpy as np
import pandas as pd
import torch
from torch import nn
from tqdm import tqdm
import pyBigWig
from pyliftover import LiftOver
from selene_sdk.sequences import Genome


class BigWigDataset:
    def __init__(self, bigwig_paths):
        self.bws = [pyBigWig.open(p) for p in bigwig_paths]

    def get(self, chrom, start, end):
        vals = []
        for bw in self.bws:
            v = bw.values(chrom, int(start), int(end), numpy=True)
            vals.append(np.nan_to_num(v, nan=0.0))
        return np.stack(vals, axis=0)


def one_hot_encode(seq):
    table = {
        "A": [1, 0, 0, 0],
        "C": [0, 1, 0, 0],
        "G": [0, 0, 1, 0],
        "T": [0, 0, 0, 1],
    }
    return [table.get(base.upper(), [0, 0, 0, 0]) for base in seq]


class ConvBlock(nn.Module):
    def __init__(self, inp, oup, expand_ratio=2):
        super().__init__()
        hidden_dim = round(inp * expand_ratio)
        self.conv = nn.Sequential(
            nn.Conv1d(inp, hidden_dim, 9, padding=4, bias=False),
            nn.GroupNorm(1, hidden_dim),
            nn.SiLU(inplace=False),
            nn.Conv1d(hidden_dim, oup, 1, bias=False),
            nn.GroupNorm(1, oup),
        )

    def forward(self, x):
        return x + self.conv(x)


class Melody(nn.Module):
    def __init__(self, n_track=1, cpg_cls_n=7):
        super().__init__()
        self.n_track = n_track

        self.uplblocks = nn.ModuleList([
            nn.Sequential(nn.Conv1d(4, 256, 17, padding=8), nn.GroupNorm(1, 256)),
            nn.Sequential(nn.Conv1d(256, 256, 17, stride=4, padding=8), nn.GroupNorm(1, 256)),
            nn.Sequential(nn.Conv1d(256, 256, 17, stride=5, padding=8), nn.GroupNorm(1, 256)),
            nn.Sequential(nn.Conv1d(256, 256, 17, stride=5, padding=8), nn.GroupNorm(1, 256)),
            nn.Sequential(nn.Conv1d(256, 256, 17, stride=4, padding=8), nn.GroupNorm(1, 256)),
            nn.Sequential(nn.Conv1d(256, 256, 17, stride=5, padding=8), nn.GroupNorm(1, 256)),
        ])

        self.upblocks = nn.ModuleList([
            nn.Sequential(ConvBlock(256, 256), ConvBlock(256, 256))
            for _ in range(6)
        ])

        self.downlblocks = nn.ModuleList([
            nn.Sequential(nn.Upsample(scale_factor=5), nn.Conv1d(256, 256, 17, padding=8), nn.GroupNorm(1, 256)),
            nn.Sequential(nn.Upsample(scale_factor=4), nn.Conv1d(256, 256, 17, padding=8), nn.GroupNorm(1, 256)),
            nn.Sequential(nn.Upsample(scale_factor=5), nn.Conv1d(256, 256, 17, padding=8), nn.GroupNorm(1, 256)),
            nn.Sequential(nn.Upsample(scale_factor=5), nn.Conv1d(256, 256, 17, padding=8), nn.GroupNorm(1, 256)),
            nn.Sequential(nn.Upsample(scale_factor=4), nn.Conv1d(256, 256, 17, padding=8), nn.GroupNorm(1, 256)),
        ])

        self.downblocks = nn.ModuleList([
            nn.Sequential(ConvBlock(256, 256), ConvBlock(256, 256))
            for _ in range(5)
        ])

        self.final = nn.Sequential(
            nn.Conv1d(256, 256, 1),
            nn.GroupNorm(1, 256),
            nn.ReLU(inplace=True),
            nn.Conv1d(256, n_track, 1),
        )

        self.final_100_cg_count = nn.Sequential(
            nn.MaxPool1d(100, 100),
            nn.Conv1d(256, 256, 1),
            nn.GroupNorm(1, 256),
            nn.ReLU(inplace=True),
            nn.Conv1d(256, cpg_cls_n, 1),
        )

        self.final_100_methy_avg = nn.Sequential(
            nn.MaxPool1d(100, 100),
            nn.Conv1d(256, 256, 1),
            nn.GroupNorm(1, 256),
            nn.ReLU(inplace=True),
            nn.Conv1d(256, 128, 1),
            nn.GroupNorm(1, 128),
            nn.ReLU(inplace=True),
            nn.Conv1d(128, n_track, 1),
        )

    def forward(self, x):
        out = x
        encodings = []

        for lconv, conv in zip(self.uplblocks, self.upblocks):
            out = conv(lconv(out))
            encodings.append(out)

        for enc, lconv, conv in zip(reversed(encodings[:-1]), self.downlblocks, self.downblocks):
            out = conv(lconv(out))
            out = enc + out

        return self.final(out), self.final_100_cg_count(out), self.final_100_methy_avg(out)


def convert_hg19_to_hg38(data):
    lo = LiftOver("hg19", "hg38")

    def convert_row(row):
        if pd.isna(row["chr"]) or pd.isna(row["start"]) or pd.isna(row["end"]):
            return pd.Series([None, None, None])

        start_res = lo.convert_coordinate(row["chr"], int(row["start"]))
        end_res = lo.convert_coordinate(row["chr"], int(row["end"]))

        if not start_res or not end_res:
            return pd.Series([None, None, None])

        chrom, start, _, _ = start_res[0]
        _, end, _, _ = end_res[0]

        return pd.Series([chrom, start, end])

    data[["chr_hg38", "start_hg38", "end_hg38"]] = data.apply(convert_row, axis=1)
    return data


def load_region_data(excel_path, sheet_name="Table S2"):
    data = pd.read_excel(excel_path, sheet_name=sheet_name, skiprows=1, header=1)
    data = convert_hg19_to_hg38(data)
    data = data.dropna(subset=["chr_hg38", "start_hg38", "end_hg38"])
    return data


def prepare_sequences(data, genome, input_length=10000):
    rows = list(data.itertuples(index=False))

    seq_tensors = []
    pred_starts = []
    pred_ends = []
    cpg_number = []

    for r in rows:
        start_id = int(r.start_hg38)
        end_id = int(r.end_hg38)
        center = (start_id + end_id) // 2
        start_pred = center - input_length // 2

        seq = genome.get_encoding_from_coords(
            r.chr_hg38,
            start_pred,
            start_pred + input_length,
            strand="+",
            pad=True,
        )

        tensor = torch.tensor(seq, dtype=torch.float32).T.unsqueeze(0)
        seq_tensors.append(tensor)

        pred_starts.append(start_id - start_pred)
        pred_ends.append(end_id - start_pred)
        cpg_number.append(int(r.endCpG) - int(r.startCpG))

    return (
        torch.cat(seq_tensors, dim=0),
        np.array(pred_starts),
        np.array(pred_ends),
        np.array(cpg_number),
    )


def predict_in_batches(model, seqs, batch_size, device):
    preds = []

    with torch.no_grad():
        for i in range(0, len(seqs), batch_size):
            batch = seqs[i:i + batch_size].to(device)
            out = torch.sigmoid(model(batch)[0])
            preds.append(out.cpu().numpy())

    return np.concatenate(preds, axis=0)


def compute_motif_effects(
    model,
    all_seqs,
    pred_starts,
    pred_ends,
    cpg_number,
    motifs,
    archetype_clusters,
    batch_size,
    positions,
    device,
):
    motif_groups = motifs.groupby("Cluster_ID")
    results = {}
    n_samples = all_seqs.shape[0]

    for _, cluster in archetype_clusters.iterrows():
        cluster_id = cluster["Cluster_ID"]
        cluster_name = cluster["Name"]

        print(f"Processing motif cluster: {cluster_name}")

        motif_group = motif_groups.get_group(cluster_id)
        motif_consensus = motif_group["Consensus"].values[0]

        motif_onehot = np.array(one_hot_encode(motif_consensus))
        motif_len = motif_onehot.shape[0]
        motif_tensor = torch.tensor(motif_onehot, dtype=torch.float32).T.unsqueeze(0)

        out_orig = predict_in_batches(model, all_seqs, batch_size, device)

        motif_effect = []

        for ins in tqdm(positions, desc=f"{cluster_name} insertion"):
            muts = all_seqs.clone()

            for i in range(n_samples):
                if ins >= 0:
                    st = pred_ends[i] + ins + 1
                else:
                    st = pred_starts[i] - motif_len + ins - 1

                if st < 0 or st + motif_len > muts.shape[-1]:
                    continue

                muts[i, :, st:st + motif_len] = motif_tensor

            deltas = []

            with torch.no_grad():
                for i in range(0, n_samples, batch_size):
                    mb = muts[i:i + batch_size].to(device)
                    out_mut = torch.sigmoid(model(mb)[0]).cpu().numpy()

                    bs = out_mut.shape[0]
                    row_idx = np.arange(i, i + bs)

                    for j, row_id in enumerate(row_idx):
                        s = pred_starts[row_id]
                        e = pred_ends[row_id]

                        diff = out_mut[j, :, s:e] - out_orig[row_id, :, s:e]
                        deltas.append(diff.sum(axis=1) / cpg_number[row_id])

            motif_effect.append(np.stack(deltas, axis=0))

        results[cluster_name] = np.array(motif_effect)

    return results


def main(args):
    os.makedirs(args.output_dir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")

    genome = Genome(input_path=args.genome_fasta)

    data = load_region_data(args.region_excel)
    data = data.sample(args.n_samples, random_state=args.seed)

    data.to_csv(os.path.join(args.output_dir, "sampled_regions.csv"), index=False)

    motifs = pd.read_excel(args.motif_excel, sheet_name="Motifs")
    archetype_clusters = pd.read_excel(args.motif_excel, sheet_name="Archetype clusters")

    all_seqs, pred_starts, pred_ends, cpg_number = prepare_sequences(
        data,
        genome,
        input_length=args.input_length,
    )

    model = Melody(n_track=args.n_tracks)
    state_dict = torch.load(args.checkpoint, map_location="cpu")
    model.load_state_dict(state_dict)
    model = model.to(device)
    model.eval()

    if torch.cuda.device_count() > 1 and not args.cpu:
        model = nn.DataParallel(model)

    positions = np.arange(args.insert_start, args.insert_end, args.insert_step)

    results = compute_motif_effects(
        model=model,
        all_seqs=all_seqs,
        pred_starts=pred_starts,
        pred_ends=pred_ends,
        cpg_number=cpg_number,
        motifs=motifs,
        archetype_clusters=archetype_clusters,
        batch_size=args.batch_size,
        positions=positions,
        device=device,
    )

    output_path = os.path.join(args.output_dir, args.output_name)
    torch.save(results, output_path)

    print(f"Saved results to: {output_path}")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate methylation changes after motif insertion."
    )

    parser.add_argument("--genome-fasta", required=True)
    parser.add_argument("--region-excel", required=True)
    parser.add_argument("--motif-excel", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output-dir", required=True)

    parser.add_argument("--output-name", default="motif_effects.pth")
    parser.add_argument("--n-tracks", type=int, default=39)
    parser.add_argument("--n-samples", type=int, default=2500)
    parser.add_argument("--input-length", type=int, default=10000)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument("--insert-start", type=int, default=-600)
    parser.add_argument("--insert-end", type=int, default=600)
    parser.add_argument("--insert-step", type=int, default=5)

    parser.add_argument("--cpu", action="store_true")

    return parser.parse_args()

# usage example:
# python run_motif_insertion.py \
#   --genome-fasta /path/to/GRCh38.fa \
#   --region-excel /path/to/data_supplementary.xlsx \
#   --motif-excel /path/to/vierstra_2020_motif_info.xlsx \
#   --checkpoint /path/to/checkpoint.pth \
#   --output-dir ./results \
#   --batch-size 256

if __name__ == "__main__":
    main(parse_args())