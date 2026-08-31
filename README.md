# Melody — manuscript figure reproduction

This repository reproduces the quantitative figure panels of the published Melody article,
[Decoding the sequence determinants of locus-specific DNA methylation across human tissues](https://doi.org/10.1038/s41467-026-76744-5),
directly from source-data CSV files. The CSVs mirror the Source Data workbook published by
Nature Communications (worksheet title/preamble rows are omitted so each CSV remains a flat table).

## Layout

```
Data/
  Figure2/   figure_2AB.csv ... figure_2J.csv
  Figure3/   figure_3C.csv  ... figure_3I.csv
  Figure4/   figure_4B.csv  ... figure_4E.csv
  Figure5/   figure_5A.csv  ... figure_5H.csv
  Supplementary/  figure_S2.csv ... figure_S7C.csv
Reproduce_figure/
  Figure2.ipynb  Figure3.ipynb  Figure3_AB_ism.ipynb  Figure4.ipynb  Figure5.ipynb  Supplementary.ipynb
  ism_plotting.py   (ISM plotting helper imported by Figure3_AB_ism.ipynb)
Reproduce_data/
  figure2_benchmark.py  figure2_perchr.py  figure3_meqtl.py
  figure5_afgh.py  figure5_b.py  figure5_de.py
  Figure4/   insert_motif.py  region_table.xlsx
```

Each notebook reads only the CSVs in `Data/<figure>/` — no model checkpoints, genome, or
server access required. Every panel has a markdown cell describing what it shows and the data
it uses, followed by a self-contained plotting cell.

`Reproduce_data/` goes one step further back and shows **how the source-data CSVs are produced
from the Melody model and the raw methylation / meQTL data** (these scripts need a GPU machine
with the model, a checkpoint, the genome, and the methylation tracks — see
`Reproduce_data/README.md`). They are not required to reproduce the figures.

## Environment

```bash
pip install numpy pandas matplotlib seaborn jupyter
```

Then open any notebook in `Reproduce_figure/` and run all cells, or execute headless:

```bash
jupyter nbconvert --to notebook --execute --inplace Reproduce_figure/Figure2.ipynb
```

## Archived release

The citable archive of this repository is available at
[Zenodo DOI 10.5281/zenodo.21386856](https://doi.org/10.5281/zenodo.21386856).
