# DiscoveryAgent reproduction

Code and inputs for Figures **4c–f, 5c–d and 6c–f** of the [DiscoveryAgent paper](https://arxiv.org/abs/2611.21950).
The scripts compute case-study statistics and render the corresponding panels.

| Panel | ID | Analysis |
|---|---|---|
| 4c | A6 | Validation AUROC across training epochs |
| 4d | A9 | PhotoFusion model comparison |
| 4e | A12 | Redshift probability integral transform (PIT) |
| 4f | A14 | Architecture comparison |
| 5c | E1 | Insurance burden across six states |
| 5d | E6 | Florida premiums in 2018 and 2023 |
| 6c | G5 | Network degree and autoimmune GWAS enrichment |
| 6d | G7 | Feedback and directional asymmetry |
| 6e | G9 | Network topology and genetic constraint |
| 6f | G12 | Network topology and tractability |

## Quick start

Use Python 3.12:

```sh
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements-render.txt
python tools/render_figures.py --panel A6 A9 A12 A14 --data-root inputs/astronomy --output-dir outputs/figures
python tools/render_figures.py --panel E1 --output-dir outputs/figures
python tools/render_figures.py --panel G5 G7 --data-root inputs/genomics --output-dir outputs/figures
```

These seven panels render from bundled summaries, predictions and tables.
The repository includes 25 study input files (about 127 MB). Download the original
Census data for E6 using the command below. The full genomics
`GWCD4i.DE_stats.h5ad` input is built from the public
[Virtual Cells Platform](https://virtualcellmodels.cziscience.com/dataset/genome-scale-tcell-perturb-seq)
file plus bundled v1.0 metadata, because the public file has since changed; see
[DATA_REQUIREMENTS.md](DATA_REQUIREMENTS.md#genomics).
Model training and prediction generation require separate experiment code and data.

## Run an analysis

Install the additional dependencies and download the Census inputs into an
economics workspace:

```sh
python -m pip install -r requirements-analysis.txt
python tools/download_economics.py --data-root outputs/economics
python tools/check_inputs.py --panel E1 E6 --data-root outputs/economics
python tools/run_analysis.py --panel E1 --data-root outputs/economics
python tools/run_analysis.py --panel E6 --data-root outputs/economics
python tools/render_figures.py --panel E6 --data-root outputs/economics --output-dir outputs/figures
```

The analysis writes tables to the workspace's `data/processed/`, `results/` or
`reports/analysis_outputs/`, and figures to `reports/figures/`. The runner executes
the scripts in dependency order. G7 runs two separate tests; both require the
inputs listed in the data guide. A6 and A14 read training/evaluation summaries
through `render_figures.py`.

For astronomy, copy `inputs/astronomy` and run A9 or A12 in the same way.
For A12, the analysis exports the subgroup histogram counts used by the
manuscript renderer. Render from that workspace to use the regenerated counts:

```sh
mkdir -p outputs/astronomy
cp -R inputs/astronomy/. outputs/astronomy/
python tools/run_analysis.py --panel A12 --data-root outputs/astronomy
python tools/render_figures.py --panel A12 --data-root outputs/astronomy --output-dir outputs/figures
```

The quick-start invocation using `inputs/astronomy`, or an A12 invocation
without `--data-root`, retains bundled-histogram rendering.

For genomics, copy `inputs/genomics` and create the v1.0 H5AD (16.8 GB download):

```sh
mkdir -p outputs
cp -r inputs/genomics outputs/genomics
python tools/download_genomics.py --data-root outputs/genomics
python tools/check_inputs.py --panel G5 G7 G9 G12 --data-root outputs/genomics
python tools/run_analysis.py --panel G5 --data-root outputs/genomics
python tools/run_analysis.py --panel G7 --data-root outputs/genomics
python tools/run_analysis.py --panel G9 --data-root outputs/genomics
python tools/run_analysis.py --panel G12 --data-root outputs/genomics
python tools/render_figures.py --panel G5 G7 --data-root outputs/genomics --output-dir outputs/figures
```

Do not use the public H5AD directly: its metadata changed after the paper's
analyses (see [DATA_REQUIREMENTS.md](DATA_REQUIREMENTS.md#genomics)). G9 and G12
produce their source figures through `run_analysis.py`.
The G7 analysis writes its recomputed feedback counts and perturbation metadata
to the workspace's `data/processed/` directory for manuscript rendering.

Analysis code is in `case_studies/*/analysis/`; manuscript
formatting is in `figure_rendering/`; numerical inputs are
in `case_studies/*/data/` and `inputs/<study>/`. Panel IDs retain the paper's research-round numbers.

For the economics data, use the [Census source citations](DATA_REQUIREMENTS.md#data-citations).

Code and documentation: [MIT License](LICENSE). External datasets and dependencies
retain their respective licenses.
