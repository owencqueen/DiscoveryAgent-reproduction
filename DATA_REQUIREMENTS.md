# Data

The repository includes 25 input files under `inputs/<study>/`, totaling about
127 MB. `INPUTS.json` records whether each input is bundled, its size and SHA-256
hash. Its relative paths are resolved from the study directory supplied as
`--data-root` (for example, `inputs/astronomy`).

| Study | Bundled inputs | Size |
|---|---|---|
| Astronomy | All 11 prediction NPZs | 8.2 MB |
| Economics | 839-PUMA summary table; raw ACS files downloaded below | 375 MB external |
| Genomics | All 13 listed annotation and intermediate tables; v1.0 metadata for the H5AD | 119 MB |

The seven raw economics CSVs and `data/GWCD4i.DE_stats.h5ad` are external.
The public H5AD has changed since the paper's analyses. Create it with
`tools/download_genomics.py` as described under [Genomics](#genomics); do not
use the file downloaded directly from the dataset page.
G5 and G7 figure rendering uses bundled tables and does not require the H5AD.
No model weights are needed for the included analyses of saved predictions and tables.

The ACS files are Census public-use microdata; genomics annotations include
Open Targets, GeneBayes and gnomAD data. Third-party data retain their respective
licenses and attribution requirements; the code's MIT license does not relicense
them.

## Astronomy

A6 uses the bundled `prior_experiment_histories.json` (four AUROC curves, 25 epochs).
A14 uses the three bundled architecture result JSONs and the two reference model
rows in its script. A12 can render from the bundled PIT histogram JSON or recompute
PIT statistics from `reports/analysis_outputs/v3_binned_val_predictions.npz`.
That NPZ must contain `bin_probs`, `bin_centers`, `z_true`, `mag_r`, `mag_g` and
`is_dwarf`. A9 requires the ten prediction NPZs listed in `INPUTS.json`.

The A12 analysis also writes `reports/analysis_outputs/round12_pit_histograms.json`
in its workspace, using the same PIT values, subgroup masks and 20 histogram bins.
`render_figures.py --panel A12 --data-root <workspace>` reads this generated file.
An unprocessed workspace must run the A12 analysis first; bundled rendering
remains available without `--data-root` or with the bundled `inputs/astronomy`.

## Economics

The analyses use U.S. Census Bureau **ACS 1-Year PUMS housing unit records**:
six states in 2023 for E1, and Florida in 2018 and 2023 for E6. Download the
original Census ZIPs below and extract the named CSV into the indicated directory
relative to your economics workspace.

| Vintage | State | Original archive | Extracted file | Directory |
|---|---|---|---|---|
| 2023 | California | [csv_hca.zip](https://www2.census.gov/programs-surveys/acs/data/pums/2023/1-Year/csv_hca.zip) | `psam_h06.csv` | `data/downloads/pums_housing_2023/` |
| 2023 | Florida | [csv_hfl.zip](https://www2.census.gov/programs-surveys/acs/data/pums/2023/1-Year/csv_hfl.zip) | `psam_h12.csv` | `data/downloads/pums_housing_2023/` |
| 2023 | Louisiana | [csv_hla.zip](https://www2.census.gov/programs-surveys/acs/data/pums/2023/1-Year/csv_hla.zip) | `psam_h22.csv` | `data/downloads/pums_housing_2023/` |
| 2023 | Minnesota | [csv_hmn.zip](https://www2.census.gov/programs-surveys/acs/data/pums/2023/1-Year/csv_hmn.zip) | `psam_h27.csv` | `data/downloads/pums_housing_2023/` |
| 2023 | Ohio | [csv_hoh.zip](https://www2.census.gov/programs-surveys/acs/data/pums/2023/1-Year/csv_hoh.zip) | `psam_h39.csv` | `data/downloads/pums_housing_2023/` |
| 2023 | Texas | [csv_htx.zip](https://www2.census.gov/programs-surveys/acs/data/pums/2023/1-Year/csv_htx.zip) | `psam_h48.csv` | `data/downloads/pums_housing_2023/` |
| 2018 | Florida | [csv_hfl.zip](https://www2.census.gov/programs-surveys/acs/data/pums/2018/1-Year/csv_hfl.zip) | `psam_h12.csv` | `data/downloads/pums_housing_2018/` |

From the repository root, download and verify all seven CSVs with Python 3.12:

```sh
python tools/download_economics.py --data-root outputs/economics
python tools/check_inputs.py --panel E1 E6 --data-root outputs/economics
```

The downloader uses only the Python standard library, extracts the specified CSV
from each ZIP, and checks its SHA-256 against `INPUTS.json`. Use `--panel E6` to
fetch only the two Florida files. Downloads total about 100 MB compressed and
375 MB extracted. E1's bundled aggregate contains 839 PUMAs and can be plotted
without downloading the raw data.

### Data citations

- U.S. Census Bureau. “Housing Unit Records: California, Florida, Louisiana,
  Minnesota, Ohio, and Texas.” *American Community Survey 1-Year Public Use
  Microdata Sample (PUMS)*, 2023. [2023 source directory](https://www2.census.gov/programs-surveys/acs/data/pums/2023/1-Year/),
  individual files linked above. Accessed February 27, 2026.
- U.S. Census Bureau. “Housing Unit Records: Florida (`csv_hfl.zip`).”
  *American Community Survey 1-Year Public Use Microdata Sample (PUMS)*, 2018.
  [2018 Florida source archive](https://www2.census.gov/programs-surveys/acs/data/pums/2018/1-Year/csv_hfl.zip).
  Accessed February 27, 2026.

For figures and derived tables: “Source: Authors' calculations using U.S. Census
Bureau, 2018 and/or 2023 ACS 1-Year PUMS housing unit records.” Use the vintage(s)
actually analyzed. The Census Bureau supplied the underlying data; the authors
are responsible for the calculations and conclusions.

These references follow the Census Bureau's
[citation guidance for downloaded data files](https://www.census.gov/about/policies/citation.html):
identify the agency, product and file, vintage, source URL, and access date.

## Genomics

Recomputing the full analyses requires `data/GWCD4i.DE_stats.h5ad` and the annotation snapshots listed
in `INPUTS.json`.

### Download the CD4+ T cell differential expression results

The analyses used **v1.0** of the **Differential Expression Results**
(`GWCD4i.DE_stats.h5ad`) from the
[Primary Human CD4+ T Cell Perturb-seq dataset](https://virtualcellmodels.cziscience.com/dataset/genome-scale-tcell-perturb-seq),
released on December 22, 2025. On May 28, 2026, the public file was replaced
with an updated version at the same address; the dataset page still lists v1.0.
The replacement has the same perturbations, measured genes and
differential expression layers, with bit-identical values, but it
changes the metadata the analyses depend on:

- `.obs` no longer has `offtarget_flag` or `ontarget_effect_category`, which the
  G5, G7, G9 and G12 quality filters use. A direct download therefore fails with
  `KeyError: 'offtarget_flag'`. The new `neighboring_gene_KD` and
  `distal_offtarget_flag` columns are not equivalent: substituting them changes
  the filtered perturbation set (18,763 in v1.0).
- `ontarget_significant` differs for 7 perturbation–condition pairs, so `n_downstream` and the
  `varm` regulator counts differ slightly.
- Rows are in a different order, and 51 observation names were changed.

The v1.0 file is no longer publicly downloadable, so the repository bundles its
metadata (`.obs`, `.var` and `.varm`, in v1.0 row order) as
`inputs/genomics/data/GWCD4i.DE_stats.v1.0_metadata.h5ad` (7 MB).
`tools/download_genomics.py` downloads the public file (16.8 GB), reorders its
layers to match v1.0, attaches the v1.0 metadata and verifies the result
against a content hash of the original file:

```sh
mkdir -p outputs
cp -r inputs/genomics outputs/genomics
python tools/download_genomics.py --data-root outputs/genomics
python tools/check_inputs.py --panel G5 G7 G9 G12 --data-root outputs/genomics
```

The result is written to `outputs/genomics/data/GWCD4i.DE_stats.h5ad`. The
download resumes if interrupted; rerun the same command. If you already have
the public file, pass it with `--public-h5ad PATH` to skip the download. The
tool needs about 34 GB of free disk space while it runs (17 GB after it removes
the public copy; keep it with `--keep-public`) and about 6 GB of memory. Reading
the H5AD requires `anndata` 0.12 or later.

The restored file matches the original in content, but not byte for byte,
because HDF5 layout depends on the writing library. `INPUTS.json` therefore
records a `content_sha256` over the layers, `.obs`, `.var` and `.varm`, which
`check_inputs.py` checks. If the public file changes again, the tool stops
rather than writing data that differs from v1.0.

The data are from Zhu, Dann, Yan et al. (2026), "Genome-scale perturb-seq in
primary human CD4+ T cells maps context-specific regulators of T cell programs
and human immune traits," *Cell*,
[doi:10.1016/j.cell.2026.08.002](https://doi.org/10.1016/j.cell.2026.08.002),
released under the MIT License (Copyright (c) 2025 Emma Dann; see
`inputs/genomics/data/GWCD4i.DE_stats.v1.0_metadata.LICENSE`).

### Panels

Use the specified snapshots: G5 uses nine autoimmune diseases
and 1,307 labeled genes; G7 uses eight diseases and 1,212 labeled genes.

| ID | Inputs and dependency order |
|---|---|
| G5 | H5AD + `autoimmune_l2g_predictions.csv`; topology → enrichment → degree-bin analysis. The runner generates `reports/gene_topology.csv` and `data/processed/gene_topology_with_gwas.csv`. |
| G7 | H5AD + `autoimmune_l2g_predictions_score_gt_0.5.csv` + bundled `data/processed/stim8hr_feedback_topology_gwas.csv` for the analysis's reference comparison. The analysis writes recomputed feedback counts and `data/processed/perturbation_summary.csv` for the manuscript renderer. |
| G9 | H5AD + GeneBayes s_het + gnomAD + `autoimmune_l2g_unique_genes.csv`; network metrics → constraint analysis → summary figure. The runner generates `merged_constraint_network.csv` and `stim8hr_constraint_annotated.csv`. |
| G12 | H5AD + `results/opentargets_druggability.csv`; topology classification → tractability analysis. |

The intermediate tables are also bundled, so figure rendering can use their
recorded values directly. Full recomputation regenerates them in the working
copy. The genomics input check reports the missing H5AD until it is supplied. Gene sets, topology
cutoffs and comparison groups follow the respective paper rounds. G5's
manuscript out-degree bins keep tied degrees together.
