#!/usr/bin/env python3
"""Run a case-study analysis in its data workspace."""
import argparse
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
ANALYSES = {
    'A9': ('astronomy', ['final_deliverables_part2.py']),
    'A12': ('astronomy', ['v3_binned_pit_analysis.py']),
    'E1': ('economics', ['r1_puma_income_burden.py']),
    'E6': ('economics', ['r6_florida_premium_distribution.py']),
    'G5': ('genomics', ['round5/gene_topology_analysis.py', 'round5/topology_gwas_enrichment.py', 'round5/degree_decile_gwas_analysis.py']),
    'G7': ('genomics', ['round7/logistic_regression_dai_gwas.py', 'round7/feedback_dai_chain_test.py']),
    'G9': ('genomics', ['round9/compute_network_constraint_metrics.py', 'round9/constraint_network_analyses.py', 'round9/analysis_10_summary_figure.py']),
    'G12': ('genomics', ['round12/gene_topology_classification.py', 'round12/topology_druggability_analysis.py']),
}

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--panel', choices=ANALYSES, required=True)
    parser.add_argument('--data-root', type=Path, required=True)
    args = parser.parse_args()
    base = args.data_root.resolve()
    if not base.is_dir():
        parser.error('--data-root must be an existing study workspace; see DATA_REQUIREMENTS.md.')
    for directory in ['reports/figures', 'reports/analysis_outputs', 'data/processed', 'results']:
        (base/directory).mkdir(parents=True, exist_ok=True)
    study, scripts = ANALYSES[args.panel]
    env = dict(os.environ, REPRO_DATA_ROOT=str(base), MPLBACKEND='Agg')
    for script in scripts:
        subprocess.run([sys.executable, str(ROOT/'case_studies'/study/'analysis'/script)],
                       cwd=base, env=env, check=True)

if __name__ == '__main__':
    main()
