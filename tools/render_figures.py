#!/usr/bin/env python3
"""Render paper panels from numerical summaries or a study workspace."""
import argparse
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = {
    'A6': ('astronomy', 'plot_round6_auroc_panel_c.py'),
    'A9': ('astronomy', 'make_d10_model_comparison_bar.py'),
    'A12': ('astronomy', 'plot_round12_pit_panel_e.py'),
    'A14': ('astronomy', 'arch_scale_all_models_ranked.py'),
    'E1': ('economics', 'r1_insurance_burden_income_only.py'),
    'E6': ('economics', 'r6_F2_premium_escalation_overlaid.py'),
    'G5': ('genomics', 'plot_combined_degree_panels_tie_preserving.py'),
    'G7': ('genomics', 'plot_feedback_dai_panel_b.py'),
}

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--panel', nargs='+', choices=SCRIPTS, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--data-root', type=Path)
    args = parser.parse_args()
    a12_histogram = None
    if 'A12' in args.panel and args.data_root:
        candidate = args.data_root/'reports/analysis_outputs/round12_pit_histograms.json'
        if candidate.is_file():
            a12_histogram = candidate.resolve()
        elif args.data_root.resolve() != (ROOT/'inputs/astronomy').resolve():
            parser.error(f'A12 workspace histogram not found: {candidate}. '
                         'Run tools/run_analysis.py --panel A12 --data-root <workspace> first.')
    if set(args.panel) & {'A9', 'E6', 'G5', 'G7'}:
        if args.data_root is None or not args.data_root.is_dir():
            parser.error('These panels require --data-root pointing to a study workspace; see DATA_REQUIREMENTS.md.')
        if len({SCRIPTS[p][0] for p in args.panel}) > 1:
            parser.error('Use a separate invocation for each study workspace.')
    out = args.output_dir.resolve()
    out.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='discoveryagent-') as td:
        temp = Path(td)
        env = dict(os.environ, MPLBACKEND='Agg', MPLCONFIGDIR=str(temp/'mpl'),
                   REPRO_OUTPUT_DIR=str(out))
        for panel in dict.fromkeys(args.panel):
            study, name = SCRIPTS[panel]
            if panel == 'E1':
                data = temp/'economics/data/processed'
                data.mkdir(parents=True, exist_ok=True)
                shutil.copy2(ROOT/'case_studies/economics/data/r1_puma_income_burden.csv',
                             data/'puma_insurance_burden_6states.csv')
                env['REPRO_DATA_ROOT'] = str(temp/'economics')
            elif args.data_root:
                env['REPRO_DATA_ROOT'] = str(args.data_root.resolve())
            command = [sys.executable, str(ROOT/'case_studies'/study/'figure_rendering'/name)]
            if panel in {'A6', 'A12'}:
                command += ['--output-dir', str(out)]
            if panel == 'A12' and a12_histogram is not None:
                command += ['--histogram-json', str(a12_histogram)]
            subprocess.run(command, env=env, check=True)

if __name__ == '__main__':
    main()
