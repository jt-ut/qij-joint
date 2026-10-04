"""Agent D self-check (1): with `feedback=False` (the default), the bead
cloudfil p1 draw 1 eps 0.01 run (study settings: survey='moments',
quantized_start='full-data', gptrend='quadratic', gpwidth='local',
fit_weights='mass') reproduces L=463, n_evals=924 and every non-wall/
busy-time column of the committed reference run exactly
(Research/QIJ_joint/runs/scratch/feedback_check2/cloudfil_p1_N10000/
qij_eps0.01_widthlocal_massfit_nosigma/s00001.parquet).

Run via scripts/run.py (one draw), then diff the two parquet files.
"""
import glob
import subprocess
import sys

import pandas as pd

REF = ('/Users/jtaylor/Dropbox/Research/QIJ_joint/runs/scratch/feedback_check2/'
       'cloudfil_p1_N10000/qij_eps0.01_widthlocal_massfit_nosigma/s00001.parquet')
OUT = 'runs/scratch/d_check1_out'


def main():
    subprocess.run([
        sys.executable, 'scripts/run.py', 'cloudfil', 'p1', 'qij',
        '--N', '10000', '--draws', '1:2', '--out', OUT, '--eps', '0.01',
        '--survey', 'moments', '--quantized-start', 'full-data',
        '--gptrend', 'quadratic', '--gpwidth', 'local', '--fit-weights', 'mass',
        '--force',
    ], check=True)

    new = pd.read_parquet(glob.glob(f'{OUT}/cloudfil_p1_N10000/*/s00001.parquet')[0])
    ref = pd.read_parquet(REF)
    print('new L, n_evals:', new['L'].iloc[0], new['n_evals'].iloc[0])
    print('ref L, n_evals:', ref['L'].iloc[0], ref['n_evals'].iloc[0])

    diffs = []
    for c in new.columns:
        if 'wall' in c.lower() or 'busy' in c.lower():
            continue
        if c not in ref.columns:
            diffs.append((c, 'missing_in_ref (agent E addition, not a regression)'))
            continue
        a, b = new[c].iloc[0], ref[c].iloc[0]
        if pd.isna(a) and pd.isna(b):
            continue
        if isinstance(a, float):
            if abs(a - b) > 1e-12:
                diffs.append((c, a, b))
        elif a != b:
            diffs.append((c, a, b))

    print(f'n diffs (excl wall/busy, excl agent-E-added columns): '
          f'{sum(1 for d in diffs if "missing_in_ref" not in str(d))}')
    for d in diffs:
        print(d)


if __name__ == '__main__':
    main()
