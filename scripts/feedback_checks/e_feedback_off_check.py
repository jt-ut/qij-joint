"""Agent E self-check (interface section 5).

`feedback` off (the default): run.py on bead cloudfil p1 draw 1 eps 0.01 with
the study flags (--survey moments --quantized-start full-data --gptrend
quadratic --gpwidth local --fit-weights mass) must produce an s00001.parquet
whose every non-wall-time, non-busy-time column equals the committed
reference run's (new columns -- feedback, n_passes, evals_prototype_refine,
rows_prototype_refine, M_X_final -- excepted):

    Research/QIJ_joint/runs/scratch/feedback_check2/cloudfil_p1_N10000/
    qij_eps0.01_widthlocal_massfit_nosigma/s00001.parquet

Also checks: no `s00001.passes.parquet` is written under `feedback=False`,
and the method folder's name carries no `_feedback` suffix.

    PYTHONPATH=src OMP_NUM_THREADS=1 python3.9 scripts/feedback_checks/e_feedback_off_check.py
"""
import os
import subprocess
import sys
import tempfile

import numpy as np
import pandas as pd

REF = ("/Users/jtaylor/Dropbox/Research/QIJ_joint/runs/scratch/feedback_check2/"
       "cloudfil_p1_N10000/qij_eps0.01_widthlocal_massfit_nosigma/s00001.parquet")
FOLDER = "qij_eps0.01_widthlocal_massfit_nosigma"


def main() -> None:
    with tempfile.TemporaryDirectory() as out_dir:
        cmd = [
            sys.executable, "scripts/run.py", "cloudfil", "p1", "qij",
            "--N", "10000", "--draws", "1:2", "--seed", "0", "--workers", "1",
            "--out", out_dir, "--eps", "0.01", "--diag-draws", "1:2",
            "--survey", "moments", "--quantized-start", "full-data",
            "--gptrend", "quadratic", "--gpwidth", "local", "--fit-weights", "mass",
        ]
        env = dict(os.environ)
        env.setdefault("PYTHONPATH", "src")
        env.setdefault("OMP_NUM_THREADS", "1")
        subprocess.run(cmd, check=True, env=env)

        method_dir = os.path.join(out_dir, "cloudfil_p1_N10000", FOLDER)
        new_path = os.path.join(method_dir, "s00001.parquet")
        assert os.path.exists(new_path), f"missing {new_path}"

        passes_path = os.path.join(method_dir, "s00001.passes.parquet")
        assert not os.path.exists(passes_path), (
            "feedback=False must never write a passes.parquet")

        ref = pd.read_parquet(REF)
        new = pd.read_parquet(new_path)

        ref_cols, new_cols = set(ref.columns), set(new.columns)
        new_only = sorted(new_cols - ref_cols)
        missing = sorted(ref_cols - new_cols)
        expected_new = {"feedback", "n_passes", "evals_prototype_refine",
                         "rows_prototype_refine", "M_X_final"}
        assert missing == [], f"columns missing from the new run: {missing}"
        assert set(new_only) == expected_new, (
            f"unexpected new-only columns: {sorted(set(new_only) - expected_new)}; "
            f"missing expected new columns: {sorted(expected_new - set(new_only))}")

        common = sorted(ref_cols & new_cols)
        skip = [c for c in common if "wall" in c.lower() or "busy" in c.lower()]
        compare_cols = [c for c in common if c not in skip]

        mismatches = []
        for c in compare_cols:
            rv, nv = ref[c].iloc[0], new[c].iloc[0]
            if isinstance(rv, float) and isinstance(nv, float):
                ok = (rv == nv) or (np.isnan(rv) and np.isnan(nv))
            else:
                ok = rv == nv
            if not ok:
                mismatches.append((c, rv, nv))

        print(f"common non-wall/busy columns compared: {len(compare_cols)}")
        print(f"new-only columns (expected): {sorted(new_only)}")
        print(f"mismatches: {len(mismatches)}")
        for m in mismatches:
            print("  ", m)
        assert not mismatches, "feedback=False must be byte-identical to the reference"
        print("PASS: feedback=False is byte-identical to the committed reference "
              "(new columns excepted); no passes.parquet written; folder name unchanged.")


if __name__ == "__main__":
    main()
