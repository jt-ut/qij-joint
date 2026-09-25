# Trace notes
Total tasks: 16

## Tasks and wall times
- `pareto/shape s=0`: 5.83s
- `pareto/shape s=1`: 5.86s
- `pareto/tail s=0`: 5.78s
- `pareto/tail s=1`: 5.80s
- `mvt/nu s=0`: 6.37s
- `mvt/nu s=1`: 6.41s
- `mvt/tail s=0`: 6.36s
- `mvt/tail s=1`: 6.35s
- `fp/fp s=0`: 6.67s
- `fp/fp s=1`: 6.66s
- `imf/chabrier s=0`: 51.25s
- `imf/chabrier s=1`: 60.79s
- `mvt/nu s=342 (failed: n_failed>0 & NaN V_btw)`: 2.96s
- `mvt/nu s=420 (failed: n_failed>0 & NaN V_btw)`: 2.97s
- `check`: 2.61s
- `gmm`: 11.78s

## Failing draws chosen (mvt/nu)
Read `QIJ_WSOM2026/runs/main/mvt/nu/qij.parquet` and `truth.parquet` (read-only). Only `mvt/nu` had any failing draws in the whole main study: `n_failed > 0` AND at least one NaN `V_btw_*` coincide on the SAME three draws, s = 342, 420, 523 (no separate failure kind -- no `theta_hat` NaN appears anywhere in any case's `truth.parquet`). Chose s=342 and s=420 (two of the three), both of the one failure kind available (n_failed>0 & NaN V_btw together); s=523 was left untraced since only up to two extra draws were requested.

## structsynhd
structsynhd was importable on this machine (`/Users/jtaylor/Dropbox/Software/JT_Py_Pkgs/struct-syn-hd/src`); used `ChaconMixGenerator(9).sample(2000, random_state=0)`, K=3 components (mixture 9's own `num_components`), structsynhd_used=True.

## Exceptions
None -- every task completed without raising.

## Per-file executed-line counts (union over all tasks)
- `src/qij/__init__.py`: 4 lines
- `src/qij/bootstrap.py`: 27 lines
- `src/qij/check.py`: 27 lines
- `src/qij/core/__init__.py`: 1 lines
- `src/qij/core/counter.py`: 24 lines
- `src/qij/core/differences.py`: 26 lines
- `src/qij/core/influence_model.py`: 514 lines
- `src/qij/core/intervals.py`: 6 lines
- `src/qij/core/ivq.py`: 111 lines
- `src/qij/core/outputs.py`: 9 lines
- `src/qij/core/refine.py`: 254 lines
- `src/qij/core/xvq.py`: 74 lines
- `src/qij/datasets.py`: 70 lines
- `src/qij/estimators.py`: 406 lines
- `src/qij/gmm.py`: 418 lines
- `src/qij/qij.py`: 105 lines
- `src/qij/result.py`: 48 lines
- `src/qij/study.py`: 158 lines
