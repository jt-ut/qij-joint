#!/usr/bin/env python3.9
"""V2 (spec/QIJ_qijt_spec.md section 12): `cloudfil` draws 0 and 1,
N = 10000, M_X = 1094, budget 600/200/100, workers 14 -- items (a)-(e).

    PYTHONPATH=src python3.9 scripts/qijt_validate_v2.py \\
        --out /Users/jtaylor/Dropbox/Research/QIJ_joint/qijt_validation/v2

(a) the curve against the stored `ij` V_ij,o; (b) V_tot,o / V_ij,o at
the end; (c) every node the tree measured at depth <= 5 is re-measured
on the FULL data (one evaluation per node, outside the method, via
`parallel.call_T` on a `Pool`): slope and r^2 of y^full against
a_o*y^Q (regression through the origin, matching a_o's own least-
squares definition, spec 6.2); the within-term PAIRS bought in the
SUBJECT component's cells (the `pairs` table, section 11.2a) get the
SAME contrast re-measured on the full data, per pair and SIGNED:
slope/r^2 of D^full against a_o*D_o (D_o the pair's own stored,
unscaled response). The direction is recomputed on the leaf's own
native member points -- the j-th principal axis (j >= 1) of the
leaf's covariance, or, for the quadratic contrast (j = 0, spec 7.3),
the Mahalanobis quadratic form orthogonalized against every principal
direction up to the leaf's own rank -- since a leaf that is not yet a
single point or an opened cell has no native rows on the quantized
side either. Also the distribution (median, 90th percentile, max) of
|open_shift_o| / sqrt(V_ij,o) over opened cells (the `nodes` table's
own `open_shift_o`, spec 5.3 -- no extra evaluation). The subject
component is P2 (cloudfil's on-filament source); its cells are those
whose prototype lies within 3 standard deviations (Mahalanobis
distance <= 3, using P2's own fitted covariance, reconstructed from
the measured log_reff/log_axis_ratio/pa outputs) of P2's fitted mean.
(d) root_drift, a_scatter, step_ratio, b_hat, accel, c_q (the scalar
row's own values). (e) busy/wall at workers 14 (this run) and workers
1 (a separate, forced single-draw run under `<out>/runs_w1`) for
draw 0.

Every `run.py` call is wrapped in a `perl alarm` hard kill: 10 min for
`oracle`, 20 min for `ij` (2 draws, N = 10000), 3 h for `qijt` at
workers 14 (2 draws, budget 600), 6 h for the workers-1 draw-0 rerun
(serial, no pool speedup).
"""
import os

for _v in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS',
           'VECLIB_MAXIMUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ.setdefault(_v, '1')

import argparse
import json
import subprocess
import sys
import time

import numpy as np
import pandas as pd

from qij_joint import products, registry
from qij_joint.core.differences import forward_step, perturbed_weights, step_parameter
from qij_joint.core.xvq import fit_xvq
from qij_joint.parallel import Pool, call_T

_RUN_PY = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'run.py')
_SRC = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'src')

DATASET, ESTIMATOR = 'cloudfil', 'p2'
N = 10000
M_X = 1094
DRAWS = (0, 1)
SEED = 0
WORKERS = 14
BUDGET, BUDGET_WIN, BUDGET_QUAD = 600, 200, 100
MAX_DEPTH_C = 5
TIMEOUT_ORACLE, TIMEOUT_IJ, TIMEOUT_QIJT, TIMEOUT_QIJT_W1 = 600, 1200, 10800, 21600


def _run_cmd(cmd: list, timeout_s: int) -> None:
    """Hard-kill wrapper (spec section 12; see module docstring):
    `perl`'s `alarm` is pending across `exec`, so it reaches the real
    process rather than leaving a plain-timeout's pool workers
    running."""
    env = dict(os.environ, PYTHONPATH=_SRC)
    guarded = ['perl', '-e', 'alarm(shift @ARGV); exec(@ARGV) or die $!', str(timeout_s)] + cmd
    result = subprocess.run(guarded, env=env)
    if result.returncode != 0:
        raise RuntimeError(f'command failed (code {result.returncode}, timeout {timeout_s}s): '
                            f'{" ".join(cmd)}')


def _ensure(method: str, runs_root: str, draws: tuple, workers: int,
            extra_args: list, timeout_s: int, force: bool = False) -> None:
    md = products.method_dir(runs_root, DATASET, ESTIMATOR, N, method)
    if not force and all(products.is_done(md, s) for s in draws):
        return
    cmd = [sys.executable, _RUN_PY, DATASET, ESTIMATOR, method,
           '--N', str(N), '--draws', f'{draws[0]}:{draws[-1] + 1}',
           '--seed', str(SEED), '--workers', str(workers), '--out', runs_root]
    if force:
        cmd.append('--force')
    _run_cmd(cmd + extra_args, timeout_s)


def _output_names(df: pd.DataFrame, prefix: str) -> list:
    """The output-name suffixes of every `<prefix>_<o>` column, in the
    column order the writer used (a dict-of-columns DataFrame keeps
    insertion order)."""
    lead = prefix + '_'
    return [c[len(lead):] for c in df.columns if c.startswith(lead)]


def _p2_ellipse(row: pd.Series) -> tuple:
    """(mu (2,), Sigma (2,2)) of P2's fitted component, rebuilt from the
    measured p2_x/p2_y/p2_log_reff/p2_log_axis_ratio/p2_pa outputs
    (cloudfil.py's own closed forms, inverted): det = exp(4*log_reff),
    ratio = lam2/lam1 = exp(log_axis_ratio), pa the major-axis angle."""
    mu = np.array([row['theta_hat_p2_x'], row['theta_hat_p2_y']])
    det = np.exp(4.0 * row['theta_hat_p2_log_reff'])
    ratio = np.exp(row['theta_hat_p2_log_axis_ratio'])
    lam1 = np.sqrt(det / ratio)
    lam2 = ratio * lam1
    pa = row['theta_hat_p2_pa']
    c, sn = np.cos(pa), np.sin(pa)
    R = np.array([[c, -sn], [sn, c]])
    Sigma = R @ np.diag([lam1, lam2]) @ R.T
    return mu, Sigma


def subject_mask(centers: np.ndarray, mu: np.ndarray, Sigma: np.ndarray) -> np.ndarray:
    """Mahalanobis distance <= 3 (i.e. squared distance <= 9) of each
    cell prototype `centers` from `mu` under `Sigma`, by `solve` (E3),
    never an explicit inverse: the rule for "the subject component's
    cells" (spec section 12, V2c)."""
    diffs = centers - mu
    solved = np.linalg.solve(Sigma, diffs.T).T
    maha2 = np.sum(diffs * solved, axis=1)
    return maha2 <= 9.0


def _children(nodes_df: pd.DataFrame) -> dict:
    """{parent node id: [child ids, ascending]} from the `nodes`
    product's own `parent` column."""
    out = {}
    for parent, node in zip(nodes_df['parent'].astype(int), nodes_df['node'].astype(int)):
        if parent >= 0:
            out.setdefault(parent, []).append(node)
    return {k: sorted(v) for k, v in out.items()}


def descendant_leaves(children: dict, node: int) -> list:
    """Every terminal id under `node` (no entry in `children`): §7.1's
    leaves, found by walking the `nodes` table's parent links down from
    `node` rather than any node kind/cell bookkeeping."""
    if node not in children:
        return [node]
    out = []
    for child in children[node]:
        out.extend(descendant_leaves(children, child))
    return out


def _full_task(T, case, X, task):
    """One full-data evaluation, outside the method: `task` = (key,
    omega, start, eta). This task's own failure boundary (R7)."""
    key, omega, start, eta = task
    t0 = time.perf_counter()
    try:
        value = np.asarray(call_T(T, X, omega, start, eta), dtype=float)
        failed = bool(np.any(np.isnan(value)))
    except Exception:
        value = np.full(len(T.outputs), np.nan)
        failed = True
    return key, value, failed, time.perf_counter() - t0


def full_data_node_check(nodes_df: pd.DataFrame, points_df: pd.DataFrame, X: np.ndarray,
                          T, theta_hat_full: np.ndarray, eta_full: float,
                          outputs: list, pool: Pool) -> dict:
    """{node id: y_full (q,)} for every measured node at depth <=
    MAX_DEPTH_C, by one full-data forward evaluation of its measured
    child per node (spec section 12, V2c; formula spec 4.1), processed
    depth by depth so a node's own U_full is known (from its parent's
    evaluation, root at 0) before it is used."""
    children = _children(nodes_df)
    depth = dict(zip(nodes_df['node'].astype(int), nodes_df['depth'].astype(int)))
    mass = dict(zip(nodes_df['node'].astype(int), nodes_df['p'].astype(float)))
    meas_flag = dict(zip(nodes_df['node'].astype(int), nodes_df['measured_child'].astype(int)))
    root = int(nodes_df.loc[nodes_df['depth'] == 0, 'node'].iloc[0])
    targets = sorted(
        nodes_df.loc[nodes_df['node_measured'].astype(bool) & (nodes_df['depth'] <= MAX_DEPTH_C),
                     'node'].astype(int).tolist(),
        key=lambda n: depth[n])
    outputs_idx = [T.outputs.index(o) for o in outputs]
    delta_f = forward_step(eta_full)
    N_full = X.shape[0]

    U_full = {root: np.zeros(len(outputs))}
    y_full = {}
    pool.share(X)
    for level in sorted(set(depth[n] for n in targets)):
        level_nodes = [n for n in targets if depth[n] == level]
        tasks, meta = [], {}
        for c in level_nodes:
            a0, a1 = children[c]
            a = a0 if meas_flag[c] == 0 else a1
            b = a1 if a == a0 else a0
            member = points_df.loc[points_df['leaf'].isin(descendant_leaves(children, a)),
                                    'i'].to_numpy()
            mask = np.zeros(N_full, dtype=bool)
            mask[member] = True
            p_a = mass[a]
            t_a = step_parameter(delta_f, p_a)
            omega = perturbed_weights(np.ones(N_full), mask, t_a)
            tasks.append((c, omega, theta_hat_full, eta_full))
            meta[c] = (a, b, p_a, t_a)
        for c, value, failed, _ in pool.map(_full_task, tasks):
            a, b, p_a, t_a = meta[c]
            if failed:
                continue
            U_a = (value[outputs_idx] - theta_hat_full[outputs_idx]) / t_a
            U_c = U_full[c]
            p_c, p_b = p_a + mass[b], mass[b]
            U_b = (p_c * U_c - p_a * U_a) / p_b
            y_full[c] = U_a - U_b
            U_full[a], U_full[b] = U_a, U_b
    return y_full


def _leaf_geometry(X: np.ndarray, member: np.ndarray) -> tuple:
    """(Xl, mu, vals, vecs, rank) on a leaf's own native member points,
    mass 1/N each: eigenpairs of the leaf's covariance, descending, and
    rank = #{j : vals[j] > d_x*vals[0]*eps} (spec 7.1, evaluated on the
    full data rather than the quantized rows)."""
    Xl = X[member]
    mu = Xl.mean(axis=0)
    C = np.cov(Xl.T, bias=True)
    vals, vecs = np.linalg.eigh(np.atleast_2d(C))
    order = np.argsort(-vals)
    vals, vecs = vals[order], vecs[:, order]
    rank = int(np.sum(vals > Xl.shape[1] * vals[0] * np.finfo(float).eps)) if vals.size else 0
    return Xl, mu, vals, vecs, rank


def _pair_direction(Xl: np.ndarray, mu: np.ndarray, vals: np.ndarray, vecs: np.ndarray,
                     rank: int, j: int) -> np.ndarray:
    """The pair's own direction h (mass-normalized over the leaf, spec
    7.2-7.3): the (j-1)-th principal axis for j >= 1, or, for j = 0
    (the quadratic contrast), the Mahalanobis quadratic form
    orthogonalized against every principal direction up to `rank`.
    None if the direction cannot be built on this (full-data) geometry
    (rank too small, or a degenerate leaf)."""
    if j >= 1:
        if j > rank or vals[j - 1] <= 0:
            return None
        return (Xl - mu) @ vecs[:, j - 1] / np.sqrt(vals[j - 1])
    if rank == 0:
        return None
    diffs = Xl - mu
    proj = diffs @ vecs[:, :rank]
    m = np.sum(proj ** 2 / vals[:rank], axis=1)
    h_dirs = proj / np.sqrt(vals[:rank])
    beta = np.mean(m[:, None] * h_dirs, axis=0)
    h_tilde = m - m.mean() - h_dirs @ beta
    scale = np.sqrt(np.mean(h_tilde ** 2))
    return None if scale == 0.0 else h_tilde / scale


def full_data_pair_check(pairs_df: pd.DataFrame, points_df: pd.DataFrame, X: np.ndarray,
                          T, theta_hat_full: np.ndarray, eta_full: float, outputs: list,
                          pool: Pool, in_subject: np.ndarray) -> pd.DataFrame:
    """Per bought pair (spec section 12, V2c) whose leaf has any member
    point in `in_subject`: the SAME contrast re-measured on the full
    data, signed. Returns `leaf, j, D_full_<o>, D_q_<o>` (D_q the
    pair's own stored, unscaled response)."""
    outputs_idx = [T.outputs.index(o) for o in outputs]
    delta_f = forward_step(eta_full)
    N_full = X.shape[0]
    pool.share(X)

    geometry, tasks, task_meta = {}, [], []
    for row in pairs_df.itertuples():
        leaf_id = int(row.leaf)
        member = points_df.loc[points_df['leaf'] == leaf_id, 'i'].to_numpy()
        if member.size < 2 or not in_subject[member].any():
            continue
        if leaf_id not in geometry:
            geometry[leaf_id] = (member,) + _leaf_geometry(X, member)
        member, Xl, mu, vals, vecs, rank = geometry[leaf_id]
        j = int(row.j)
        h_local = _pair_direction(Xl, mu, vals, vecs, rank, j)
        if h_local is None:
            continue
        t = delta_f / np.max(np.abs(h_local))
        omega = np.ones(N_full)
        omega[member] = 1.0 + t * h_local
        tasks.append((leaf_id, omega, theta_hat_full, eta_full))
        task_meta.append((leaf_id, j, t))

    if not tasks:
        return pd.DataFrame()
    rows = []
    for (leaf_id, j, t), (_, value, failed, _) in zip(task_meta, pool.map(_full_task, tasks)):
        if failed:
            continue
        D_full = (value[outputs_idx] - theta_hat_full[outputs_idx]) / t
        stored = pairs_df.loc[(pairs_df['leaf'] == leaf_id) & (pairs_df['j'] == j)].iloc[0]
        row = {'leaf': leaf_id, 'j': j}
        for i, o in enumerate(outputs):
            row[f'D_full_{o}'] = D_full[i]
            row[f'D_q_{o}'] = float(stored[f'D_{o}'])
        rows.append(row)
    return pd.DataFrame(rows)


def open_shift_stats(nodes_df: pd.DataFrame, V_ij: dict, outputs: list) -> dict:
    """Per output: median, 90th percentile and max of |open_shift_o| /
    sqrt(V_ij,o) over opened cells (spec 5.3's `open_shift_o`, non-NaN
    exactly there)."""
    stats = {}
    for o in outputs:
        vals = nodes_df[f'open_shift_{o}'].dropna().to_numpy()
        scaled = np.abs(vals) / np.sqrt(V_ij[o])
        if scaled.size == 0:
            stats[o] = {'median': float('nan'), 'p90': float('nan'), 'max': float('nan')}
        else:
            stats[o] = {'median': float(np.median(scaled)),
                         'p90': float(np.percentile(scaled, 90)), 'max': float(np.max(scaled))}
    return stats


def reg_through_origin(x: np.ndarray, y: np.ndarray) -> tuple:
    """(slope, r2) of y ~ slope*x with no intercept: slope =
    sum(x*y)/sum(x*x); r2 = 1 - sum(resid^2)/sum(y^2), the no-intercept
    convention (SS_tot taken about 0, not about the mean), matching
    a_o's own least-squares definition (spec 6.2)."""
    finite = np.isfinite(x) & np.isfinite(y)
    x, y = x[finite], y[finite]
    if x.size < 2 or np.sum(x * x) == 0.0:
        return float('nan'), float('nan')
    slope = float(np.sum(x * y) / np.sum(x * x))
    resid = y - slope * x
    ss_tot = np.sum(y * y)
    r2 = float(1.0 - np.sum(resid ** 2) / ss_tot) if ss_tot > 0 else float('nan')
    return slope, r2


def _md_table(headers: list, rows: list) -> str:
    lines = ['| ' + ' | '.join(headers) + ' |', '|' + '|'.join('---' for _ in headers) + '|']
    for r in rows:
        lines.append('| ' + ' | '.join(str(v) for v in r) + ' |')
    return '\n'.join(lines)


def _draw_report(s: int, runs_root: str, T, pool: Pool, full_outputs: list, measured: list,
                  oracle_df: pd.DataFrame, ij_df: pd.DataFrame, qijt_df: pd.DataFrame,
                  curve_df: pd.DataFrame, anchors_df: pd.DataFrame) -> dict:
    """Items (a)-(d) for one draw."""
    row_qijt, row_ij, row_oracle = qijt_df.loc[s], ij_df.loc[s], oracle_df.loc[s]
    theta_hat_full = np.array([row_oracle[f'theta_hat_{o}'] for o in full_outputs])
    eta_full = float(row_qijt['eta_full'])

    curve_s = curve_df[curve_df['s'] == s].sort_values('round')
    a_curve = {o: curve_s[f'V_btw_{o}'].tolist() for o in measured}
    V_ij = {o: float(row_ij[f'V_ij_{o}']) for o in measured}
    ratio = {o: float(row_qijt[f'V_tot_{o}']) / V_ij[o] for o in measured}

    X = registry.case(DATASET, ESTIMATOR).draw(N, SEED + s)
    nodes_df = products.collect_array(runs_root, DATASET, ESTIMATOR, N, 'qijt', 'nodes', draws=[s])
    pairs_df = products.collect_array(runs_root, DATASET, ESTIMATOR, N, 'qijt', 'pairs', draws=[s])
    points_df = products.collect_array(runs_root, DATASET, ESTIMATOR, N, 'qijt', 'points', draws=[s])

    mu_p2, Sigma_p2 = _p2_ellipse(row_qijt)
    xvq = fit_xvq(X, M_X, SEED + s, WORKERS)
    in_subject = subject_mask(xvq.centers, mu_p2, Sigma_p2)[xvq.bmu]

    y_full = full_data_node_check(nodes_df, points_df, X, T, theta_hat_full, eta_full,
                                   measured, pool)
    node_rows = [
        {'node': node_id, 'output': o, 'y_full': y_f[i],
         'a_yQ': float(row_qijt[f'a_scale_{o}'])
         * float(nodes_df.loc[nodes_df['node'] == node_id, f'y_{o}'].iloc[0])}
        for node_id, y_f in y_full.items() for i, o in enumerate(measured)
    ]
    node_df = pd.DataFrame(node_rows)
    node_fit = {o: reg_through_origin(node_df.loc[node_df['output'] == o, 'a_yQ'].to_numpy(),
                                       node_df.loc[node_df['output'] == o, 'y_full'].to_numpy())
                for o in measured}

    pair_full = full_data_pair_check(pairs_df, points_df, X, T, theta_hat_full,
                                      eta_full, measured, pool, in_subject)
    pair_fit = {o: (reg_through_origin(
                        float(row_qijt[f'a_scale_{o}']) * pair_full[f'D_q_{o}'].to_numpy(),
                        pair_full[f'D_full_{o}'].to_numpy())
                     if not pair_full.empty else (float('nan'), float('nan')))
                for o in measured}

    anchors_s = anchors_df[anchors_df['s'] == s]
    step_ratio_stats = {o: {'mean': float(anchors_s[f'step_ratio_{o}'].mean()),
                             'min': float(anchors_s[f'step_ratio_{o}'].min()),
                             'max': float(anchors_s[f'step_ratio_{o}'].max())}
                         for o in measured}
    open_shift = open_shift_stats(nodes_df, V_ij, measured)

    return {
        'a_curve': a_curve, 'V_ij': V_ij, 'V_tot_over_V_ij': ratio,
        'node_slope_r2': node_fit, 'n_nodes_checked': len(y_full),
        'pair_slope_r2': pair_fit, 'n_pairs_checked': int(len(pair_full)),
        'open_shift': open_shift,
        'root_drift': {o: float(row_qijt[f'root_drift_{o}']) for o in measured},
        'a_scatter': {o: float(row_qijt[f'a_scatter_{o}']) for o in measured},
        'b_hat': {o: float(row_qijt[f'b_hat_{o}']) for o in measured},
        'accel': {o: float(row_qijt[f'accel_{o}']) for o in measured},
        'c_q': {o: float(row_qijt[f'c_q_{o}']) for o in measured},
        'step_ratio': step_ratio_stats,
    }


def write_report(report: dict, measured: list, out_dir: str) -> None:
    with open(os.path.join(out_dir, 'v2.json'), 'w') as f:
        json.dump(report, f, indent=2)

    lines = [f'# qijt validation V2 (cloudfil, N={N}, M_X={M_X}, '
             f'budget {BUDGET}/{BUDGET_WIN}/{BUDGET_QUAD}, workers {WORKERS})', '']
    for s, d in report['draws'].items():
        lines.append(f'## draw {s}')
        lines.append('')
        rows = [[o, round(d['V_ij'][o], 4), round(d['V_tot_over_V_ij'][o], 4),
                 round(d['node_slope_r2'][o][0], 4), round(d['node_slope_r2'][o][1], 4),
                 round(d['pair_slope_r2'][o][0], 4), round(d['pair_slope_r2'][o][1], 4),
                 round(d['root_drift'][o], 4), round(d['a_scatter'][o], 4),
                 round(d['b_hat'][o], 6), round(d['accel'][o], 6), round(d['c_q'][o], 6)]
                for o in measured]
        lines.append(_md_table(['output', 'V_ij', 'V_tot/V_ij', 'node slope', 'node r2',
                                 'pair slope', 'pair r2', 'root_drift', 'a_scatter',
                                 'b_hat', 'accel', 'c_q'], rows))
        lines.append('')
        lines.append(f"{d['n_nodes_checked']} nodes and {d['n_pairs_checked']} pairs checked "
                      f"against the full data.")
        lines.append('')
        shift_rows = [[o, round(d['open_shift'][o]['median'], 4),
                       round(d['open_shift'][o]['p90'], 4), round(d['open_shift'][o]['max'], 4)]
                      for o in measured]
        lines.append('|open_shift_o| / sqrt(V_ij,o) over opened cells:')
        lines.append(_md_table(['output', 'median', 'p90', 'max'], shift_rows))
        lines.append('')
    lines.append('## busy/wall')
    lines.append(_md_table(['workers', 'busy_time', 'wall_time'],
                            [[w, round(v['busy_time'], 2), round(v['wall_time'], 2)]
                             for w, v in report['busy_wall'].items()]))
    with open(os.path.join(out_dir, 'v2.md'), 'w') as f:
        f.write('\n'.join(lines))


def main(argv=None) -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--out', default='/Users/jtaylor/Dropbox/Research/QIJ_joint/qijt_validation/v2')
    p.add_argument('--runs', default=None)
    args = p.parse_args(argv)
    out_dir = args.out
    runs_root = args.runs or os.path.join(out_dir, 'runs')
    runs_root_w1 = os.path.join(out_dir, 'runs_w1')
    os.makedirs(out_dir, exist_ok=True)

    diag = ['--diag-draws', f'{DRAWS[0]}:{DRAWS[-1] + 1}']
    budget_args = ['--M-X', str(M_X), '--budget', str(BUDGET), '--budget-win', str(BUDGET_WIN),
                   '--budget-quad', str(BUDGET_QUAD)]
    _ensure('oracle', runs_root, DRAWS, WORKERS, [], TIMEOUT_ORACLE)
    _ensure('ij', runs_root, DRAWS, WORKERS, [], TIMEOUT_IJ)
    _ensure('qijt', runs_root, DRAWS, WORKERS, budget_args + diag, TIMEOUT_QIJT)
    _ensure('qijt', runs_root_w1, (0,), 1, budget_args + ['--diag-draws', '0:1'],
            TIMEOUT_QIJT_W1, force=True)

    oracle_df = products.collect(runs_root, DATASET, ESTIMATOR, N, 'oracle').set_index('s')
    ij_df = products.collect(runs_root, DATASET, ESTIMATOR, N, 'ij').set_index('s')
    qijt_df = products.collect(runs_root, DATASET, ESTIMATOR, N, 'qijt').set_index('s')
    qijt_w1_row = products.collect(runs_root_w1, DATASET, ESTIMATOR, N, 'qijt').set_index('s').loc[0]
    curve_df = products.collect_array(runs_root, DATASET, ESTIMATOR, N, 'qijt', 'curve',
                                       draws=list(DRAWS))
    anchors_df = products.collect_array(runs_root, DATASET, ESTIMATOR, N, 'qijt', 'anchors',
                                         draws=list(DRAWS))

    full_outputs = _output_names(ij_df, 'V_ij')
    measured = _output_names(qijt_df, 'V_btw')
    T = registry.case(DATASET, ESTIMATOR).make_T()
    pool = Pool(WORKERS, case=(DATASET, ESTIMATOR))

    report = {'draws': {
        s: _draw_report(s, runs_root, T, pool, full_outputs, measured, oracle_df, ij_df,
                         qijt_df, curve_df, anchors_df)
        for s in DRAWS
    }}
    pool.close()
    report['busy_wall'] = {
        'workers_14': {'busy_time': float(qijt_df.loc[0, 'busy_time']),
                       'wall_time': float(qijt_df.loc[0, 'wall_time'])},
        'workers_1': {'busy_time': float(qijt_w1_row['busy_time']),
                      'wall_time': float(qijt_w1_row['wall_time'])},
    }
    write_report(report, measured, out_dir)


if __name__ == '__main__':
    main()
