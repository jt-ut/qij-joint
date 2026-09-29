#!/usr/bin/env python3.9
"""V2 (spec/QIJ_qijdt_spec.md section 12, plus three planner additions
of 29 Sept): `cloudfil` draws 0 and 1, N = 10000, eps = 0.01,
workers 14.

Spec items: the curve against the stored `ij` V_ij,o; V_tot,o / V_ij,o;
evals by stage; n_below_closed and P_unbought,o against V_ij,o; the
per-node y_o against the exact influence's contrast for every measured
node at depth <= 5 (member sets rebuilt by the method's own bisection,
`core.tree.bisect`, on the same Z; the contrast = mean psi over A minus
mean psi over B from the `ij` product's stored psi array; no
evaluation); busy/wall at workers 14 and workers 1 for draw 0.

Planner additions (29 Sept):
  (a) draw 0, offline: the noise-inflation sum per output,
      (1/N)*sum_leaves p_l*s_l^2 (s_l = the leaf's parent row's own
      stored s), next to V_btw - V_ij.
  (b) draw 0: the ten depth<=5 nodes with the largest
      |y-y_exact|/s (max over outputs), each re-measured at delta_f/2
      and 2*delta_f (20 full-data evaluations total), reported per
      node/output alongside the stored y (at delta_f), y_exact and s.
  (c) draw 1 gets the same depth<=5 node check as (b)'s first part
      (the y vs y_exact comparison) and its V_tot/V_ij, side by side
      with draw 0.

    PYTHONPATH=src python3.9 scripts/qijdt_validate_v2.py \\
        --out /Users/jtaylor/Dropbox/Research/QIJ_joint/qijdt_validation/v2

`ij` products already exist and are read directly, not copied: draw 0
at .../qijt_validation/fp_fix/cloudfil/cloudfil_p2_N10000/ij, draw 1 at
.../qijt_validation/fp_fix/cloudfil_d1/cloudfil_p2_N10000/ij. `qijdt`
is run fresh for both draws at workers 14, and again for draw 0 alone
at workers 1 (a separate `--out`).

Every `run.py` call is wrapped in a `perl alarm` hard kill: 1 h for the
workers-14 call (2 draws), 1 h for the workers-1 draw-0 rerun (per the
task's own cap, "may take ~20+ min").
"""
import os

for _v in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS',
           'VECLIB_MAXIMUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ.setdefault(_v, '1')

import argparse
import json
import subprocess
import sys

import numpy as np
import pandas as pd

from qij_joint import products, registry
from qij_joint.core.differences import forward_step, step_parameter
from qij_joint.core.tree import bisect as tree_bisect
from qij_joint.core.tree import fixed_point, set_weights
from qij_joint.parallel import Pool, call_T
from qij_joint.qij import _wrap

_RUN_PY = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'run.py')
_SRC = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'src')

DATASET, ESTIMATOR = 'cloudfil', 'p2'
N = 10000
DRAWS = (0, 1)
SEED = 0
WORKERS = 14
EPS = 0.01
MAX_DEPTH_C = 5
N_ESCALATE = 10
TIMEOUT_W14, TIMEOUT_W1 = 3600, 3600

IJ_SOURCE = {0: '/Users/jtaylor/Dropbox/Research/QIJ_joint/qijt_validation/fp_fix/cloudfil',
             1: '/Users/jtaylor/Dropbox/Research/QIJ_joint/qijt_validation/fp_fix/cloudfil_d1'}

QIJDT_STAGES = ('full_fit', 'eta_full', 'tree', 'bias', 'within', 'quadratic', 'curvature')


def _run_cmd(cmd: list, timeout_s: int) -> None:
    """Hard-kill wrapper: `perl`'s `alarm` is pending across `exec`, so
    it reaches the real process rather than leaving a plain-timeout's
    pool workers running."""
    env = dict(os.environ, PYTHONPATH=_SRC)
    guarded = ['perl', '-e', 'alarm(shift @ARGV); exec(@ARGV) or die $!', str(timeout_s)] + cmd
    result = subprocess.run(guarded, env=env)
    if result.returncode != 0:
        raise RuntimeError(f'command failed (code {result.returncode}, timeout {timeout_s}s): '
                            f'{" ".join(cmd)}')


def _ensure_qijdt(runs_root: str, draws: tuple, workers: int, timeout_s: int,
                   force: bool = False) -> None:
    md = products.method_dir(runs_root, DATASET, ESTIMATOR, N, 'qijdt')
    if not force and all(products.is_done(md, s) for s in draws):
        return
    cmd = [sys.executable, _RUN_PY, DATASET, ESTIMATOR, 'qijdt',
           '--N', str(N), '--draws', f'{draws[0]}:{draws[-1] + 1}', '--seed', str(SEED),
           '--workers', str(workers), '--out', runs_root, '--eps', str(EPS)]
    if force:
        cmd.append('--force')
    _run_cmd(cmd, timeout_s)


def _ij_row_and_psi(s: int, outputs: list) -> tuple:
    """(row (Series), psi (N, len(outputs))) for draw `s`, read directly
    from `IJ_SOURCE[s]` (an existing product, not copied)."""
    root = IJ_SOURCE[s]
    row = products.collect(root, DATASET, ESTIMATOR, N, 'ij').set_index('s').loc[s]
    psi_df = products.collect_array(root, DATASET, ESTIMATOR, N, 'ij', 'psi', draws=[s])
    psi = np.column_stack([psi_df[f'psi_{o}'].to_numpy() for o in outputs])
    return row, psi


def _children_map(nodes_df: pd.DataFrame) -> dict:
    """{parent id: [child0, child1] ascending} from the `nodes` table's
    own `parent` column (ascending child ids match spec 3.2's "A0
    before A1" breadth-first id order)."""
    out = {}
    for parent, node in zip(nodes_df['parent'].astype(int), nodes_df['node'].astype(int)):
        if parent >= 0:
            out.setdefault(parent, []).append(node)
    return {k: sorted(v) for k, v in out.items()}


def rebuild_members(X: np.ndarray, nodes_df: pd.DataFrame) -> dict:
    """{node id: (N_ell,) point indices}, replaying qijdt's own lazy
    bisection (spec 3.1-3.2, `core.tree.bisect`, imported not
    reimplemented). Ids are assigned breadth-first, so every parent id
    is smaller than its children's; processing node ids in ascending
    order therefore always has a node's member set ready before it is
    split. qijdt's Z = X (cloudfil has no vq_transform), mass 1/N per
    point throughout (an all-point tree)."""
    N_full = X.shape[0]
    children = _children_map(nodes_df)
    measured = set(nodes_df.loc[nodes_df['node_measured'].astype(bool), 'node'].astype(int))
    members = {0: np.arange(N_full)}
    for node_id in sorted(nodes_df['node'].astype(int).tolist()):
        if node_id not in members or node_id not in measured or node_id not in children:
            continue
        idx = members[node_id]
        z = X[idx]
        m = np.full(len(idx), 1.0 / N_full)
        first = tree_bisect(z, m)
        c0, c1 = children[node_id]
        members[c0] = idx[first]
        members[c1] = idx[~first]
    return members


def verify_sizes(nodes_df: pd.DataFrame, members: dict) -> list:
    """Rows (dicts) for any node at depth <= MAX_DEPTH_C whose rebuilt
    member-set size disagrees with the stored `n_points` (should be
    empty; a mismatch would be a method bug, not a validation-script
    issue -- flagged, not fixed)."""
    bad = []
    shallow = nodes_df[nodes_df['depth'] <= MAX_DEPTH_C]
    for row in shallow.itertuples():
        node_id = int(row.node)
        if node_id not in members:
            bad.append({'node': node_id, 'reason': 'no rebuilt member set'})
            continue
        n_rebuilt = len(members[node_id])
        if n_rebuilt != int(row.n_points):
            bad.append({'node': node_id, 'n_points_stored': int(row.n_points),
                        'n_points_rebuilt': n_rebuilt})
    return bad


def node_y_exact(nodes_df: pd.DataFrame, members: dict, psi: np.ndarray, outputs: list,
                  node_ids) -> pd.DataFrame:
    """Per measured node in `node_ids`: y_exact_o = mean psi_o over A
    minus mean psi_o over B (A = the measured child, B the other;
    spec 4.1, section 12 V2), alongside the stored y_o and s_o."""
    children = _children_map(nodes_df)
    node_row_by_id = {int(r.node): r for r in nodes_df.itertuples()}
    rows = []
    for node_id in node_ids:
        r = node_row_by_id[node_id]
        kids = children.get(node_id)
        if kids is None or node_id not in members:
            continue
        A_id = kids[0] if int(r.measured_child) == 0 else kids[1]
        B_id = kids[1] if A_id == kids[0] else kids[0]
        if A_id not in members or B_id not in members:
            continue
        A, B = members[A_id], members[B_id]
        row = {'node': node_id, 'depth': int(r.depth)}
        for j, o in enumerate(outputs):
            y_exact = float(psi[A, j].mean() - psi[B, j].mean())
            row[f'y_meas_{o}'] = float(getattr(r, f'y_{o}'))
            row[f'y_exact_{o}'] = y_exact
            row[f's_{o}'] = float(getattr(r, f's_{o}'))
        rows.append(row)
    return pd.DataFrame(rows)


def reg_through_origin(x: np.ndarray, y: np.ndarray) -> tuple:
    """(slope, r2) of y ~ slope*x with no intercept: slope =
    sum(x*y)/sum(x*x); r2 = 1 - sum(resid^2)/sum(y^2) (spec section
    12's own convention)."""
    finite = np.isfinite(x) & np.isfinite(y)
    x, y = x[finite], y[finite]
    if x.size < 2 or np.sum(x * x) == 0.0:
        return float('nan'), float('nan')
    slope = float(np.sum(x * y) / np.sum(x * x))
    resid = y - slope * x
    ss_tot = np.sum(y * y)
    r2 = float(1.0 - np.sum(resid ** 2) / ss_tot) if ss_tot > 0 else float('nan')
    return slope, r2


def node_check_fit(node_df: pd.DataFrame, outputs: list) -> dict:
    """{output: (slope, r2)} of measured y against y_exact (x=y_exact,
    y=y_meas), and {depth: median ratio} pooled over every (node,
    output) pair at that depth."""
    fit = {}
    for o in outputs:
        fit[o] = reg_through_origin(node_df[f'y_exact_{o}'].to_numpy(),
                                     node_df[f'y_meas_{o}'].to_numpy())
    ratios_by_depth: dict = {}
    for row in node_df.itertuples():
        for o in outputs:
            ye, ym = getattr(row, f'y_exact_{o}'), getattr(row, f'y_meas_{o}')
            if ye == 0 or not np.isfinite(ye) or not np.isfinite(ym):
                continue
            ratios_by_depth.setdefault(int(row.depth), []).append(ym / ye)
    per_depth_median = {d: float(np.median(r)) for d, r in sorted(ratios_by_depth.items())}
    return {'slope_r2': fit, 'per_depth_median_ratio': per_depth_median}


def full_depth_diagnostic(node_df_all: pd.DataFrame, outputs: list) -> pd.DataFrame:
    """Beyond spec 12's own depth<=5 requirement (free -- no evaluation,
    only bisection and means already computed): per depth, the median
    ratio y_meas/y_exact and the median/max of max_o|y_meas-y_exact|/s,
    to localize which depths drive the measured V_btw's excess over the
    exact leaf-mean value."""
    rows = []
    for depth, g in node_df_all.groupby('depth'):
        ratios, zscores = [], []
        for row in g.itertuples():
            for o in outputs:
                ye, ym, s = (getattr(row, f'y_exact_{o}'), getattr(row, f'y_meas_{o}'),
                             getattr(row, f's_{o}'))
                if np.isfinite(ye) and ye != 0 and np.isfinite(ym):
                    ratios.append(ym / ye)
                if s > 0 and np.isfinite(ye) and np.isfinite(ym):
                    zscores.append(abs(ym - ye) / s)
        rows.append({'depth': int(depth), 'n_nodes': int(len(g)),
                      'median_ratio': float(np.median(ratios)) if ratios else float('nan'),
                      'median_z': float(np.median(zscores)) if zscores else float('nan'),
                      'max_z': float(np.max(zscores)) if zscores else float('nan')})
    return pd.DataFrame(rows).sort_values('depth')


def V_btw_exact(leaves_df: pd.DataFrame, members: dict, psi: np.ndarray, outputs: list,
                 N_full: int) -> dict:
    """V_btw,o = (1/N) sum_leaves p_l * (mean_l psi_o)^2, from the EXACT
    leaf means of the ij psi (spec section 12 V2 localization item).
    `core.tree.leaf_V_btw` (the method's own implementation, spec 8.2)
    divides by N a SECOND time beyond p_l = n_l/N itself -- this mirrors
    that exactly, so the result is on the same scale as the method's own
    stored V_btw and as V_ij."""
    out = {o: 0.0 for o in outputs}
    for row in leaves_df.itertuples():
        leaf_id = int(row.leaf)
        if leaf_id not in members:
            continue
        idx = members[leaf_id]
        p_l = len(idx) / N_full
        for j, o in enumerate(outputs):
            mean_psi = psi[idx, j].mean()
            out[o] += p_l * mean_psi ** 2
    return {o: v / N_full for o, v in out.items()}


def noise_inflation_sum(nodes_df: pd.DataFrame, leaves_df: pd.DataFrame, outputs: list,
                         N_full: int) -> dict:
    """(a) Planner addition: (1/N)*sum_leaves p_l*s_l^2, s_l = the
    leaf's PARENT row's own stored s (the split's noise, spec 4.2) --
    "the measured child's s, or the derived sibling's s, as the nodes
    table stores them for that leaf": one s per split, applied to
    whichever child terminates as that leaf. Same double-N scaling as
    `V_btw_exact` (p_l already n_l/N, plus the explicit outer 1/N), so
    it lands on the same scale as V_btw - V_ij."""
    parent_of = dict(zip(nodes_df['node'].astype(int), nodes_df['parent'].astype(int)))
    node_row_by_id = {int(r.node): r for r in nodes_df.itertuples()}
    out = {o: 0.0 for o in outputs}
    for row in leaves_df.itertuples():
        leaf_id = int(row.leaf)
        parent_id = parent_of.get(leaf_id)
        if parent_id is None or parent_id not in node_row_by_id:
            continue
        prow = node_row_by_id[parent_id]
        p_l = float(row.p)
        for o in outputs:
            s = float(getattr(prow, f's_{o}'))
            if np.isfinite(s):
                out[o] += p_l * s ** 2
    return {o: v / N_full for o, v in out.items()}


def _base_fit(T, X: np.ndarray, eta_full: float) -> np.ndarray:
    """Recomputes the full-width theta_hat exactly as qijdt's own base
    fit does (spec section 2): theta_hat = T(X, 1_N), then, when T
    takes a start, the fixed-point rule (spec 2.1, `core.tree.
    fixed_point`, imported not reimplemented) at the SAME eta_full
    already stored in the qijdt product. Deterministic (spec: nothing
    in the method is random), so this reproduces the run's own
    theta_hat bit-for-bit."""
    N_full = X.shape[0]
    theta0 = np.asarray(call_T(T, X, np.ones(N_full)), dtype=float)
    if not getattr(T, 'takes_start', False):
        return theta0
    theta, _n, _r, _ok = fixed_point(
        lambda th: call_T(T, X, np.ones(N_full), start=th, eta=eta_full), theta0, eta_full)
    return theta


def escalate_nodes(node_df: pd.DataFrame, outputs: list, n_pick: int) -> list:
    """The `n_pick` node ids with the largest max-over-outputs
    |y_meas-y_exact|/s (planner addition (b))."""
    scores = []
    for row in node_df.itertuples():
        vals = [abs(getattr(row, f'y_meas_{o}') - getattr(row, f'y_exact_{o}'))
                / getattr(row, f's_{o}') for o in outputs if getattr(row, f's_{o}') > 0]
        scores.append((max(vals) if vals else 0.0, int(row.node)))
    scores.sort(key=lambda t: -t[0])
    return [n for _, n in scores[:n_pick]]


def _remeasure_task(T, case, X, task):
    """One full-data evaluation, outside the method (planner addition
    (b)): `task` = (key, omega, start, eta)."""
    key, omega, start, eta = task
    value = np.asarray(call_T(T, X, omega, start, eta), dtype=float)
    failed = bool(np.any(np.isnan(value)))
    return key, value, failed


def escalate_remeasure(nodes_df: pd.DataFrame, members: dict, node_ids: list, X: np.ndarray,
                        theta_hat_full: np.ndarray, eta_full: float, measured_idx: list,
                        outputs: list, pool: Pool) -> dict:
    """Re-measures each of `node_ids` at delta_f/2 and 2*delta_f
    (planner addition (b), 20 evaluations total for 10 nodes): member
    set A = the node's own measured child (already rebuilt), t =
    step_parameter(delta, p_A), U_B and y by spec 4.1 using the node's
    stored U_c. Returns {node id: {'delta_half': y (q,), 'delta_2x': y}}."""
    N_full = X.shape[0]
    delta_f = forward_step(eta_full)
    node_row_by_id = {int(r.node): r for r in nodes_df.itertuples()}
    children = _children_map(nodes_df)

    tasks, meta = [], []
    for node_id in node_ids:
        r = node_row_by_id[node_id]
        kids = children[node_id]
        A_id = kids[0] if int(r.measured_child) == 0 else kids[1]
        B_id = kids[1] if A_id == kids[0] else kids[0]
        A, B = members[A_id], members[B_id]
        p_A, p_B = len(A) / N_full, len(B) / N_full
        p_c = p_A + p_B
        U_c = np.array([getattr(r, f'U_{o}') for o in outputs])
        for tag, delta in (('delta_half', delta_f / 2.0), ('delta_2x', delta_f * 2.0)):
            t = step_parameter(delta, p_A)
            omega = set_weights(np.ones(N_full), A, p_A, t)
            tasks.append(((node_id, tag), omega, theta_hat_full, eta_full))
            meta.append((node_id, tag, t, p_A, p_B, p_c, U_c))

    results = pool.map(_remeasure_task, tasks)
    out: dict = {}
    for (_key, value, failed), (node_id, tag, t, p_A, p_B, p_c, U_c) in zip(results, meta):
        if failed:
            y = np.full(len(outputs), np.nan)
        else:
            U_A = (value[measured_idx] - theta_hat_full[measured_idx]) / t
            U_B = (p_c * U_c - p_A * U_A) / p_B
            y = U_A - U_B
        out.setdefault(node_id, {})[tag] = y
    return out


def _md_table(headers: list, rows: list) -> str:
    lines = ['| ' + ' | '.join(headers) + ' |', '|' + '|'.join('---' for _ in headers) + '|']
    for r in rows:
        lines.append('| ' + ' | '.join(str(v) for v in r) + ' |')
    return '\n'.join(lines)


def draw_report(s: int, runs_root: str, X: np.ndarray, T, outputs: list, qijdt_row: pd.Series,
                 nodes_df: pd.DataFrame, leaves_df: pd.DataFrame, curve_df: pd.DataFrame) -> dict:
    ij_row, psi = _ij_row_and_psi(s, outputs)
    V_ij = {o: float(ij_row[f'V_ij_{o}']) for o in outputs}
    V_tot = {o: float(qijdt_row[f'V_tot_{o}']) for o in outputs}
    V_btw = {o: float(qijdt_row[f'V_btw_{o}']) for o in outputs}
    ratio = {o: V_tot[o] / V_ij[o] for o in outputs}

    members = rebuild_members(X, nodes_df)
    size_mismatches = verify_sizes(nodes_df, members)

    shallow_ids = nodes_df.loc[(nodes_df['node_measured'].astype(bool))
                                & (nodes_df['depth'] <= MAX_DEPTH_C), 'node'].astype(int).tolist()
    node_df = node_y_exact(nodes_df, members, psi, outputs, shallow_ids)
    check = node_check_fit(node_df, outputs)

    all_ids = nodes_df.loc[nodes_df['node_measured'].astype(bool), 'node'].astype(int).tolist()
    node_df_all = node_y_exact(nodes_df, members, psi, outputs, all_ids)
    depth_diag = full_depth_diagnostic(node_df_all, outputs)

    v_btw_exact = V_btw_exact(leaves_df, members, psi, outputs, X.shape[0])
    noise_sum = noise_inflation_sum(nodes_df, leaves_df, outputs, X.shape[0])

    evals_by_stage = {stg: int(qijdt_row[f'evals_{stg}']) for stg in QIJDT_STAGES}
    P_unbought = {o: float(qijdt_row[f'P_unbought_{o}']) for o in outputs}
    curve_s = curve_df.sort_values('round')
    curve_rows = [{'round': int(r.round), 'evals_total': int(r.evals_total), 'L': int(r.L),
                   **{f'V_btw_{o}': float(getattr(r, f'V_btw_{o}')) for o in outputs}}
                  for r in curve_s.itertuples()]

    return {
        's': s, 'V_ij': V_ij, 'V_btw': V_btw, 'V_tot': V_tot, 'V_tot_over_V_ij': ratio,
        'V_btw_exact': v_btw_exact, 'noise_inflation_sum': noise_sum,
        'evals_by_stage': evals_by_stage, 'n_below_closed': int(qijdt_row['n_below_closed']),
        'P_unbought': P_unbought, 'n_nodes_checked': len(node_df),
        'n_nodes_all': len(node_df_all),
        'size_mismatches': size_mismatches, 'node_check': check, 'curve': curve_rows,
        'depth_diagnostic': depth_diag.to_dict('records'), 'node_df': node_df,
    }


def main(argv=None) -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--out', default='/Users/jtaylor/Dropbox/Research/QIJ_joint/qijdt_validation/v2')
    p.add_argument('--runs', default=None)
    args = p.parse_args(argv)
    out_dir = args.out
    runs_root = args.runs or os.path.join(out_dir, 'runs')
    runs_root_w1 = os.path.join(out_dir, 'runs_w1')
    os.makedirs(out_dir, exist_ok=True)

    _ensure_qijdt(runs_root, DRAWS, WORKERS, TIMEOUT_W14)
    _ensure_qijdt(runs_root_w1, (0,), 1, TIMEOUT_W1)

    qijdt_df = products.collect(runs_root, DATASET, ESTIMATOR, N, 'qijdt').set_index('s')
    qijdt_w1_row = products.collect(runs_root_w1, DATASET, ESTIMATOR, N, 'qijdt') \
        .set_index('s').loc[0]
    outputs = [c[len('V_tot_'):] for c in qijdt_df.columns if c.startswith('V_tot_')]

    T_raw = registry.case(DATASET, ESTIMATOR).make_T()
    T = _wrap(T_raw)
    measured_idx = list(T.measured)

    draws_data = {}
    for s in DRAWS:
        X = registry.case(DATASET, ESTIMATOR).draw(N, SEED + s)
        nodes_df = products.collect_array(runs_root, DATASET, ESTIMATOR, N, 'qijdt',
                                           'nodes', draws=[s])
        leaves_df = products.collect_array(runs_root, DATASET, ESTIMATOR, N, 'qijdt',
                                            'leaves', draws=[s])
        curve_df = products.collect_array(runs_root, DATASET, ESTIMATOR, N, 'qijdt',
                                           'curve', draws=[s])
        draws_data[s] = draw_report(s, runs_root, X, T, outputs, qijdt_df.loc[s],
                                     nodes_df, leaves_df, curve_df)

    # planner addition (b): escalate the 10 worst depth<=5 nodes on draw 0, 20 evaluations.
    s0 = 0
    d0 = draws_data[s0]
    X0 = registry.case(DATASET, ESTIMATOR).draw(N, SEED + s0)
    nodes_df0 = products.collect_array(runs_root, DATASET, ESTIMATOR, N, 'qijdt',
                                        'nodes', draws=[s0])
    members0 = rebuild_members(X0, nodes_df0)
    node_df0 = d0['node_df']
    escalate_ids = escalate_nodes(node_df0, outputs, N_ESCALATE)
    eta_full = float(qijdt_df.loc[s0, 'eta_full'])
    theta_hat_full = _base_fit(T, X0, eta_full)
    pool = Pool(WORKERS, case=(DATASET, ESTIMATOR))
    pool.share(X0)
    escalation = escalate_remeasure(nodes_df0, members0, escalate_ids, X0, theta_hat_full,
                                     eta_full, measured_idx, outputs, pool)
    pool.close()

    escalation_rows = []
    for node_id in escalate_ids:
        row = node_df0[node_df0['node'] == node_id].iloc[0]
        for j, o in enumerate(outputs):
            escalation_rows.append({
                'node': node_id, 'output': o,
                'y_delta_half': float(escalation[node_id]['delta_half'][j]),
                'y_delta_f': float(row[f'y_meas_{o}']),
                'y_delta_2x': float(escalation[node_id]['delta_2x'][j]),
                'y_exact': float(row[f'y_exact_{o}']), 's': float(row[f's_{o}']),
            })

    report = {
        'draws': {s: {k: v for k, v in d.items() if k != 'node_df'} for s, d in draws_data.items()},
        'escalation_node_ids': escalate_ids, 'escalation': escalation_rows,
        'busy_wall': {
            'workers_14': {'busy_time': float(qijdt_df.loc[0, 'busy_time']),
                           'wall_time': float(qijdt_df.loc[0, 'wall_time'])},
            'workers_1': {'busy_time': float(qijdt_w1_row['busy_time']),
                          'wall_time': float(qijdt_w1_row['wall_time'])},
        },
    }
    with open(os.path.join(out_dir, 'v2.json'), 'w') as f:
        json.dump(report, f, indent=2, default=str)

    lines = [f'# qijdt validation V2 (cloudfil, N={N}, eps={EPS}, workers={WORKERS})', '']
    for s, d in draws_data.items():
        lines.append(f'## draw {s}')
        lines.append('')
        rows = [[o, round(d['V_ij'][o], 4), round(d['V_btw'][o], 4), round(d['V_tot'][o], 4),
                 round(d['V_tot_over_V_ij'][o], 4), round(d['V_btw_exact'][o], 4),
                 round(d['V_btw'][o] - d['V_ij'][o], 6), round(d['noise_inflation_sum'][o], 6),
                 round(d['P_unbought'][o], 6)]
                for o in outputs]
        lines.append(_md_table(['output', 'V_ij', 'V_btw', 'V_tot', 'V_tot/V_ij', 'V_btw_exact',
                                'V_btw-V_ij', 'noise_infl_sum', 'P_unbought'], rows))
        lines.append('')
        lines.append(f"n_below_closed={d['n_below_closed']}, "
                      f"nodes checked (depth<=5)={d['n_nodes_checked']}, "
                      f"size mismatches={len(d['size_mismatches'])}")
        lines.append('')
        fit_rows = [[o, round(d['node_check']['slope_r2'][o][0], 4),
                     round(d['node_check']['slope_r2'][o][1], 4)] for o in outputs]
        lines.append('Node check (depth<=5), y_meas vs y_exact, regression through origin:')
        lines.append(_md_table(['output', 'slope', 'r2'], fit_rows))
        lines.append('')
        pd_rows = [[depth, round(ratio, 4)]
                   for depth, ratio in d['node_check']['per_depth_median_ratio'].items()]
        lines.append('Per-depth median ratio y_meas/y_exact (pooled over outputs, depth<=5):')
        lines.append(_md_table(['depth', 'median_ratio'], pd_rows))
        lines.append('')
        lines.append(f"Beyond-spec diagnostic (free, no evaluation): ALL {d['n_nodes_all']} "
                      f"measured nodes, per depth, to localize the V_btw excess:")
        diag_rows = [[r['depth'], r['n_nodes'], round(r['median_ratio'], 4),
                      round(r['median_z'], 3), round(r['max_z'], 3)]
                     for r in d['depth_diagnostic']]
        lines.append(_md_table(['depth', 'n_nodes', 'median_ratio', 'median|y-y_exact|/s',
                                'max|y-y_exact|/s'], diag_rows))
        lines.append('')
        lines.append('Evals by stage:')
        lines.append(_md_table(['stage'] + list(d['evals_by_stage'].keys()),
                                [['evals'] + list(d['evals_by_stage'].values())]))
        lines.append('')
        lines.append('Curve (round, evals_total, L, V_btw per output):')
        curve_headers = ['round', 'evals_total', 'L'] + [f'V_btw_{o}' for o in outputs]
        curve_rows_fmt = [[r['round'], r['evals_total'], r['L']]
                           + [round(r[f'V_btw_{o}'], 4) for o in outputs] for r in d['curve']]
        lines.append(_md_table(curve_headers, curve_rows_fmt))
        lines.append('')

    lines.append('## planner addition (b): draw 0, ten worst depth<=5 nodes, '
                  'escalated at delta_f/2 and 2*delta_f')
    lines.append(f"Escalated node ids: {escalate_ids}")
    esc_rows = [[r['node'], r['output'], f"{r['y_delta_half']:.4g}", f"{r['y_delta_f']:.4g}",
                 f"{r['y_delta_2x']:.4g}", f"{r['y_exact']:.4g}", f"{r['s']:.4g}"]
                for r in escalation_rows]
    lines.append(_md_table(['node', 'output', 'y@delta_f/2', 'y@delta_f (stored)',
                            'y@2*delta_f', 'y_exact', 's'], esc_rows))
    lines.append('')

    lines.append('## busy/wall, draw 0')
    lines.append(_md_table(['workers', 'busy_time', 'wall_time'],
                            [[w, round(v['busy_time'], 2), round(v['wall_time'], 2)]
                             for w, v in report['busy_wall'].items()]))
    with open(os.path.join(out_dir, 'v2.md'), 'w') as f:
        f.write('\n'.join(lines))


if __name__ == '__main__':
    main()
