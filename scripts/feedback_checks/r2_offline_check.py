"""REVISION 2 self-check a (interface section 6): offline, against stored
products, no estimator run.

Reproduces, with `core.feedback.attribute_intracell` (rule 2'), the
coordinator's numbers on
Research/QIJ_joint/runs/scratch/feedback_check2/cloudfil_u_p2_N10000/
qij_eps0.01_widthlocal_massfit_nosigma/s00045.{splits,points,rounds}.parquet:
treating ALL splits up to round r as "since last pass" (no pass ever
happening, so the marker never advances), for every round r = 1..10:

  - round_trigger (unchanged, section 0 step 1) from the round's own D/floor
    against SW/Vtot read from `rounds` at round r-1 (as
    `scripts/feedback_checks/a_feedback_check.py` does);
  - if it fires, attribute_intracell(splits_so_far, bmu, fire) -- `bmu` is
    `points.bmu` throughout (the stored run never refines, so it is also the
    CURRENT bmu at every round); cumulative selection is the union of the
    cells selected at every firing round.

Parent point sets (needed for `idx_small`/`idx_large` per split, since
`splits.parquet` stores no point indices): a node's points are the union of
its final leaves' points; leaves are the node ids that never appear in
`splits.parent`; leaf k (in ascending node-id order) <-> `points.bin_label`
== k. Node ids are assigned in strictly increasing order as they are
created (`run_joint`'s own `next_id`), so processing all node ids in
DESCENDING order correctly computes every node's point set bottom-up
(child ids always exceed their parent's).

Expected (coordinator, work order "REVISION 2" section a): first selection
at round 6; cumulatively by round 10, 21 cells / 217 points, 73 of those
points in leaves (final leaves, i.e. bin_label groups) of size >= 2.

    PYTHONPATH=src OMP_NUM_THREADS=1 python3.9 scripts/feedback_checks/r2_offline_check.py
"""
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, "src")
from qij_joint.core.feedback import attribute_intracell, round_trigger  # noqa: E402

BASE = ("/Users/jtaylor/Dropbox/Research/QIJ_joint/runs/scratch/feedback_check2/"
        "cloudfil_u_p2_N10000/qij_eps0.01_widthlocal_massfit_nosigma/s00045")
EPS = 0.01
OUTPUTS = ["p2_x", "p2_y", "p2_log_reff", "p2_log_axis_ratio", "p2_pa",
           "p2_logit_w", "p2_log_contrast"]


def build_node_points(sp: pd.DataFrame, bin_label: np.ndarray):
    """Node id -> point-index array, for every node (internal and leaf),
    by bottom-up union over the final tree (module docstring)."""
    node_children = {}
    all_ids = {0}
    for _, row in sp.iterrows():
        node_children[int(row["parent"])] = (int(row["child_a"]), int(row["child_b"]))
        all_ids.add(int(row["child_a"]))
        all_ids.add(int(row["child_b"]))

    leaf_ids = sorted(all_ids - set(node_children.keys()))
    rank_of_leaf = {lid: i for i, lid in enumerate(leaf_ids)}
    points_of_leaf = {lid: np.where(bin_label == rank_of_leaf[lid])[0] for lid in leaf_ids}

    node_points = {}
    for nid in sorted(all_ids, reverse=True):
        if nid in node_children:
            a, b = node_children[nid]
            node_points[nid] = np.concatenate([node_points[a], node_points[b]])
        else:
            node_points[nid] = points_of_leaf[nid]

    leaf_size_of_leaf = {lid: points_of_leaf[lid].size for lid in leaf_ids}
    leaf_size = np.empty(bin_label.shape[0], dtype=int)
    for lid in leaf_ids:
        leaf_size[points_of_leaf[lid]] = leaf_size_of_leaf[lid]
    return node_points, leaf_size


def main() -> None:
    sp = pd.read_parquet(BASE + ".splits.parquet")
    rd = pd.read_parquet(BASE + ".rounds.parquet").sort_values("round").reset_index(drop=True)
    pt = pd.read_parquet(BASE + ".points.parquet")

    bmu = pt["bmu"].to_numpy()
    bin_label = pt["bin_label"].to_numpy()
    N = len(pt)

    node_points, leaf_size = build_node_points(sp, bin_label)
    assert node_points[0].size == N, "root's point set must be every point"

    # Build the full (ordered) split list once: child_a is always the
    # SMALL (measured) child, child_b the LARGE (conservation) child
    # (core/joint.py's own convention, `_qij_splits`'s docstring).
    splits_all = []
    for _, row in sp.iterrows():
        idx_small = node_points[int(row["child_a"])]
        idx_large = node_points[int(row["child_b"])]
        D = row[["D_" + o for o in OUTPUTS]].to_numpy(dtype=float)
        floor = row[["floor_" + o for o in OUTPUTS]].to_numpy(dtype=float)
        W_parent = row[["W_parent_" + o for o in OUTPUTS]].to_numpy(dtype=float)
        splits_all.append(dict(idx_small=idx_small, idx_large=idx_large,
                                D=D, floor=floor, W_parent=W_parent,
                                round=int(row["round"])))

    n_rounds = int(sp["round"].max())
    cumD = {o: 0.0 for o in OUTPUTS}
    first_selection_round = None
    cum_cells = set()

    for r in range(1, n_rounds + 1):
        rows = sp[sp["round"] == r]
        row0 = rd[rd["round"] == r - 1].iloc[0]
        D_round = rows[["D_" + o for o in OUTPUTS]].to_numpy()
        floor_round = rows[["floor_" + o for o in OUTPUTS]].to_numpy()
        SW = np.array([row0["V_win_" + o] / row0["kappa_" + o] for o in OUTPUTS])
        Vtot = np.array([cumD[o] + row0["V_win_" + o] for o in OUTPUTS])
        fire, ratio, excess = round_trigger(D_round, floor_round, SW, Vtot, EPS)

        if np.any(fire):
            since_last_pass = [s for s in splits_all if s["round"] <= r]
            plan, _scores = attribute_intracell(since_last_pass, bmu, fire)
            sel = {j for j, _ in plan}
            if first_selection_round is None and sel:
                first_selection_round = r
            cum_cells |= sel
            print(f"round {r:2d}: fire={fire.tolist()} n_selected={len(sel):3d} "
                  f"cumulative_cells={len(cum_cells):3d}")
        else:
            print(f"round {r:2d}: no fire")

        for o in OUTPUTS:
            cumD[o] += D_round[:, OUTPUTS.index(o)].sum()

    cum_points = set()
    for j in cum_cells:
        cum_points |= set(np.where(bmu == j)[0].tolist())
    pts_arr = np.array(sorted(cum_points))
    n_ge2 = int(np.sum(leaf_size[pts_arr] >= 2)) if pts_arr.size else 0

    print()
    print(f"first selection round: {first_selection_round}")
    print(f"cumulative selected cells (by round {n_rounds}): {len(cum_cells)}")
    print(f"cumulative points in those cells: {len(cum_points)}")
    print(f"of those, in leaves (bin_label groups) of size >= 2: {n_ge2}")

    assert first_selection_round == 6, first_selection_round
    assert len(cum_cells) == 21, len(cum_cells)
    assert len(cum_points) == 217, len(cum_points)
    assert n_ge2 == 73, n_ge2
    print("\nPASS: matches the coordinator's reported numbers "
          "(first selection round 6; 21 cells / 217 points cumulative; "
          "73 of those points in leaves >= 2).")


if __name__ == "__main__":
    main()
