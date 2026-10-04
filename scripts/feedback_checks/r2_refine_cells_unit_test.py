"""REVISION 2 self-check b (interface section 6): synthetic unit tests for
`core.xvq_refine.refine_cells`'s NEW signature, `refine_cells(Z, xvq, plan,
seed)` with `plan` = list of (j, Q_j, k_j) (rule 3').

Three toy cells, one `refine_cells` call:

  - cell 0 ("normal"): three well-separated blobs (k_j=3) PLUS two
    singleton points sitting right on top of blob 0 but NOT included in
    Q_0 (as rule 3' leaves a leaf-of-1 out of the k-means fit) -- checks
    that they are still reassigned to the nearest of the 3 new sub-centers
    (not left behind at the old single center, not dropped).
  - cell 1 ("untouched"): not in `plan` at all -- checks its bmu/center are
    byte-identical after the call.
  - cell 2 ("forced failure"): Q_2 has only two DISTINCT positions but
    k_j=3 is requested anyway (a precondition `_FeedbackRefiner` would
    normally enforce via k_j <= #distinct rows -- violated here on purpose,
    at the `refine_cells` unit level, to exercise its own empty-sub-cell
    defence) -- checks cell 2 is reported skipped with NO mutation to its
    bmu/center.

Also checks the id-appending rule: new ids are appended in ascending j,
then sub-cell order (here, only cell 0 is actually refined, so its two new
ids should be exactly M_old, M_old+1, in that order for sub-cells 1, 2).

    PYTHONPATH=src OMP_NUM_THREADS=1 python3.9 scripts/feedback_checks/r2_refine_cells_unit_test.py
"""
import sys

import numpy as np
from scipy.sparse import csr_matrix

sys.path.insert(0, "src")
from qij_joint.core.xvq import XVQ  # noqa: E402
from qij_joint.core.xvq_refine import refine_cells  # noqa: E402

SEED = 0


def build_toy():
    rng = np.random.default_rng(SEED)

    # Cell 0: three blobs (Q_0, well separated along x) + two singleton
    # points sitting on blob 0's location but excluded from Q_0.
    blob0 = rng.normal(loc=[-10.0, 0.0], scale=0.1, size=(8, 2))
    blob1 = rng.normal(loc=[0.0, 0.0], scale=0.1, size=(8, 2))
    blob2 = rng.normal(loc=[10.0, 0.0], scale=0.1, size=(8, 2))
    Q0_points = np.vstack([blob0, blob1, blob2])
    singles0 = np.array([[-10.05, 0.02], [-9.95, -0.03]])  # on top of blob0

    # Cell 1: untouched, its own well-separated blob.
    cell1_points = rng.normal(loc=[0.0, 50.0], scale=0.1, size=(10, 2))

    # Cell 2: forced failure -- only 2 distinct positions (one repeated),
    # k_j=3 requested anyway.
    cell2_points = np.array([[100.0, 100.0], [100.0, 100.0], [100.5, 100.0]])

    Z = np.vstack([Q0_points, singles0, cell1_points, cell2_points])
    N = Z.shape[0]

    n0 = Q0_points.shape[0]
    n0s = singles0.shape[0]
    n1 = cell1_points.shape[0]
    n2 = cell2_points.shape[0]

    idx_Q0 = np.arange(0, n0)
    idx_singles0 = np.arange(n0, n0 + n0s)
    idx_cell0_all = np.concatenate([idx_Q0, idx_singles0])
    idx_cell1 = np.arange(n0 + n0s, n0 + n0s + n1)
    idx_cell2 = np.arange(n0 + n0s + n1, N)

    bmu = np.zeros(N, dtype=np.intp)
    bmu[idx_cell0_all] = 0
    bmu[idx_cell1] = 1
    bmu[idx_cell2] = 2

    M_old = 3
    centers = np.zeros((M_old, 2))
    centers[0] = Z[idx_cell0_all].mean(axis=0)
    centers[1] = Z[idx_cell1].mean(axis=0)
    centers[2] = Z[idx_cell2].mean(axis=0)

    # bmu2/conn: not exercised by this test beyond being present and the
    # right shape (refine_cells recomputes both unconditionally).
    bmu2 = np.roll(bmu, 1)
    data = np.ones(N, dtype=int)
    conn = csr_matrix((data, (bmu, bmu2)), shape=(M_old, M_old))

    p = np.bincount(bmu, minlength=M_old).astype(float) / N
    xvq = XVQ(centers=centers, labels=bmu.copy(), p=p, bmu=bmu, bmu2=bmu2,
              conn=conn, M_requested=M_old, M_used=M_old)
    return Z, xvq, idx_Q0, idx_singles0, idx_cell0_all, idx_cell1, idx_cell2


def main() -> None:
    Z, xvq, idx_Q0, idx_singles0, idx_cell0_all, idx_cell1, idx_cell2 = build_toy()
    M_old = xvq.M_used
    bmu_old = xvq.bmu.copy()
    centers_old = xvq.centers.copy()

    # plan: cell 0 with k_0=3 (Q_0 only, singletons excluded); cell 2 with
    # k_2=3 requested over only 2 distinct positions (forced failure). Cell
    # 1 is not in the plan at all (untouched).
    plan = [(0, idx_Q0, 3), (2, np.arange(idx_cell2.size) + idx_cell2[0], 3)]

    xvq_new, refined, skipped = refine_cells(Z, xvq, plan, SEED)

    print(f"M_used: {M_old} -> {xvq_new.M_used}")
    print(f"refined: {refined}")
    print(f"skipped: {skipped}")

    # (1) cell 2 ("forced failure"): skipped, no mutation.
    assert 2 in skipped, skipped
    assert not any(j == 2 for j, _ in refined), refined
    assert np.array_equal(xvq_new.bmu[idx_cell2], bmu_old[idx_cell2]), \
        "cell 2's points must keep their old bmu (skipped, no mutation)"
    assert np.allclose(xvq_new.centers[2], centers_old[2]), \
        "cell 2's center must be unchanged (skipped, no mutation)"
    print("PASS: forced-failure cell (k_j > #distinct positions) is "
          "skipped with no mutation to its bmu/center.")

    # (2) cell 1 ("untouched"): not in `plan` at all -- bmu/center identical.
    assert np.array_equal(xvq_new.bmu[idx_cell1], bmu_old[idx_cell1]), \
        "untouched cell's points must keep their old bmu"
    assert np.allclose(xvq_new.centers[1], centers_old[1]), \
        "untouched cell's center must be unchanged"
    print("PASS: untouched cell's bmu/center are unchanged.")

    # (3) cell 0: refined into k_0=3 sub-cells; ids appended ascending,
    # sub-cell order -- since cell 0 is the only one actually refined, the
    # two new ids must be exactly M_old, M_old+1 (sub-cells 1, 2).
    pair0 = [ids for j, ids in refined if j == 0]
    assert len(pair0) == 1, refined
    ids0 = pair0[0]
    assert ids0[0] == 0, ids0  # sub-cell 0 keeps id j
    assert ids0[1:] == [M_old, M_old + 1], ids0  # appended ascending
    assert xvq_new.M_used == M_old + 2, xvq_new.M_used
    print(f"PASS: cell 0 refined into ids {ids0}, ids appended correctly "
          f"({M_old}, {M_old + 1}).")

    # Three DISTINCT sub-bmu values now live among cell 0's original points
    # (Q_0's three blobs), each blob entirely homogeneous in its new bmu
    # (the blobs are far enough apart that k-means cannot have split one).
    new_bmu0 = xvq_new.bmu[idx_cell0_all]
    assert set(np.unique(new_bmu0).tolist()) == set(ids0), (np.unique(new_bmu0), ids0)
    n_blob = idx_Q0.size // 3
    blob_labels = [new_bmu0[i * n_blob:(i + 1) * n_blob] for i in range(3)]
    for b in blob_labels:
        assert np.unique(b).size == 1, "each blob must map to exactly one sub-cell"
    assert len({int(b[0]) for b in blob_labels}) == 3, "the three blobs must map to 3 distinct sub-cells"
    print("PASS: cell 0's three blobs each map to one, and all three distinct, sub-cells.")

    # (4) the two singleton points (excluded from Q_0's own k-means fit)
    # ARE reassigned -- to one of the 3 new sub-cells, not left at a stale
    # id and not dropped -- and specifically to the sub-cell nearest them
    # (they sit on top of blob 0).
    singles_new_bmu = xvq_new.bmu[idx_singles0]
    assert set(singles_new_bmu.tolist()) <= set(ids0), singles_new_bmu
    blob0_sub_id = int(blob_labels[0][0])
    assert np.all(singles_new_bmu == blob0_sub_id), (
        f"singleton points (on top of blob 0) must land on blob 0's own "
        f"sub-cell {blob0_sub_id}, got {singles_new_bmu}")
    print(f"PASS: the two singleton points (excluded from the k-means fit) "
          f"are reassigned to the nearest sub-cell ({blob0_sub_id}), matching blob 0.")

    print("\nALL PASS")


if __name__ == "__main__":
    main()
