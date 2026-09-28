# qijt build: shared state and interfaces (coordinator, for the parallel build)

This is not a spec. `QIJ_qijt_spec.md` defines the method. This file fixes the names every agent codes against, so seven agents can build at once. Where this file and the spec disagree, the spec wins and the agent reports it.

## Files during the build, joined at integration

- `core/tree_h.py` (agent H), `core/tree_g.py` (G), `core/tree_n.py` (N), `core/tree_w.py` (W) are concatenated by the coordinator, in that order, into `core/tree.py` (≤ 700 lines in total, so aim at ≤ 175 each).
- Inside the build, the later parts import the earlier ones as `from .tree_h import ...`. The coordinator rewrites those imports when joining.
- `qijt.py` (agent Q) imports `from .core.tree import ...`; during the build it imports from the part files.
- `pipeline.run_qijt` and `scripts/run.py` belong to agent P; the validation scripts `scripts/qijt_validate_*.py` to agent V.

## The state: `TreeState` (a dataclass in tree_h.py; every other part reads and mutates it)

Data, fixed per draw:
- `X` (N, dx): native coordinates; `Z` (N, d): quantizer coordinates.
- `bmu` (N,) int: each point's cell; `centers_x` (M, dx): prototypes in native coordinates (`inverse(xvq.centers)`); `centers_z` (M, d).
- `N`, `M`, `q_full` ints; `measured` (q,) int array: the absolute output indices.
- `cell_order` (N,) int and `cell_start` (M+1,) int: the points of cell j are `cell_order[cell_start[j]:cell_start[j+1]]`, ascending point index.

Rows, preallocated to `row_cap = M + N`; a removed row stays in the table as inactive:
- `row_x` (row_cap, dx), `row_z` (row_cap, d), `omega0` (row_cap,), `row_cell` (row_cap,) int, `row_point` (row_cap,) int (−1 for a prototype row), `row_active` (row_cap,) bool, `n_rows` int (rows ever allocated).
- `cell_row` (M,) int: the row id of cell j's prototype row, or −1 once cell j is opened.
- The EVALUATION SET is the active rows in ascending row id (`active_rows(state)`). Every weight vector handed to T is over that compact order.

Nodes, preallocated to `node_cap` (the coordinator passes `M` + `N` + 1):
- `node_parent`, `node_child0`, `node_child1` (−1 for none), `node_depth`, `node_kind` (0 = cells, 1 = points), `node_cell` (−1 above the cell level), `node_lo`, `node_hi` (a member range into `cell_perm` (M,) for kind 0, or into `point_perm` (N,) for kind 1), `n_nodes`.
- Measurement (filled by G, N, W): `measured` bool, `node_round` int (−1 unmeasured), `meas_child` (0 or 1, the child of larger mass per §3.5), `node_t`, `node_status` (object array of `fit_status` strings), `U` (node_cap, q_full) (NaN until known; the root's is 0), `y`, `s`, `E` (node_cap, q) unscaled, `passes` (node_cap, q) bool.
- Base quantities: `theta_hat` (q_full,), `theta_Q` (q_full,) (current), `eta_full`, `eta_Q`, `delta_f`, `z_n` = 5.0 (declared once, here).

## H: `tree_h.py`, the hierarchy and the rows (spec 2.1, 3, 5.3 rows part)

- `TreeState` (above).
- `build_state(X, Z, bmu, centers_x, centers_z, measured, q_full, node_cap) -> TreeState`: the initial rows (one per cell, ω0 = n_j) and the complete cell-level tree (§3.2), with breadth-first ids and root 0.
- `bisect(z, m) -> first (n,) bool`: §3.1 exactly (the members in the first child); degenerate handling per §3.4 is for the caller.
- `is_degenerate(state, c) -> bool`: every member at one position (§3.4). `is_point_leaf(state, c) -> bool`.
- `node_mass(state, c) -> float`: p of the node (cells: Σ p_j; points: count / N).
- `measured_child(state, c) -> (A, B)`: §3.5.
- `node_rows(state, c) -> (k,) int`: the active row ids of node c's members, ascending (a cell node's prototype row, or its native points' rows once opened).
- `open_cells(state, cells) -> None`: §5.3 for the rows and the within-cell tree (§3.3, new ids continue breadth-first per opening, the cell node keeps its id). Does NOT re-evaluate θ_Q (that is G's job).
- `active_rows(state) -> (row_ids (R,), rows_x (R, dx), omega0 (R,))`.

## G: `tree_g.py`, measurement and growth (spec 2.3, 4, 5, 9.2)

- `set_weights(omega0, member_pos, p_K, t) -> ω (R,)`: §2.3 (member_pos = positions in the compact order).
- `run_batch(T, counter, pool, shared, tasks) -> list[(key, value (q_full,), failed, status, wall)]`: §9.2's contract, consumed by position. It calls `pool.share(shared)` when given a pool (skipping when unchanged is the caller's choice), runs `_eval_task` per task (key, ω, start, eta) through `pool.map`, or through the counter in the same order when `pool is None`, and calls `counter.add` per task in the parent. Every stage uses it. Status per task by `parallel.fit_status`.
- `measure_nodes(state, T, counter, pool, nodes, round_) -> None`: §4.1–4.2 for a batch on the current rows (one evaluation per node along its measured child).
- `reevaluate_theta_Q(state, T, counter) -> bool`: after openings (§5.3); False means θ_Q is NaN and the draw fails with 'opening_failed'.
- `grow(state, T, counter, pool, budget) -> (curve_rows: list[dict], tree_status, n_rounds, evals_tree)`: §5.2, one batch per round, openings before the batch. Stage attribution is the caller's (Q snapshots the counter around each stage); G returns the count of `opening` evaluations so Q can split them out.

## N: `tree_n.py`, anchors, bias and drift (spec 6)

- `anchors(state, T, counter, pool) -> dict(a (q,), scatter (q,), table: list[dict], b_hat (q,), evals_anchor_Q int)`: §6.1–6.2 and the bias of §6.4 (the table rows per §11.5). Full-data batches share X; the quantized batch shares the rows.
- `drift(state, T, counter, pool) -> root_drift (q,)`: §6.3, one quantized evaluation of the root on the final rows (the root's initial y is `state.y[0]`).

## W: `tree_w.py`, within-term and reconstruction (spec 7, 8.1)

- `leaves(state) -> leaf ids (ascending)`: §7.1's definition.
- `within(state, T, counter, pool, budget_win, budget_quad, V_btw_unscaled (q,)) -> dict(pairs: list[dict], quads: list[dict], opened: list[int], leaf_rows: list[dict])`: §7.1–7.3, including the openings bought pairs require (it calls H's `open_cells` and G's `reevaluate_theta_Q`, the latter's evaluation counted in `opening` by the caller's snapshot: W returns how many it made). pairs rows: leaf, j, D (q_full,), W (q,) unscaled, t, status. quads rows: leaf, Q (q_full,), contribution (q,) unscaled.
- `reconstruct(state, pairs, quads, a (q,)) -> (psi_rows (R, q_full), psi_points (N, q), leaf_of_point (N,))`: §8.1, rows in compact order, unmeasured outputs 0 in psi_rows.

## Q: `qijt.py` (spec 1.1, 8.2–8.4, 9.1, 10, 11.1 as attributes)

- `QIJT(M_X, budget, budget_win, budget_quad, seed=0, vq_transform=None).fit(X, T, pool=None) -> QIJTResult`.
- `QIJTResult`: every scalar of §11.1 as an attribute ((q,) arrays per output), the tables `nodes`, `leaves`, `curve`, `anchors` as DataFrames, `psi_hat` (N, q), `leaf_of_point` (N,), and `interval(level)`, `interval_btw(level)`, `abc_interval(level)`, `variance`.
- Stage order and per-stage `evals_by_stage`/`rows_by_stage`/`wall_time_by_stage` keyed by §9.1's names (Q snapshots the counter around each stage); the failures of §10.

## P: pipeline and run.py (spec 1.2, 11)

- `pipeline.run_qijt(dataset, estimator, N, draws, seed, out_dir, M_X, budget, budget_win, budget_quad, diag_draws, force, workers)`, plus writers for §11.1–11.6 through `products.write_draw` (arrays: `nodes`, `leaves`, `curve`, `anchors`, and `points` for diag draws). Also the search audit on θ̂, as the other methods have it (`audit.search_audit`).
- `scripts/run.py`: method `qijt`, with `--budget`, `--budget-win`, `--budget-quad` and `--M-X` required for it.
