# Pilot and floor options — build interface (4 October 2026)

Two OPT-IN options for the unified loop. Defaults must stay byte-identical to main 63c2b0d
(every product column, every number). Author approved both "as options".

## Option 1: `pilot` — `'gp'` (default, today) | `'affine'`

The pilot fills the loop's starting state vector psi0_all (N, q) from the survey's prototype
influences I_proto. `'gp'` = today's `influence_model.fit_influence_model` + `psi0`. `'affine'` =
the removed affine pilot (commit 6c8fb74, `src/qij_joint/core/affine_pilot.py`; see
`git show 6c8fb74:src/qij_joint/core/affine_pilot.py`), ported WITHOUT its bridge table / bridge
pricing (the unified loop has no adjacency splits):

    per cell j (X-VQ prototype), per output c: mu_j = mean of the cell's points in Z (bmu
    membership); neighbours N(j) = cells adjacent to j in xvq.conn (either direction);
    gradient g_j = weighted LS of I_k - I_j ~ g_j^T (mu_k - mu_j), k in N(j), weight p_k
    (g_j = 0 if fewer than d neighbours or singular at rel tol 1e-12); pilot at point i in
    cell j: psi0(z_i) = I_j + g_j^T (z_i - mu_j). The cell's own point-average equals I_j.

Module `core/affine_pilot.py` (Agent A) exposes:

    fit_affine_pilot(Z, xvq, I_proto, theta_Q) -> AffinePilot
    affine_psi0(pilot: AffinePilot, Z) -> ndarray (N, q)

`AffinePilot` must carry the attributes the loop and qij.fit read from the GP model:
`constant_path` (q,) bool and `offset` (q,) float, with the SAME definitions as
`influence_model.InfluenceModel` (read how fit_influence_model sets them; e.g. constant path when a
coordinate's prototype influences are constant; offset = whatever psi0's centring offset is there),
plus `gradients` (M, q, d) and `n_flat_cells` (int: cells with g_j = 0) for products. Cost must be
O(N d + M d^2) — no M x M work. Mass-centred I_proto is the input as stored.

## Option 2: `gp_floor` — `'isotropic'` (default, today) | `'mass'`

Today the declared-noise floor lam_floor,c (`_lambda_floor`) is solved on the UNSCALED projected
kernel eigenbasis with one n_c2 = 2 eta^2 theta_Q[c]^2 median_j(1/t_j^2) — i.e. every prototype
equally noisy — even under fit_weights='mass', costing a second eigh per width candidate.

`'mass'`: the floor is solved in the SAME eigenbasis the mass-weighted fit uses (Lambda_t, V_t of
the rescaled K~ = D^-1/2 K D^-1/2, z = V_t^T ytilde_proj[c]), with the declared noise referred to
that rescaled problem:

    n_c2_t = 2 eta^2 theta_Q[c]^2 * median_j( mtilde_j / t_j^2 ),   mtilde_j = p_j / mean(p_j)

(the per-prototype declared noise 2 eta^2 theta^2 / t_j^2, times mtilde_j because the rescaled
problem's noise is lam*s^2 homoscedastic where the original's is lam*s^2/mtilde_j). Under
fit_weights='mass' + gp_floor='mass' the unscaled A/eigh per candidate is NOT computed. Under
fit_weights='none', mtilde_j = 1 and V_t = V, so 'mass' equals 'isotropic' exactly — allow it.
Owner: Agent B, `core/influence_model.py` only: `fit_influence_model(..., gp_floor='isotropic')`,
threaded into `_width_grid_candidate`; anything else that recomputes the floor (e.g. the final
per-coordinate solve) uses the same rule.

## Plumbing (Agent C)

`QIJ(..., pilot='gp', gp_floor='isotropic')` in qij.py; pipeline.run_qij; scripts/run.py
`--pilot {gp,affine}` and `--gp-floor {isotropic,mass}`; config.json / method-folder naming via the
existing products config mechanism (non-default settings appended to the folder name, defaults
leave today's names unchanged); QIJResult/product rows gain `pilot` and `gp_floor` columns ONLY if
adding columns keeps default products otherwise identical (if the products writer compares column
sets, add them with defaults — report what you did). Under pilot='affine': no GP is fitted;
psi0_all = affine_psi0(...); sigma_all = NaN (N, q); every GP-model-derived product column
(width, lam, c, h, at_bound, ...) = NaN/False; the loop is called exactly as today with the
AffinePilot in place of `model`. Import names exactly as above (A and B build in parallel).

## Rules for all agents
- Repo /Users/jtaylor/Dropbox/Software/JT_Py_Pkgs/qij_joint, branch main. Do NOT commit, push,
  branch, or `git add -A`. Leave untracked spec files alone. python3.9, `PYTHONPATH=src`.
- Default behaviour byte-identical: verify on ONE draw (fp all, N 2000, draw 0, seed 0, eps
  0.03162277660168379, flags `--survey moments --quantized-start full-data --gptrend quadratic
  --gpwidth local --fit-weights mass`) against a run of unmodified code where your piece allows.
- No new tunable constants beyond those named here; if you need one, STOP and report.
- Do not run studies, oracle, boot or IJ.
