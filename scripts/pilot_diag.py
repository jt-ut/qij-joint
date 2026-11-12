"""Pilot diagnostic (no method change; one draw each): is the quantized pilot's error the quantizer (within-cell structure
erased) or the quantized refit (basin, step, convergence, weight total)? Plain script, no pytest.

The method no longer uses quantized evaluations (its pilot is M full-data evaluations). This script is kept as the
admissibility test for a user who wants to know whether quantized evaluations are legitimate for their statistic: item 3
compares the quantized evaluation with the full-data evaluation on a few cells, item 4 does the whole comparison for fp and mvt
(fp: admissible; the shearmix subject and mvt's nu: not).

Shearmix draw 0, N = 10^4, ('cloudfil_u', 'p2'), the bed's 1086 cells, exact psi = the bed's parquet for that draw.
  1. base fits: theta_q (the fit on the M weighted cell means) against theta-hat; the full-data objective at both; the Newton
     finish from theta_q on the full data; convergence of the quantized fit; the step and its normalisation.
  2. the error structure over all cells.
  3. step sensitivity on 10 cells (half, double, central) and, as the reference without quantization, the same forward
     difference on the FULL data for those cells; plus the whole pilot with the rows' weights summing to N instead of M.
  4. controls: ('fp', 'all') and ('mvt', 'all') at N = 10^4 draw 0, exact influence T.influence(X, ones), M = 1086 cells.
Run: PYTHONPATH=src python3.9 scripts/pilot_diag.py   (output: scripts/pilot_diag.log)"""
import os
for _v in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS'):
    os.environ[_v] = '1'
import time, warnings
warnings.filterwarnings('ignore')
import numpy as np
import pandas as pd

from qij_joint import registry
from qij_joint.core import pilot as pl
from qij_joint.core.counter import Counter
from qij_joint.core.differences import forward_step, step_parameter, evaluation_weights
from qij_joint.core.xvq import prototype_influences, _field_step_weights, _measure_eta_Q

CASE = '/Users/jtaylor/Dropbox/Software/JT_Py_Pkgs/vqboot/fable/bed/cache/shearmix_s00000_M1086/case.npz'
PSI = '/Users/jtaylor/Dropbox/Research/QIJ_joint/runs/shearmix4/cloudfil_u_p2_N10000/ij/s00000.psi.parquet'
OUTS = ['p2_x', 'p2_y', 'p2_log_reff', 'p2_log_axis_ratio', 'p2_pa', 'p2_logit_w', 'p2_log_contrast']
SHORT = ['x', 'y', 'lreff', 'laxr', 'pa', 'lgtw', 'lcon']
T0 = time.time()


def native_cell_means(X, pil):
    """The quantized data: each cell's mean in the statistic's own coordinates, (C, d)."""
    X = np.asarray(X, float); flat = X.ndim == 1; X2 = X[:, None] if flat else X
    W = np.stack([np.bincount(pil.inv, X2[:, a], pil.C) for a in range(X2.shape[1])], 1) / pil.ncell[:, None]
    return W[:, 0] if flat else W


def f(v, fm='%.3g'):
    return '[' + ' '.join(fm % x for x in np.asarray(v, float)) + ']'


def cell_stats(P, inv, C):
    n = np.bincount(inv, minlength=C).astype(float)
    m = np.stack([np.bincount(inv, P[:, j], C) for j in range(P.shape[1])], 1) / n[:, None]
    m2 = np.stack([np.bincount(inv, P[:, j] ** 2, C) for j in range(P.shape[1])], 1) / n[:, None]
    return n, m, np.maximum(m2 - m ** 2, 0.0)


def summary(name, q, ex, n, V, outs):
    p = n / n.sum(); d = q - ex
    print('%s  per output %s' % (name, ' '.join('%8s' % o for o in outs)))
    print('  sum_c p_c d_c^2 / V         ' + ' '.join('%8.3f' % v for v in (p[:, None] * d ** 2).sum(0) / V))
    print('  sum_c p_c exact^2 / V       ' + ' '.join('%8.3f' % v for v in (p[:, None] * ex ** 2).sum(0) / V))
    print('  median |d_c| / |exact_c|    ' + ' '.join('%8.3f' % v for v in np.median(np.abs(d) / np.abs(ex), 0)))
    print('  corr(quantized, exact)      ' + ' '.join('%8.3f' % np.corrcoef(q[:, j], ex[:, j])[0, 1] for j in range(len(outs))))


def shearmix():
    c = registry.case('cloudfil_u', 'p2'); T = c.make_T(); mix = T._mix
    X = np.asarray(c.draw(10000, 0), float); N = len(X)
    z = np.load(CASE); inv = z['inv'].astype(int)
    P = pd.read_parquet(PSI)[['psi_' + o for o in OUTS]].to_numpy(float); V = (P ** 2).mean(0)
    meas = list(T.measured)
    pil = pl.build_cells(X, cells=inv); C = pil.C; n = pil.ncell; p = n / N
    rows = native_cell_means(X, pil); w0 = C * p
    cnt = Counter(T, N)
    t = time.time(); th = cnt(X, np.ones(N)); i_full = mix.last_fit_info; t_th = time.time() - t

    print('=' * 100); print('1. BASE FITS (shearmix draw 0, N = 10^4, M = %d cells)' % C); print('=' * 100)
    t = time.time(); thq = cnt(rows, w0, start=th); i_q = mix.last_fit_info; t_q = time.time() - t
    t = time.time(); thq_cold = cnt(rows, w0); i_qc = mix.last_fit_info; t_qc = time.time() - t
    thqN = cnt(rows, N * p, start=th); i_qN = mix.last_fit_info
    back = cnt(X, np.ones(N), start=thq); i_b = mix.last_fit_info
    ll = lambda th_: T.loglik(X, np.ones(N), th_)
    names = list(T.outputs)
    print('%-22s %12s %12s %12s %12s' % ('parameter', 'theta-hat', 'theta_q', 'theta_q cold', 'theta_q W=N'))
    for i, nm in enumerate(names):
        print('%-22s %12.5g %12.5g %12.5g %12.5g' % (nm, th[i], thq[i], thq_cold[i], thqN[i]))
    print('full-data objective (penalized ll per unit weight): theta-hat %.8f  theta_q %.8f  theta_q cold %.8f  theta_q(W=N) %.8f'
          % (ll(th), ll(thq), ll(thq_cold), ll(thqN)))
    print('derived outputs, (theta_q - theta-hat) / sqrt(V/N) (the sampling sd):  ' + f((thq - th)[meas] / np.sqrt(V / N), '%.2f'))
    print('                 (theta_q(W=N) - theta-hat) / sqrt(V/N):               ' + f((thqN - th)[meas] / np.sqrt(V / N), '%.2f'))
    print('Newton finish from theta_q on the full data: max |theta - theta-hat| = %.2e over all 37 outputs (status %s)'
          % (float(np.nanmax(np.abs(back - th))), i_b.status))
    for nm, i_, tt in (('theta-hat (cold, full data)', i_full, t_th), ('theta_q (from theta-hat)', i_q, t_q),
                       ('theta_q (cold, multistart)', i_qc, t_qc), ('theta_q (W = N)', i_qN, 0)):
        print('  %-28s status %-10s score %.2e  EM iters %4d  Newton iters %3d  %s' % (nm, i_.status, i_.score, i_.n_iter_em, i_.n_iter_newton,
                                                                                 ('%.2f s' % tt) if tt else ''))
    eQ, _ = _measure_eta_Q(cnt, rows, w0, np.arange(C), p, thq); dlt = forward_step(eQ)
    jt = int(np.argsort(n)[C // 2]); tj = step_parameter(dlt, p[jt])
    wj = _field_step_weights(w0, np.arange(C), jt, p[jt], tj)
    print('step: eta_Q %.2e -> delta = 2 sqrt(eta_Q) = %.2e; for cell %d (n %d, p %.2e): t = delta p/(1-p) = %.2e; the cell\'s '
          'weight goes %.6g -> %.6g (x(1+delta)), every other row x(1-t); weights sum to %.6g (= M) before and %.6g after'
          % (eQ, dlt, jt, n[jt], p[jt], tj, w0[jt], wj[jt], w0.sum(), wj.sum()))
    print('normalisation: (T(t) - theta_Q) / t is the derivative along e_c/p_c - 1 = (cell mass relative change), which is the '
          'cell MEAN influence m_c (not n_c m_c): d/dt T = sum_i (dw_i/dt) dT/dw_i = (1/n_c) sum_{i in c} psi_i - mean psi')

    print('\n' + '=' * 100); print('2. ERROR STRUCTURE, all %d cells' % C); print('=' * 100)
    t = time.time()
    _, I, _, sv = prototype_influences(rows, cnt, p, T.eta, quantized_start='full-data', theta_hat=th)
    t_pil = time.time() - t
    q = I[:, meas]; nn, ex, var = cell_stats(P, inv, C); d = q - ex
    print('pilot: %d refits on the rows in %.1f s (%.1f ms each); step doubling ratios (expect 2) on 5 cells: %s'
          % (C + 14, t_pil, 1000 * t_pil / (C + 14), f(np.nanmedian(sv.step_ratio[:, meas], 0), '%.3f')))
    centre = th[meas[:2]]; reff = np.exp(th[meas[2]])
    dist = np.sqrt(((rows - centre) ** 2).sum(1)) / reff
    summary('shearmix, rows weighted to M', q, ex, n, V, SHORT)
    print('  corr(d_c^2, Var_c(psi))     ' + ' '.join('%8.3f' % np.corrcoef(d[:, j] ** 2, var[:, j])[0, 1] for j in range(7)))
    print('  corr(|d_c|, 1/n_c)          ' + ' '.join('%8.3f' % np.corrcoef(np.abs(d[:, j]), 1 / n)[0, 1] for j in range(7)))
    sh = []
    for j in range(7):
        e = p * d[:, j] ** 2; o = np.argsort(-e)[:20]; sh.append(e[o].sum() / e.sum())
    print('  share of sum p d^2, 20 worst' + ' '.join('%8.3f' % v for v in sh))
    big = (np.abs(d) / np.sqrt(V)).max(1); o = np.argsort(-big)
    print('  distance from the subject centre (units of its r_eff = %.3g): all cells median %.1f; 20 worst cells median %.1f, '
          '%d of the 20 within 3 r_eff (%d of all %d cells are)' % (reff, np.median(dist), np.median(dist[o[:20]]),
                                                                    int((dist[o[:20]] < 3).sum()), int((dist < 3).sum()), C))
    print('\n10 worst cells (by max_j |d_cj| / sqrt V_j); exact, quantized, sd_c = sqrt Var_c(psi), all in units of sqrt V; '
          'outputs x, lreff, laxr, lcon')
    print('%5s %4s %6s | %-27s | %-27s | %-27s' % ('cell', 'n', 'dist', 'exact', 'quantized', 'within-cell sd'))
    js = [0, 2, 3, 6]; sV = np.sqrt(V)
    for cc in o[:10]:
        print('%5d %4d %6.2f | %s | %s | %s' % (cc, n[cc], dist[cc], f(ex[cc, js] / sV[js], '%6.2f'), f(q[cc, js] / sV[js], '%6.2f'),
                                               f(np.sqrt(var[cc, js]) / sV[js], '%6.2f')))

    print('\n' + '=' * 100); print('3. STEP SENSITIVITY on 10 cells spanning the error range (values in units of sqrt V)'); print('=' * 100)
    pick = o[np.linspace(0, C - 1, 10).astype(int)]; rf = np.arange(C)
    thQ = cnt(rows, w0, start=th)
    from qij_joint.core.eta import measure_eta_full
    dfull = forward_step(measure_eta_full(cnt, X, th)[0])

    def q_at(cc, dl, central=False):
        tj = step_parameter(dl, p[cc])
        up = cnt(rows, _field_step_weights(w0, rf, cc, p[cc], tj), start=thQ)
        if not central: return ((up - thQ) / tj)[meas]
        dn = cnt(rows, _field_step_weights(w0, rf, cc, p[cc], -tj), start=thQ)
        return ((up - dn) / (2 * tj))[meas]
    print('%5s %4s  %-10s %s' % ('cell', 'n', 'variant', ' '.join('%7s' % s for s in SHORT)))
    worst_spread = 0.0; fd_gap = []
    for cc in pick:
        vs =dict(default=q_at(cc, dlt), half=q_at(cc, dlt / 2), double=q_at(cc, 2 * dlt), central=q_at(cc, dlt, True))
        u = np.zeros(N); u[inv == cc] = 1.0 / n[cc]; om, tt = evaluation_weights(u, dfull)
        full = ((cnt(X, om, start=th) - th) / tt)[meas]
        fd_gap.append(np.abs(full - ex[cc]) / sV)
        for k, v in vs.items():
            print('%5d %4d  %-10s %s' % (cc, n[cc], k, ' '.join('%7.3f' % x for x in v / sV)))
        spread = max(np.abs(v - vs['default']).max() / max(np.abs(vs['default']).max(), 1e-300) for v in vs.values()); worst_spread = max(worst_spread, spread)
        print('%5s %4s  %-10s %s' % ('', '', 'full data', ' '.join('%7.3f' % x for x in full / sV)))
        print('%5s %4s  %-10s %s' % ('', '', 'exact', ' '.join('%7.3f' % x for x in ex[cc] / sV)))
    print('largest relative change among default / half / double / central (relative to max |default|): %.2e' % worst_spread)
    print('full-data forward difference of the cell mean vs exact: max |diff| / sqrt V over the 10 cells and outputs: %.2e' % np.max(fd_gap))
    # the whole pilot again with base weights N p_c (sum N, the full data's weight total, so the penalty a = 1/W is the full
    # data's); the same step rule, the same mass fractions p_c, every row continued from theta_q(W = N)
    t = time.time(); wN = N * p
    thQN = cnt(rows, wN, start=th); eN, _ = _measure_eta_Q(cnt, rows, wN, rf, p, thQN); dN = forward_step(eN)
    I_N = np.empty((C, len(th)))
    for cc in range(C):
        tj = step_parameter(dN, p[cc]); I_N[cc] = (cnt(rows, _field_step_weights(wN, rf, cc, p[cc], tj), start=thQN) - thQN) / tj
    I_N -= (p[:, None] * I_N).sum(0)
    print('\nthe whole pilot with the rows\' weights summing to N instead of M (eta_Q %.1e), %.1f s:' % (eN, time.time() - t))
    summary('shearmix, rows weighted to N', I_N[:, meas], ex, n, V, SHORT)
    return None


def control(dataset):
    c = registry.case(dataset, 'all'); T = c.make_T()
    X = np.asarray(c.draw(10000, 0), float); N = len(X)
    P = np.asarray(T.influence(X, np.ones(N)), float); V = (P ** 2).mean(0)
    pil = pl.build_cells(X, M=1086, vq_transform=c.vq_transform, seed=0); C = pil.C; p = pil.ncell / N
    rows = native_cell_means(X, pil); cnt = Counter(T, N); th = cnt(X, np.ones(N))
    t = time.time()
    _, I, _, _ = prototype_influences(rows, cnt, p, T.eta, quantized_start=('full-data' if cnt.takes_start else 'multistart'), theta_hat=th)
    _, ex, _ = cell_stats(P, pil.inv, C)
    print('\n%s: N %d, M %d cells, pilot %.1f s, influence mean / sqrt V %s' % (dataset, N, C, time.time() - t, f(P.mean(0) / np.sqrt(V), '%.1e')))
    summary(dataset, I, ex, pil.ncell, V, list(T.outputs))


if __name__ == '__main__':
    shearmix()
    print('\n' + '=' * 100); print('4. CONTROLS: the same pilot on fp/all and mvt/all, N = 10^4 draw 0, exact T.influence'); print('=' * 100)
    control('fp'); control('mvt')
    print('\npilot_diag total %.1f s' % (time.time() - T0))
