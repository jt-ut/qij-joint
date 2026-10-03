"""cloudfil_G_U_P3_v1: Gaussian cloud + ONE smooth sheared-Gaussian
filament + three protostars (spec/QIJ_shearmix_interface.md section 5).
The author's replacement for `cloudfil_G_B6_P3_v1`'s six straight
Gaussian beads: a single filament component whose density is

    f_fil(x, y) = N(x | m, s2x) * N(y - h(x) | 0, s2p),
    h(x) = b0 + b1 sin(omega x) + b2 cos(omega x)

so an estimator's model can CONTAIN it exactly (well-specified). The
cloud and the three protostars are copied EXACTLY (weight, mean,
covariance) from `cloudfil_G_B6_P3_v1.npz` by role, so only the
filament's parametric form changes between the two datasets.

Run: python3.9 make_cloudfil_G_U_P3_v1.py -> .npz and .txt beside it."""
import os

import numpy as np

here = os.path.dirname(os.path.abspath(__file__))

with np.load(os.path.join(here, "cloudfil_G_B6_P3_v1.npz"), allow_pickle=True) as _src:
    _roles_src = [str(r) for r in _src["role"]]
    _weights_src = _src["weights"].astype(float)
    _means_src = _src["means"].astype(float)
    _covs_src = _src["covs"].astype(float)


def _pick(role):
    k = _roles_src.index(role)
    return float(_weights_src[k]), _means_src[k].copy(), _covs_src[k].copy()


roles = ["cloud", "P1 embedded", "P2 on-filament", "P3 off-filament", "filament"]

w, mu, S = [], [], []
for r in roles[:4]:
    wk, muk, Sk = _pick(r)
    w.append(wk)
    mu.append(muk)
    S.append(Sk)

# The filament: weight 0.38 (exactly the six-bead total it replaces),
# m = 0, s2x = 1.25**2, b = (0, 0.55, 0) (the same along-filament
# amplitude as the B6 curve, h(x) = 0.55 sin(omega x)), s2p = 0.08**2
# (the same across-filament spread as a B6 bead), omega = pi/2.64.
omega = float(np.pi / 2.64)
w_fil = 0.38
fil = np.array([0.0, 1.25 ** 2, 0.0, 0.55, 0.0, 0.08 ** 2])

weights = np.array(w + [w_fil])
means = np.array(mu)   # (4, 2): cloud, P1, P2, P3
covs = np.array(S)     # (4, 2, 2)
assert abs(weights.sum() - 1.0) < 1e-12

np.savez(
    os.path.join(here, "cloudfil_G_U_P3_v1.npz"),
    weights=weights, means=means, covs=covs, role=np.array(roles),
    fil=fil, omega=omega, name="cloudfil_G_U_P3_v1", N_design=10000,
    subject="P2 on-filament",
)

with open(os.path.join(here, "cloudfil_G_U_P3_v1.txt"), "w") as f:
    f.write("cloudfil_G_U_P3_v1 -- Gaussian cloud, ONE smooth sheared-Gaussian filament, three protostars (0.05 x 0.025)\n")
    f.write("design N = 10000; subject core = P2; weights sum to 1\n")
    f.write("filament: f_fil(x,y) = N(x|m,s2x) * N(y - h(x)|0,s2p), h(x) = b0 + b1 sin(omega x) + b2 cos(omega x)\n\n")
    f.write(f"{'k':>2} {'role':<18} {'weight':>8} {'mean_x':>8} {'mean_y':>8} {'semi_a':>7} {'semi_b':>7} {'angle_deg':>9} {'E[n]@1e4':>9}\n")
    for k in range(4):
        ev, V = np.linalg.eigh(covs[k])
        a, b = np.sqrt(ev[1]), np.sqrt(ev[0])
        ang = np.degrees(np.arctan2(V[1, 1], V[0, 1]))
        f.write(f"{k:>2} {roles[k]:<18} {weights[k]:8.4f} {means[k, 0]:8.3f} {means[k, 1]:8.3f} {a:7.3f} {b:7.3f} {ang:9.1f} {10000 * weights[k]:9.0f}\n")
    f.write(f"{4:>2} {'filament':<18} {weights[4]:8.4f}\n\n")
    f.write("filament parameters (data coordinates):\n")
    f.write(f"  m = {fil[0]:.4f}  s2x = {fil[1]:.4f} (sigma_x = {np.sqrt(fil[1]):.4f})\n")
    f.write(f"  b0, b1, b2 = {fil[2]:.4f}, {fil[3]:.4f}, {fil[4]:.4f}\n")
    f.write(f"  s2p = {fil[5]:.6f} (sigma_perp = {np.sqrt(fil[5]):.4f})\n")
    f.write(f"  omega = pi/2.64 = {omega:.6f}\n")
    f.write(f"  E[n]@1e4 (filament) = {10000 * weights[4]:.0f}\n")

print(open(os.path.join(here, "cloudfil_G_U_P3_v1.txt")).read())
