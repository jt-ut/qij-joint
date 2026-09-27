"""cloudfil_G_B6_P3_v1: Gaussian cloud + six-bead Gaussian filament + three protostars.
Author's design, 27-28 September 2026. Run: python3.9 make_cloudfil_G_B6_P3_v1.py -> .npz and .txt beside it."""
import numpy as np, os
def rot(a):
    c, s = np.cos(a), np.sin(a); return np.array([[c, -s], [s, c]])
def cov(a, b, ang):
    R = rot(ang); return R @ np.diag([a*a, b*b]) @ R.T
roles, w, mu, S = [], [], [], []
# cloud
roles.append("cloud"); w.append(0.585); mu.append([0.0, 0.0]); S.append(cov(2.0, 1.2, np.deg2rad(30)))
# filament: six beads on x = 2.2 t, y = 0.55 sin(pi t / 1.2); along-axis semi-axis 0.35, across 0.08
for t in (-1.0, -0.6, -0.2, 0.2, 0.6, 1.0):
    x, y = 2.2*t, 0.55*np.sin(np.pi*t/1.2)
    dx, dy = 2.2, 0.55*np.cos(np.pi*t/1.2)*np.pi/1.2
    roles.append(f"filament t={t:+.1f}"); w.append(0.38/6); mu.append([x, y]); S.append(cov(0.35, 0.08, np.arctan2(dy, dx)))
# protostars
roles.append("P1 embedded");    w.append(0.020); mu.append([0.0, 0.0]);                                   S.append(cov(0.05, 0.025, np.deg2rad(20)))
roles.append("P2 on-filament"); w.append(0.010); mu.append([2.2*0.6, 0.55*np.sin(np.pi*0.6/1.2)]);        S.append(cov(0.05, 0.025, np.deg2rad(-50)))
roles.append("P3 off-filament");w.append(0.005); mu.append([-1.2, -1.1]);                                  S.append(cov(0.05, 0.025, np.deg2rad(70)))
w = np.array(w); mu = np.array(mu); S = np.array(S)
assert abs(w.sum() - 1) < 1e-12
here = os.path.dirname(os.path.abspath(__file__))
np.savez(os.path.join(here, "cloudfil_G_B6_P3_v1.npz"), weights=w, means=mu, covs=S, role=np.array(roles),
         name="cloudfil_G_B6_P3_v1", N_design=10000, subject="P2 on-filament")
with open(os.path.join(here, "cloudfil_G_B6_P3_v1.txt"), "w") as f:
    f.write("cloudfil_G_B6_P3_v1 -- Gaussian cloud, six-bead Gaussian filament (along 0.35 / across 0.08), three protostars (0.05 x 0.025)\n")
    f.write("design N = 10000; subject core = P2; weights sum to 1\n\n")
    f.write(f"{'k':>2} {'role':<18} {'weight':>8} {'mean_x':>8} {'mean_y':>8} {'semi_a':>7} {'semi_b':>7} {'angle_deg':>9} {'E[n]@1e4':>9}\n")
    for k in range(len(w)):
        ev, V = np.linalg.eigh(S[k]); a, b = np.sqrt(ev[1]), np.sqrt(ev[0]); ang = np.degrees(np.arctan2(V[1,1], V[0,1]))
        f.write(f"{k:>2} {roles[k]:<18} {w[k]:8.4f} {mu[k,0]:8.3f} {mu[k,1]:8.3f} {a:7.3f} {b:7.3f} {ang:9.1f} {10000*w[k]:9.0f}\n")
print(open(os.path.join(here, "cloudfil_G_B6_P3_v1.txt")).read())
