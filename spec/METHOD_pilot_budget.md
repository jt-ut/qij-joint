# The pilot's budget: M as a function of ε

Proposed 2026-11-07; ruled the method's default on 2026-11-12 (`core/pilot.py`, `pilot_budget`; `method.run` uses it when
M is not given). Replaces M = round(N / ln N) (bed/METHOD_qij.md step 1) and the original cost rule
M_X = ⌈√((1 + 2 q M_ref) N / 2)⌉, M_ref = ⌈√(2.7 / ε)⌉ (spec/archive, QIJ_method_outline.md step 1 and
QIJ_writeup_notes.md, revision 3.3 of 16 September 2026).

## The rule

    M(ε) = 3 · K_r(ε),        K_r(ε) = (κ_r / ε)^(r/2),        r = min(d, q),

    κ_r = 2 · Γ(r/2 + 1)^(2/r) · ((r + 2) / r)^((r + 2)/2) / (r (r + 2)).

M is the number of pilot cells (k-means prototypes on the clustering coordinates), each evaluated once. d is the
dimension of the clustering coordinates, q the number of outputs the method works on. M does not depend on N.

| r | κ_r | K_r at ε = 0.01 / 0.02 / 0.03 / 0.045 | M = 3 K_r |
|---|---|---|---|
| 1 | 2.72 | 16 / 12 / 10 / 8 | 49 / 35 / 29 / 23 |
| 2 | 1.00 | 100 / 50 / 33 / 22 | 300 / 150 / 100 / 67 |
| 3 | 0.578 | 440 / 155 / 85 / 46 | 1319 / 466 / 254 / 138 |
| 4 | 0.398 | 1584 / 396 / 176 / 78 | 4752 / 1188 / 528 / 235 |

For the paper's cases at N = 10⁴: shearmix (d = 2, q = 7, r = 2) 300, mvt (d = 10, q = 2, r = 2) 300,
fp (d = 3, q = 4, r = 3) 1319 at ε = 0.01. The standing pilots were 1086 on all three.

## Origin of each term

### The budget principle, and the factor 3

The pilot gets the same cost as the measurement it serves. This is the original rule's principle (revision 3.3): the
cheap stage, which finds where ψ varies, is given what the expensive stage, which measures it, will spend. The
original rule balanced rows, because its pilot evaluations were refits on M prototypes and were counted as M / N of
a full refit, 2 M² rows against (1 + 2 q M_ref) N rows for the measurement. Measured on 2026-11-06/07 (shearmix,
N = 10⁴): a refit on M prototypes costs the same wall time as a refit on the N points, because the cost is the
iterations, not the rows. Every evaluation therefore costs one refit, and the balance is in evaluation counts.

The measurement is the tree. The factor 3 was fixed on 2026-11-07 as the tree's upper count per final leaf: 2 per
split, as both children of a split were then evaluated (K − 1 splits for K leaves), and 1 test evaluation per leaf,
counted for every leaf although only the leaves that reach the test receive one. Later the same day the split went
back to one evaluation (the smaller child; the sibling inferred), with repairs added, so the count per leaf is now 1 per
split + 1 test + repairs; measured on the suite of 2026-11-07 at M = 300, (E − M − q) / K was 1.5 on shearmix and 1.9
on mvt, with no repair. The rule keeps 3, as ruled: the budget is one number, and 3 provisions the pilot at 1.5 to 2
times the tree's measured cost. Hence M = 3 K, where K is the number of leaves the tree will have at the stop.

### K_r(ε): Zador's law for the leaves

K is not known before the run; the tree finds it. The budget substitutes the count for normal-valued influence,
which is in closed form by Zador's law. For the optimal quantizer of a density f on R^r with K groups, the
high-rate distortion is

    D(K) = C_r · ‖f‖_{r/(r+2)} · K^(−2/r),        ‖f‖_{r/(r+2)} = (∫ f^(r/(r+2)))^((r+2)/r),

C_r the normalized second moment of the optimal cell shape (the cell-shape factor; it carries nothing about f).
The method works in units in which every output has unit variance (ψ_j / √V_j), so the substitute density is the
standard normal in r dimensions, φ_r. Its norm is in closed form:

    ∫ φ_r^a = (2π)^(−ra/2) (2π/a)^(r/2),  a = r/(r+2)   ⇒   ‖φ_r‖_{r/(r+2)} = 2π ((r + 2)/r)^((r + 2)/2).

The within variance per output is D(K)/r. Setting it equal to ε and solving for K gives K_r(ε) = (κ_r / ε)^(r/2)
with κ_r = C_r ‖φ_r‖_{r/(r+2)} / r.

### κ_r: the constant

κ_r is Zador's law evaluated for the standard normal, per coordinate. The one number in it that is not exactly known
for r ≥ 3 is C_r. The rule uses the sphere bound C_r = Γ(r/2 + 1)^(2/r) / ((r + 2) π), one expression for every r:
exact at r = 1 (1/12), 0.0796 against the exact hexagonal 0.0802 at r = 2, and two to five percent below the best
known lattices for r ≥ 3 (body-centred cubic 0.0785 at r = 3, D4 0.0766 at r = 4; the large-r limit is 1/(2πe)).
Substituting the sphere bound gives the closed form of κ_r above. The margin against the lattice values is three
percent in M at r = 3 and five at r = 4, uniformly on the optimistic side; the k-means partition itself falls short
of the optimal quantizer by more. As r grows, κ_r → 1/r, with no change of form. κ_1 = (∫ φ^(1/3))³ / 12 = 2.72 is
the 2.7 of the original rule.

### r = min(d, q): the dimension of the influence values

The influence vectors ψ(x) lie in R^q, but ψ is a function of x ∈ R^d, so their set has dimension at most
min(d, q), the largest possible rank of the Jacobian of ψ. The rule takes that maximum: the influence values are
assumed to fill every dimension the data and the outputs allow, with no correlation between outputs (the spherical
normal). For a normal with correlation matrix R across the outputs the count is K_r(ε) · |R|^(1/2), and
|R|^(1/2) ≤ 1, so the spherical case is the largest count among normals of that dimension; the budget errs on the
side of provisioning the pilot. R is a property of the estimator and the data, not known before any evaluation,
which is why it is not in the rule.

## What the rule assumes, and what reports on it

The substitute is normal-valued influence of full dimension r. Influence values that are more concentrated, of
lower dimension, or correlated across outputs need fewer leaves than K_r, and the pilot is provisioned beyond what
the tree then uses; heavier-tailed values need more, and the tree closes the gap, with ψ̂ improving at every split.
Neither affects the certificate: the stop is decided by the test, not by the budget. Two quantities the method
already has report afterwards which held: the number of leaves at the stop against K_r(ε), and |R|^(1/2) computed
from the pilot's own cell means.

Cost of the whole method under the rule: about 5 K_r(ε) evaluations (3 K_r pilot, 1.5 to 2 K_r tree), scaling as
ε^(−r/2), against the bootstrap's 2 / ε².
