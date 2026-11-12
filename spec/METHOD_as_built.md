# The method as built, from the paper onward

Written 2026-11-07 for the author, who asked what the method is actually doing and what drives it; read and accepted
2026-11-12, when the budget rule became the code's default M and the quantized path left the code. Plain statement, the
paper's words, one symbol per quantity. The code is `src/qij_joint/method.py` (`run`). Detailed step document:
`spec/METHOD_qij.md`; the pilot's size: `spec/METHOD_pilot_budget.md`.

## 1. The paper

The paper's QIJ: quantize the data into M receptive fields; the variance of the estimator splits exactly as

    V  =  V_btw(M)  +  V_win(M),

V_btw the variance of the receptive-field mean influences (weighted by the fields' masses), V_win the variance of the
influence inside the fields. V_btw is computable without the influence formula: one estimator evaluation per receptive
field, each a finite difference of T along that field's mass, M evaluations in all. V_win was the problem: averages over
receptive fields do not see the spread inside them, and on the multivariate t and the Fundamental Plane the inside spread
is 54 to 80 % of V. Everything built since the paper exists to measure V_win without the influence formula, and to know
when it has been measured to a stated precision ε.

## 2. What the method does now, in five steps

Every "evaluation" below is one refit of the estimator on the full data with the weights of a group of points perturbed,
divided by the step: it returns the mean influence of that group. It costs one refit. There is no other kind of
measurement anywhere in the method.

**Step 1, the pilot (the paper's QIJ).** k-means on the data into M cells. One evaluation per cell: the cell's mean
influence. From these, V_btw(M) exactly, as in the paper. M is set before anything else by the budget rule
(`spec/METHOD_pilot_budget.md`): M = 3 K_r(ε), K_r the number of groups Zador's law says a normal influence of dimension
r = min(d, q) needs at precision ε. For the paper's cases at ε = 0.01: 300, 300, 1319.

**Step 2, the influence at every point, ψ̂.** A kernel interpolant through the M cell means: the one function that
reproduces every measured group mean exactly and is smoothest in between. ψ̂ is the method's current estimate of the
influence of every data point. Every later evaluation is added to it as one more group mean it must reproduce, so ψ̂ gets
better with every evaluation, and it is never anything but the interpolant of what has been measured.

**Step 3, the tree (measuring V_win).** Repeatedly: rank the current leaves (the root is all the data) by how much ψ̂
varies inside them, cut the top ones in two along ψ̂ (a two-means cut of the leaf's ψ̂ values), and evaluate the smaller
child. The larger child's mean is the parent's mean minus the smaller child's, by subtraction: one evaluation per split.
Each split moves variance from "inside the leaves" to "between the leaves", that is from V_win to V_btw, exactly as
refining M does in the paper, but only where ψ̂ says the influence varies. Up to k splits per round (k = 16, the pool
size), then ψ̂ is refitted.

**Step 4, the stop (knowing when V_win is captured).** The method stops when the influence variance still inside the
leaves is below ε V. It cannot read that variance directly, so it tests it: on every leaf it makes one evaluation with
random, mean-zero weights inside the leaf (a centred contrast), which reads how much ψ varies inside the leaf beyond what
ψ̂ already says. Those readings, summed with their known chi-square law, give an 84 % upper limit on the remaining
inside variance. When that limit is below ε V on every output, the method stops. The remaining inside variance is the
error of ψ̂, called R; its upper limit is called U; the threshold is δ = ε times a lower bound on V. "The test passed"
means U ≤ δ. One test evaluation per leaf that reaches the test.

**Step 5, the report.** q evaluations, one per output, with weights ψ̂ itself. They give V̂ = V − R exactly, one-sided:
V̂ is below V by exactly the error of ψ̂ that the test bounded. So the statement at the stop is V̂ ≤ V ≤ V̂ + U with U ≤
ε V at 84 %. The interval on the parameter is θ̂ ± z √(V̂/N).

**What it costs.** M pilot evaluations, one per split (K − 1 for K leaves), one test per leaf, q for the report:
about M + 2K. Today on shearmix at N = 10⁴, ε = 0.01: median 676 evaluations (M 300, K about 250) against 1376 with the
old pilot of 1086 and against 20 000 bootstrap replicates; the interval was within ε on 20 of 20 draws.

## 3. What drives it

- **ε** sets everything: the stop (δ = ε V), the number of leaves the tree needs (K grows like ε^(−r/2) for an influence of
  dimension r, Zador's law in influence space, measured today: ε^(−0.5) on shearmix and mvt, ε^(−1) on fp), and through
  K the pilot's budget M.
- **ψ̂'s quality** sets how many splits are spent. A coarse pilot makes early cuts in the wrong places and the tree pays
  with more leaves (K 250 at M = 300 against 179 at M = 1086 on shearmix) but recovers, because every split's
  evaluation corrects ψ̂. The tree reaches ε either way; the pilot only changes the route.
- **η, the estimator's accuracy**, sets the one departure from exactness. A refit solved to relative accuracy η makes the
  finite difference agree with the mean influence only to the step term, of relative order √η. For an estimator solved to
  machine precision (fp, P_tail) the evaluations are exact. For shearmix (η about 1e-6) each evaluation carries a step term
  of about half a percent of √V, and a value obtained by subtraction inherits the terms of the rows it was subtracted from.
  The order √η is derived; the constant in front of it is not, and the method's own measurements put it at 10 to 50
  (`METHOD_qij.md` section 4, item 3). The allowance the stop carries for it is therefore not yet justified; the author
  has asked for the justification (2026-11-12) and it is open.

## 4. Where it deviates from the paper

1. V_win is measured by a tree of further evaluations instead of being left out. This is the whole point of the work
   since the paper.
2. The influence is carried at every point by ψ̂, the interpolant of the measured group means, instead of only by the
   receptive-field means. ψ̂ is what lets the tree cut where the influence varies.
3. There is a stop with a stated level: the test of step 4. The paper had no criterion for M; the method now has one for
   when to stop refining.
4. Every evaluation is a refit on the full data. The paper's pilot refits on the M prototypes with their masses (cheap
   when a refit's cost is its rows). Measured this week: at N = 10⁴ a refit on 2705 prototypes costs the same wall time as
   one on the data, because the cost is the iterations; and the quantized refit returns the influence of a different
   problem where the statistic depends on the spread inside cells (the shearmix subject, the t's ν), which put V̂ off by
   4 to 40 % on 16 of 16 shearmix draws. So quantized evaluations are gone from the method.
5. M comes from the budget rule, not from N / ln N. The old defaults 1094 / 588 / 828 were the paper-era cost rule with a
   pilot evaluation counted as M / N of a refit; with a refit's true cost the balance is in evaluation counts and
   independent of N.

## 5. The quantities that appeared today

| name | what it is |
|---|---|
| V, V̂ | the influence variance per output, and the method's estimate V̂ = V − R |
| R | the error of ψ̂: the influence variance still inside the leaves at the stop; V − V̂ exactly |
| U | the 84 % upper limit on R from the test; the stop is U ≤ δ |
| δ | ε times the lower bound on V the method has at the time (so δ ≈ ε V) |
| η, η_full | the estimator's relative accuracy, measured once on the full data at the start |
| the step term | the amount by which a finite-difference evaluation differs from the exact mean influence, of relative order √η |
| the offset Ω | the allowance added to the stop for values obtained by subtraction, built from a bound √η per measured row; ruled today |
| the repair | evaluating a subtracted leaf directly when the offset alone blocks the stop; built today, never triggered |
| x, y, log_reff, log_axis_ratio, pa, logit_w, log_contrast | the 7 outputs of the filament estimator: the subject Gaussian's position, size, shape, angle, weight and contrast |
| ν, P_tail | the two outputs of the multivariate-t estimator: degrees of freedom and a tail probability |
| K | the number of leaves at the stop |
| M, k, q, N, d, r | pilot cells; splits per round; outputs; points; data dimension; min(d, q) |

## 6. The issues I raised today, judged

1. **Quantized evaluations do not give the interval.** Legitimate and measured (16 of 16 draws outside ε, no speed gain).
   This removed the paper's cheap pilot and forced the question of M.
2. **The pilot cost a bootstrap; M must depend on ε.** Legitimate; your question; answered by the budget rule.
3. **Two child evaluations per split.** My change of 6 November, made on a simulated-noise argument, not ruled by you.
   Your ruling today restored one child with η carried in the stop. On every draw today the allowance was below a
   thousandth of δ: the second evaluation was never needed for the interval. That change of mine was over-caution.
4. **"V − V̂ exceeds U on 12 of 20 draws" for one output.** Real but internal. The interval at ε held on 20 of 20 draws;
   what fails is the stop's own 84 % statement about R, because the test's centred contrasts cannot see a constant
   offset of ψ̂ over a subtracted leaf, and the allowance for it, built from √η, is 10 to 50 times smaller than the
   discrepancies the method itself measures on rows it finds redundant. It affects the method's claim to a stated level,
   not today's intervals. The remedy is to take the bound from those measured discrepancies instead of from √η. It can
   wait for your decision; nothing in the results today depends on it.
5. **The mvt misses on 5 draws.** Not an issue; the design level, as you said.
6. **The suite runs slowly.** Not the method. A draw takes 63 s alone and 180 to 345 s when 10 to 16 run at once, with the
   refit itself 3 to 5 times slower under load. That is the machine, threads or memory, and `top` during a suite will
   show which.

My own errors today, for the record: the √N pilot verdict in the morning was pedantic; the "first round measures k
groups" and "M realized by the tree" proposals changed the method while presented as rules for M; the symbol σ and the
word "error" for exact measurements; and a suite cap sized from a single-process timing.
