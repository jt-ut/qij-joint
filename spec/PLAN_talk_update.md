# Plan: update the QIJ talk figures with the method as built

Written 2026-11-12 (Thursday). The talk is given again next week. This file is the one place that says where the
work stands; it is updated as items close. Dataset: sheared filament (cloudfil_u), **Core 1 (p1)**, N = 10⁴, as in
the talk of 4 October. The deck: `Talks/2026_Oct04_QIJ-SciMLWorkshop/2026_Oct04_QIJ-SciMLWorkshop.pdf` (21 slides).

## 1. Where the code stands today

qij_joint main, commit 8a41bbf (2026-11-12). The method is `src/qij_joint/method.py` `run`; the plain account is
`spec/METHOD_as_built.md`; the steps are `spec/METHOD_qij.md`; the pilot's size is `spec/METHOD_pilot_budget.md`.

| piece | state | evidence |
|---|---|---|
| the method, single process | works | shearmix Core 2, 20 draws, ε = 0.01: within ε on 20/20, median 676 evaluations, 63 s per draw alone (2026-11-07) |
| the method on Core 1 (the talk's core) | **not run since the port** | the 20-draw suite used Core 2 (p2); Core 1 differs in its influence (exact ψ for 1000 draws is on disk: runs/shearmix4/cloudfil_u_p1_N10000/ij) |
| the parallel path (pool of workers) | **not run since the port** | the sandbox here cannot start the pool; every run so far was one process |
| the study driver (scripts/run.py → pipeline.run_qij → record) | **not run since the port** | the record's columns changed (V̂, U, δ, offset, counts instead of V_btw, V_win_hat, V_tot_hat, κ) |
| the summary (analysis/summarize_shearmix.py) and the frontier figure | **read the old columns** | they read V_tot_hat_<o>, V_btw_<o>, wall_time_total, normalized_rows |
| the budget rule M(ε) = 3 (κ_r/ε)^(r/2) | default in the code | 300 at ε = 0.01 for this dataset (r = 2) |
| TACC submission | **stale** | submit lines pass flags the driver no longer has; staged package is October's |
| Fundamental Plane, mvt | not needed for the talk | fp never run through the port; mvt 3/5 within ε on 5 draws |

## 2. Is it an improvement over the code that made the 4 October figures?

Same dataset and ε = 0.01, N = 10⁴. Old = the October study (runs/shearmix4, Core 1, 999 draws, TACC one core).
New = the port (Core 2, 20 draws, laptop one process). The Core 1 numbers for the new code are step 1 below.

| | old (4 Oct) | new (as built) |
|---|---|---|
| evaluations per draw, every refit counted | 1733 (1108 pilot on quantized data + 624 tree + 1) | 676 median (300 pilot + about 370 tree and test + 7) |
| evaluations as the old study counted them | 1154 (a quantized refit counted as M/N of a refit) | 676 (nothing is discounted) |
| seconds per draw, one core | 125 (TACC) | 63 (laptop) |
| V̂ within ε of the exact V | 99.6 % of draws (any output) | 20 of 20 |
| what certifies the stop | the check rule V̂_win / V̂_tot < ε, with κ | the test: one centred-contrast evaluation per leaf, 84 % one-sided limit on the error of ψ̂ |
| the pilot | 1094 refits on prototypes, GP regression, scale correction | 300 refits on the data (the cells' mean influences), kernel interpolant through them |

So: half the evaluations and half the time at the same accuracy, with a certificate in place of a check rule. Two
things the old study claimed are gone: the quantized refit's cheapness (measured: a refit on the prototypes costs
the same as on the data at N = 10⁴, the cost is the iterations) and its correctness (the quantized evaluations put V̂
off by 4 to 40 % when the whole loop ran on them; the October loop survived because its tree evaluations were on
the data and the pilot only steered).

## 3. What each talk slide depends on

| slides | content | figure script | data it reads | with the new method |
|---|---|---|---|---|
| 1–9 | title, cloud, Core 1, intervals, bootstrap / jackknife / IJ, analytic influence, the identity | data/, schematic/, resampling/, influence/ | the dataset, the analytic influence, old boot/ij replicates (demos) | **unchanged** |
| 10 | prototypes by k-means, M by a budget argument | qij/stage1_explore.py (prototypes) | runs/scratch/talk_canonical, draw 871 | regenerate: M = 300 cells |
| 11–12 | derivatives at prototypes; regress and predict ψ̂ | stage1_explore.py (pilot ψ̂ scatter) | same | regenerate: the pilot evaluations are the cells' mean influences; ψ̂ is the interpolant |
| 13–14 | "predicted ψ̂ orders influence space but not the scale" | stage1_explore.py (pilot vs measured) | same | regenerate; the statement becomes "the pilot ψ̂ under-predicts the variance; the tree closes it" |
| 15–18 | the tree: start, cut, measure, correct, stop | qij/stage2_strip.py | runs/scratch/talk_canonical_shape (shape-only run of draw 871) | regenerate from a new shape-only run; step 4 becomes "refit ψ̂ through the new means; the test; stop at U ≤ ε V" |
| 19–20 | Core 1 performance: accuracy and cost against Boot and the oracle | frontier/frontier_shearmix.py | analysis/summaries/shearmix4_p1 (from the shearmix4 qij study, boot B = 50…10000, oracle, ij) | **new study** on TACC, new summaries, same figure script |
| 21 | conclusions | text | | rewrite three lines |

Boot, oracle and ij references are frozen; only the qij study is rerun.

## 4. The work, in order

| # | what | needs | time | done when |
|---|---|---|---|---|
| 1 | **Readiness on Core 1.** 20 draws of Core 1 at ε = 0.01 through the study driver with 8 workers on the laptop (`scripts/run.py cloudfil_u p1 qij --N 10000 --draws 0:20 --eps 0.01 --workers 8 --out runs/scratch/talk_port`): exercises the pool and the driver, writes the records, gives the real wall time; the records' V̂ are then read against V_ij on the ij rows. | the author's go (given 2026-11-12) | 5–8 min, 10-min cap | within ε of V_ij on ≥ 16/20 and E in the 600s |
| 2 | **Summary and figure code.** summarize_shearmix.py reads the new record (V̂_<o>, E, wall_s); the honest-ε table compares V̂ with V_ij as before; frontier_shearmix.py unchanged. The cost axis counts every refit as one. | no run | 2 h | runs on the old records' shape with the new names (checked on step 1's record) |
| 3 | **TACC study.** Stage main 8a41bbf; new submit lines (`scripts/run.py cloudfil_u p1 qij --N 10000 --draws a:b --eps E --workers 1 --out <root>`); study root runs/talk2 (new root, convention); Core 1, draws 0..999, the 11 ε of the old grid (0.01 … 0.2). | the author's go; reference_tacc.md | about 6 core-minutes per draw for all 11 ε → about 100 core-hours, one node-hour on Stampede3; queue time dominates | 11 × 1000 records in runs/talk2 |
| 4 | **Summaries and the frontier figures.** | step 3's records | 1 h | figs/frontier/sheared/frontier_{accratio,costratio}_dark regenerated; crossover B read off |
| 5 | **Canonical-draw figures (slides 10–18).** A driver `scripts/talk_canonical.py`: run draw 871 of Core 1 with record= and progress=, export the cells, the pilot values, ψ̂ at the pilot and at the stop, the first split's cut and measurement, the test's terms; a shape-only run for the strip. Adapt stage1_explore.py and stage2_strip.py to read the export. Decide whether 871 stays canonical (re-rank by the new method's own result). | the author, figure by figure | 1–2 days | the eight dark figures regenerated at the deck's sizes |
| 6 | **Slide text.** Steps on slides 10–18; numbers on 19–20; conclusions. | the author | ½ day | |

Steps 1–2 today; step 3 submitted tomorrow; 4 when the records are down; 5 in parallel with 3; 6 last.

## 5. Decisions needed from the author

1. Go for step 1 (two small runs on the laptop, 11 minutes).
2. The study: Core 1, 1000 draws, the 11 ε, root runs/talk2. Or fewer ε (the frontier plots B = 50 … 10⁴, which is
   ε = 0.2 … 0.014; ε = 0.01 is off the plotted range but is the paper's number).
3. Canonical draw: keep 871, or re-pick by the new method's result.
4. Slides 15–18: present the test and the certificate as the stop (the method as built), or keep the loop's
   presentation ending at the refit of ψ̂ with the stop stated in words.
5. Delete the four stale files in scripts/ (port_check.py, port_check.log, quantized_run.py, quantized_run.log).

## 6. What changes in the talk's story

- The pilot is M evaluations of the cells' mean influences on the data, M = 3 (κ_r/ε)^(r/2) with r = min(d, q):
  slide 10's "cost-budgeting argument + asymptotic VQ theory" is now literally the rule. Slide 11's "derivatives at
  mass-weighted prototypes, M ≪ N, cheap" goes: each pilot evaluation is a refit.
- ψ̂ is the kernel interpolant that reproduces every measured group mean; no regression, no scale correction, no κ.
  Slides 13–14's "right order, wrong scale" becomes "the pilot ψ̂ under-predicts the variance inside the cells; the
  tree measures it".
- The loop: rank leaves by the variance of ψ̂ inside them, cut by two-means on ψ̂, measure one child (the sibling by
  subtraction), refit ψ̂; the test reads what ψ̂ still misses, leaf by leaf; stop when its 84 % limit is below ε V.
  Report V̂ = V − R: the stated statement is V̂ ≤ V ≤ V̂ + U.
- Cost: about 5 K_r(ε) evaluations, scaling as ε^(−r/2), against the bootstrap's 2/ε². At ε = 0.01 on this dataset
  about 650 against 20 000.

## 7. The runs directory: what is there, and the proposed standard (2026-11-12)

Inventory of `Research/QIJ_joint/runs/` (52 GB, 12 roots, 70 scratch folders). Layout rule already in force:
`runs/<study>/<dataset>_<estimator>_N<N>/<method>_<config>/s<draw>.parquet`; the mess is the number of roots and
the scratch folders, not the layout inside a root.

| root | what | draws | GB | status |
|---|---|---|---|---|
| shearmix3 | oracle, sheared filament, N = 10⁴ (raw θ; serves both cores) | 10 000 | 0.6 | **reference** |
| shearmix4 | boot B = 5000 (raw θ per replicate), ij Core 1 and Core 2 (Core 2's ij holds per-point ψ, 3.7 GB), qij October method Core 1 and Core 2, 11 and 10 ε | 1000 | 11.6 | boot + ij **reference**; qij = the 4 Oct figures, superseded once the new study is down |
| shearmix_b10k | boot B = 10 000 (raw θ), one qij folder | 1000 | 4.0 | boot **reference**; qij folder superseded |
| shearmix | oracle, 628 draws, before shearmix3 | 628 | 0.04 | superseded |
| shearmix_fb_s1000 | qij "feedback" variant, Core 2 | 1000 | 4.1 | superseded |
| speed_s1000 | qij affine-trend variant: sheared Core 1, imf, mvt | 1000 | 2.8 | superseded |
| paper_n2000 | oracle, ij, boot, qij for pareto / mvt / imf / fp at N = 2000 (25 Sept code) | 1000 | 0.4 | oracle + ij **reference** for the paper datasets |
| paper_s1000 | boot B = 5000 and qij October method, 9 ε, four paper datasets at N = 2000 | 1000 | 3.1 | boot **reference**; qij superseded |
| cloudfil_s1000 | six-bead dataset: oracle 10 000, boot, ij, ijfd, 24 qij variants | 1000 | 20.1 | bead, retired (your rule: not deleted, not used) |
| total_s1000, unified_s1000 | six-bead dataset, qij variants | 1000 | 4.3 | bead, retired |
| scratch | 70 folders of checks 27 Sept – 5 Oct of code that no longer exists; plus talk_canonical, talk_canonical_shape, talk_canonical_speed, jackknife_shearmix (the animation's input), talk_port (today) | | 0.9 | all but the five named: delete |

**Are the bootstrap results sufficient?** For the figures as designed, yes. Sheared filament Core 1 at N = 10⁴:
boot B = 5000 and B = 10 000 with raw θ per replicate on draws 0..999, so every B ≤ 10 000 is a subsample of
what is on disk, on the same draws the qij study uses, with the oracle's 10 000 draws for the truth. Paper
datasets at N = 2000: boot B = 5000 on draws 0..999 and the oracle / ij reference. They would stop being
sufficient only if N changes, or if the ε = 0.01 point is to be shown with its matched bootstrap (B = 20 000,
never run, never plotted), or if bootstrap seconds are wanted on different hardware from TACC's (the new qij
study is on TACC too, so the seconds stay comparable).

### 7.1 The canonical tree (proposed 2026-11-12, awaiting approval)

The author's ruling: the talk covers the sheared filament (working example), mvt and fp; the results tree must
hold qij, boot and ij together, per case, with no study level; the six-bead results are wrong and are deleted;
the project must not occupy 50 GB. The rule of 2026-10 (one root per study) is withdrawn by this.

One folder per data set and size; every method's products beneath it; the estimator in the folder name only
where methods differ by estimator (the two cores of the sheared filament):

    runs/
      README.md                          what each folder holds, which code made it, when; kept current
      cloudfil_u_N10000/                 the sheared filament, N = 10⁴ (the talk's working example)
        oracle/                          10 000 draws, the mixture fit's raw θ (both cores derive from it)
        boot_B5000/  boot_B10000/        1000 draws, raw θ per replicate
        ij_p1/  ij_p2/                   1000 draws each; ij_p2 carries per-point ψ (3.7 GB), ij_p1 rows only
        jackknife/                       the animation's delete-one run (draw 1)
        qij_p1_eps<ε>/                   the method of record, one folder per ε (the new study)
        canonical/                       the draw-871 export for the method slides
      mvt_N<N>/  fp_N<N>/  imf_N<N>/  pareto_N<N>/    the four paper datasets, all computed the same way
        oracle/  ij/  boot_B5000/  qij_eps<ε>/        (author 2026-11-12: imf and pareto stay)

    N for the paper datasets is the author's decision (2026-11-12): every 1000-draw study of them on disk is at
    N = 2000, the paper's design (paper_n2000 of 25 Sept, paper_s1000 of 4 Oct, speed_s1000). N = 10⁴ exists only
    as (a) the test bed's suites, mvt 20 draws and fp 5 draws with exact influences, no bootstrap
    (vqboot/fable/bed/runs, bed/cache), and (b) the paper's own single-draw cost-against-N series for fp and imf
    (Research/QIJ_WSOM2026/runs/cost_vs_n*, N = 1000 … 76 997, September paper code). There is no 1000-draw mvt
    or fp study at N = 10⁴ with oracle, ij and bootstrap anywhere on disk. Running one is cheap: these estimators
    refit in about a millisecond, so oracle + ij + boot B5000 at N = 10⁴ is about two core-hours per dataset, and
    the qij study minutes per ε. Recommendation: N = 10⁴ for all four, the same N as the working example.
      scratch/                           one check at a time; emptied when its question is answered

Rules: a folder's name says what is in it; nothing is stored by the TACC job that made it; a qij study that
supersedes another replaces it (the old one is deleted, its summary's numbers stay in the paper/talk record);
`products.method_dir` writes this layout, so new runs land in it without renaming.

### 7.2 What moves, what is rerun, what is deleted

| action | items | GB |
|---|---|---|
| move into the tree | shearmix3 oracle → cloudfil_u_N10000/oracle; shearmix4 boot → boot_B5000; shearmix_b10k boot → boot_B10000; shearmix4 ij p1, p2 → ij_p1, ij_p2; scratch/jackknife_shearmix → jackknife; talk_port's 11 records → qij_p1_eps0.01 (replaced by the study) | 9.7 kept |
| rerun with the current code, into the tree (consistency: one code version for the paper datasets; cheap, sub-ms refits) | mvt_N2000 and fp_N2000: oracle (1000 draws), ij, boot_B5000 (1000 × 5000 refits ≈ 1–2 core-hours each) | ≈ 0.5 |
| the new qij studies, into the tree | cloudfil_u_N10000/qij_p1_eps<ε> (11 ε × 1000 draws, ≈ 100 core-hours); mvt_N2000 and fp_N2000 qij_eps<ε> (9 ε × 1000 draws, minutes each) | ≈ 3 |
| **delete now** | the six-bead roots cloudfil_s1000, total_s1000, unified_s1000 (wrong, your ruling); shearmix (628-draw oracle); shearmix_fb_s1000; speed_s1000; shearmix_b10k's qij folder; paper_n2000's and paper_s1000's qij folders; scratch except jackknife_shearmix, talk_canonical*, talk_port | 32 |
| **delete after step 4** (the 4 Oct figures rest on them until the new ones exist) | shearmix4's qij folders (Core 1 and Core 2); paper_s1000 boot and paper_n2000 oracle/ij once the reruns above are down and checked against them; scratch/talk_canonical* once the new export exists | 10 |
| beside runs/: analysis/summaries | one folder per case matching the tree; the old summary folders (cloudfil bead, compare_total_unified, speed, full_warm, qij_pilot_compare, shearmix4, shearmix4_p1, paper_s1000) deleted with their studies; the cache rebuilt once (30 min) for the new paths | |
| beside runs/: talk/figs | frontier/bead, imf_speed, mvt_speed, sheared_speed and the _archive folder deleted | |

After: about 13 GB, all of it in the tree, all of it from one code version except the sheared filament's
oracle and bootstrap (October, same estimator code).

### 7.3 Code that follows the tree

- `qij_joint/src/qij_joint/products.py` `method_dir`: `runs/<dataset>_N<N>/<method>[_<estimator>][_<tag>]`, boot
  tagged by B; the driver's `--out` becomes `runs/` itself.
- `analysis/summarize_shearmix.py`, `analysis/summarize_paper.py`: defaults to the tree's paths; the paper summary
  reads the new 'all' oracle/ij rather than the per-output folders of paper_n2000.
- `talk/figs/frontier/*.py`: default roots.
- The TACC submit lines: `--out <scratch>/runs`, staged back into the tree.

**Done 2026-11-12:** the delete-now set, approved by the author ("delete list approved"), removed: runs/ went
from 50 GB to 16 GB. Remaining roots: paper_n2000 (oracle, ij, boot), paper_s1000 (boot only), shearmix3
(oracle), shearmix4 (boot, ij ×2, the two October qij studies), shearmix_b10k (boot), scratch (the five talk
folders). Not yet removed, pending the same approval: the matching summary folders under analysis/summaries
(cloudfil_*, compare_total_unified, speed_s1000, speed_s1000_p1) and talk/figs/frontier/{bead, imf_speed,
mvt_speed, sheared_speed}; two summary folders of unknown provenance (full_warm, qij_pilot_compare) are listed,
not touched.

### 7.4 Order

1. Approve this section. 2. Delete-now set (one command, listed first for your eyes). 3. Create the tree and
move the references; update products.method_dir and the two summaries; rebuild the cache in the background.
4. Rerun oracle / ij / boot for mvt and fp (laptop overnight or TACC, your choice). 5. Submit the qij studies
into the tree. 6. Regenerate summaries and figures. 7. Delete-after set. 8. runs/README.md written; the layout
rule saved to memory in place of the 2026-10 rule.

## 9. The runs, on Vista (proposed 2026-11-12, awaiting approval)

Checked on TACC (2026-11-12, over the author's open Vista connection; Lustre scratch not searched, per the
author): the work filesystem holds the same study roots as the laptop and nothing at N = 10⁴ for mvt or fp.
So the paper datasets' references and every qij study are new runs.

**Vista facts.** aarch64 (Grace); system python3.9.25; partitions gg (251 CPU nodes, 144 cores each), gh, gh-dev,
gh-shared, gb; allocation AST24021 with 75 127 SUs. No qij environment exists on Vista (the Stampede3 venv is
x86); the sesnaimpute Vista build script is the template (binary wheels only, pinned to Stampede3's versions).
The qij_joint clone on work is at ee50223 (4 Oct); main 8a41bbf is local only, one commit ahead of origin.

**What runs.** All at N = 10⁴ (recommendation of §7.1, to be confirmed), draws 0..999:

| case | oracle | ij | boot B5000 | qij | note |
|---|---|---|---|---|---|
| cloudfil_u_N10000 (Core 1) | on disk | on disk | on disk (+ B10000) | 11 ε | about 100 core-hours |
| mvt_N10000 | new | new | new | 11 ε | ν by profile likelihood: a few ms per refit |
| fp_N10000 | new | new | new | 11 ε | r = 3: M = 1319, about 1300 leaves at ε = 0.01; ms refits |
| imf_N10000 | new | new | new | 11 ε | the one optimizer: the bootstrap is the expensive item (5 × 10⁶ refits) |
| pareto_N10000 | new | new | new | 11 ε | closed form; r = 1 |

Rough total under 1000 core-hours, a few gg node-hours. Job layout as in October: one job per (case, method,
ε), 1000 draws, one draw per core, `--workers 1`; outputs to `$SCRATCH/QIJ_JOINT/runs/<case>/<method>/`
(the canonical layout, written by the updated products.method_dir), staged to `$WORK/projects/QIJ_JOINT/runs`
after each job and pulled to the laptop with rsync. Reference jobs (oracle, ij, boot) first; qij jobs after a
one-draw smoke of each case on gh-dev.

**Steps.** V1 push main to origin (author's word; add scikit-learn to pyproject's dependencies first, the pilot
uses it). V2 build the venv on Vista from the template; clone to `/work/08772/jjt277/vista/software/qij_joint`
at main. V3 products.method_dir → the canonical layout (also what the local tree needs). V4 submit scripts for
Vista's gg partition (adapted from the October `slurm_s1000` scripts on work). V5 smoke, then submit (each
submission on an explicit go). V6 rsync back into the canonical tree; summaries; figures.

## 8. Checklist (ticked as items close)

- [x] 1 Core 1 through the driver, 8 workers (2026-11-12, 09:30–09:40): 11 draws finished inside the 10-min cap
      (draws 0–10), **within ε of V_ij on 11 of 11** (max 0.47 ε; medians per output 0.03–0.37 ε, largest on
      logit_w and log_contrast; V̂ below V on most, as the one-sided statement says); E median 783 (717–899), K 284
      (260–391), M 300; rounds 37–90 because k defaults to the pool's 8 workers; wall 37–66 s per draw, median 55 s,
      against 63 s single-process: the pool gave 1.1–1.7×, not 8×, because the shearmix refit slows 3–5× when 8 run
      at once on this laptop (pilot: 300 evaluations in 8–15 s). The driver, the pool and the record all work;
      records at runs/scratch/talk_port/cloudfil_u_p1_N10000/qij_eps0.01 (s00000–s00010 + leaves/rounds parquet).
      Note for the summary: the ij rows' V_ij is the variance of the estimate, V / N.
- [x] 2 summarize_shearmix.py reads both record formats (as built: Vhat_<o>/N, V_lb_<o>/N, U_<o>/N, E, total_s;
      October: V_tot_hat, V_btw, normalized_rows, wall_time_total), detected per folder from the first record;
      --qij-glob default qij_eps*; frontier_shearmix.py's folder pattern likewise. Checked on the 11 records
      (2026-11-12): report_err/ε per draw identical to the direct computation (0.47, 0.34, 0.39, …), evals = E,
      seconds = total_s; 77 interval rows, coverage 0.974, median width ratio 1.067 (boot B5000 on the same
      draws 0.987 / 1.131). Also fixed: the script's cache fingerprint compared directory paths as strings, so a
      run given absolute roots missed the cache written from relative ones and rebuilt the oracle scan and the
      bootstrap derivation (30+ min); paths are now compared resolved, and a miss prints which key differs. The
      summary now takes 5 s. Outputs of the check: the session scratchpad summ_check/ (not the summaries dir).
- [ ] 3 TACC: staged, submitted, down
- [ ] 4 summaries + frontier figures
- [ ] 5 canonical-draw export + stage 1 figures + stage 2 strip
- [ ] 6 slide text
- [x] runs/ delete-now set removed (50 → 16 GB, 2026-11-12)
- [x] canonical tree (approved "ok go", 2026-11-12): products.method_dir writes `runs/<dataset>_N<N>/<folder>`;
      config.json carries the code commit and refuses a mismatch; manifest.json per folder and a generated
      README.md (commits a964327, 6f4d897, pushed). The sheared filament's references moved in with manifests
      written from their runs.log (oracle 10 000 draws, boot_B5000 and boot_B10000 1000, ij_p1 and ij_p2 1000;
      all Stampede3, 3–4 Oct) and a `notes` line naming where each came from; jackknife/ moved beside them
      (no manifest: not a driver product). summarize_shearmix.py and frontier_shearmix.py read the tree
      (--runs, qij_<core>_eps*); the summary cache's stored paths remapped, cache hit confirmed, numbers
      unchanged. Attributable summary and figure folders of the deleted studies removed; full_warm and
      qij_pilot_compare (unknown provenance) left. Still under shearmix4/: the two October qij studies.
- [x] Vista: venv /home1/08772/jjt277/envs/qij_vista (numpy 1.24.4, scipy 1.13.1, pandas 2.1.1, pyarrow 21,
      h5py 3.7, scikit-learn 1.5.2); the shared work clone at main, installed editable; job generator
      scripts/tacc/make_jobs_vista.sh wrote three sets under $WORK/projects/QIJ_JOINT/jobs: smoke (1 job),
      refs (8), qij (15). Nothing submitted.
- [x] the smoke set dropped (author: a failed real job writes no product; deficient jobs create bloat). In its place a
      local one-draw check of mvt, imf, fp at N = 10⁴ through the driver (2026-11-12, "yes go"): oracle, ij, boot
      B = 20 and qij at ε = 0.01, into the session scratchpad. Oracle, ij, boot: all three fine. qij: mvt fine
      (703 evaluations, within 0.7 ε). **Two defects found, both method-level, awaiting the author's ruling:**
      (1) **fp**: the tree reaches 1280 leaves before the test, and the first test fit (3598 rows: 1319 pilot,
      1280 leaves, 1100 contrasts) fails the interpolant's reproduction assert (residual 1e-5, tolerance 1.5e-7).
      Mechanism measured: 20 pivots of the tail factor below 1e-4 of the largest; the worst rows are mean rows and
      contrasts on leaves of 2–3 points whose kernel columns are nearly dependent; |α| = 2e5 and the residual is
      exactly jitter × α. With exact influences the values are consistent and α stays small (the bed passed fp);
      with the forward difference's truncation (fp's η = 1e-12, step 2e-6) the near-dependent system is slightly
      inconsistent. Remedies: (i) the redundancy rule in the kernel's metric: a row whose Schur-complement pivot
      relative to its own Gram diagonal is ≤ 1e-8 is kept as a measurement, not a constraint (recommended; the same
      rule as core/span.py, applied where the interpolant is actually singular); (ii) central differences on fp
      (truncation 1e-12; doubles the evaluation cost); (iii) a looser reproduction tolerance (hides it).
      (2) **imf at k = 1**: with one split per round (what the driver runs with one worker, since k defaults to the
      pool size) the stop misses on 2 of 6 contrast seeds by 3 and 5 ε; with k = 16 all 6 pass. Measured mechanism:
      at k = 1 the tree stops at 40–52 leaves with one low-mass leaf (0.002–0.054 M☉, 50–143 stars) holding 96–99 %
      of the model's error; that leaf's single contrast came out at 0.02 and 0.055 of its own standard deviation, so
      its term is tiny, and the plug-in quantile, which weights leaves by their measured terms, cannot widen for it
      (METHOD_qij.md §4 approximation 1). The evaluations themselves are exact to 4 decimals (checked at three leaf
      sizes and three steps), so this is the test's accounting, not the estimator. Remedies: (i) k becomes the
      method's own parameter, default 16 as every bed suite ran, independent of the pool (the minimum for the runs);
      (ii) the plug-in weakness itself is a design matter, to be read off the 1000-draw study's miss rate.
- [x] rulings (2026-11-12): (1) "ok do this" → Interpolant.kernel_test built (the redundancy rule in the kernel's
      metric; leaving rows registered before each round's candidates; accepted columns cached for the fit);
      (2) "k can be a parameter, but it should default to however many n_jobs the user specifies" → --k on the driver,
      default --workers; the qij folder carries _k<k>; the Vista qij jobs run 16 workers per draw, 9 draws per node.
      Reruns through the driver at 16 workers, N = 10⁴, ε = 0.01, draw 0: fp K 1281, E 3750, within 0.23 ε (211
      kernel-redundant rows); imf K 75, E 223, within 0.13 ε; mvt K 220, E 741, within 1.31 ε (P_tail; design level);
      sheared Core 1 K 234, E 709, within 0.47 ε. Commit 40f1a4e pushed; Vista clone at it; jobs regenerated.
- [x] time axis on equal footing (author, 2026-11-13: the talk shows cost in evaluations AND wall time). The October
      figure's seconds were single-core-equivalent for both: boot's busy_time (sum of the replicates' own refit times;
      October boot ran 112 workers, 656 s busy for 7 s wall) and qij at one worker (wall = busy). The port recorded only
      wall, so at 16 workers qij would have been flattered up to 16×. Fix: method records busy_s = total_s with the
      pool's wall replaced by the sum of its tasks' refit times (bootstrap's rule); pipeline writes it; the summary's
      seconds for as-built qij = busy_s. Check: Core 1 draw 0 at 16 workers, busy_s 352 s, wall 52 s, E 709, K 234
      (unchanged). Commit 7881404 pushed; Vista pulled; jobs regenerated at it. Slide text: label the axis CPU seconds
      (core-seconds slightly favour qij: boot parallelises perfectly, qij's refit is serial).
- [x] SUBMITTED 2026-11-13 (author's "ok go"): refs 8 jobs + qij 15 jobs, Slurm 1141544–1141566, all pending at
      submit. Vista clone's remote set to github.com/jjtphd/qij-joint (author ran it; the account was renamed).
- [ ] jobs down → rsync $WORK/projects/QIJ_JOINT/runs into Research/QIJ_joint/runs → manifests/README → summaries → figures
- [ ] budget rule note: imf and pareto are 1-D data, so r = 1 and M = 49 at ε = 0.01; fp r = 3, M = 1319;
      mvt clusters in its 10-D coordinates, r = 2, M = 300
- [ ] stale files deleted; vqboot branches (separate, not for the talk)
