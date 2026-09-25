# QIJ glossary and notation

The single source for the spec, the code and any downstream tables or
figures. Where a symbol is given, prose uses the name and equations the
symbol. Subscripts: i for data points, j for X-VQ prototypes, k for
I-VQ bins.

## Terms

| term | symbol | meaning |
|---|---|---|
| data space | X | the space the data live in, in the quantizer's coordinates (whitened) |
| influence space | I | the axis of influence values |
| X-VQ | | the vector quantizer of the data (k-means, centroids); stage 1 |
| I-VQ | | the one-dimensional vector quantizer of the initial influence estimate; stage 2, on which the finite differences are taken |
| prototype | w_j | a prototype of the X-VQ |
| receptive field | RF_j | the data points nearest prototype j; its mass p_j = n_j/N |
| connectivity graph | CADJ, CONN | the X-VQ's adjacency from second-nearest prototypes (directed, symmetric) |
| bin | | a cell of the I-VQ: the data points whose initial influence estimate lies in one interval; mass p_k = n_k/N |
| prototype influence | I_j | the finite difference of T along prototype j's mass, T evaluated on the prototypes as a weighted dataset; mass-centred |
| initial influence estimate | psi0(x) | the Gaussian-process model through the I_j, evaluated at any point |
| its uncertainty | sigma_i | the posterior standard deviation of psi0 at x_i (a diagnostic; not used by the refinement's split test) |
| within-bin posterior variance | v_k | the variance of the model's error inside bin k about the error common to the bin: mean of the bin's posterior covariance diagonal minus its grand mean; prices the refinement's proposals |
| declared noise floor | n_c^2 | the evaluation-noise level the declared eta implies for the prototype influences, 2 eta^2 theta_Q,c^2 * median_j(1/t_j^2); the influence model's noise ratio is not fitted below it |
| bin influence | U_k | the finite difference of T along bin k's mass on the full data: the average influence over the bin |
| refined influence estimate | psi_hat(x_i) | U_{k(i)} + rho*(psi0(x_i) - psi0_bar,k): the per-point refined influence estimate |
| FD-to-prediction scale | rho | sqrt(V_btw / sum_k p_k psi0_bar,k^2): the ratio of finite-differenced to predicted between-bin spread |
| between-bin term | V_btw | (1/N) sum_k p_k U_k^2: the part of the variance the bin influences account for; finite-differenced; a lower bound on V_tot |
| within-bin term | V_win | (1/N) sum_k p_k Var_k(psi): the part inside the bins; not observable from bin influences |
| predicted within-bin term | V_win_hat | rho^2 (1/N) sum_k p_k gamma_k [Var_k(psi0) + v_k] over bins with more than one point; v_k the within-bin posterior variance of psi0; gamma_k = min(1, g/g_hat) of the split that created the bin, 1 if never split |
| variance estimate | V_tot_hat | V_btw + V_win_hat |
| QIJ interval | h^QIJ_q | the q*100% normal interval theta_hat +/- z*sqrt(V_btw) on the measured between-bin variance; V_tot_hat is a diagnostic, not an interval input |
| bootstrap interval | h^boot_q | the percentile interval from B replicates |
| declared accuracy | eta | the relative accuracy of one estimator evaluation, declared by the user |
| tolerance | eps | the share of the variance the user accepts leaving inside the bins (default 0.01) |
| difference step | delta | the relative change of the mass being moved: (3 eta)^(1/3) for the central stencil, delta_f = 2*sqrt(eta) for a single forward step; for a bin or prototype of mass p the weight parameter is t = delta*p/(1 - p), so its mass becomes p(1 + delta) |
| prototype count | M_X | ceil(sqrt((1 + 2 q M_ref) N / 2)), floor 20, cap N/2 |
| initial bin count | M_I | ceil(sqrt(2.7/eps)) = 17 at the default |
| reference bin count | M_ref | the same number, as it enters the prototype-count rule |
| final bin count | L | the number of bins after refinement |
| expected gain | g_hat_k | the predicted increase in V_btw from splitting bin k |
| gain | g_k | the finite-differenced increase in V_btw the split produced |
| gain-driven refinement | | the loop that splits the bin with the largest expected gain and stops when gains fall below the share tau = eps V_btw / L |
| level split | | a split of a bin by psi0 (two-means on the bin's values), proposed when Var_k(psi0) > v_k; expected gain rho^2*(between-children variance of psi0)/N |
| adjacency split | | a split of a bin by data-space adjacency: points whose second-nearest prototype has a higher prototype influence than their nearest, against the rest; proposed otherwise; expected gain rho^2 p_k v_k/N |
| closing rule | | a lineage closes after two consecutive splits whose realized gain is below tau; one such split cannot tell a constant-influence bin from one whose variation was divided evenly between its children |
| tail probability | S_99 | the survival function at the population's 0.99 quantile (of x for the Pareto, of radius for the MVT) |
| FP scatter | s | the intrinsic scatter about the plane |
| estimator evaluation | T(X, omega) | one run of the estimator on the data with weights omega >= 0, sum(omega) = N |
| normalized rows | | sum over evaluations of (rows in the evaluation)/N: full-data-evaluation equivalents |

## Dropped from this package

a_bca (the BCa acceleration constant), B_hat (the between-bin bias
term and the second difference Delta^2 T_k it was built from), the
FITC sparse influence model, and the natural-support interval clip are
not part of this package; where they appear in a derivation elsewhere,
they are historical, not current.
