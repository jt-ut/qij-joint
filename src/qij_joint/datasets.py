"""The four paper draws (pareto, mvt, fp, imf), the MVT VQ transform, the
`mix11` demo mixture (spec/method_notes.md section 5), and the
`cloudfil_G_B6_P3_v1` demo (spec/QIJ_mods_waves.md A11); neither demo is
one of the paper's four.

Parametric draws (`pareto`, `mvt`, `mix11`, `cloudfil_G_B6_P3_v1`) sample
the named law directly. Population draws (`fp`, `imf`) resample with
replacement from the pool files shipped under `data/`, loaded once per
process (E5's named exception for constant data read from the package's
own files) and never re-read per draw.
"""

import functools
import pathlib
from typing import Callable, Tuple

import h5py
import numpy as np

_DATA_DIR = pathlib.Path(__file__).parent / 'data'

PARETO_ALPHA = 2.0
PARETO_X_MIN = 1.0

MVT_D = 10
MVT_NU = 5.0

# Chabrier's own break and the shipped pool's minimum mass, fixed here
# exactly as the pool is fixed for `imf`: never a draw's own minimum,
# which would make the estimator's normalization depend on which points
# a resample happened to keep.
CHABRIER_M_B = 1.0
CHABRIER_M_MIN = 0.0017582750879228115
CHABRIER_BOUNDS = ((0.01, 5.0), (0.02, 5.0), (0.05, 10.0))


def pareto(N: int, seed: int) -> np.ndarray:
    """N i.i.d. draws from Pareto(alpha=2.0, x_min=1.0). Returns (N,)."""
    rng = np.random.default_rng(seed)
    U = rng.uniform(0.0, 1.0, size=N)
    return PARETO_X_MIN * (1.0 - U) ** (-1.0 / PARETO_ALPHA)


def mvt(N: int, seed: int) -> np.ndarray:
    """N i.i.d. draws from a d=10, nu=5.0 multivariate t. Returns (N, 10)."""
    rng = np.random.default_rng(seed)
    Z = rng.normal(size=(N, MVT_D))
    V = rng.chisquare(MVT_NU, size=(N, 1))
    return Z / np.sqrt(V / MVT_NU)


# Chacon mixture 11's own means/covariances (structsynhd's loaded 2-D
# order), with the talk demo's weights --
# the level sets stay close to the file's own, every component keeping
# at least 50 points at N = 2000.
MIX11_MIXNUM = 11
MIX11_WEIGHTS = np.array(
    [0.3775, 0.3775, 0.025, 0.035, 0.050, 0.025, 0.050, 0.035, 0.025])


def mix11(N: int, seed: int) -> np.ndarray:
    """N draws from Chacon mixture 11 (`ChaconMixGenerator`, its own
    means/covariances and component order), weights replaced by
    `MIX11_WEIGHTS` -- the talk's demo mixture, not one of the paper's
    four draws. `expand_dimension` is never called (GMM2D is 2-D), so
    overriding the instance's own `weights` before `sample` is enough.
    structsynhd is imported here, not at module level: only mix11 needs
    it, and the package must import where it is not installed."""
    from structsynhd import ChaconMixGenerator
    gen = ChaconMixGenerator(MIX11_MIXNUM)
    gen.weights = MIX11_WEIGHTS
    X, _ = gen.sample(N, random_state=seed)
    return X


@functools.lru_cache(maxsize=1)
def _cloudfil_G_B6_P3_v1_pop() -> dict:
    """The packaged `cloudfil_G_B6_P3_v1.npz` (weights, means, covs),
    loaded once per process (E5's named exception for constant data read
    from the package's own files)."""
    with np.load(_DATA_DIR / 'cloudfil_G_B6_P3_v1.npz', allow_pickle=True) as data:
        return dict(weights=data['weights'].astype(float),
                    means=data['means'].astype(float),
                    covs=data['covs'].astype(float))


def cloudfil_G_B6_P3_v1(N: int, seed: int) -> np.ndarray:
    """N draws from the `cloudfil_G_B6_P3_v1` mixture (spec/QIJ_mods_waves.md
    A11): a multinomial component count from the file's weights, then a
    multivariate normal draw per component. Returns (N, 2)."""
    rng = np.random.default_rng(seed)
    pop = _cloudfil_G_B6_P3_v1_pop()
    counts = rng.multinomial(N, pop['weights'])
    X = np.empty((N, 2))
    start = 0
    for k, n_k in enumerate(counts):
        if n_k == 0:
            continue
        X[start:start + n_k] = rng.multivariate_normal(pop['means'][k], pop['covs'][k], size=n_k)
        start += n_k
    return X


@functools.lru_cache(maxsize=1)
def _fp_pool() -> np.ndarray:
    data = np.load(_DATA_DIR / 'fp_sdss.npz')
    return data['fp_data']   # (76997, 3): [log_sigma, log_I_e, log_R_half]


def fp(N: int, seed: int) -> np.ndarray:
    """N galaxies drawn with replacement from the 76,997-galaxy SDSS pool."""
    rng = np.random.default_rng(seed)
    pool = _fp_pool()
    idx = rng.choice(len(pool), size=N, replace=True)
    return pool[idx]


@functools.lru_cache(maxsize=1)
def _imf_pool() -> np.ndarray:
    with h5py.File(_DATA_DIR / 'stars.h5', 'r') as f:
        return f['data/BH_Mass'][()]   # (19857,)


def imf(N: int, seed: int) -> np.ndarray:
    """N stellar masses drawn with replacement from the 19,857-star pool."""
    rng = np.random.default_rng(seed)
    pool = _imf_pool()
    idx = rng.choice(len(pool), size=N, replace=True)
    return pool[idx]


def mvt_vq_transform(X: np.ndarray) -> Tuple[np.ndarray, Callable[[np.ndarray], np.ndarray]]:
    """Per-coordinate mean/sd whitening of X into the quantizer's own
    coordinates. Returns (Z, inverse); `inverse` closes over this call's
    own mean/std and maps a whitened prototype position back to X's
    native coordinates."""
    mean = X.mean(axis=0)
    std = X.std(axis=0)
    Z = (X - mean) / std

    def inverse(W: np.ndarray) -> np.ndarray:
        return W * std + mean

    return Z, inverse
