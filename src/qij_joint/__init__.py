"""qij_joint: variance for estimators without a closed-form influence
function. The method is `qij_joint.method.run` (spec/METHOD_qij.md);
`run_qij` is its per-draw study runner."""
from importlib.util import find_spec as _find_spec

from .bootstrap import Bootstrap
from .pipeline import run_qij

__all__ = ['Bootstrap', 'run_qij']

# The method module is imported only once it exists, so the package
# imports while the port is in progress; an error inside it propagates.
if _find_spec(__name__ + '.method') is not None:
    from .method import run as run_method
    __all__.append('run_method')
