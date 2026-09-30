"""The only module that knows the product layout (Section 2): one
directory per (dataset, estimator, N), one subdirectory per method, one
scalar-row parquet and zero or more array parquets per draw, plus a
JSON-lines run log.
"""
from __future__ import annotations

import json
import os
import platform
import socket
import subprocess
from datetime import datetime
from typing import Optional

import pandas as pd
import pyarrow.dataset as ds


def method_dir(out_dir: str, dataset: str, estimator: str, N: int, method: str,
               tag: str = '') -> str:
    """The directory `<out_dir>/<dataset>_<estimator>_N<N>/<folder>/`,
    where `folder` is `method` if `tag == ''`, else `f'{method}_{tag}'`
    -- `dir_tag` builds `tag` from a run's configuration, so different
    configurations of a method land in different folders."""
    folder = method if tag == '' else f'{method}_{tag}'
    return os.path.join(out_dir, f'{dataset}_{estimator}_N{N}', folder)


# The S=1000 cloudfil study's fixed qij choices (spec/QIJ_mods_waves.md);
# `dir_tag` names only the settings that differ from these.
QIJ_CANONICAL = dict(survey='moments', quantized_start='full-data', gptrend='quadratic',
                      gpwidth='global', ivqbins='joint', refine_schedule='rounds',
                      sigma_points=True, M_X=None)


def _num(x) -> str:
    return format(float(x), 'g')


def dir_tag(method: str, config: dict) -> str:
    """The method-folder suffix a configuration maps to (Section 2):
    configurations must never share a folder, so `run.py` derives this
    from `config` rather than letting `--out` be used to separate them.
    `qij` -> `<pilot>_eps<eps>`, e.g. `gp_eps0.02`, then one suffix per
    setting that differs from `QIJ_CANONICAL`, in a fixed order:
    `survey`, `quantized_start`, `gptrend`, `gpwidth`, `ivqbins`,
    `refine_schedule`, `sigma_points` (named only when False, as
    `_nosigma`), `M_X` (named only when not None, as `_MX<int>`), then
    `check_rule` (spec/QIJ_joint_check_measured_spec.md 3) as a
    non-canonical suffix, named only when `'measured'`, as `_measured`
    -- so a `check_rule='predicted'` run keeps today's folder name. `boot`
    -> `''` (`B` is not in the name -- replicates are a deterministic
    prefix, so a smaller B is read from a larger run's folder; `B` is
    still recorded in config.json and so still guarded). `ijfd` -> `''`,
    or `'curv'` under `point_curvature`. `qijdt` -> `eps<eps>`. `qijt`
    -> `MX<M_X>_B<budget>_W<budget_win>_Q<budget_quad>`. `oracle`, `ij`,
    anything else -> `''`."""
    if method == 'qij':
        tag = f"{config['pilot']}_eps{_num(config['eps'])}"
        if config['survey'] != QIJ_CANONICAL['survey']:
            tag += f"_survey{config['survey']}"
        if config['quantized_start'] != QIJ_CANONICAL['quantized_start']:
            tag += f"_start{config['quantized_start']}"
        if config['gptrend'] != QIJ_CANONICAL['gptrend']:
            tag += f"_trend{config['gptrend']}"
        if config['gpwidth'] != QIJ_CANONICAL['gpwidth']:
            tag += f"_width{config['gpwidth']}"
        if config['ivqbins'] != QIJ_CANONICAL['ivqbins']:
            tag += f"_bins{config['ivqbins']}"
        if config['refine_schedule'] != QIJ_CANONICAL['refine_schedule']:
            tag += f"_sched{config['refine_schedule']}"
        if config['sigma_points'] != QIJ_CANONICAL['sigma_points']:
            tag += '_nosigma'
        if config['M_X'] != QIJ_CANONICAL['M_X']:
            tag += f"_MX{int(config['M_X'])}"
        if config['check_rule'] == 'measured':
            tag += '_measured'
        return tag
    if method == 'boot':
        return ''
    if method == 'ijfd':
        return 'curv' if config.get('point_curvature') else ''
    if method == 'qijdt':
        return f"eps{_num(config['eps'])}"
    if method == 'qijt':
        return (f"MX{config['M_X']}_B{config['budget']}_"
                f"W{config['budget_win']}_Q{config['budget_quad']}")
    return ''


def ensure_config(md: str, config: dict) -> None:
    """Write `md/config.json` if absent (atomically, like `_atomic_write`:
    `config.json.tmp` then `os.replace`); if present, load it and compare
    to `config` after a JSON round-trip of `config` (so tuples/lists and
    ints/floats compare as stored), raising `RuntimeError` naming `md`
    and listing each differing key -- including one present on only one
    side -- as `key: stored=<v> requested=<v>` if they differ, and
    telling the caller to choose a different --out study root or fix the
    arguments. Many single-draw processes may start against a new folder
    at once; each writes its own `config.json.<pid>.tmp`, so racing
    creators never share a temp file, and they write identical content,
    so whichever `os.replace` lands last is harmless."""
    os.makedirs(md, exist_ok=True)
    path = os.path.join(md, 'config.json')
    requested = json.loads(json.dumps(config))
    if not os.path.exists(path):
        tmp = f'{path}.{os.getpid()}.tmp'
        with open(tmp, 'w') as f:
            f.write(json.dumps(config, indent=1, sort_keys=True))
        os.replace(tmp, path)
        return
    with open(path) as f:
        stored = json.load(f)
    if 'check_rule' in requested and 'check_rule' not in stored:
        # A qij folder from before this build recorded no `check_rule`
        # (spec/QIJ_joint_check_measured_spec.md 3): read as 'predicted',
        # so today's `check_rule='predicted'` runs still match it. Only a
        # configuration that carries the key (qij) gets the default.
        stored = dict(stored, check_rule='predicted')
    if stored != requested:
        keys = sorted(set(stored) | set(requested))
        diffs = '; '.join(
            f'{k}: stored={stored.get(k)} requested={requested.get(k)}'
            for k in keys if stored.get(k) != requested.get(k))
        raise RuntimeError(
            f'{md}: recorded configuration in config.json differs from the '
            f'requested configuration ({diffs}); choose a different --out '
            f'study root or fix the arguments.')


def _scalar_path(md: str, s: int) -> str:
    return os.path.join(md, f's{s:05d}.parquet')


def _array_path(md: str, s: int, kind: str) -> str:
    return os.path.join(md, f's{s:05d}.{kind}.parquet')


def _cpu_model() -> str:
    """The running machine's CPU model name: `sysctl` on macOS, the first
    `model name` line of `/proc/cpuinfo` on Linux, `platform.processor()`
    otherwise or if neither yields one."""
    system = platform.system()
    if system == 'Darwin':
        return subprocess.check_output(
            ['sysctl', '-n', 'machdep.cpu.brand_string']).decode().strip()
    if system == 'Linux':
        with open('/proc/cpuinfo') as f:
            for line in f:
                if line.startswith('model name'):
                    return line.split(':', 1)[1].strip()
    return platform.processor()


def is_done(md: str, s: int) -> bool:
    """A draw is done once its scalar row exists -- the done marker."""
    return os.path.exists(_scalar_path(md, s))


def _atomic_write(path: str, df: pd.DataFrame) -> None:
    tmp = path + '.tmp'
    df.to_parquet(tmp, index=False)
    os.replace(tmp, path)


def write_draw(md: str, s: int, row: dict, arrays: Optional[dict] = None) -> None:
    """Write one draw's files: every array first, the scalar row last
    (the done marker), each via `<name>.tmp` then `os.replace` (Section 2)."""
    os.makedirs(md, exist_ok=True)
    for kind, df in (arrays or {}).items():
        _atomic_write(_array_path(md, s, kind), df)
    _atomic_write(_scalar_path(md, s), pd.DataFrame([row]))


def append_log(md: str, argv: list, params: dict, written: int, skipped: int) -> None:
    """One JSON line in `runs.log`: time, host, CPU, argv, parameters,
    draws written/skipped (Section 2)."""
    os.makedirs(md, exist_ok=True)
    record = {
        'time': datetime.now().isoformat(timespec='seconds'),
        'host': socket.gethostname(), 'cpu_model': _cpu_model(),
        'cpu_count': os.cpu_count(), 'argv': list(argv), 'params': params,
        'written': written, 'skipped': skipped,
    }
    with open(os.path.join(md, 'runs.log'), 'a') as f:
        f.write(json.dumps(record) + '\n')


def collect(run_dir: str, dataset: str, estimator: str, N: int, method: str,
            tag: str = '') -> pd.DataFrame:
    """Every draw's scalar row, as one pyarrow dataset over the method's
    `s*.parquet` files, sorted by `s`."""
    md = method_dir(run_dir, dataset, estimator, N, method, tag)
    if not os.path.isdir(md):
        return pd.DataFrame()
    paths = sorted(
        os.path.join(md, name) for name in os.listdir(md)
        if name.endswith('.parquet') and name.count('.') == 1
    )
    if not paths:
        return pd.DataFrame()
    table = ds.dataset(paths, format='parquet').to_table()
    return table.to_pandas().sort_values('s').reset_index(drop=True)


def collect_array(run_dir: str, dataset: str, estimator: str, N: int, method: str,
                   kind: str, draws: Optional[list] = None, tag: str = '') -> pd.DataFrame:
    """Every draw's `kind` array, concatenated with an `s` column added,
    optionally restricted to `draws`."""
    md = method_dir(run_dir, dataset, estimator, N, method, tag)
    if not os.path.isdir(md):
        return pd.DataFrame()
    frames = []
    for name in sorted(os.listdir(md)):
        if not name.endswith(f'.{kind}.parquet'):
            continue
        s = int(name[1:name.index('.')])
        if draws is not None and s not in draws:
            continue
        df = pd.read_parquet(os.path.join(md, name))
        df.insert(0, 's', s)
        frames.append(df)
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)
