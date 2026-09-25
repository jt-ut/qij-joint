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


def method_dir(out_dir: str, dataset: str, estimator: str, N: int, method: str) -> str:
    """The directory `<out_dir>/<dataset>_<estimator>_N<N>/<method>/`."""
    return os.path.join(out_dir, f'{dataset}_{estimator}_N{N}', method)


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


def collect(run_dir: str, dataset: str, estimator: str, N: int, method: str) -> pd.DataFrame:
    """Every draw's scalar row, as one pyarrow dataset over the method's
    `s*.parquet` files, sorted by `s`."""
    md = method_dir(run_dir, dataset, estimator, N, method)
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
                   kind: str, draws: Optional[list] = None) -> pd.DataFrame:
    """Every draw's `kind` array, concatenated with an `s` column added,
    optionally restricted to `draws`."""
    md = method_dir(run_dir, dataset, estimator, N, method)
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
