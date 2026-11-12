"""The only module that knows the product layout (Section 2): under the
runs root, one case directory per (dataset, N), named `<dataset>_N<N>`,
and inside it one folder per method configuration (`oracle`,
`boot_B<B>`, `ij_<estimator>`, `qij_<estimator>_eps<eps>[_level<level>][_M<M>]`).
Each folder holds one scalar-row parquet and zero or more array parquets
per draw, the configuration manifest config.json, a JSON-lines run log
runs.log, and the provenance summary manifest.json. The runs root also
carries a README.md generated from every manifest.json beneath it.
"""
from __future__ import annotations

import glob
import json
import os
import re
import platform
import socket
import subprocess
from datetime import datetime
from typing import Optional

import pandas as pd
import pyarrow.dataset as ds


def method_dir(out_dir: str, dataset: str, estimator: str, N: int, method: str,
               tag: str = '') -> str:
    """The directory `<out_dir>/<dataset>_N<N>/<folder>/`. The folder name
    is the method, then the estimator for `ij` and `qij` only (the oracle
    and the bootstrap record theirs in config.json), then `tag` when it
    is not empty, joined by underscores: `oracle`, `boot_B5000`, `ij_p1`,
    `qij_p1_eps0.01`. `dir_tag` builds `tag` from a run's configuration,
    so different configurations of a method land in different folders."""
    parts = [method]
    if method in ('ij', 'qij'):
        parts.append(estimator)
    if tag != '':
        parts.append(tag)
    return os.path.join(out_dir, f'{dataset}_N{N}', '_'.join(parts))


def _num(x) -> str:
    return format(float(x), 'g')


def dir_tag(method: str, config: dict) -> str:
    """The method-folder suffix a configuration maps to (Section 2):
    configurations must never share a folder, so `run.py` derives this
    from `config` rather than letting `--out` be used to separate them.
    `qij` -> `eps<eps>`, then `_level<level>` when the level is not
    0.84 and `_M<M>` when the pilot size is given (not the method's
    own). `boot` -> `B<B>`, from `config['B']`. `oracle`, `ij`,
    anything else -> `''`."""
    if method == 'qij':
        tag = f"eps{_num(config['eps'])}"
        if config.get('level', 0.84) != 0.84:
            tag += f"_level{_num(config['level'])}"
        if config.get('M') is not None:
            tag += f"_M{int(config['M'])}"
        # k, the cap on splits per round, defaults to the number of workers the evaluations run on (the author's rule,
        # 2026-11-12); the folder name carries the value that applies, so two runs with different k never share a folder
        k_eff = config.get('k') if config.get('k') is not None else config.get('workers')
        if k_eff is not None:
            tag += f"_k{int(k_eff)}"
        return tag
    if method == 'boot':
        return f"B{int(config['B'])}"
    return ''


def code_commit() -> str:
    """The git commit of the package source: the 12-character short hash
    of HEAD in the repository holding this module, with `-dirty` appended
    when `git status --porcelain` reports changes to tracked files, or
    'unknown' when git is unavailable or the source is not in a
    repository."""
    here = os.path.dirname(os.path.abspath(__file__))
    try:
        head = subprocess.check_output(
            ['git', '-C', here, 'rev-parse', '--short=12', 'HEAD'],
            stderr=subprocess.DEVNULL).decode().strip()
        status = subprocess.check_output(
            ['git', '-C', here, 'status', '--porcelain', '--untracked-files=no'],
            stderr=subprocess.DEVNULL).decode().strip()
    except (OSError, subprocess.CalledProcessError):
        return 'unknown'
    return head + ('-dirty' if status else '')


def ensure_config(md: str, config: dict, ignore: tuple = ()) -> None:
    """Write `md/config.json` if absent (atomically, like `_atomic_write`:
    `config.json.<pid>.tmp` then `os.replace`); if present, load it and
    compare to `config` after a JSON round-trip of `config` (so
    tuples/lists and ints/floats compare as stored), raising
    `RuntimeError` naming `md` and listing each differing key --
    including one present on only one side -- as `key: stored=<v>
    requested=<v>`, and telling the caller to fix the arguments (or, when
    only the commit differs, to pass --allow-commit-mismatch). Many single-draw processes may
    start against a new folder at once; each writes its own temp file,
    and they write identical content, so whichever `os.replace` lands
    last is harmless. The comparison is exact: a folder written by a
    method removed before the port (its config carries keys today's
    config never sets) does not match, so new products never land
    beside old ones. Keys named in `ignore` are left out of the
    comparison (the driver passes `('code_commit',)` only when the
    author asks to continue a folder across a commit)."""
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
    stored_cmp = {k: v for k, v in stored.items() if k not in ignore}
    requested_cmp = {k: v for k, v in requested.items() if k not in ignore}
    if stored_cmp != requested_cmp:
        keys = sorted(set(stored_cmp) | set(requested_cmp))
        diffs = '; '.join(
            f'{k}: stored={stored.get(k)} requested={requested.get(k)}'
            for k in keys if stored.get(k) != requested.get(k))
        raise RuntimeError(
            f'{md}: recorded configuration in config.json differs from the '
            f'requested configuration ({diffs}). A folder holds one configuration '
            f'and one code version: fix the arguments, or, if only code_commit '
            f'differs and the commit did not touch the method, pass '
            f'--allow-commit-mismatch.')


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
    draws written/skipped, and the code commit (Section 2)."""
    os.makedirs(md, exist_ok=True)
    record = {
        'time': datetime.now().isoformat(timespec='seconds'),
        'host': socket.gethostname(), 'cpu_model': _cpu_model(),
        'cpu_count': os.cpu_count(), 'argv': list(argv), 'params': params,
        'written': written, 'skipped': skipped, 'code_commit': code_commit(),
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


def _draw_ranges(draws: list) -> str:
    """A compact string for a sorted list of draw indices: runs of
    consecutive draws as `a..b`, single draws as `a`, joined by commas
    (for example "0..43,88..131"); '' when there are none."""
    parts = []
    i = 0
    while i < len(draws):
        j = i
        while j + 1 < len(draws) and draws[j + 1] == draws[j] + 1:
            j += 1
        parts.append(str(draws[i]) if i == j else f'{draws[i]}..{draws[j]}')
        i = j + 1
    return ','.join(parts)


def write_manifest(md: str) -> dict:
    """Rewrite `md/manifest.json`, the folder's provenance summary, from
    what is on disk: config.json, runs.log (if present) and the
    s<draw>.parquet scalar rows. It records the case folder name, the
    folder, dataset, estimator, N and method (from the configuration),
    the configuration itself, how many draws are present and which, the
    times of the first and last runs.log entries, the hosts and code
    commits seen in runs.log, and the first entry's argv. Without a
    runs.log the times, hosts and argv are null and the commit list is
    empty. Written atomically; the summary is also returned."""
    with open(os.path.join(md, 'config.json')) as f:
        config = json.load(f)
    draws = sorted(int(m.group(1)) for m in
                   (re.fullmatch(r's(\d+)\.parquet', n) for n in os.listdir(md))
                   if m)
    records = []
    log = os.path.join(md, 'runs.log')
    if os.path.exists(log):
        with open(log) as f:
            records = [json.loads(line) for line in f if line.strip()]
    manifest = {
        'case': os.path.basename(os.path.dirname(os.path.abspath(md))),
        'folder': os.path.basename(os.path.abspath(md)),
        'dataset': config.get('dataset'), 'estimator': config.get('estimator'),
        'N': config.get('N'), 'method': config.get('method'),
        'config': config,
        'n_draws': len(draws), 'draws': _draw_ranges(draws),
        'created': records[0]['time'] if records else None,
        'last_write': records[-1]['time'] if records else None,
        'hosts': sorted({r['host'] for r in records if 'host' in r}) if records else None,
        'code_commits': sorted({r['code_commit'] for r in records if 'code_commit' in r}),
        'argv_first': records[0].get('argv') if records else None,
    }
    path = os.path.join(md, 'manifest.json')
    tmp = f'{path}.{os.getpid()}.tmp'
    with open(tmp, 'w') as f:
        f.write(json.dumps(manifest, indent=1))
    os.replace(tmp, path)
    return manifest


def write_readme(runs_root: str) -> str:
    """Write `<runs_root>/README.md` from every manifest.json found two
    levels down (`<case>/<folder>/manifest.json`): a header saying the
    file is generated by this function and when, the layout rule, and
    one table per case folder listing each method folder's method,
    estimator, number of draws, draw ranges, first and last write,
    hosts and code commits. Returns the path written."""
    by_case = {}
    for path in sorted(glob.glob(os.path.join(runs_root, '*', '*', 'manifest.json'))):
        with open(path) as f:
            m = json.load(f)
        case = os.path.basename(os.path.dirname(os.path.dirname(path)))
        by_case.setdefault(case, []).append(m)

    def hosts_cell(hosts):
        """Hosts summarised by system: a TACC node name like c512-051.stampede3.tacc.utexas.edu counts
        towards 'stampede3'; anything else towards its own name; e.g. 'stampede3 (23 nodes)'."""
        if not hosts:
            return ''
        systems = {}
        for h in hosts:
            parts = h.split('.')
            system = parts[1] if len(parts) >= 3 and parts[0][:1] == 'c' and parts[0][1:].replace('-', '').isdigit() else parts[0]
            systems[system] = systems.get(system, 0) + 1
        return ', '.join(f'{s} ({n} node{"s" if n != 1 else ""})' if n > 1 else s for s, n in sorted(systems.items()))

    def cell(c, v):
        if c == 'hosts':
            return hosts_cell(v)
        if v is None:
            return ''
        if isinstance(v, list):
            return ', '.join(str(x) for x in v)
        return str(v)

    cols = ['folder', 'method', 'estimator', 'n_draws', 'draws', 'created',
            'last_write', 'hosts', 'code_commits', 'notes']
    lines = [
        '# Runs',
        '',
        f'Generated by `qij_joint.products.write_readme` on '
        f'{datetime.now().isoformat(timespec="seconds")}; do not edit by hand.',
        '',
        'Each case folder is `<dataset>_N<N>`, and inside it each method '
        'configuration has its own folder: `oracle`, `boot_B<B>`, '
        '`ij_<estimator>`, or `qij_<estimator>_eps<eps>[_level<level>][_M<M>]`. '
        'A folder holds one `s<draw>.parquet` per draw (plus '
        '`s<draw>.<kind>.parquet` arrays), its `config.json`, `runs.log` '
        'and `manifest.json`.',
    ]
    for case in sorted(by_case):
        lines += ['', f'## {case}', '',
                  '| ' + ' | '.join(cols) + ' |',
                  '|' + '---|' * len(cols)]
        for m in sorted(by_case[case], key=lambda m: m.get('folder', '')):
            lines.append('| ' + ' | '.join(cell(c, m.get(c)) for c in cols) + ' |')
    path = os.path.join(runs_root, 'README.md')
    tmp = f'{path}.{os.getpid()}.tmp'
    with open(tmp, 'w') as f:
        f.write('\n'.join(lines) + '\n')
    os.replace(tmp, path)
    return path
