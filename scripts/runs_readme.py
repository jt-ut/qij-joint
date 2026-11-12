#!/usr/bin/env python3.9
"""Regenerate `<runs_root>/README.md` from every manifest.json found two
levels down (`<case>/<folder>/manifest.json`).

    python scripts/runs_readme.py /Users/jtaylor/Dropbox/Research/QIJ_joint/runs
"""
import sys

from qij_joint import products


if __name__ == '__main__':
    if len(sys.argv) != 2:
        sys.exit(__doc__)
    print(products.write_readme(sys.argv[1]))
