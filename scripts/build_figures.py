#!/usr/bin/env python3
"""Copy paper metric JSON into a dated snapshot under outputs/figures.

This repository stores machine-readable table data in results/paper/metrics.
Manuscript figures are produced from the paper sources, which are not in this
code release.
"""
from __future__ import annotations

import argparse
import shutil
import sys
from datetime import date
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _bootstrap import setup_path

ROOT = setup_path()
SRC = ROOT / "results" / "paper" / "metrics"


def main():
    p = argparse.ArgumentParser(description="Snapshot paper metric files")
    p.add_argument("--out", default=None)
    args = p.parse_args()
    out = Path(args.out) if args.out else ROOT / "outputs" / "figures" / date.today().isoformat()
    out.mkdir(parents=True, exist_ok=True)
    copied = []
    for f in sorted(SRC.glob("*.json")):
        shutil.copy2(f, out / f.name)
        copied.append(f.name)
    print("copied", copied, "to", out)


if __name__ == "__main__":
    main()
