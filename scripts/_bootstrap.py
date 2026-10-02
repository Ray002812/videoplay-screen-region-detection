"""Locate the repository root and put src/ plus the local Ultralytics tree on sys.path."""
from __future__ import annotations

import sys
from pathlib import Path


def repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def setup_path() -> Path:
    root = repo_root()
    src = str(root / "src")
    ultra = str(root / "third_party" / "ultralytics")
    if src not in sys.path:
        sys.path.insert(0, src)
    if ultra not in sys.path:
        sys.path.insert(0, ultra)
    return root
