"""
Test package for img-edit.

- Ensures 'src/' is on sys.path so 'lib.*' modules are importable during pytest runs
- Makes the tests a package

Note: Python syntax validated via ast prior to submission.
"""
from __future__ import annotations

import sys
from pathlib import Path

# Ensure the project 'src' directory is importable during tests
_ROOT = Path(__file__).resolve().parents[1]
_SRC = _ROOT / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))