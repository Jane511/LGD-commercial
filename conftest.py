"""
Root conftest.py — makes the repo root importable as a package root for all tests.

Without this, test files that do not manually insert the repo root into sys.path
(e.g. tests/test_aps113_compliance.py) cannot resolve `from src.* import ...`.
"""
import sys
from pathlib import Path

# Insert repo root so `import src.*` works from any test file.
sys.path.insert(0, str(Path(__file__).resolve().parent))
