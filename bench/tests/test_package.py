import tomllib
from pathlib import Path

import inferrail_bench

REPO_ROOT = Path(__file__).resolve().parents[2]


def test_version_matches_cargo_workspace():
    workspace = tomllib.loads((REPO_ROOT / "Cargo.toml").read_text())
    assert inferrail_bench.__version__ == workspace["workspace"]["package"]["version"]
