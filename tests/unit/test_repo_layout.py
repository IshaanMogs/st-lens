"""Phase 0 gate: repository artefacts required before Phase 1 exist and are wired up."""

import ast
import tomllib
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
SRC = REPO_ROOT / "src" / "stlens"


@pytest.mark.parametrize(
    "relpath",
    [
        "docs/ST-LENS_Technical_Specification.md",
        "docs/related_work.md",
        "docs/data_source_binance.md",
        "docs/decisions.md",
        ".github/workflows/ci.yml",
        "configs/data",
        "configs/features",
        "configs/labels",
        "configs/models",
        "configs/experiments",
        "tests/leakage",
        "tests/integration",
        "tests/fixtures",
    ],
)
def test_required_path_exists(relpath):
    assert (REPO_ROOT / relpath).exists(), relpath


def test_ci_runs_ruff_and_pytest_on_python_311():
    ci = (REPO_ROOT / ".github/workflows/ci.yml").read_text(encoding="utf-8")
    assert 'python-version: "3.11"' in ci
    assert "ruff check" in ci
    assert "ruff format --check" in ci
    assert "run: pytest" in ci


def test_pyproject_targets_python_311():
    config = tomllib.loads((REPO_ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    assert config["project"]["requires-python"] == ">=3.11"
    assert config["tool"]["ruff"]["target-version"] == "py311"


def test_data_directory_is_git_ignored():
    ignored = (REPO_ROOT / ".gitignore").read_text(encoding="utf-8").splitlines()
    assert "/data/" in ignored


def _imported_modules(path: Path) -> set[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    names: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            names.update(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module and node.level == 0:
            names.add(node.module.split(".")[0])
    return names


@pytest.mark.parametrize(
    "path", sorted(SRC.rglob("*.py")), ids=lambda p: p.relative_to(SRC).as_posix()
)
def test_library_never_imports_notebooks(path):
    # Spec section N: notebooks are for exploration only and are never imported.
    assert "notebooks" not in _imported_modules(path)
