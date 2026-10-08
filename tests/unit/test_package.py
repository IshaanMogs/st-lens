"""Phase 0 gate: the installed package imports and has the layout the spec requires."""

import importlib
import importlib.metadata
import sys
from pathlib import Path

import pytest

import stlens

# Section N of the specification: one package per pipeline stage.
SPEC_SUBPACKAGES = [
    "ingestion",
    "schemas",
    "book",
    "features",
    "labeling",
    "datasets",
    "models",
    "models.classical",
    "models.baselines",
    "models.stlens_net",
    "training",
    "evaluation",
    "tracking",
    "inference",
    "scoring",
    "explain",
    "utils",
]


def test_python_version_is_supported():
    assert sys.version_info >= (3, 11)


def test_version_matches_installed_metadata():
    assert stlens.__version__ == importlib.metadata.version("stlens")


def test_package_is_imported_from_src_layout():
    # Guards against importing a stray copy from the repo root instead of the installed package.
    assert Path(stlens.__file__).parent.parent.name == "src"


@pytest.mark.parametrize("name", SPEC_SUBPACKAGES)
def test_spec_subpackage_imports(name):
    module = importlib.import_module(f"stlens.{name}")
    assert module.__doc__, f"stlens.{name} needs a docstring stating its purpose"
