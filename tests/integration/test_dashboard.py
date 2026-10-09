"""Dashboard smoke test: renders every tab from a tiny real pipeline run without errors."""

import os
from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

from stlens.dashboard_data import confusion, list_days, load_day, load_results, metrics_rows
from stlens.datasets.build import DataConfig
from stlens.experiment import ExperimentConfig, run
from stlens.training.trainer import TrainConfig

pytestmark = pytest.mark.integration
APP = Path(__file__).resolve().parents[2] / "dashboard" / "app.py"


@pytest.fixture(scope="module")
def artifacts(tmp_path_factory):
    out = tmp_path_factory.mktemp("artifacts")
    cfg = ExperimentConfig(
        data=DataConfig(n_days=3, n_val_days=1, n_test_days=1, sim_steps_per_day=3_000),
        train=TrainConfig(epochs=1),
        seeds=(0,),
    )
    run(cfg, out, final=True, models=("STLENS_full",), log=lambda _: None)
    return out


def test_pipeline_artifacts_load(artifacts):
    r = load_results(artifacts)
    assert r["final_test_evaluated"]
    assert {"B0_prior", "B1_rule_engine", "B3_xgboost", "STLENS_full"} <= set(r["summary"]["test"])
    (day,) = list_days(artifacts)
    dv = load_day(artifacts, day)
    assert dv.depth.shape[0] == len(dv.prob) == len(dv.y)
    assert confusion(r, "test", "STLENS_full").sum() == r["data"]["test"]["windows"]
    assert metrics_rows(r, "val")


def test_dashboard_renders(artifacts):
    os.environ["STLENS_ARTIFACTS"] = str(artifacts)
    try:
        at = AppTest.from_file(str(APP), default_timeout=120).run()
    finally:
        os.environ.pop("STLENS_ARTIFACTS", None)
    assert not at.exception, [e.value for e in at.exception]
    assert len(at.tabs) == 4
    assert "SYNTHETIC" in at.warning[0].value
