"""ST-LENS research dashboard (read-only). Run: ``streamlit run dashboard/app.py``.

Reads artefacts from ``$STLENS_ARTIFACTS`` (default ``artifacts/``). Never trains or scores.
"""

import os
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

from stlens.dashboard_data import (
    confusion,
    downsample,
    list_days,
    load_alerts,
    load_day,
    load_pr_curves,
    load_results,
    metrics_rows,
)

ROOT = Path(os.environ.get("STLENS_ARTIFACTS", "artifacts"))
GRID_S = 0.25
LIMITATION = (
    "**Research prototype on SYNTHETIC data.** Scores indicate a *spoof-like pattern* in a "
    "simulated order book with injected episodes. They do not identify traders, intent or rule "
    "violations, and say nothing about real markets."
)

st.set_page_config(page_title="ST-LENS", layout="wide")
st.title("ST-LENS - spoof-like pattern scoring (prototype)")
st.warning(LIMITATION)


@st.cache_data
def _results(root: str):
    return load_results(Path(root))


@st.cache_data
def _day(root: str, day: int):
    return load_day(Path(root), day)


results = _results(str(ROOT))
if results is None:
    st.error(f"No results found in `{ROOT}`. Run `python scripts/run_experiment.py --final` first.")
    st.stop()

days = list_days(ROOT)
alerts = load_alerts(ROOT)
split = "test" if results["final_test_evaluated"] else "val"

tab_market, tab_alerts, tab_models, tab_data = st.tabs(
    ["Market monitor", "Alerts", "Models & metrics", "Data health"]
)

# ------------------------------------------------------------------ market monitor
with tab_market:
    if not days:
        st.info("No replayed days (run with --final).")
    else:
        day = st.selectbox("Test day", days)
        dv = _day(str(ROOT), day)
        T = len(dv.mid)
        lo, hi = st.slider("Time range (grid steps of 250 ms)", 0, T - 1, (0, min(T - 1, 2400)))
        sl = slice(lo, hi + 1)
        idx = np.arange(lo, hi + 1)[downsample(hi - lo + 1, 1200)]
        t_min = idx * GRID_S / 60.0
        fig = make_subplots(
            rows=3,
            cols=1,
            shared_xaxes=True,
            row_heights=[0.5, 0.2, 0.3],
            vertical_spacing=0.04,
            subplot_titles=(
                "Ladder depth (bid buckets | ask buckets around mid)",
                "Mid price (ticks)",
                "Scores",
            ),
        )
        P = dv.depth.shape[1]
        labels = [f"bid+{P // 2 - 1 - i}" for i in range(P // 2)] + [
            f"ask+{i}" for i in range(P // 2)
        ]
        fig.add_trace(
            go.Heatmap(
                z=np.log1p(dv.depth[idx]).T,
                x=t_min,
                y=labels,
                colorscale="Viridis",
                showscale=False,
            ),
            row=1,
            col=1,
        )
        fig.add_trace(
            go.Scatter(x=t_min, y=dv.mid[idx], name="mid", line={"width": 1}), row=2, col=1
        )
        fig.add_trace(
            go.Scatter(x=t_min, y=dv.prob[idx], name="ST-LENS prob", line={"width": 1}),
            row=3,
            col=1,
        )
        fig.add_trace(
            go.Scatter(x=t_min, y=dv.smooth[idx], name="smoothed", line={"width": 2}), row=3, col=1
        )
        fig.add_trace(
            go.Scatter(
                x=t_min,
                y=np.where(dv.y[idx] > 0, 1.0, np.nan),
                name="injected spoof label",
                mode="markers",
                marker={"symbol": "line-ns-open", "size": 10},
            ),
            row=3,
            col=1,
        )
        for ep in dv.episodes:
            c = ep.get("cancel_step")
            if c is None or not lo <= c <= hi:
                continue
            colour = "red" if ep["positive"] else "grey"
            fig.add_vline(x=c * GRID_S / 60.0, line_color=colour, line_dash="dot", opacity=0.5)
        for a in alerts:
            if a["day"] == day and lo <= a["start_step"] <= hi:
                fig.add_vrect(
                    x0=a["start_step"] * GRID_S / 60,
                    x1=a["end_step"] * GRID_S / 60,
                    fillcolor="orange",
                    opacity=0.25,
                    line_width=0,
                    row=3,
                    col=1,
                )
        fig.update_xaxes(title_text="minutes since session start", row=3, col=1)
        fig.update_layout(height=750, margin={"t": 40, "b": 30})
        st.plotly_chart(fig, use_container_width=True)
        st.caption(
            "Dotted lines: injected episode ends (red = ground-truth spoof, grey = hard negative). "
            "Orange: alerts from offline replay."
        )

# ------------------------------------------------------------------ alerts
with tab_alerts:
    if not alerts:
        st.info("No alerts (run with --final).")
    else:
        df = pd.DataFrame(alerts)
        df["duration_s"] = (df["end_step"] - df["start_step"] + 1) * GRID_S
        st.dataframe(
            df[
                [
                    "alert_id",
                    "day",
                    "start_step",
                    "duration_s",
                    "side",
                    "peak_score",
                    "status",
                    "model",
                ]
            ],
            use_container_width=True,
        )
        chosen = st.selectbox("Alert detail", df["alert_id"])
        a = df[df["alert_id"] == chosen].iloc[0]
        st.markdown(
            f"**{a['alert_id']}**: side `{a['side']}`, peak score {a['peak_score']:.3f}, "
            f"status `{a['status']}`"
        )
        for r in a["reasons"]:
            st.markdown(f"- Evidence: {r}")
        dv = _day(str(ROOT), int(a["day"]))
        w0, w1 = max(0, int(a["start_step"]) - 40), min(len(dv.mid), int(a["end_step"]) + 20)
        P = dv.depth.shape[1]
        ylab = [f"bid+{P // 2 - 1 - i}" for i in range(P // 2)] + [
            f"ask+{i}" for i in range(P // 2)
        ]
        c1, c2 = st.columns(2)
        c1.plotly_chart(
            go.Figure(
                go.Heatmap(z=np.log1p(dv.depth[w0:w1]).T, y=ylab, colorscale="Viridis")
            ).update_layout(title="Depth around alert", height=350),
            use_container_width=True,
        )
        c2.plotly_chart(
            go.Figure(
                go.Heatmap(z=np.log1p(dv.cancel_vol[w0:w1]).T, y=ylab, colorscale="Reds")
            ).update_layout(title="Inferred cancellations around alert", height=350),
            use_container_width=True,
        )
        st.info(LIMITATION)

# ------------------------------------------------------------------ models
with tab_models:
    st.subheader(f"Metrics ({split}; mean over seeds)")
    rows = pd.DataFrame(metrics_rows(results, split))
    if not rows.empty:
        show = [
            c
            for c in [
                "model",
                "pr_auc",
                "pr_auc_std",
                "roc_auc",
                "recall_at_budget",
                "episode_recall",
                "hard_negative_flag_rate",
                "median_latency_s",
                "alerts_per_hour",
            ]
            if c in rows
        ]
        st.dataframe(rows[show].sort_values("pr_auc", ascending=False), use_container_width=True)
        st.caption(
            "Operating threshold per model and seed = the threshold maximising window-level F1 on "
            "the VALIDATION day, applied unchanged to test. alerts_per_hour = flagged windows per "
            "hour of evaluated market time (one window per 250 ms step, up to 14,400/h; "
            "consecutive flags counted separately, not de-duplicated alerts). Read episode_recall "
            "together with alerts_per_hour. recall_at_budget uses a separate validation threshold "
            "fixed at the configured window budget."
        )
    curves = load_pr_curves(ROOT)
    if curves:
        fig = go.Figure()
        for name, c in curves.items():
            fig.add_trace(go.Scatter(x=c["recall"], y=c["precision"], name=name, mode="lines"))
        fig.update_layout(
            title="Precision-recall (test, first seed)",
            xaxis_title="recall",
            yaxis_title="precision",
            height=450,
        )
        st.plotly_chart(fig, use_container_width=True)
    models = list(results["per_run"][split])
    if models:
        m = st.selectbox(
            "Confusion matrix for",
            models,
            index=models.index("STLENS_full") if "STLENS_full" in models else 0,
        )
        cm = confusion(results, split, m)
        st.dataframe(pd.DataFrame(cm, index=["true 0", "true 1"], columns=["pred 0", "pred 1"]))
    if results.get("explanation"):
        st.subheader("Explanation checks (test true positives)")
        st.json(results["explanation"])

# ------------------------------------------------------------------ data health
with tab_data:
    st.subheader("Splits and labels")
    st.dataframe(pd.DataFrame(results["data"]).T, use_container_width=True)
    if results.get("replay"):
        st.subheader("Replay")
        st.dataframe(pd.DataFrame(results["replay"]).T, use_container_width=True)
    st.caption(
        f"Generated {results['generated_utc']}; "
        f"final test evaluated: {results['final_test_evaluated']}."
    )
