"""End-to-end fast-track experiment: data -> features -> labels -> splits -> baselines ->
deep models -> evaluation -> explanation -> replay -> artefacts for the dashboard.

Discipline enforced here:
* every threshold, temperature and early-stopping decision uses VALIDATION data only;
* test days are scored and evaluated once, only when ``final=True``; each final run is
  logged to ``final_runs.log``;
* every number written to the results files is computed by this code.
"""

import json
import time
from collections.abc import Callable
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import precision_recall_curve

from stlens.datasets.build import DataConfig, DayData, build_days, fit_standardizer, window_index
from stlens.datasets.torch_data import WindowDataset
from stlens.evaluation.metrics import EvalSet, evaluate, threshold_for_budget
from stlens.explain.attribution import (
    bucket_importance,
    integrated_gradients,
    localisation_hit_rate,
    occlusion_buckets,
    randomised_copy,
    reason_codes,
)
from stlens.features.tabular import RULE_LIKE, window_features
from stlens.models.baselines.deep import DeepLOBLite, SpatialCNN, TemporalTCN
from stlens.models.classical.baselines import fit_logistic, fit_xgboost, prior_scores
from stlens.models.stlens_net.model import STLENSNet
from stlens.scoring.replay import AlertConfig, replay_day
from stlens.training.trainer import TrainConfig, calibrate, fit_temperature, predict, train_model


@dataclass(frozen=True)
class ExperimentConfig:
    data: DataConfig = field(default_factory=DataConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    seeds: tuple[int, ...] = (0, 1, 2)
    alerts_per_hour: float = 600.0  # window-level alert budget for recall@budget
    explain_max_windows: int = 200


DEEP_MODELS: dict[str, Callable[[int, int, int, int], torch.nn.Module]] = {
    "B4_spatial_cnn": lambda C, P, F, L2: SpatialCNN(C),
    "B5_temporal_tcn": lambda C, P, F, L2: TemporalTCN(C, F),
    "B6_deeplob_lite": lambda C, P, F, L2: DeepLOBLite(L2),
    "STLENS_full": lambda C, P, F, L2: STLENSNet(C, P, F),
    "STLENS_spatial_only": lambda C, P, F, L2: STLENSNet(C, P, F, use_temporal=False),
    "STLENS_temporal_only": lambda C, P, F, L2: STLENSNet(C, P, F, use_spatial=False),
}


def _eval_set(days: list[DayData], idx: np.ndarray, grid_s: float) -> EvalSet:
    lab = [days[p].labels for p, _ in idx]
    episode = np.array(
        [
            f"{p}:{days[p].labels.episode[t]}" if days[p].labels.episode[t] >= 0 else ""
            for p, t in idx
        ]
    )
    return EvalSet(
        y=np.array([lb.y[t] for lb, (_, t) in zip(lab, idx, strict=True)], dtype=int),
        step=idx[:, 1].copy(),
        day=idx[:, 0].copy(),
        kind=np.array([lb.kind[t] for lb, (_, t) in zip(lab, idx, strict=True)]),
        episode=episode,
        grid_s=grid_s,
        hours=len(idx) * grid_s / 3600.0,
    )


def _tabular(days: list[DayData], idx: np.ndarray, window: int) -> tuple[np.ndarray, list[str]]:
    blocks, names = [], []
    for pos in np.unique(idx[:, 0]):
        d = days[pos]
        ends = idx[idx[:, 0] == pos, 1]
        X, names = window_features(d.grid, d.ladder, d.context, d.scale, ends, window)
        blocks.append(X)
    return np.concatenate(blocks), names


def _best_f1_threshold(scores: np.ndarray, y: np.ndarray) -> float:
    p, r, thr = precision_recall_curve(y, scores)
    f1 = 2 * p[:-1] * r[:-1] / np.maximum(p[:-1] + r[:-1], 1e-12)
    return float(thr[int(np.nanargmax(f1))]) if len(thr) else 0.5


def _summ(values: list[float]) -> dict[str, float]:
    a = np.asarray(values, float)
    if not np.isfinite(a).any():
        return {"mean": float("nan"), "std": float("nan"), "n": len(a)}
    return {"mean": float(np.nanmean(a)), "std": float(np.nanstd(a)), "n": len(a)}


def run(
    cfg: ExperimentConfig,
    out_dir: Path,
    final: bool = False,
    models: tuple[str, ...] | None = None,
    log: Callable[[str], None] = print,
) -> dict:
    t_start = time.time()
    out_dir.mkdir(parents=True, exist_ok=True)
    dc = cfg.data
    grid_s = dc.grid_us / 1e6
    torch.set_num_threads(max(1, torch.get_num_threads()))

    log("[1/8] simulating synthetic days, reconstructing books, features, labels ...")
    days = build_days(dc)
    idx = {
        "train": window_index(days, "train", dc.window, dc.train_stride),
        "val": window_index(days, "val", dc.window, dc.eval_stride),
        "test": window_index(days, "test", dc.window, dc.eval_stride),
    }
    es = {s: _eval_set(days, idx[s], grid_s) for s in idx}
    y = {s: es[s].y for s in idx}
    data_summary = {
        s: {
            "days": [d.day_index for d in days if d.split == s],
            "windows": len(idx[s]),
            "positive_windows": int(y[s].sum()),
            "prevalence": float(y[s].mean()),
            "episodes": {
                k: int(
                    sum(
                        r.spec.kind.value == k for d in days if d.split == s for r in d.sim.episodes
                    )
                )
                for k in ("spoof", "hn_cancel_no_payoff", "hn_executed")
            },
            "positive_episodes": int(
                sum(r.is_positive for d in days if d.split == s for r in d.sim.episodes)
            ),
            "book_gaps": int(sum(d.grid.health["gaps"] for d in days if d.split == s)),
        }
        for s in idx
    }
    counts = {s: (v["windows"], v["positive_windows"]) for s, v in data_summary.items()}
    log(f"      windows / positive windows: {json.dumps(counts)}")

    log("[2/8] window features + classical baselines (B0-B3) ...")
    X, names = {}, []
    for s in idx:
        X[s], names = _tabular(days, idx[s], dc.window)
    no_rule = np.array([n not in RULE_LIKE for n in names])
    rule = {s: np.array([days[p].rule_score[t] for p, t in idx[s]]) for s in idx}
    scores: dict[str, dict[str, list[np.ndarray]]] = {}  # model -> split -> per-seed scores

    def add(model: str, split: str, values: np.ndarray) -> None:
        scores.setdefault(model, {}).setdefault(split, []).append(values)

    for seed in cfg.seeds:
        for s in ("val", "test"):
            add(
                "B0_prior",
                s,
                prior_scores(len(idx[s]), seed=1_000 + seed + (7 if s == "test" else 0)),
            )
    for s in ("val", "test"):
        add("B1_rule_engine", s, rule[s])
    for tag, cols in (("", slice(None)), ("_no_rule_feats", no_rule)):
        lr = fit_logistic(X["train"][:, cols], y["train"])
        xgb = fit_xgboost(X["train"][:, cols], y["train"], X["val"][:, cols], y["val"])
        for s in ("val", "test"):
            add(f"B2_logistic{tag}", s, lr.predict_proba(X[s][:, cols])[:, 1])
            add(f"B3_xgboost{tag}", s, xgb.predict_proba(X[s][:, cols])[:, 1])

    log("[3/8] deep models (B4-B6, ST-LENS-Net + ablations) ...")
    std = {a: fit_standardizer(days, a, 1) for a in ("ladder", "context", "level")}
    ds = {
        s: WindowDataset(days, idx[s], dc.window, std["ladder"], std["context"], std["level"])
        for s in idx
    }
    C, P = days[0].ladder.shape[1:]
    F = days[0].context.shape[1]
    L2 = days[0].level.shape[2]
    histories, trained, temps = {}, {}, {}
    for name in models or tuple(DEEP_MODELS):
        for seed in cfg.seeds:
            t0 = time.time()
            torch.manual_seed(seed)
            model = DEEP_MODELS[name](C, P, F, L2)
            tc = TrainConfig(**{**asdict(cfg.train), "seed": seed})
            model, hist = train_model(model, ds["train"], ds["val"], y["train"], y["val"], tc)
            val_logits = predict(model, ds["val"])
            temp = fit_temperature(val_logits, y["val"])
            add(name, "val", calibrate(val_logits, temp))
            histories[f"{name}/seed{seed}"] = hist
            trained[(name, seed)], temps[(name, seed)] = model, temp
            log(
                f"      {name} seed {seed}: best val PR-AUC "
                f"{max(h['val_pr_auc'] for h in hist):.4f} "
                f"({len(hist)} epochs, {time.time() - t0:.0f}s)"
            )

    log("[4/8] thresholds from validation; validation metrics ...")
    results: dict[str, dict] = {"val": {}, "test": {}}
    thresholds: dict[str, list[float]] = {}
    for name, per_split in scores.items():
        thresholds[name] = [_best_f1_threshold(v, y["val"]) for v in per_split["val"]]
        results["val"][name] = [
            evaluate(v, es["val"], thr, probs=name != "B1_rule_engine").as_dict()
            for v, thr in zip(per_split["val"], thresholds[name], strict=True)
        ]
    budget_thr = {
        name: [
            threshold_for_budget(v, es["val"].hours, cfg.alerts_per_hour) for v in per_split["val"]
        ]
        for name, per_split in scores.items()
    }

    summary: dict = {"val": {}, "test": {}}
    for split in ("val", "test") if final else ("val",):
        if split == "test":
            log("[5/8] FINAL: scoring and evaluating test days once ...")
            with (out_dir / "final_runs.log").open("a", encoding="utf-8") as fh:
                fh.write(f"{datetime.now(UTC).isoformat()} final test evaluation\n")
            for (name, seed), model in trained.items():
                add(name, "test", calibrate(predict(model, ds["test"]), temps[(name, seed)]))
            for name, per_split in scores.items():
                results["test"][name] = [
                    evaluate(v, es["test"], thr, probs=name != "B1_rule_engine").as_dict()
                    for v, thr in zip(per_split["test"], thresholds[name], strict=True)
                ]
        for name, runs in results[split].items():
            recall_budget = [
                evaluate(v, es[split], thr).recall
                for v, thr in zip(scores[name][split], budget_thr[name], strict=True)
            ]
            summary[split][name] = {
                metric: _summ([r[metric] for r in runs])
                for metric in (
                    "pr_auc",
                    "roc_auc",
                    "precision",
                    "recall",
                    "episode_recall",
                    "hard_negative_flag_rate",
                    "median_latency_s",
                    "alerts_per_hour",
                )
            } | {"recall_at_budget": _summ(recall_budget), "prevalence": runs[0]["prevalence"]}

    explanation, alerts_all, replay_eval = {}, [], {}
    best_seed = None
    if "STLENS_full" in {n for n, _ in trained}:
        best_seed = max(
            cfg.seeds, key=lambda s: max(h["val_pr_auc"] for h in histories[f"STLENS_full/seed{s}"])
        )
    if final and best_seed is not None:
        model = trained[("STLENS_full", best_seed)]
        thr = thresholds["STLENS_full"][cfg.seeds.index(best_seed)]
        test_prob = scores["STLENS_full"]["test"][cfg.seeds.index(best_seed)]

        log("[6/8] explanations on test true positives (IG, occlusion, localisation) ...")
        tp = np.nonzero((test_prob >= thr) & (y["test"] == 1))[0][: cfg.explain_max_windows]
        if len(tp):
            items = [ds["test"][int(i)] for i in tp]
            batch = {k: torch.stack([it[k] for it in items]) for k in items[0]}
            ig_imp = bucket_importance(integrated_gradients(model, batch))
            occ_imp = occlusion_buckets(model, batch).numpy()
            rnd_imp = bucket_importance(integrated_gradients(randomised_copy(model), batch))
            true_loc = batch["loc"].numpy()
            corr = [
                float(np.corrcoef(a, b)[0, 1]) if np.std(a) > 0 and np.std(b) > 0 else float("nan")
                for a, b in zip(ig_imp, rnd_imp, strict=True)
            ]
            explanation = {
                "n_true_positive_windows": len(tp),
                "n_with_known_location": int((true_loc >= 0).sum()),
                "ig_localisation_hit_rate_pm1": localisation_hit_rate(ig_imp, true_loc, 1),
                "occlusion_localisation_hit_rate_pm1": localisation_hit_rate(occ_imp, true_loc, 1),
                "random_weights_ig_localisation_hit_rate_pm1": localisation_hit_rate(
                    rnd_imp, true_loc, 1
                ),
                "chance_hit_rate_pm1": 3.0 / P,
                "mean_corr_trained_vs_random_ig": float(np.nanmean(corr)),
            }
            out = model(batch)
            if "loc" in out:
                head = out["loc"].argmax(1).numpy()
                keep = true_loc >= 0
                explanation["location_head_hit_rate_pm1"] = (
                    float(np.mean(np.abs(head[keep] - true_loc[keep]) <= 1))
                    if keep.any()
                    else float("nan")
                )
                side_pred = out["side"].argmax(1).numpy()
                explanation["side_head_accuracy"] = float(
                    np.mean(side_pred == batch["side"].numpy())
                )

        log("[7/8] offline replay + alerts on test days ...")
        acfg = AlertConfig(high=thr, low=0.75 * thr, medium=0.5 * thr)
        for pos in sorted({int(p) for p in idx["test"][:, 0]}):
            d = days[pos]
            sel = idx["test"][:, 0] == pos
            steps = idx["test"][sel, 1]
            prob = np.full(d.grid.n_steps, np.nan)
            prob[steps] = test_prob[sel]
            items = WindowDataset(
                days, idx["test"][sel], dc.window, std["ladder"], std["context"], std["level"]
            )
            heads_side = np.zeros(d.grid.n_steps, int)
            heads_loc = np.full(d.grid.n_steps, -1)
            with torch.no_grad():
                for start in range(0, len(items), 512):
                    chunk = [items[i] for i in range(start, min(start + 512, len(items)))]
                    b = {k: torch.stack([c[k] for c in chunk]) for k in chunk[0]}
                    o = model(b)
                    st = steps[start : start + len(chunk)]
                    heads_side[st] = o["side"].argmax(1).numpy()
                    heads_loc[st] = o["loc"].argmax(1).numpy()
            evidence = np.zeros(d.grid.n_steps, bool)
            for t in steps[prob[steps] >= acfg.medium]:
                evidence[t] = any(c.supports_high for c in reason_codes(d, int(t), dc.window))
            smooth, band, alerts = replay_day(
                d.day_index,
                d.grid.grid_times_us,
                prob,
                evidence,
                acfg,
                heads_side,
                heads_loc,
                reasons_at=lambda t, d=d: [c.text() for c in reason_codes(d, t, dc.window)[:1]],
                model=f"STLENS_full/seed{best_seed}",
            )
            # Alert-level precision: an alert is a hit if it overlaps a positive label span.
            pos_steps = set(np.nonzero(d.labels.y)[0].tolist())
            for a in alerts:
                a.status = (
                    "hit" if pos_steps & set(range(a.start_step, a.end_step + 1)) else "false_alert"
                )
            alerts_all += [a.as_dict() for a in alerts]
            pos_eps = [r for r in d.sim.episodes if r.is_positive]
            caught = sum(
                any(
                    a.start_step <= d.labels.cancel_step[r.spec.episode_id] + dc.horizon - 1
                    and a.end_step >= d.labels.cancel_step[r.spec.episode_id]
                    for a in alerts
                )
                for r in pos_eps
            )
            replay_eval[str(d.day_index)] = {
                "alerts": len(alerts),
                "hits": sum(a.status == "hit" for a in alerts),
                "positive_episodes": len(pos_eps),
                "episodes_alerted": int(caught),
            }
            np.savez_compressed(
                out_dir / f"day{d.day_index}.npz",
                grid_times_us=d.grid.grid_times_us,
                mid=d.grid.mid,
                best_bid=d.grid.best_bid,
                best_ask=d.grid.best_ask,
                depth=d.grid.depth.astype(np.float32),
                cancel_vol=d.grid.cancel_vol.astype(np.float32),
                y=d.labels.y,
                rule_score=d.rule_score,
                prob=prob,
                smooth=smooth,
                band=band,
            )
            (out_dir / f"day{d.day_index}_episodes.json").write_text(
                json.dumps(
                    [
                        {
                            "episode_id": r.spec.episode_id,
                            "kind": r.spec.kind.value,
                            "side": r.spec.side.value,
                            "positive": r.is_positive,
                            "placed": r.placed,
                            "size": r.spec.size,
                            "layers": r.spec.layers,
                            "prices_ticks": r.prices_ticks,
                            "executed_size": r.executed_size,
                            "place_step": d.labels.place_step.get(r.spec.episode_id),
                            "cancel_step": d.labels.cancel_step.get(r.spec.episode_id),
                        }
                        for r in d.sim.episodes
                    ],
                    indent=1,
                ),
                encoding="utf-8",
            )
        # PR curves + confusion matrices for the dashboard.
        curves = {}
        for name, per_split in scores.items():
            prec, rec, _ = precision_recall_curve(y["test"], per_split["test"][0])
            k = max(1, len(prec) // 400)
            curves[name] = {"precision": prec[::k].tolist(), "recall": rec[::k].tolist()}
        (out_dir / "pr_curves.json").write_text(json.dumps(curves), encoding="utf-8")
        (out_dir / "alerts.json").write_text(json.dumps(alerts_all, indent=1), encoding="utf-8")

    log("[8/8] writing artefacts ...")
    report = {
        "generated_utc": datetime.now(UTC).isoformat(),
        "final_test_evaluated": final,
        "runtime_s": round(time.time() - t_start, 1),
        "config": json.loads(json.dumps(asdict(cfg), default=str)),
        "data": data_summary,
        "summary": summary,
        "per_run": results,
        "thresholds_from_val": thresholds,
        "best_stlens_seed_by_val": best_seed,
        "explanation": explanation,
        "replay": replay_eval,
        "training_histories": histories,
        "feature_names": names,
        "rule_like_features": list(RULE_LIKE),
    }
    (out_dir / "results.json").write_text(
        json.dumps(report, indent=1, default=float), encoding="utf-8"
    )
    return report
