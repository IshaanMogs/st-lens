"""Offline replay scoring and alerting (spec L, fast-track version).

Replay walks a day's grid steps in time order and, at each step, uses only the model
probability of the window ending at that step:

    calibrated probability -> EMA smoothing -> band -> alert with hysteresis + cooldown

Deferred (documented): incremental event-by-event feature state and live WebSocket
input. Replay here consumes the causal per-step features computed offline; their
causality is established by the truncation tests.
"""

from dataclasses import asdict, dataclass, field

import numpy as np


@dataclass(frozen=True)
class AlertConfig:
    ema_alpha: float = 0.5
    high: float = 0.5  # open threshold (set on validation)
    low: float = 0.25  # close threshold (hysteresis)
    medium: float = 0.25  # band boundary for display
    cooldown_steps: int = 8  # after closing, ignore re-triggers for this many steps
    require_evidence: bool = True  # High requires a supporting reason code


@dataclass
class Alert:
    alert_id: str
    day: int
    start_step: int
    end_step: int
    start_time_us: int
    end_time_us: int
    peak_score: float
    peak_step: int
    band: str
    side: str
    location_bucket: int
    reasons: list[str] = field(default_factory=list)
    model: str = ""
    status: str = "open"

    def as_dict(self) -> dict:
        return asdict(self)


class StreamingScorer:
    """Stateful per-step scorer. Feed probabilities in time order; never looks ahead."""

    def __init__(self, cfg: AlertConfig) -> None:
        self.cfg = cfg
        self.ema: float | None = None
        self.open_since: int | None = None
        self.cooldown_until = -1

    def update(self, step: int, prob: float, evidence_ok: bool) -> tuple[float, str, str]:
        """Returns (smoothed score, band, event) with event in {"", "open", "close"}."""
        c = self.cfg
        if np.isnan(prob):
            return (self.ema if self.ema is not None else float("nan")), "none", ""
        self.ema = prob if self.ema is None else c.ema_alpha * prob + (1 - c.ema_alpha) * self.ema
        s = self.ema
        band = "high" if s >= c.high else "medium" if s >= c.medium else "low"
        if band == "high" and c.require_evidence and not evidence_ok:
            band = "medium"
        event = ""
        if self.open_since is None:
            if band == "high" and step >= self.cooldown_until:
                self.open_since, event = step, "open"
        elif s < c.low:
            self.open_since, event = None, "close"
            self.cooldown_until = step + c.cooldown_steps
        return s, band, event


def ema_batch(prob: np.ndarray, alpha: float) -> np.ndarray:
    """Vectorised-equivalent EMA used to check the streaming scorer (nan-skipping)."""
    out = np.full(len(prob), np.nan)
    ema = None
    for i, p in enumerate(prob):
        if np.isnan(p):
            out[i] = np.nan if ema is None else ema
            continue
        ema = p if ema is None else alpha * p + (1 - alpha) * ema
        out[i] = ema
    return out


def replay_day(
    day_index: int,
    grid_times_us: np.ndarray,
    prob: np.ndarray,
    evidence_ok: np.ndarray,
    cfg: AlertConfig,
    side_pred: np.ndarray | None = None,
    loc_pred: np.ndarray | None = None,
    reasons_at=None,
    model: str = "",
) -> tuple[np.ndarray, np.ndarray, list[Alert]]:
    """Score one day step by step. Returns (smoothed score, band, alerts)."""
    scorer = StreamingScorer(cfg)
    T = len(prob)
    smooth = np.full(T, np.nan)
    band = np.full(T, "none", dtype="<U6")
    alerts: list[Alert] = []
    current: Alert | None = None
    for t in range(T):
        s, b, ev = scorer.update(t, float(prob[t]), bool(evidence_ok[t]))
        smooth[t], band[t] = s, b
        if ev == "open":
            side = (
                {1: "bid", 2: "ask"}.get(int(side_pred[t]), "none")
                if side_pred is not None
                else "?"
            )
            current = Alert(
                alert_id=f"d{day_index}-a{len(alerts):04d}",
                day=day_index,
                start_step=t,
                end_step=t,
                start_time_us=int(grid_times_us[t]),
                end_time_us=int(grid_times_us[t]),
                peak_score=s,
                peak_step=t,
                band="high",
                side=side,
                location_bucket=int(loc_pred[t]) if loc_pred is not None else -1,
                reasons=reasons_at(t) if reasons_at else [],
                model=model,
            )
            alerts.append(current)
        if current is not None:
            current.end_step, current.end_time_us = t, int(grid_times_us[t])
            if s > current.peak_score:
                current.peak_score, current.peak_step = s, t
            if ev == "close":
                current.status = "closed"
                current = None
    return smooth, band, alerts
