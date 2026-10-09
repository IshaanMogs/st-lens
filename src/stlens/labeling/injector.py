"""Controlled spoof injection schedule and window labels (Phase 4, spec F Source 1).

Ground truth comes ONLY from the injector's episode log (what the simulator actually did
with each injected order). Heuristic weak labels come from ``labeling.rules`` and are a
separate, explicitly weaker label source.

Episode parameters are drawn to resemble the background's own large orders so the
injected orders are not trivially distinguishable at placement (checked by the artefact
probe): sizes come from the background size distribution conditioned on being "large",
distances from the background's geometric distance law.
"""

from dataclasses import dataclass

import numpy as np

from stlens.book.builder import GridDay, ladder_bucket
from stlens.ingestion.synthetic import (
    EpisodeKind,
    EpisodeRecord,
    InjectionSpec,
    MarketParams,
    SimulatedDay,
)
from stlens.schemas.events import Side


@dataclass(frozen=True)
class InjectionParams:
    episodes_per_minute: float = 4.0
    p_spoof: float = 0.5
    p_hn_cancel: float = 0.25  # remainder -> HN_EXECUTED
    rest_steps: tuple[int, int] = (10, 40)  # simulator steps (100 ms): 1-4 s
    payoff_delay_frac: tuple[float, float] = (0.3, 0.8)
    payoff_size_mult: tuple[float, float] = (3.0, 8.0)  # x background market-order median
    layer_probs: tuple[float, float, float] = (0.7, 0.2, 0.1)  # 1, 2, 3 layers
    large_quantile_z: float = 2.326  # "very large" = above the background 99th size percentile
    min_gap_steps: int = (
        130  # 13 s > window (10 s) + label horizon (2 s): episodes never share a window
    )


def large_size_threshold(market: MarketParams, z: float) -> float:
    """Size above which an order counts as "large" for injection and the artefact probe."""
    return float(np.exp(np.log(market.size_median) + z * market.size_sigma))


def _large_size(rng: np.random.Generator, market: MarketParams, z: float) -> float:
    """Background lognormal size conditioned on exceeding its z-quantile (rejection)."""
    mu, sigma = np.log(market.size_median), market.size_sigma
    while True:
        s = float(rng.lognormal(mu, sigma))
        if s >= np.exp(mu + z * sigma):
            return round(s, 5)


def schedule(
    params: InjectionParams,
    market: MarketParams,
    n_steps: int,
    seed: int,
    tag: str,
) -> list[InjectionSpec]:
    """Non-overlapping injection specs for one day, deterministic in ``seed``."""
    rng = np.random.default_rng(seed)
    steps_per_min = 60_000 // market.step_ms
    mean_gap = steps_per_min / params.episodes_per_minute
    specs: list[InjectionSpec] = []
    k = int(rng.integers(params.min_gap_steps, params.min_gap_steps * 2))
    while True:
        rest = int(rng.integers(params.rest_steps[0], params.rest_steps[1] + 1))
        if k + rest + params.min_gap_steps >= n_steps:
            break
        u = rng.random()
        if u < params.p_spoof:
            kind = EpisodeKind.SPOOF
        elif u < params.p_spoof + params.p_hn_cancel:
            kind = EpisodeKind.HN_CANCEL_NO_PAYOFF
        else:
            kind = EpisodeKind.HN_EXECUTED
        side = Side.BID if rng.random() < 0.5 else Side.ASK
        layers = int(rng.choice([1, 2, 3], p=params.layer_probs))
        distance = 0 if kind is EpisodeKind.HN_EXECUTED else int(rng.geometric(market.distance_p))
        size = _large_size(rng, market, params.large_quantile_z) * layers
        payoff = delay = 0
        if kind is EpisodeKind.SPOOF:
            payoff = round(
                market.size_median
                * market.market_size_mult
                * float(rng.uniform(*params.payoff_size_mult)),
                5,
            )
            delay = max(1, int(rest * float(rng.uniform(*params.payoff_delay_frac))))
        specs.append(
            InjectionSpec(
                episode_id=f"{tag}-ep{len(specs):04d}",
                kind=kind,
                side=side,
                place_step=k,
                rest_steps=rest,
                distance=distance,
                layers=layers,
                size=size,
                payoff_size=float(payoff),
                payoff_delay_steps=delay,
            )
        )
        k += rest + max(params.min_gap_steps, int(rng.exponential(mean_gap)))
    return specs


def message_time_us(day: SimulatedDay, t_us: int) -> int:
    """Time of the depth message that carries an action at ``t_us`` (end of its step)."""
    step = day.params.step_ms * 1_000
    return day.start_time_us + -(-(t_us - day.start_time_us) // step) * step


def grid_index(g: GridDay, t_us: int) -> int:
    """First grid step whose time is >= t_us (where the event becomes visible)."""
    return int(np.searchsorted(g.grid_times_us, t_us, side="left"))


@dataclass
class DayLabels:
    """Per-grid-step labels. A window ending at t takes the labels of step t."""

    y: np.ndarray  # [T] int8: 1 if a positive episode's cancel is in (t - h, t]
    side: np.ndarray  # [T] int8: 0 none, 1 bid, 2 ask
    loc: np.ndarray  # [T] int16: ladder bucket of the spoof order (-1 unknown/none)
    episode: np.ndarray  # [T] int32: index into ``episodes`` (-1 none), any kind
    kind: np.ndarray  # [T] <U24: episode kind for steps in an episode's label horizon
    cancel_step: dict[str, int]  # episode_id -> grid step of cancellation/end
    place_step: dict[str, int]
    episodes: list[EpisodeRecord]


def label_day(day: SimulatedDay, g: GridDay, horizon: int) -> DayLabels:
    """Ground-truth window labels from the injector's episode log (spec D)."""
    T, H = g.n_steps, g.half_buckets
    y = np.zeros(T, np.int8)
    side = np.zeros(T, np.int8)
    loc = np.full(T, -1, np.int16)
    episode = np.full(T, -1, np.int32)
    kind = np.full(T, "", dtype="<U24")
    cancel_step, place_step = {}, {}
    for i, rec in enumerate(day.episodes):
        if rec.end_time_us == 0:
            continue
        c = grid_index(g, message_time_us(day, rec.end_time_us))
        a = grid_index(g, message_time_us(day, rec.place_time_us))
        cancel_step[rec.spec.episode_id], place_step[rec.spec.episode_id] = c, a
        if c >= T:
            continue
        span = slice(c, min(c + horizon, T))
        episode[span] = i
        kind[span] = rec.spec.kind.value
        if rec.is_positive:
            y[span] = 1
            side[span] = 1 if rec.spec.side is Side.BID else 2
            ref = max(c - 1, 0)
            if g.valid[ref]:
                b = ladder_bucket(rec.spec.side, rec.prices_ticks[0], g.mid[ref], H)
                loc[span] = -1 if b is None else b
    return DayLabels(y, side, loc, episode, kind, cancel_step, place_step, list(day.episodes))
