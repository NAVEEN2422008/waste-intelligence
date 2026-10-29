"""Reproducible SYNTHETIC data generator (24 zones x 730 days by default).

Everything produced here is SIMULATED. It is NOT city data. Every output row
carries ``is_simulated = True`` so the dashboard can label it.

Simulation design (mirrors the simulation described in the paper/report):
  * 4 zone archetypes (6 zones each): market, commercial, residential, industrial
  * archetype base volume / category shares are taken from the paper's Table 4
    centroids (IMPLEMENTATION CHOICE: Glass/Other split the remaining share
    55% / 45%, because the paper table lists only four of the six categories)
  * weekly cycle, annual swing, slow upward trend, planned-holiday uplift,
    multiplicative noise, day-to-day composition variation
  * one 5-day UNANNOUNCED festival (up to 1.75x volume) placed in the test
    period and deliberately NOT included in the planned-holiday flag
  * synthetic contamination audit counts (see recommendations.py for the metric)

Usage:
    python generate_data.py                  # 24 zones, 730 days, seed 42
    python generate_data.py --seed 7 --csv   # also write a CSV copy
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

import config
from calendar_utils import is_planned_holiday

# (base items/day, shares of paper/plastic/organic/metal) from paper Table 4
ARCHETYPES = {
    "market": {
        "description": "Market, very high volume, organic-dominant",
        "base": 1455,
        "shares": {"paper": 0.10, "plastic": 0.15, "organic": 0.60, "metal": 0.03},
        "weekly": [0.90, 0.92, 0.95, 1.00, 1.10, 1.18, 1.00],
    },
    "commercial": {
        "description": "Commercial, high volume, plastic-heavy",
        "base": 1029,
        "shares": {"paper": 0.20, "plastic": 0.35, "organic": 0.22, "metal": 0.05},
        "weekly": [1.08, 1.10, 1.10, 1.08, 1.05, 0.85, 0.74],
    },
    "residential": {
        "description": "Residential, mid volume, organic-heavy",
        "base": 670,
        "shares": {"paper": 0.18, "plastic": 0.20, "organic": 0.42, "metal": 0.04},
        "weekly": [0.95, 0.93, 0.95, 0.97, 1.02, 1.12, 1.06],
    },
    "industrial": {
        "description": "Industrial, low volume, paper/metal-heavy",
        "base": 477,
        "shares": {"paper": 0.30, "plastic": 0.12, "organic": 0.08, "metal": 0.25},
        "weekly": [1.12, 1.12, 1.10, 1.08, 1.05, 0.75, 0.68],
    },
}


def _full_shares(shares: dict[str, float]) -> np.ndarray:
    """Return the 6-category share vector (paper, plastic, glass, metal, organic, other)."""
    remaining = 1.0 - sum(shares.values())
    full = {**shares, "glass": remaining * 0.55, "other": remaining * 0.45}
    return np.array([full[c] for c in config.CATEGORY_COLS], dtype=float)


def generate_dataset(
    n_zones: int = config.N_ZONES,
    n_days: int = config.N_DAYS,
    seed: int = config.RANDOM_SEED,
    start_date: str = config.START_DATE,
    test_days: int = config.TEST_DAYS,
    capacity_quantile: float = config.CAPACITY_QUANTILE,
    capacity_multiplier: float = config.CAPACITY_MULTIPLIER,
    n_contaminated_zones: int = 5,
    include_unannounced_event: bool = True,
    event_days: int = 5,
    event_peak: float = 1.75,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Generate (daily, zones) DataFrames. Fully reproducible for a given seed."""
    if n_zones < 1 or n_days < 30:
        raise ValueError("Need at least 1 zone and 30 days.")
    rng = np.random.default_rng(seed)
    dates = pd.date_range(start_date, periods=n_days, freq="D")
    dow = dates.dayofweek.to_numpy()
    doy = dates.dayofyear.to_numpy()
    day_idx = np.arange(n_days)
    holiday = is_planned_holiday(dates)

    # Unannounced event (festival) inside the test period; NOT in the holiday flag.
    event_mult = np.ones(n_days)
    event_flag = np.zeros(n_days, dtype=bool)
    if include_unannounced_event and n_days > test_days + 40:
        test_start = n_days - test_days
        start_i = min(test_start + 109, n_days - event_days - 1)
        ramp = np.interp(np.linspace(0, 1, event_days), [0, 0.5, 1], [1.30, event_peak, 1.30])
        event_mult[start_i:start_i + event_days] = ramp
        event_flag[start_i:start_i + event_days] = True

    arch_keys = list(ARCHETYPES)
    n_bad = min(n_contaminated_zones, n_zones)
    bad_zones = set(rng.choice(n_zones, size=n_bad, replace=False).tolist())

    zone_rows, daily_frames = [], []
    for zi in range(n_zones):
        zone_id = f"Z{zi + 1:02d}"
        arch_key = arch_keys[zi % len(arch_keys)]
        arch = ARCHETYPES[arch_key]
        zone_scale = float(rng.lognormal(0.0, 0.08))
        zone_shares = rng.dirichlet(_full_shares(arch["shares"]) * 400)
        weekly = np.asarray(arch["weekly"], dtype=float)
        weekly = weekly / weekly.mean()
        annual = 1 + 0.08 * np.sin(2 * np.pi * (doy - 60) / 365.25 + rng.normal(0, 0.2))
        trend = 1 + 0.05 * day_idx / max(n_days - 1, 1)
        noise = rng.lognormal(0.0, 0.05, n_days)
        total = (arch["base"] * zone_scale * weekly[dow] * annual * trend
                 * np.where(holiday, 1.15, 1.0) * event_mult * noise)
        total = np.maximum(np.round(total), 1).astype(int)

        comp = rng.dirichlet(zone_shares * 200, size=n_days)       # daily composition
        counts = np.vstack([rng.multinomial(t, p) for t, p in zip(total, comp)])

        true_rate = (rng.uniform(0.20, 0.30) if zi in bad_zones else rng.uniform(0.05, 0.12))
        recyc_bin = np.round(0.35 * total).astype(int)             # items in recyclable-designated bins
        nonrecyc = rng.binomial(recyc_bin, true_rate)              # of which non-recyclable

        frame = pd.DataFrame(counts, columns=config.CATEGORY_COLS)
        frame.insert(0, "zone_id", zone_id)
        frame.insert(0, "date", dates)
        frame["total_items"] = counts.sum(axis=1)
        frame["recyc_bin_items"] = recyc_bin
        frame["recyc_bin_nonrecyc_items"] = nonrecyc
        frame["is_holiday"] = holiday
        frame["sim_unannounced_event"] = event_flag
        frame["is_simulated"] = True
        daily_frames.append(frame)

        train_n = n_days - test_days if n_days > test_days else n_days
        cap = float(np.quantile(frame["total_items"].to_numpy()[:train_n], capacity_quantile))
        zone_rows.append({
            "zone_id": zone_id,
            "zone_name": f"Zone {zi + 1:02d}",
            "capacity_items_per_day": int(round(cap * capacity_multiplier)),
            "capacity_basis": (f"SIMULATED: {int(capacity_quantile * 100)}th percentile of "
                               f"training-period daily volume x {capacity_multiplier:g}"),
            "sim_archetype": arch_key,
            "sim_true_contamination_rate": round(float(true_rate), 4),
            "is_simulated": True,
        })

    daily = pd.concat(daily_frames, ignore_index=True)
    return daily, pd.DataFrame(zone_rows)


def save_dataset(daily: pd.DataFrame, zones: pd.DataFrame, data_dir: Path | None = None,
                 write_csv: bool = False) -> dict[str, Path]:
    """Write Parquet (and optionally CSV) files; returns the written paths."""
    data_dir = Path(data_dir) if data_dir else config.DATA_DIR
    data_dir.mkdir(parents=True, exist_ok=True)
    paths = {
        "daily": data_dir / config.SYNTHETIC_DAILY_PATH.name,
        "zones": data_dir / config.SYNTHETIC_ZONES_PATH.name,
    }
    daily.to_parquet(paths["daily"], index=False)
    zones.to_parquet(paths["zones"], index=False)
    if write_csv:
        paths["daily_csv"] = data_dir / config.SYNTHETIC_DAILY_CSV.name
        daily.to_csv(paths["daily_csv"], index=False)
    return paths


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Generate SYNTHETIC waste data (not real city data).")
    p.add_argument("--zones", type=int, default=config.N_ZONES)
    p.add_argument("--days", type=int, default=config.N_DAYS)
    p.add_argument("--seed", type=int, default=config.RANDOM_SEED)
    p.add_argument("--start", default=config.START_DATE)
    p.add_argument("--capacity-quantile", type=float, default=config.CAPACITY_QUANTILE)
    p.add_argument("--capacity-multiplier", type=float, default=config.CAPACITY_MULTIPLIER)
    p.add_argument("--no-event", action="store_true", help="omit the unannounced festival")
    p.add_argument("--csv", action="store_true", help="also write a CSV copy")
    p.add_argument("--out", type=Path, default=None, help="output folder (default: data/)")
    a = p.parse_args(argv)
    daily, zones = generate_dataset(
        n_zones=a.zones, n_days=a.days, seed=a.seed, start_date=a.start,
        capacity_quantile=a.capacity_quantile, capacity_multiplier=a.capacity_multiplier,
        include_unannounced_event=not a.no_event)
    paths = save_dataset(daily, zones, a.out, write_csv=a.csv)
    print(f"[SIMULATED] Generated {len(daily):,} zone-day rows "
          f"({a.zones} zones x {a.days} days, seed={a.seed}).")
    print(f"[SIMULATED] Total synthetic items: {int(daily['total_items'].sum()):,}")
    for k, v in paths.items():
        print(f"  wrote {k}: {v}")
    print("Reminder: this is synthetic data, not real city data.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
