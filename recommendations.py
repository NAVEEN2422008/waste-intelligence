"""Decision-support rules (Module 9, paper Section 4.1.3 "Decision-support rules").

Three transparent rules. Every evaluation records the rule text, the input values
and the reason, so each recommendation can be traced to a number on the dashboard.

IMPORTANT: these are illustrative rule outputs on the data you loaded. They are NOT
real municipal actions, work orders or schedules.

Rule 1  CAPACITY       forecast V_hat(z, t+Delta) > trigger_margin x theta_z
Rule 2  CONTAMINATION  rho(z) > rho_max  (default 0.15)
Rule 3  SCHEDULE       dominant recyclable category in the zone's cluster centroid

Contamination metric rho (documented definition, from the paper):
    rho(z, w) = (items in recyclable-designated bins that were labelled NON-recyclable)
                / (all items in recyclable-designated bins)
    summed over the trailing window w (default 28 days) for zone z.
In this project the two counts come from the columns ``recyc_bin_nonrecyc_items`` and
``recyc_bin_items``. In the SIMULATED dataset they are generated synthetically. With a
real classifier, the numerator would come from classifier labels, which are biased
upward by misclassification - see calibrate_contamination().
"""
from __future__ import annotations

import numpy as np
import pandas as pd

import analytics
import config

DISCLAIMER = ("Illustrative decision-support output computed from the loaded data and the thresholds "
              "in the sidebar. These are not real municipal actions or schedules.")


# ---------------------------------------------------------------------------
# Contamination metric
# ---------------------------------------------------------------------------
def contamination_rates(daily: pd.DataFrame, window_days: int = config.CONTAMINATION_WINDOW_DAYS) -> pd.DataFrame:
    """rho per zone over the trailing window ending at the last observed day."""
    cols = {"recyc_bin_items", "recyc_bin_nonrecyc_items"}
    if not cols.issubset(daily.columns):
        return pd.DataFrame(columns=["zone_id", "bin_items", "nonrecyc_items", "rho",
                                     "window_start", "window_end"])
    end = daily["date"].max()
    start = end - pd.Timedelta(days=window_days - 1)
    w = daily[(daily["date"] >= start) & (daily["date"] <= end)]
    g = w.groupby("zone_id")[["recyc_bin_items", "recyc_bin_nonrecyc_items"]].sum().reset_index()
    g = g.rename(columns={"recyc_bin_items": "bin_items", "recyc_bin_nonrecyc_items": "nonrecyc_items"})
    g["rho"] = np.where(g["bin_items"] > 0, g["nonrecyc_items"] / g["bin_items"].replace(0, np.nan), np.nan)
    g["window_start"], g["window_end"] = start, end
    return g


def calibrate_contamination(rho_observed: float, sensitivity: float, specificity: float) -> float:
    """Correct an observed rate for classifier error (Rogan-Gladen estimator).

    true = (observed + specificity - 1) / (sensitivity + specificity - 1), clipped to [0, 1].
    Assumes sensitivity/specificity are known (measure them on a labelled validation set).
    """
    denom = sensitivity + specificity - 1.0
    if denom <= 0:
        raise ValueError("sensitivity + specificity must exceed 1 for a meaningful correction.")
    return float(np.clip((rho_observed + specificity - 1.0) / denom, 0.0, 1.0))


# ---------------------------------------------------------------------------
# Rules
# ---------------------------------------------------------------------------
def _row(zone, rule_id, rule, inputs, triggered, reason, action, severity):
    return {"zone_id": zone, "rule_id": rule_id, "rule": rule, "inputs": inputs, "triggered": bool(triggered),
            "reason": reason, "recommendation": action if triggered else "No action from this rule",
            "severity": severity if triggered else "none"}


def capacity_rule(forecast: pd.DataFrame, zones: pd.DataFrame, horizon: int = config.HORIZON,
                  margin: float = config.TRIGGER_MARGIN, capacity_multiplier: float = 1.0) -> list[dict]:
    rows = []
    cap = zones.set_index("zone_id")["capacity_items_per_day"] * capacity_multiplier
    fc = forecast[forecast["horizon"] == horizon]
    path_max = forecast.groupby("zone_id")["forecast"].max()
    for r in fc.itertuples():
        theta = float(cap.get(r.zone_id, np.nan))
        trigger = margin * theta
        trig = r.forecast > trigger
        over = (r.forecast / theta - 1) * 100 if theta else np.nan
        rows.append(_row(
            r.zone_id, "R1-CAPACITY",
            f"IF forecast volume on day t+{horizon} > {margin:g} x capacity THEN recommend extra collection capacity",
            {"forecast_date": str(r.target_date.date()), "forecast_items": round(float(r.forecast)),
             "capacity_items_per_day": round(theta), "trigger_margin": margin,
             "trigger_value": round(trigger), "max_forecast_in_7_days": round(float(path_max[r.zone_id])),
             "forecast_vs_capacity_%": round(float(over), 1)},
            trig,
            (f"Forecast {r.forecast:,.0f} items for {r.target_date.date()} exceeds the trigger "
             f"{trigger:,.0f} ({margin:g} x capacity {theta:,.0f})." if trig else
             f"Forecast {r.forecast:,.0f} is within the trigger {trigger:,.0f}."),
            f"Add extra collection capacity for {r.zone_id} before {r.target_date.date()}", "high"))
    return rows


def contamination_rule(rates: pd.DataFrame, rho_max: float = config.CONTAMINATION_MAX,
                       calibration: tuple[float, float] | None = None) -> list[dict]:
    rows = []
    for r in rates.itertuples():
        rho = r.rho
        rho_used = rho
        extra = {}
        if calibration is not None and rho == rho:
            rho_used = calibrate_contamination(rho, *calibration)
            extra = {"rho_calibrated": round(rho_used, 4), "sensitivity": calibration[0],
                     "specificity": calibration[1]}
        trig = bool(rho_used == rho_used and rho_used > rho_max)
        rows.append(_row(
            r.zone_id, "R2-CONTAMINATION",
            f"IF contamination rate rho > {rho_max:g} THEN recommend segregation-awareness action or bin-signage review",
            {"recyc_bin_items": int(r.bin_items), "nonrecyc_items_in_recyc_bins": int(r.nonrecyc_items),
             "rho_observed": None if rho != rho else round(float(rho), 4), **extra,
             "rho_max": rho_max, "window": f"{r.window_start.date()} to {r.window_end.date()}"},
            trig,
            ("rho is not defined (no items in recyclable bins)." if rho != rho else
             f"rho = {int(r.nonrecyc_items):,} / {int(r.bin_items):,} = {rho:.3f}"
             + (f" (calibrated {rho_used:.3f})" if extra else "")
             + (f" exceeds the limit {rho_max:g}." if trig else f" is within the limit {rho_max:g}.")),
            f"Run a segregation-awareness action or review bin signage in {r.zone_id}", "medium"))
    return rows


def schedule_rule(clustering: analytics.ClusteringResult, daily: pd.DataFrame,
                  min_share: float = config.DOMINANT_RECYCLABLE_MIN_SHARE) -> list[dict]:
    rows = []
    cent = clustering.centroids.set_index("cluster")
    for z in clustering.zone_assignment.itertuples():
        c = cent.loc[z.cluster]
        shares = {cat: float(c[f"share_{cat.lower()}"]) for cat in config.RECYCLABLE_CATEGORIES}
        top_cat = max(shares, key=shares.get)
        trig = shares[top_cat] >= min_share
        peak_day = None
        if trig:
            g = daily[daily["zone_id"] == z.zone_id]
            prof = g.groupby(g["date"].dt.dayofweek)[top_cat.lower()].mean()
            peak_day = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"][int(prof.idxmax())]
        rows.append(_row(
            z.zone_id, "R3-SCHEDULE",
            f"IF a recyclable category share in the zone's cluster centroid >= {min_share:.0%} THEN recommend a matching pickup schedule",
            {"cluster": int(z.cluster), "cluster_description": c["description"],
             **{f"centroid_share_{k.lower()}": round(v, 3) for k, v in shares.items()},
             "dominant_recyclable": top_cat, "min_share": min_share,
             "historical_peak_weekday": peak_day},
            trig,
            (f"Cluster {int(z.cluster)} ({c['description']}): {top_cat} share {shares[top_cat]:.1%} "
             f"{'>=' if trig else '<'} {min_share:.0%}."),
            (f"{config.PICKUP_TEMPLATES[top_cat]} for {z.zone_id}"
             + (f", aligned to its historical {top_cat.lower()} peak ({peak_day})" if peak_day else "")
             + " (illustrative template)"), "info"))
    return rows


def build_recommendations(forecast: pd.DataFrame, zones: pd.DataFrame, daily: pd.DataFrame,
                          clustering: analytics.ClusteringResult, margin: float = config.TRIGGER_MARGIN,
                          capacity_multiplier: float = 1.0, rho_max: float = config.CONTAMINATION_MAX,
                          window_days: int = config.CONTAMINATION_WINDOW_DAYS,
                          min_share: float = config.DOMINANT_RECYCLABLE_MIN_SHARE,
                          calibration: tuple[float, float] | None = None) -> pd.DataFrame:
    """Evaluate all three rules for all zones. Returns one row per (zone, rule)."""
    rows = capacity_rule(forecast, zones, config.HORIZON, margin, capacity_multiplier)
    rows += contamination_rule(contamination_rates(daily, window_days), rho_max, calibration)
    rows += schedule_rule(clustering, daily, min_share)
    return pd.DataFrame(rows)
