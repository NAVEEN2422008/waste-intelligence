import numpy as np
import pandas as pd
import pytest

import analytics as a
import config
import recommendations as r


def _daily():
    dates = pd.date_range("2025-01-01", periods=10)
    return pd.DataFrame({"date": dates, "zone_id": "Z01", "recyc_bin_items": 100, "recyc_bin_nonrecyc_items": 20})


def test_contamination_rate_definition():
    rates = r.contamination_rates(_daily(), window_days=5)
    row = rates.iloc[0]
    assert row["bin_items"] == 500 and row["nonrecyc_items"] == 100
    assert row["rho"] == pytest.approx(0.20)


def test_contamination_missing_columns_is_graceful():
    out = r.contamination_rates(_daily().drop(columns=["recyc_bin_items"]))
    assert out.empty


def test_contamination_rule_threshold_is_strict():
    rates = pd.DataFrame({"zone_id": ["A", "B"], "bin_items": [100, 100], "nonrecyc_items": [15, 16],
                          "rho": [0.15, 0.16], "window_start": pd.Timestamp("2025-01-01"),
                          "window_end": pd.Timestamp("2025-01-28")})
    rows = r.contamination_rule(rates, 0.15)
    assert [x["triggered"] for x in rows] == [False, True]          # exactly 0.15 does not exceed
    assert "rho_max" in rows[1]["inputs"] and rows[1]["reason"]


def test_calibration_roundtrip():
    true, sens, spec = 0.10, 0.9, 0.95
    observed = true * sens + (1 - true) * (1 - spec)
    assert r.calibrate_contamination(observed, sens, spec) == pytest.approx(true)
    with pytest.raises(ValueError):
        r.calibrate_contamination(0.2, 0.5, 0.4)


def test_capacity_rule_and_margin():
    zones = pd.DataFrame({"zone_id": ["A", "B"], "capacity_items_per_day": [1000, 1000]})
    fc = pd.DataFrame({"zone_id": ["A", "B"], "origin_date": pd.Timestamp("2025-01-01"),
                       "target_date": pd.Timestamp("2025-01-08"), "horizon": 7, "forecast": [1001.0, 960.0]})
    nominal = r.capacity_rule(fc, zones, 7, 1.0)
    early = r.capacity_rule(fc, zones, 7, 0.95)
    assert [x["triggered"] for x in nominal] == [True, False]
    assert [x["triggered"] for x in early] == [True, True]
    assert nominal[0]["inputs"]["capacity_items_per_day"] == 1000


def test_schedule_rule(small_data):
    daily, zones = small_data
    cres = a.cluster_zones(daily, "zone", k=4)
    rows = r.schedule_rule(cres, daily, 0.20)
    assert len(rows) == daily["zone_id"].nunique()
    trig = [x for x in rows if x["triggered"]]
    assert trig and all(x["inputs"]["dominant_recyclable"] in config.RECYCLABLE_CATEGORIES for x in trig)
    none = r.schedule_rule(cres, daily, 0.99)
    assert not any(x["triggered"] for x in none)
