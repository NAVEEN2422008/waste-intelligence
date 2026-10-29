import numpy as np
import pandas as pd

import config
import generate_data as g


def test_default_shape_and_columns():
    daily, zones = g.generate_dataset()
    assert daily["zone_id"].nunique() == 24
    assert daily["date"].nunique() == 730
    assert len(daily) == 24 * 730
    for c in config.CATEGORY_COLS + ["total_items", "is_holiday", "is_simulated"]:
        assert c in daily.columns
    assert daily["is_simulated"].all() and zones["is_simulated"].all()
    assert not daily.isna().any().any()


def test_reproducible_and_seed_changes_output():
    a, _ = g.generate_dataset(n_zones=4, n_days=100, seed=3)
    b, _ = g.generate_dataset(n_zones=4, n_days=100, seed=3)
    c, _ = g.generate_dataset(n_zones=4, n_days=100, seed=4)
    pd.testing.assert_frame_equal(a, b)
    assert not a["total_items"].equals(c["total_items"])


def test_totals_and_nonnegative(small_data):
    daily, _ = small_data
    assert (daily[config.CATEGORY_COLS].sum(axis=1) == daily["total_items"]).all()
    assert (daily[config.CATEGORY_COLS] >= 0).all().all()


def test_zone_patterns_differ_and_capacity(small_data):
    daily, zones = small_data
    means = daily.groupby("zone_id")["total_items"].mean()
    assert means.max() / means.min() > 1.5
    assert (zones["capacity_items_per_day"] > 0).all()
    assert zones["capacity_basis"].str.startswith("SIMULATED").all()


def test_contamination_ground_truth_split():
    _, zones = g.generate_dataset()
    assert (zones["sim_true_contamination_rate"] > 0.15).sum() == 5


def test_event_in_test_period_not_in_holiday_flag():
    daily, _ = g.generate_dataset()
    ev = daily[daily["sim_unannounced_event"]]
    assert ev["date"].nunique() == 5
    assert ev["date"].min() >= daily["date"].max() - pd.Timedelta(days=config.TEST_DAYS)
    assert not ev["is_holiday"].any()


def test_save_dataset(tmp_path, small_data):
    daily, zones = small_data
    paths = g.save_dataset(daily, zones, tmp_path, write_csv=True)
    assert paths["daily"].exists() and paths["zones"].exists() and paths["daily_csv"].exists()
    back = pd.read_parquet(paths["daily"])
    assert len(back) == len(daily)
