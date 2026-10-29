import numpy as np
import pandas as pd
import pytest

import analytics as a
import config


def test_composition_shares_sum_to_one(small_data):
    daily, _ = small_data
    comp = a.composition(daily, "zone_id")
    sums = comp[[f"share_{c}" for c in config.CATEGORY_COLS]].sum(axis=1)
    assert np.allclose(sums, 1.0)


def test_daily_long_format(small_data):
    daily, _ = small_data
    long = a.daily_zone_category_long(daily)
    assert set(long["category"]) == set(config.CATEGORIES)
    assert long["items"].sum() == daily["total_items"].sum()


def test_k_search_silhouette_validity():
    X = np.random.default_rng(0).normal(size=(5, 2))
    t = a.k_search(X, 2, 10)
    assert t.loc[t["k"] >= 5, "silhouette"].isna().all()      # K >= n is not valid
    assert t.loc[t["k"] == 2, "silhouette"].notna().all()


def test_zone_clustering_recovers_archetypes(small_data):
    daily, zones = small_data
    res = a.cluster_zones(daily, "zone", k=4, zones=zones)
    assert res.chosen_k == 4
    assert res.ari_vs_simulation is not None and res.ari_vs_simulation > 0.8
    assert res.centroids["description"].str.len().gt(0).all()
    assert set(res.zone_assignment["zone_id"]) == set(daily["zone_id"])


def test_forecast_frame_has_no_lookahead(small_data):
    daily, _ = small_data
    fr = a.build_forecast_frame(daily, 7)
    d = daily.set_index(["zone_id", "date"])["total_items"]
    row = fr.iloc[100]
    assert row["target_date"] == row["origin_date"] + pd.Timedelta(days=7)
    assert row["y"] == d[(row["zone_id"], row["target_date"])]
    assert row["vol_t"] == d[(row["zone_id"], row["origin_date"])]
    assert row["vol_lag_7"] == d[(row["zone_id"], row["origin_date"] - pd.Timedelta(days=7))]


def test_chronological_split_has_no_overlap(small_data):
    daily, _ = small_data
    fr = a.build_forecast_frame(daily, 7)
    train, test, cutoff = a.chronological_split(fr, 60)
    assert train["target_date"].max() <= cutoff < test["target_date"].min()
    assert test["target_date"].nunique() == 60


def test_split_rejects_too_little_data(small_data):
    daily, _ = small_data
    fr = a.build_forecast_frame(daily, 7)
    with pytest.raises(ValueError):
        a.chronological_split(fr, 10_000)


def test_regression_metrics():
    m = a.regression_metrics([100, 200], [110, 190])
    assert m["MAE"] == 10 and m["Bias"] == 0
    assert m["RMSE"] == pytest.approx(10)


def test_evaluate_and_forecast_small(small_data):
    daily, zones = small_data
    ev = a.evaluate_forecasters(daily, zones, test_days=60, n_estimators=20, k_max=5)
    assert ev.n_train > 0 and ev.n_test > 0
    assert {"MAE", "RMSE", "R2", "MAPE_%"} <= set(ev.metrics.columns)
    assert np.isfinite(ev.metrics["MAE"]).all()
    assert ev.predictions["target_date"].min() > ev.cutoff           # test strictly after training
    fc = a.forecast_future(daily, horizons=(1, 7), n_estimators=20, k_max=5)
    last = daily["date"].max()
    assert (fc["origin_date"] == last).all()
    assert (fc["target_date"] > last).all()                            # forecasts are in the future
    assert (fc["forecast"] >= 0).all()
