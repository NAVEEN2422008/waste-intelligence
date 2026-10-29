import io
from datetime import datetime

import pandas as pd
import pytest

import config
import data_io as dio


def test_validate_rejects_missing_columns_and_gaps(small_data):
    daily, _ = small_data
    with pytest.raises(dio.DataError, match="Missing"):
        dio.validate_daily(daily.drop(columns=["paper"]))
    gap = daily[~((daily["zone_id"] == "Z01") & (daily["date"] == daily["date"].iloc[50]))]
    with pytest.raises(dio.DataError, match="missing"):
        dio.validate_daily(gap)


def test_validate_rejects_duplicates_and_negatives(small_data):
    daily, _ = small_data
    with pytest.raises(dio.DataError, match="Duplicate"):
        dio.validate_daily(pd.concat([daily, daily.iloc[:1]]))
    bad = daily.copy()
    bad.loc[0, "paper"] = -5
    with pytest.raises(dio.DataError, match="negative"):
        dio.validate_daily(bad)


def test_upload_csv_roundtrip(small_data):
    daily, _ = small_data
    buf = io.BytesIO(daily.to_csv(index=False).encode())
    b = dio.load_uploaded_daily(buf, "x.csv")
    assert not b.is_simulated and "USER DATA" in b.tag
    assert b.zones["capacity_basis"].str.startswith("DERIVED").all()


def test_record_store_and_aggregation(tmp_path):
    p = tmp_path / "rec.parquet"
    for cat, conf, day in [("Paper", 0.9, 1), ("Paper", 0.95, 1), ("Metal", 0.85, 3), ("Glass", 0.5, 2)]:
        dio.append_record(cat, conf, "Z01", datetime(2025, 1, day, 12), p)
    rec = dio.load_records(p)
    assert len(rec) == 4 and set(rec.columns) == set(dio.RECORD_COLUMNS)
    daily = dio.aggregate_records_to_daily(rec)
    assert daily["paper"].sum() == 2 and daily["metal"].sum() == 1
    assert daily["glass"].sum() == 0                                  # below the 0.80 threshold, excluded
    assert len(daily) == 3                                            # gap day filled with zeros


def test_review_queue_has_metadata_only(tmp_path):
    p = tmp_path / "q.csv"
    dio.append_review_item("Plastic", 0.5, "Z02", "below threshold", p)
    df = pd.read_csv(p)
    assert list(df.columns) == ["id", "timestamp", "zone_id", "top_category", "top_confidence", "reason"]


def test_missing_synthetic_gives_friendly_error(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "SYNTHETIC_DAILY_PATH", tmp_path / "no.parquet")
    with pytest.raises(dio.DataError, match="generate_data.py"):
        dio.load_synthetic()
