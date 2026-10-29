import io

import pandas as pd
import pytest

from integrations import hdfs_storage, spark_processor


def test_modules_import_without_optional_packages():
    assert isinstance(spark_processor.spark_status(), dict)
    assert isinstance(hdfs_storage.hdfs_status(), dict)


def test_spark_unavailable_raises_clean_error(monkeypatch):
    monkeypatch.setattr(spark_processor, "spark_status", lambda: {"available": False, "message": "no spark"})
    with pytest.raises(spark_processor.SparkUnavailableError):
        spark_processor.aggregate_daily_counts("x.parquet")


def test_hdfs_without_package_raises_clean_error(monkeypatch):
    monkeypatch.setattr(hdfs_storage, "hdfs_status", lambda: {"package_installed": False, "message": "no pkg"})
    with pytest.raises(hdfs_storage.HdfsUnavailableError):
        hdfs_storage.HdfsStorage(url="http://x")


class FakeClient:
    def __init__(self):
        self.files = {}

    def write(self, path, data, overwrite, replication):
        self.files[path] = (data, replication)

    def read(self, path):
        data = self.files[path][0]

        class R:
            def __enter__(s):
                return io.BytesIO(data)

            def __exit__(s, *a):
                return False
        return R()

    def status(self, path, strict=True):
        return {} if path in self.files else None


def test_hdfs_roundtrip_with_mock_client():
    store = hdfs_storage.HdfsStorage(client=FakeClient())
    df = pd.DataFrame({"a": [1, 2], "b": ["x", "y"]})
    store.write_parquet(df, "/waste/r.parquet")
    assert store.client.files["/waste/r.parquet"][1] == 3            # replication factor 3
    pd.testing.assert_frame_equal(store.read_parquet("/waste/r.parquet"), df)
    assert store.exists("/waste/r.parquet") and not store.exists("/nope")
