"""OPTIONAL HDFS storage integration (Module 4) using WebHDFS.

Nothing here is needed to run the dashboard. The ``hdfs`` package (HdfsCLI) is imported
lazily; a missing package or unreachable cluster only returns a status message.

WebHDFS is used because it is pure Python and works on Windows without Hadoop
native libraries. Configure with environment variables (never hard-coded):

    WASTE_HDFS_URL   e.g. http://namenode-host:9870
    WASTE_HDFS_USER  e.g. hadoop

STATUS: implemented and unit-tested with a mocked client only. It has NOT been run
against a real HDFS cluster by the author of this package.
"""
from __future__ import annotations

import importlib.util
import io
import os

import pandas as pd

import config


class HdfsUnavailableError(RuntimeError):
    """Raised when the hdfs package is missing, unconfigured or the cluster is unreachable."""


def hdfs_status() -> dict:
    installed = importlib.util.find_spec("hdfs") is not None
    url = os.environ.get(config.HDFS_URL_ENV, "")
    if not installed:
        msg = "Not enabled - the 'hdfs' package is not installed (optional)."
    elif not url:
        msg = f"Not enabled - set the {config.HDFS_URL_ENV} environment variable (optional)."
    else:
        msg = f"Configured for {url} (connection not tested)."
    return {"available": installed and bool(url), "package_installed": installed,
            "url_configured": bool(url), "message": msg}


class HdfsStorage:
    """Thin wrapper to append/read Parquet files on HDFS via WebHDFS."""

    def __init__(self, url: str | None = None, user: str | None = None, client=None):
        if client is not None:               # injected client (used by tests)
            self.client = client
            return
        status = hdfs_status()
        if not status["package_installed"]:
            raise HdfsUnavailableError(status["message"])
        url = url or os.environ.get(config.HDFS_URL_ENV)
        if not url:
            raise HdfsUnavailableError(f"No HDFS URL given and {config.HDFS_URL_ENV} is not set.")
        from hdfs import InsecureClient  # lazy import
        self.client = InsecureClient(url, user=user or os.environ.get(config.HDFS_USER_ENV))

    def write_parquet(self, df: pd.DataFrame, hdfs_path: str, replication: int = config.HDFS_REPLICATION,
                      overwrite: bool = True) -> str:
        buf = io.BytesIO()
        df.to_parquet(buf, index=False)
        try:
            self.client.write(hdfs_path, data=buf.getvalue(), overwrite=overwrite, replication=replication)
        except Exception as exc:
            raise HdfsUnavailableError(f"HDFS write failed: {exc}") from exc
        return hdfs_path

    def read_parquet(self, hdfs_path: str) -> pd.DataFrame:
        try:
            with self.client.read(hdfs_path) as reader:
                return pd.read_parquet(io.BytesIO(reader.read()))
        except Exception as exc:
            raise HdfsUnavailableError(f"HDFS read failed: {exc}") from exc

    def exists(self, hdfs_path: str) -> bool:
        try:
            return self.client.status(hdfs_path, strict=False) is not None
        except Exception as exc:
            raise HdfsUnavailableError(f"HDFS status check failed: {exc}") from exc
