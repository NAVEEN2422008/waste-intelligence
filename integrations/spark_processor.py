"""OPTIONAL Apache Spark integration (Module 5).

Nothing here is needed to run the dashboard. PySpark and Java are imported / probed
lazily, so a missing installation only produces a status message - it never crashes
the main application.

STATUS: implemented for DAILY AGGREGATION only (Eq. 4.2). It has been syntax-checked
and unit-tested for its "not available" behaviour, but it has NOT been executed
against a real Spark installation by the author of this package. Running K-Means and
Random Forest through Spark MLlib (as in the paper) is PENDING - the dashboard uses
scikit-learn on a single machine instead.

To enable later:  pip install -r requirements-optional.txt  and install Java 11/17.
On Windows you may also need HADOOP_HOME/winutils for some Spark operations.
"""
from __future__ import annotations

import importlib.util
import shutil
from pathlib import Path

import pandas as pd

import config


class SparkUnavailableError(RuntimeError):
    """Raised when PySpark or Java is not available."""


def spark_status() -> dict:
    py = importlib.util.find_spec("pyspark") is not None
    java = shutil.which("java") is not None
    if py and java:
        msg = "PySpark and Java were found. (Spark jobs have not been run by this package's author.)"
    else:
        missing = [n for n, ok in (("pyspark", py), ("Java", java)) if not ok]
        msg = f"Not enabled - missing: {', '.join(missing)}. The local Pandas engine is used instead."
    return {"available": py and java, "pyspark_installed": py, "java_found": java, "message": msg}


def _get_session(app_name: str = "waste-intelligence", master: str = "local[*]"):
    status = spark_status()
    if not status["available"]:
        raise SparkUnavailableError(status["message"])
    from pyspark.sql import SparkSession  # lazy import
    return SparkSession.builder.appName(app_name).master(master).getOrCreate()


def aggregate_daily_counts(records_path: str | Path, min_confidence: float = config.CONFIDENCE_THRESHOLD,
                           master: str = "local[*]") -> pd.DataFrame:
    """Aggregate item-level records (id, category, confidence, zone_id, timestamp) in Spark
    into daily zone x category counts and return a pandas DataFrame (daily schema).

    Returned via toPandas() so no Hadoop/winutils is needed to write files on Windows.
    """
    spark = _get_session(master=master)
    from pyspark.sql import functions as F

    df = spark.read.parquet(str(records_path))
    df = (df.filter(F.col("confidence") >= min_confidence)
            .withColumn("date", F.to_date("timestamp"))
            .withColumn("category", F.lower(F.col("category"))))
    wide = (df.groupBy("zone_id", "date").pivot("category", config.CATEGORY_COLS).count().na.fill(0))
    out = wide.toPandas()
    for c in config.CATEGORY_COLS:
        if c not in out.columns:
            out[c] = 0
    out["date"] = pd.to_datetime(out["date"])
    return out[["date", "zone_id"] + config.CATEGORY_COLS].sort_values(["zone_id", "date"]).reset_index(drop=True)
