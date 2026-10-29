"""Tiny calendar helpers shared by the generator and the forecaster."""
from __future__ import annotations

import numpy as np
import pandas as pd

import config


def is_planned_holiday(dates) -> np.ndarray:
    """Boolean array: is each date on the configured planned-holiday calendar?"""
    idx = pd.DatetimeIndex(pd.to_datetime(dates))
    return np.asarray(idx.strftime("%m-%d").isin(config.PLANNED_HOLIDAYS_MMDD))
