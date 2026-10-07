"""Rolling-window helpers for daily feature series."""

from __future__ import annotations

import numpy as np


def validate_window_days(window_days: int) -> None:
	"""Reject invalid rolling window sizes."""
	if isinstance(window_days, bool) or not isinstance(window_days, int):
		raise TypeError("window_days must be an integer")
	if window_days < 1:
		raise ValueError("window_days must be a positive integer")


def rolling_sum(values: np.ndarray, window_days: int) -> np.ndarray:
	"""Return trailing sums, with NaN before the first complete window."""
	validate_window_days(window_days)
	cumulative = np.concatenate(([0.0], np.cumsum(values, dtype=np.float64)))
	result = np.full(len(values), np.nan, dtype=np.float64)
	if len(values) >= window_days:
		result[window_days - 1 :] = (
			cumulative[window_days:] - cumulative[: -window_days]
		)
	return result
