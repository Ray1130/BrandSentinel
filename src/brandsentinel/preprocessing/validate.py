"""Validation entry points for preprocessing tables."""

from __future__ import annotations

import polars as pl

from brandsentinel.core.schemas import validate
from brandsentinel.core.types import Table


def validate_clean_reviews(df: pl.DataFrame) -> pl.DataFrame:
	"""Validate and return a clean_reviews dataframe."""
	return validate(Table.CLEAN_REVIEWS, df)


def validate_daily_agg(df: pl.DataFrame) -> pl.DataFrame:
	"""Validate and return a daily_agg dataframe."""
	return validate(Table.DAILY_AGG, df)
