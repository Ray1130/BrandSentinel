from datetime import date, timedelta

import polars as pl

from brandsentinel.core.types import Table
from brandsentinel.detection.indicators.content import AspectNegativity, SafetyRisk, TopicBurst
from brandsentinel.detection.indicators.reliability import RatingTextMismatch, VerifiedRatioDrop
from brandsentinel.detection.stage import detect
from brandsentinel.testing.mock_data import make_all


def feature_series(indicator_id, feature, values, *, n_window=20):
    start = date(2025, 1, 1)
    return pl.DataFrame(
        {
            "sku": ["SKU-1"] * len(values),
            "category": ["Test_Category"] * len(values),
            "window_end": [start + timedelta(days=i) for i in range(len(values))],
            "indicator_id": [indicator_id] * len(values),
            "feature": [feature] * len(values),
            "raw_value": values,
            "n_window": [n_window] * len(values),
        },
        schema_overrides={"raw_value": pl.Float64},
    )


def test_content_reliability_detectors_run_on_mock_feature_series(cfg):
    features = make_all(seed=9, n_skus=3, days=120, cfg=cfg)[Table.FEATURE_SERIES]
    matrix = detect(features, cfg)

    assert {"I7", "I8", "I9", "I10", "I11"} <= set(matrix["indicator_id"].unique())
    assert (
        matrix.filter(pl.col("indicator_id").is_in(["I7", "I8", "I9", "I10", "I11"]))[
            "strength"
        ].min()
        >= 0
    )


def test_i7_detects_a_sustained_topic_burst(cfg):
    values = [0.0] * 40 + [5.0, 5.0, 5.0]

    result = TopicBurst(cfg.indicators.indicators["I7"], cfg).run(
        feature_series("I7", "term_burst_score", values),
        baseline=None,
    )

    assert result["triggered"].to_list()[-3:] == [False, True, True]


def test_i8_requires_hit_and_distinct_user_thresholds_and_marks_severity(cfg):
    features = pl.concat(
        [
            feature_series("I8", "risk_hits", [1.0, 2.0, 3.0, 4.0]),
            feature_series("I8", "risk_distinct_users", [3.0, 1.0, 3.0, 2.0]),
        ]
    )

    result = SafetyRisk(cfg.indicators.indicators["I8"], cfg).run(features, baseline=None)

    assert result["triggered"].to_list() == [False, False, True, True]
    assert result["strength"].to_list() == [0.0, 0.0, 1.0, 0.5]


def test_i8_firetv_and_negation_cases_with_zero_risk_features_do_not_trigger(cfg):
    # The upstream risk lexicon excludes "Fire TV Stick" and negated risk phrases.
    no_risk_features = pl.concat(
        [
            feature_series("I8", "risk_hits", [0.0, 0.0]),
            feature_series("I8", "risk_distinct_users", [0.0, 0.0]),
        ]
    )

    result = SafetyRisk(cfg.indicators.indicators["I8"], cfg).run(
        no_risk_features,
        baseline=None,
    )

    assert result["triggered"].to_list() == [False, False]


def test_i9_requires_minimum_window_reviews(cfg):
    values = [0.05] * 40 + [0.9, 0.9, 0.9]
    features = feature_series("I9", "aspect_neg_safety", values, n_window=4)

    result = AspectNegativity(cfg.indicators.indicators["I9"], cfg).run(features, baseline=None)

    assert not result["triggered"].any()


def test_i9_detects_an_aspect_specific_negative_shift(cfg):
    values = [0.05] * 40 + [0.9, 0.9, 0.9]
    features = feature_series("I9", "aspect_neg_safety", values)

    result = AspectNegativity(cfg.indicators.indicators["I9"], cfg).run(features, baseline=None)

    assert result["triggered"].to_list()[-3:] == [False, True, True]


def test_i10_detects_mismatch_rate_increase(cfg):
    values = [0.05] * 40 + [0.9, 0.9, 0.9]

    result = RatingTextMismatch(cfg.indicators.indicators["I10"], cfg).run(
        feature_series("I10", "mismatch_rate", values),
        baseline=None,
    )

    assert result["triggered"].to_list()[-3:] == [False, True, True]


def test_i11_needs_verified_drop_and_volume_spike(cfg):
    verified = [0.9] * 100 + [0.1, 0.1, 0.1]
    volume = [1.0] * 100 + [5.0, 5.0, 5.0]
    features = pl.concat(
        [
            feature_series("I11", "verified_ratio_all", verified),
            feature_series("I1", "log1p_daily_count", volume),
        ]
    )

    result = VerifiedRatioDrop(cfg.indicators.indicators["I11"], cfg).run(features, baseline=None)

    assert result["triggered"].to_list()[-3:] == [False, True, True]


def test_i11_does_not_trigger_without_volume_spike(cfg):
    verified = [0.9] * 100 + [0.1, 0.1, 0.1]
    volume = [1.0] * 103
    features = pl.concat(
        [
            feature_series("I11", "verified_ratio_all", verified),
            feature_series("I1", "log1p_daily_count", volume),
        ]
    )

    result = VerifiedRatioDrop(cfg.indicators.indicators["I11"], cfg).run(features, baseline=None)

    assert not result["triggered"].any()
