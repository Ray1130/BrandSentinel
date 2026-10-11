import polars as pl

from brandsentinel.evaluation.synthetic import (
    calibrate_category,
    inject_synthetic_crises,
)


def test_synthetic_injection_builds_four_labeled_crises_and_control(feature_series):
    suite = inject_synthetic_crises(feature_series, event_days=14)

    assert len(suite.cases) == 4
    assert {case.name for case in suite.cases} == {
        "volume",
        "rating",
        "safety_keyword",
        "review_bombing",
    }
    assert suite.control_sku in suite.feature_series["sku"].unique().to_list()
    assert set(suite.feature_series["sku"].unique()) == {
        suite.control_sku,
        *(case.sku for case in suite.cases),
    }
    assert (
        suite.feature_series.group_by(["sku", "window_end", "indicator_id", "feature"])
        .len()["len"]
        .max()
        == 1
    )
    assert (
        suite.cases[0].event_start
        == suite.feature_series.filter(pl.col("sku") == suite.control_sku)["window_end"]
        .unique()
        .sort()[-14]
    )


def test_calibration_reports_thresholds_cases_and_review_bombing_as_flag(feature_series, cfg):
    report = calibrate_category(
        feature_series,
        cfg,
        event_days=14,
        tune_triggers=False,
    )

    assert report["synthetic"] is True
    assert report["writes_config"] is False
    assert set(report["thresholds"]) >= {
        "tau1",
        "tau2",
        "used_fallback",
    }
    assert len(report["synthetic_cases"]) == 4
    bombing = next(case for case in report["synthetic_cases"] if case["name"] == "review_bombing")
    assert "I11" in bombing["indicator_recall"]
    assert bombing["review_bombing_flagged_windows"] > 0
    assert "target_false_alarm_rate" in report["control"]
    assert report["event_capture"]["total_cases"] == 4


def test_trigger_sweep_reports_kmn_i8_thresholds_and_pelt(feature_series, cfg):
    report = calibrate_category(feature_series, cfg, event_days=14, tune_triggers=True)

    sweeps = report["trigger_calibration"]
    assert {"I1", "I5", "I8", "I11"} <= set(sweeps)
    assert {"k", "m", "n", "pelt_penalty_beta"} <= set(sweeps["I5"]["selected"])
    assert {"min_hits", "min_distinct_users"} <= set(sweeps["I8"]["selected"])
    assert all("false_alarm_rate" in candidate for candidate in sweeps["I1"]["candidates"])
