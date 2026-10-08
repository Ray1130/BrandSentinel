import polars as pl

from brandsentinel.detection.indicators.volume import (
    F_COUNT,
    F_GROWTH,
    F_NEG,
    _i1_series,
    _mk_z,
    run_indicator,
)
from brandsentinel.detection.triggers import TriggerRule


def rate(res, sku, last=14):
    r = res.filter(pl.col("sku") == sku).sort("window_end")["triggered"].to_numpy()
    return r[-last:].mean(), r[:-last].mean()


def test_panel(feature_series):
    out = {
        "I1": run_indicator(
            feature_series,
            F_COUNT,
            _i1_series,
            TriggerRule(2.5, 3, 5),
            z_kw={"scale_floor": 0.05},
        ),
        "I2": run_indicator(
            feature_series,
            F_GROWTH,
            lambda d, v: v,
            TriggerRule(2.5, 3, 5),
            z_kw={"scale_floor": 0.05},
        ),
        "I3": run_indicator(
            feature_series,
            F_NEG,
            lambda d, v: _mk_z(v),
            TriggerRule(1.645, 3, 5),
            use_baseline=False,
        ),
    }
    crisis_sku = "B0MOCK0000"
    bombing_sku = "B0MOCK0001"
    normal_skus = sorted(
        set(feature_series["sku"].to_list()) - {crisis_sku, bombing_sku}
    )
    expected_columns = {"sku", "category", "window_end", "triggered", "strength"}
    for result in out.values():
        assert set(result.columns) == expected_columns

    assert rate(out["I1"], crisis_sku)[0] > 0.3
    assert rate(out["I3"], crisis_sku)[0] > 0.3
    for k in ("I1", "I2"):
        res = out[k]
        for sku in normal_skus:
            last, before = rate(res, sku)
            assert last == 0 and before <= 0.15, (k, sku, last, before)