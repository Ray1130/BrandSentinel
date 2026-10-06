"""Test tren chuoi tong hop (thay bang fixture testing/mock_data.py khi co)."""
import numpy as np
import polars as pl
import pytest

from detection.baseline import Baseline, add_robust_z, fit_baseline, robust_z, rolling_robust_z, shrink_to_industry
from detection.stl import stl_residual, volume_surge_z
from detection.trend import mk_trend, mk_z_series, neg_ratio_shrunk
from detection.triggers import TriggerRule, build_indicator_matrix, m_of_n

T, CRISIS = 220, 170  # khung hoang bat dau ngay 170


def make_counts(crisis: bool, seed=0):
    rng = np.random.default_rng(seed)
    dow = np.arange(T) % 7
    lam = 20 * (1 + 0.4 * np.sin(2 * np.pi * dow / 7))  # mua vu tuan
    if crisis:
        lam[CRISIS : CRISIS + 14] *= 4
    return rng.poisson(lam), dow


def test_robust_z_and_shrink():
    b = fit_baseline(np.r_[np.random.default_rng(1).normal(0, 1, 100), 50.0])  # outlier khong keo median/MAD
    assert abs(b.median) < 0.3 and 0.7 < b.scale < 1.3
    assert robust_z(5, b) > 3
    sku, ind = Baseline(10, 1, 5), Baseline(0, 4, 500)
    s = shrink_to_industry(sku, ind, n0=30)
    assert ind.median < s.median < sku.median  # n=5 << n0 -> gan nganh
    assert shrink_to_industry(None, ind) == ind


def test_m_of_n():
    rule = TriggerRule(k=3, m=2, n=3)
    z = np.array([0, 4, 0, 0, 4, 4, 0, np.nan, 4, np.nan])
    fired, strength = m_of_n(z, rule)
    assert fired.tolist() == [False, False, False, False, False, True, True, False, False, False]
    assert (strength[fired] > 0).all() and (strength[~fired] == 0).all()
    down, _ = m_of_n(-z, TriggerRule(3, 2, 3, direction="down"))
    assert down.tolist() == fired.tolist()


def test_i1_catches_crisis_not_normal():
    c, dow = make_counts(True)
    z, methods = volume_surge_z(c, dow=dow)
    fired, _ = m_of_n(z, TriggerRule(k=3, m=3, n=5))
    assert fired[CRISIS : CRISIS + 14].any()
    assert not fired[:CRISIS].any()
    n_c, dow = make_counts(False, seed=3)
    z2, _ = volume_surge_z(n_c, dow=dow)
    assert not m_of_n(z2, TriggerRule(3, 3, 5))[0].any()
    assert "stl" in methods


def test_stl_short_series_fallback():
    r, m = stl_residual(np.log1p([5, 6, 5, 7, 20]))
    assert m == "median_only" and len(r) == 5
    r, m = stl_residual(np.log1p([5, 6, 5, 7, 20]), dow=[0, 1, 2, 3, 4], dow_profile=np.zeros(7))
    assert m == "dow_profile"


def test_i3_mann_kendall_neg_ratio():
    rng = np.random.default_rng(5)
    n = rng.poisson(15, T) + 1
    p = np.full(T, 0.10)
    p[CRISIS:] = 0.10 + 0.01 * np.arange(T - CRISIS)  # ty le tieu cuc tang dan
    neg = rng.binomial(n, np.clip(p, 0, 1))
    r = neg_ratio_shrunk(neg, n, prior_rate=0.10)
    z = mk_z_series(r)
    fired, _ = m_of_n(z, TriggerRule(k=1.645, m=3, n=5))
    assert fired[CRISIS + 15 :].any()
    assert fired[:CRISIS].mean() < 0.15  # false positive thap
    assert mk_trend(np.arange(30.0)).trend == "increasing"
    assert mk_trend([1, 2, 3]).trend == "insufficient"


def test_polars_pipeline():
    c, dow = make_counts(True)
    dates = pl.date_range(pl.date(2025, 1, 1), pl.date(2025, 1, 1) + pl.duration(days=T - 1), eager=True)
    fs = pl.DataFrame({"sku": "B0CRISIS", "window_end": dates, "indicator_id": "I1",
                       "feature": "stl_resid_7d", "raw_value": np.log1p(c).astype(float), "n_window": c})
    z = add_robust_z(fs)
    im = build_indicator_matrix(z, {"I1": TriggerRule(k=3, m=3, n=5, feature="stl_resid_7d")})
    assert im.columns == ["sku", "window_end", "indicator_id", "triggered", "strength"]
    assert im.height == T
