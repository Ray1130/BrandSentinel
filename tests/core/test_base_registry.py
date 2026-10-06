import polars as pl
import pytest

from brandsentinel.core import registry
from brandsentinel.core.registry import RegistryError, build_indicators, register
from brandsentinel.core.schemas import DataContractError
from brandsentinel.core.types import Group, Table
from brandsentinel.detection.base import Indicator


@pytest.fixture(autouse=True)
def clean_registry():
    saved = dict(registry.REGISTRY)
    yield
    registry.REGISTRY.clear()
    registry.REGISTRY.update(saved)


def _make_i1(grp=Group.VOLUME, bad_output=False):
    @register
    class Dummy(Indicator):
        id = "I1"
        group = grp

        def compute(self, feats, baseline):
            if bad_output:
                return feats.select("sku")
            return feats.select(
                "sku",
                "category",
                "window_end",
                (pl.col("raw_value") > 5).alias("triggered"),
                pl.col("raw_value").alias("strength"),
            )

    return Dummy


def test_run_returns_contract_table(mock_tables, cfg):
    cls = _make_i1()
    out = cls(cfg.indicators.indicators["I1"], cfg).run(mock_tables[Table.FEATURE_SERIES], None)
    assert set(out["indicator_id"]) == {"I1"}
    assert out.schema["triggered"] == pl.Boolean
    assert (
        out.height
        == mock_tables[Table.FEATURE_SERIES].filter(pl.col("indicator_id") == "I1").height
    )


def test_missing_columns_rejected(mock_tables, cfg):
    cls = _make_i1(bad_output=True)
    with pytest.raises(DataContractError):
        cls(cfg.indicators.indicators["I1"], cfg).run(mock_tables[Table.FEATURE_SERIES], None)


def test_register_validates_id():
    with pytest.raises(RegistryError):

        @register
        class NoId(Indicator):
            group = Group.VOLUME

            def compute(self, feats, baseline):
                return feats

    _make_i1()
    with pytest.raises(RegistryError, match="đã được đăng ký"):

        @register
        class Another(Indicator):
            id = "I1"
            group = Group.VOLUME

            def compute(self, feats, baseline):
                return feats


def test_build_indicators_skips_unimplemented_and_checks_group(cfg):
    _make_i1()
    built = build_indicators(cfg)
    assert list(built) == ["I1"]
    with pytest.raises(RegistryError, match="chưa được cài đặt"):
        build_indicators(cfg, strict=True)
    registry.REGISTRY["I1"].group = Group.RATING  # lệch với indicators.yaml
    with pytest.raises(RegistryError, match="group"):
        build_indicators(cfg)
