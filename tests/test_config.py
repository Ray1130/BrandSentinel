import shutil
from pathlib import Path

import pytest
import yaml

from brandsentinel.core.config import ConfigError, load_config

REPO = Path(__file__).resolve().parents[1]


@pytest.fixture
def cfg_dir(tmp_path: Path) -> Path:
    dst = tmp_path / "configs"
    shutil.copytree(REPO / "configs", dst)
    return dst


def _edit(path: Path, fn) -> None:
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    fn(data)
    path.write_text(yaml.safe_dump(data, allow_unicode=True), encoding="utf-8")


def test_load_real_configs():
    cfg = load_config(REPO / "configs")
    assert list(cfg.indicators.indicators) == [f"I{i}" for i in range(1, 12)]
    assert cfg.indicators.scored_ids == [f"I{i}" for i in range(1, 10)]
    assert cfg.indicators.flag_ids == ["I10", "I11"]
    assert cfg.indicators.ids_in_group("content") == ["I7", "I8", "I9"]
    assert cfg.default.time.window_days == 7
    assert cfg.llm.provider == "openai"
    assert cfg.llm.api_key_env == "OPENAI_API_KEY"
    assert cfg.llm.monthly_budget_usd == 5.0
    assert cfg.thresholds.calibration.fallback.tau1 < cfg.thresholds.calibration.fallback.tau2
    assert cfg.path("data_raw") == REPO / "data" / "raw"


def test_env_override(cfg_dir: Path, monkeypatch):
    monkeypatch.setenv("BS_CONFIG_DIR", str(cfg_dir))
    from brandsentinel.core.config import find_config_dir

    assert find_config_dir() == cfg_dir


def test_tau_order_rejected(cfg_dir: Path):
    def bad(d):
        d["calibration"]["fallback"] = {"tau1": 0.5, "tau2": 0.4}

    _edit(cfg_dir / "thresholds.yaml", bad)
    with pytest.raises(ConfigError, match="tau1"):
        load_config(cfg_dir)


def test_unknown_priority_rejected(cfg_dir: Path):
    _edit(cfg_dir / "indicators.yaml", lambda d: d["indicators"]["I1"].update(priority="urgent"))
    with pytest.raises(ConfigError, match="priority"):
        load_config(cfg_dir)


def test_typo_key_rejected(cfg_dir: Path):
    _edit(cfg_dir / "default.yaml", lambda d: d["time"].update(window_day=7))
    with pytest.raises(ConfigError):
        load_config(cfg_dir)


def test_flag_must_not_be_scored(cfg_dir: Path):
    _edit(cfg_dir / "indicators.yaml", lambda d: d["indicators"]["I10"].update(in_score=True))
    with pytest.raises(ConfigError, match="in_score"):
        load_config(cfg_dir)


def test_missing_file(cfg_dir: Path):
    (cfg_dir / "thresholds.yaml").unlink()
    with pytest.raises(ConfigError, match="thresholds.yaml"):
        load_config(cfg_dir)


def test_llm_budget_must_be_positive(cfg_dir: Path):
    _edit(cfg_dir / "llm.yaml", lambda d: d.update(monthly_budget_usd=0))
    with pytest.raises(ConfigError, match="llm.yaml"):
        load_config(cfg_dir)
