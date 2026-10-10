"""Đọc và kiểm tra cấu hình: configs/default.yaml, indicators.yaml, thresholds.yaml.

Dùng:
    from brandsentinel.core.config import get_config
    cfg = get_config()
    cfg.default.time.window_days
    cfg.indicators.indicators["I8"].params["min_hits"]
    cfg.thresholds.calibration.fallback.tau2

Thư mục cấu hình: tham số `config_dir` > biến môi trường BS_CONFIG_DIR > <repo>/configs.
Khóa thừa hoặc sai chính tả trong YAML bị từ chối (extra="forbid") để lỗi lộ ra ngay lúc nạp.
"""

from __future__ import annotations

import os
import re
from datetime import date
from functools import lru_cache
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, ValidationError, model_validator

REPO_ROOT = Path(__file__).resolve().parents[3]
ENV_CONFIG_DIR = "BS_CONFIG_DIR"
GROUP_FLAG = "flag"


class ConfigError(ValueError):
    """Cấu hình thiếu, sai định dạng hoặc không nhất quán."""


class _Cfg(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


# ============================== default.yaml ==============================
class ProjectCfg(_Cfg):
    name: str
    seed: int


class PathsCfg(_Cfg):
    data_raw: str
    data_interim: str
    data_processed: str
    configs: str
    parquet_compression: Literal["zstd", "snappy", "lz4", "gzip", "uncompressed"] = "zstd"


class TimeCfg(_Cfg):
    timezone: Literal["UTC"]
    bin: Literal["day", "week"]
    window_days: int = Field(ge=1)
    step_days: int = Field(ge=1)
    baseline_days: int = Field(ge=1)
    exclude_flagged_windows_from_baseline: bool
    min_reviews_per_window: int = Field(ge=0)


class KeyColumnsCfg(_Cfg):
    sku: str
    user_id: str
    timestamp: str
    rating: str
    text: str
    verified_purchase: str


class CategoriesCfg(_Cfg):
    dev: list[str]
    holdout: list[str]

    @model_validator(mode="after")
    def _disjoint(self) -> CategoriesCfg:
        both = set(self.dev) & set(self.holdout)
        if both:
            raise ValueError(f"category vừa dev vừa holdout: {sorted(both)}")
        return self


class SkuSelectionCfg(_Cfg):
    min_history_days: int = Field(ge=1)
    min_total_reviews: int = Field(ge=0)
    min_median_weekly_reviews: float = Field(ge=0)


class DataCfg(_Cfg):
    source: str
    key_columns: KeyColumnsCfg
    timestamp_unit: Literal["ms", "s"]
    categories: CategoriesCfg
    sku_selection: SkuSelectionCfg
    start: date | None = None
    end: date | None = None

    @model_validator(mode="after")
    def _range(self) -> DataCfg:
        if self.start and self.end and self.start >= self.end:
            raise ValueError("data.start phải trước data.end")
        return self


class NearDuplicateCfg(_Cfg):
    enabled: bool
    jaccard_min: float = Field(gt=0, le=1)
    shingle_tokens: int = Field(ge=1)


class DedupCfg(_Cfg):
    exact_key: list[str] = Field(min_length=1)
    keep: Literal["earliest", "latest"]
    near_duplicate: NearDuplicateCfg


class SpamCfg(_Cfg):
    min_tokens: int = Field(ge=0)
    max_reviews_per_user_per_day: int = Field(ge=1)
    template_repeat_min_count: int = Field(ge=2)
    same_second_burst_min: int = Field(ge=2)


class NormalizeCfg(_Cfg):
    unicode_form: Literal["NFC", "NFD", "NFKC", "NFKD"]
    lowercase: bool
    strip_html: bool
    collapse_repeated_chars: int = Field(ge=1)
    emoji_to_token: bool
    vi_teencode_dict: str | None = None
    vi_word_segmentation: bool


class PreprocessingCfg(_Cfg):
    dedup: DedupCfg
    spam: SpamCfg
    normalize: NormalizeCfg


class PoolingCfg(_Cfg):
    enabled: bool
    prior_strength: float = Field(ge=0)


class BaselineCfg(_Cfg):
    estimator: Literal["median_mad"]
    mad_scale: float = Field(gt=0)
    min_points: int = Field(ge=2)
    pooling: PoolingCfg
    ratio_shrinkage: Literal["beta", "none"]


class StlCfg(_Cfg):
    period: int = Field(ge=2)
    robust: bool
    transform: Literal["log1p", "none"]
    min_cycles: int = Field(ge=2)
    fallback: Literal["category_seasonal_profile", "rolling_median"]
    calendar_regressors: bool


class TrendCfg(_Cfg):
    mann_kendall_alpha: float = Field(gt=0, lt=1)
    lookback_days: int = Field(ge=7)
    cusum: bool


class ChangepointCfg(_Cfg):
    library: Literal["ruptures"]
    algorithm: Literal["pelt"]
    cost_model: Literal["l1", "l2", "normal", "rbf"]
    penalty_beta: float = Field(gt=0)
    min_size: int = Field(ge=2)
    jump: int = Field(ge=1)
    max_window_days: int = Field(ge=1)
    accept_recent_days: int = Field(ge=1)
    min_effect_sigma: float = Field(ge=0)

    @model_validator(mode="after")
    def _recent_within_window(self) -> ChangepointCfg:
        if self.accept_recent_days > self.max_window_days:
            raise ValueError("accept_recent_days không được lớn hơn max_window_days")
        return self


class NlpCfg(_Cfg):
    device: Literal["auto", "cpu", "cuda"]
    batch_size: int = Field(ge=1)
    languages: list[str] = Field(min_length=1)
    sentiment_model: dict[str, str]
    embedding_model: str
    cache_embeddings: bool
    sample_for_dev: int | None = Field(default=None, ge=1)


class LlmRefCfg(_Cfg):
    config: str


class LlmConfig(_Cfg):
    provider: Literal["openai"]
    model: str = Field(min_length=1)
    api_key_env: str = Field(pattern=r"^[A-Z][A-Z0-9_]*$")
    monthly_budget_usd: float = Field(gt=0)
    cache_enabled: bool
    cache_key: Literal["sha256_canonical_evidence"]


class LoggingCfg(_Cfg):
    level: Literal["DEBUG", "INFO", "WARNING", "ERROR"]
    audit_log: bool


class DefaultConfig(_Cfg):
    project: ProjectCfg
    paths: PathsCfg
    time: TimeCfg
    data: DataCfg
    preprocessing: PreprocessingCfg
    baseline: BaselineCfg
    stl: StlCfg
    trend: TrendCfg
    changepoint: ChangepointCfg
    nlp: NlpCfg
    llm: LlmRefCfg
    logging: LoggingCfg


# ============================== indicators.yaml ==============================
class TriggerCfg(_Cfg):
    """Robust z vượt k trong ít nhất m/n cửa sổ gần nhất; k=None: luật riêng (m=n=1)."""

    direction: Literal["up", "down"]
    k: float | None = None
    m: int = Field(ge=1)
    n: int = Field(ge=1)

    @model_validator(mode="after")
    def _check(self) -> TriggerCfg:
        if self.m > self.n:
            raise ValueError(f"m ({self.m}) không được lớn hơn n ({self.n})")
        if self.k is None:
            if (self.m, self.n) != (1, 1):
                raise ValueError("k = null chỉ dùng với m = 1 và n = 1")
        elif self.k <= 0:
            raise ValueError("k phải dương")
        return self


class IndicatorCfg(_Cfg):
    name: str
    group: str
    priority: str
    enabled: bool
    in_score: bool
    method: str
    trigger: TriggerCfg
    params: dict[str, Any] = Field(default_factory=dict)


class IndicatorsConfig(_Cfg):
    version: int
    priority_points: dict[str, int]
    groups: list[str]
    indicators: dict[str, IndicatorCfg]

    @model_validator(mode="after")
    def _check(self) -> IndicatorsConfig:
        if any(p <= 0 for p in self.priority_points.values()):
            raise ValueError("priority_points phải dương")
        if GROUP_FLAG not in self.groups:
            raise ValueError(f'groups phải có nhóm "{GROUP_FLAG}"')
        for iid, ind in self.indicators.items():
            if not re.fullmatch(r"I\d+", iid):
                raise ValueError(f"mã chỉ báo không hợp lệ: {iid}")
            if ind.group not in self.groups:
                raise ValueError(f"{iid}: group '{ind.group}' không có trong groups")
            if ind.priority not in self.priority_points:
                raise ValueError(f"{iid}: priority '{ind.priority}' không có trong priority_points")
            if ind.in_score == (ind.group == GROUP_FLAG):
                raise ValueError(
                    f"{iid}: in_score phải false khi và chỉ khi group = '{GROUP_FLAG}'"
                )
        if not self.scored_ids:
            raise ValueError("cần ít nhất một chỉ báo enabled và in_score")
        return self

    @property
    def scored_ids(self) -> list[str]:
        """Chỉ báo tham gia RiskScore (enabled và in_score), theo thứ tự khai báo."""
        return [i for i, v in self.indicators.items() if v.enabled and v.in_score]

    @property
    def flag_ids(self) -> list[str]:
        return [i for i, v in self.indicators.items() if v.enabled and not v.in_score]

    def ids_in_group(self, group: str) -> list[str]:
        return [i for i, v in self.indicators.items() if v.enabled and v.group == group]


# ============================== thresholds.yaml ==============================
class TauCfg(_Cfg):
    tau1: float = Field(ge=0, le=1)
    tau2: float = Field(ge=0, le=1)

    @model_validator(mode="after")
    def _order(self) -> TauCfg:
        if self.tau1 >= self.tau2:
            raise ValueError(f"tau1 ({self.tau1}) phải nhỏ hơn tau2 ({self.tau2})")
        return self


class FloorsCfg(_Cfg):
    tau1_min: float = Field(ge=0, le=1)
    tau2_min: float = Field(ge=0, le=1)

    @model_validator(mode="after")
    def _order(self) -> FloorsCfg:
        if self.tau1_min >= self.tau2_min:
            raise ValueError("tau1_min phải nhỏ hơn tau2_min")
        return self


class ReferenceWindowsCfg(_Cfg):
    min_windows: int = Field(ge=1)
    exclude_changepoint_windows: bool


class CalibrationCfg(_Cfg):
    method: Literal["percentile"]
    tau1_percentile: float = Field(gt=0, lt=100)
    tau2_percentile: float = Field(gt=0, lt=100)
    per_category: bool
    reference_windows: ReferenceWindowsCfg
    floors: FloorsCfg
    fallback: TauCfg

    @model_validator(mode="after")
    def _order(self) -> CalibrationCfg:
        if self.tau1_percentile >= self.tau2_percentile:
            raise ValueError("tau1_percentile phải nhỏ hơn tau2_percentile")
        return self


class MediumRuleCfg(_Cfg):
    min_persist_windows: int = Field(ge=1)
    min_groups_active: int = Field(ge=1)


class HighRuleCfg(_Cfg):
    min_groups_active: int = Field(ge=1)
    allow_severe_override: bool


class TieringCfg(_Cfg):
    groups_counted: list[str] = Field(min_length=1)
    min_active_indicators_per_group: int = Field(ge=1)
    medium: MediumRuleCfg
    high: HighRuleCfg
    on_missing_confirmation: Literal["step_down_and_recheck"]

    @model_validator(mode="after")
    def _check(self) -> TieringCfg:
        n = len(self.groups_counted)
        if self.high.min_groups_active > n or self.medium.min_groups_active > n:
            raise ValueError("min_groups_active vượt quá số nhóm trong groups_counted")
        if self.high.min_groups_active < self.medium.min_groups_active:
            raise ValueError("HIGH không được yêu cầu ít nhóm hơn MEDIUM")
        return self


class HysteresisCfg(_Cfg):
    downgrade_after_windows: int = Field(ge=1)


class StateCfg(_Cfg):
    hysteresis: HysteresisCfg
    cooldown_days: int = Field(ge=0)
    escalate_immediately: bool


class FlagsCfg(_Cfg):
    verify_authenticity_from: list[str] = Field(min_length=1)
    require: Literal["any", "all"]


class LlmGateCfg(_Cfg):
    min_level: Literal["MEDIUM", "HIGH"]


class ThresholdsConfig(_Cfg):
    version: int
    calibration: CalibrationCfg
    overrides: dict[str, TauCfg] = Field(default_factory=dict)
    tiering: TieringCfg
    state: StateCfg
    flags: FlagsCfg
    llm_gate: LlmGateCfg


# ============================== gói tổng hợp ==============================
class Config(_Cfg):
    default: DefaultConfig
    indicators: IndicatorsConfig
    thresholds: ThresholdsConfig
    llm: LlmConfig
    root: Path

    @model_validator(mode="after")
    def _cross_check(self) -> Config:
        groups = set(self.indicators.groups)
        bad = [
            g for g in self.thresholds.tiering.groups_counted if g not in groups or g == GROUP_FLAG
        ]
        if bad:
            raise ValueError(f"thresholds.tiering.groups_counted có nhóm không hợp lệ: {bad}")
        missing = [
            i
            for i in self.thresholds.flags.verify_authenticity_from
            if i not in self.indicators.indicators
        ]
        if missing:
            raise ValueError(f"thresholds.flags tham chiếu chỉ báo không tồn tại: {missing}")
        not_flag = [
            i
            for i in self.thresholds.flags.verify_authenticity_from
            if i not in self.indicators.indicators
            or self.indicators.indicators[i].group != GROUP_FLAG
        ]
        if not_flag:
            raise ValueError(f"thresholds.flags chỉ được tham chiếu chỉ báo nhóm flag: {not_flag}")
        return self

    def path(self, key: Literal["data_raw", "data_interim", "data_processed", "configs"]) -> Path:
        """Đường dẫn trong `paths`, tính từ thư mục gốc repo (pathlib, chạy được trên Windows)."""
        return self.root / getattr(self.default.paths, key)


# ============================== nạp file ==============================
def find_config_dir(config_dir: Path | str | None = None) -> Path:
    if config_dir is not None:
        return Path(config_dir)
    env = os.environ.get(ENV_CONFIG_DIR)
    return Path(env) if env else REPO_ROOT / "configs"


def _read_yaml(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise ConfigError(f"không tìm thấy file cấu hình: {path}")
    with path.open(encoding="utf-8") as f:
        data = yaml.safe_load(f)
    if not isinstance(data, dict):
        raise ConfigError(f"{path.name} rỗng hoặc không phải dạng mapping")
    return data


def _validate(model: type[BaseModel], path: Path) -> Any:
    try:
        return model.model_validate(_read_yaml(path))
    except ValidationError as e:
        raise ConfigError(f"{path.name} không hợp lệ:\n{e}") from e


def load_config(config_dir: Path | str | None = None) -> Config:
    """Nạp và kiểm tra cả ba file cấu hình. Ném ConfigError nếu có lỗi."""
    cdir = find_config_dir(config_dir).resolve()
    try:
        default = _validate(DefaultConfig, cdir / "default.yaml")
        return Config(
            default=default,
            indicators=_validate(IndicatorsConfig, cdir / "indicators.yaml"),
            thresholds=_validate(ThresholdsConfig, cdir / "thresholds.yaml"),
            llm=_validate(LlmConfig, cdir.parent / default.llm.config),
            root=cdir.parent,
        )
    except ValidationError as e:  # lỗi nhất quán giữa các file
        raise ConfigError(f"cấu hình không nhất quán:\n{e}") from e


@lru_cache(maxsize=1)
def get_config() -> Config:
    """Cấu hình dùng chung của tiến trình (nạp một lần). Trong test, gọi load_config(dir)."""
    return load_config()
