"""Registry chỉ báo: mỗi chỉ báo (I1-I11) tự đăng ký bằng decorator.

    from brandsentinel.core.registry import register
    from brandsentinel.detection.base import Indicator

    @register
    class StlVolumeSpike(Indicator):
        id = "I1"
        group = Group.VOLUME
        def compute(self, feats, baseline): ...

`build_indicators(cfg)` tạo các chỉ báo đang bật trong configs/indicators.yaml mà đã được cài đặt.
"""

from __future__ import annotations

import importlib
import pkgutil
import re
from typing import TYPE_CHECKING, TypeVar

from brandsentinel.core.config import Config
from brandsentinel.core.logging import get_logger

if TYPE_CHECKING:
    from brandsentinel.detection.base import Indicator

log = get_logger(__name__)
REGISTRY: dict[str, type[Indicator]] = {}
T = TypeVar("T", bound=type)
INDICATORS_PACKAGE = "brandsentinel.detection.indicators"


class RegistryError(RuntimeError):
    pass


def register(cls: T) -> T:
    """Decorator đăng ký lớp chỉ báo theo `cls.id`. Mã trùng hoặc sai định dạng bị từ chối."""
    iid = getattr(cls, "id", None)
    if not isinstance(iid, str) or not re.fullmatch(r"I\d+", iid):
        raise RegistryError(f"{cls.__name__}: thiếu hoặc sai `id` (cần dạng 'I1'..'I11'): {iid!r}")
    if not hasattr(cls, "group"):
        raise RegistryError(f"{cls.__name__}: thiếu `group`")
    existing = REGISTRY.get(iid)
    if existing is not None and (existing.__module__, existing.__qualname__) != (
        cls.__module__,
        cls.__qualname__,
    ):
        raise RegistryError(
            f"{iid} đã được đăng ký bởi {existing.__module__}.{existing.__qualname__}"
        )
    REGISTRY[iid] = cls  # type: ignore[assignment]
    return cls


def load_indicator_modules() -> None:
    """Import mọi module trong detection/indicators để các decorator @register chạy."""
    package = importlib.import_module(INDICATORS_PACKAGE)
    for mod in pkgutil.iter_modules(package.__path__):
        importlib.import_module(f"{INDICATORS_PACKAGE}.{mod.name}")


def get_indicator_class(indicator_id: str) -> type[Indicator]:
    load_indicator_modules()
    try:
        return REGISTRY[indicator_id]
    except KeyError:
        raise RegistryError(f"chỉ báo {indicator_id} chưa được cài đặt/đăng ký") from None


def build_indicators(config: Config, *, strict: bool = False) -> dict[str, Indicator]:
    """Tạo thể hiện cho các chỉ báo `enabled`. strict=False: bỏ qua chỉ báo chưa cài đặt."""
    load_indicator_modules()
    built: dict[str, Indicator] = {}
    for iid, spec in config.indicators.indicators.items():
        if not spec.enabled:
            continue
        cls = REGISTRY.get(iid)
        if cls is None:
            if strict:
                raise RegistryError(f"{iid} bật trong indicators.yaml nhưng chưa được cài đặt")
            log.warning("bỏ qua %s: chưa cài đặt", iid)
            continue
        if cls.group.value != spec.group:
            raise RegistryError(
                f"{iid}: group trong code ({cls.group.value}) khác indicators.yaml ({spec.group})"
            )
        built[iid] = cls(spec, config)
    return built
