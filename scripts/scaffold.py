"""Tạo toàn bộ cây thư mục/file rỗng theo thiết kế đã chốt (không ghi đè file đã có).

Chạy từ thư mục gốc repo:  uv run python scripts/scaffold.py
"""

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PKG = ROOT / "src" / "brandsentinel"

MODULES = {
    "core": "config schemas types registry io logging ids",
    "preprocessing": "loader validate dedup spam normalize aggregate",
    "features": "windows volume rating reliability",
    "features/nlp": "embedder sentiment topic risk_lexicon aspect",
    "detection": "base baseline stl trend changepoint triggers",
    "detection/indicators": "volume rating content reliability",
    "scoring": "weights groups score thresholds rules state",
    "llm": "evidence prompts schemas client validate fallback",
    "evaluation": "recall_match metrics baselines ablation synthetic",
    "serving": "api dashboard notifier",
    "testing": "mock_data",
}
TOP_LEVEL_PY = ["pipeline"]  # cli.py đã có sẵn

CONFIGS = [
    "default.yaml",
    "indicators.yaml",
    "thresholds.yaml",
    "lexicon_risk_en.yaml",
    "lexicon_risk_vi.yaml",
    "lexicon_aspect.yaml",
    "llm.yaml",
]
SCRIPTS = ["download_amazon.py", "calibrate_thresholds.py", "run_eval.py"]
TEST_DIRS = [
    "core",
    "preprocessing",
    "features",
    "detection",
    "scoring",
    "llm",
    "evaluation",
    "fixtures",
]
KEEP_DIRS = ["data/raw", "data/interim", "data/processed", "notebooks"]


def touch(path: Path, text: str = "") -> None:
    if path.exists():
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    print("created", path.relative_to(ROOT))


def main() -> None:
    touch(PKG / "__init__.py", '"""BrandSentinel."""\n')
    for name in TOP_LEVEL_PY:
        touch(PKG / f"{name}.py", f'"""{name}: TODO."""\n')
    for folder, names in MODULES.items():
        touch(PKG / folder / "__init__.py")
        for name in names.split():
            touch(PKG / folder / f"{name}.py", f'"""{folder}/{name}: TODO."""\n')
    for name in CONFIGS:
        touch(ROOT / "configs" / name, "# TODO\n")
    for name in SCRIPTS:
        touch(ROOT / "scripts" / name, f'"""{name}: TODO."""\n')
    for name in TEST_DIRS:
        touch(ROOT / "tests" / name / ".gitkeep")
    for name in KEEP_DIRS:
        touch(ROOT / name / ".gitkeep")


if __name__ == "__main__":
    main()
