from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "configs" / "default.json"


@dataclass(frozen=True)
class ProjectConfig:
    dataset_dir: Path
    train_dir: Path
    test_dir: Path
    model_path: Path
    class_names_path: Path
    history_path: Path
    reports_dir: Path
    image_size: tuple[int, int]
    batch_size: int
    validation_split: float
    seed: int
    epochs: int
    learning_rate: float
    top_k: int


def _resolve_path(value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else PROJECT_ROOT / path


def load_config(config_path: str | Path = DEFAULT_CONFIG_PATH) -> ProjectConfig:
    path = Path(config_path)
    if not path.is_absolute():
        path = PROJECT_ROOT / path

    with path.open("r", encoding="utf-8") as file:
        raw = json.load(file)

    return ProjectConfig(
        dataset_dir=_resolve_path(raw["dataset_dir"]),
        train_dir=_resolve_path(raw["train_dir"]),
        test_dir=_resolve_path(raw["test_dir"]),
        model_path=_resolve_path(raw["model_path"]),
        class_names_path=_resolve_path(raw["class_names_path"]),
        history_path=_resolve_path(raw["history_path"]),
        reports_dir=_resolve_path(raw["reports_dir"]),
        image_size=tuple(raw["image_size"]),
        batch_size=int(raw["batch_size"]),
        validation_split=float(raw["validation_split"]),
        seed=int(raw["seed"]),
        epochs=int(raw["epochs"]),
        learning_rate=float(raw["learning_rate"]),
        top_k=int(raw["top_k"]),
    )
