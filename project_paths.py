from __future__ import annotations

import configparser
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
CONFIG_PATH = PROJECT_ROOT / "config.ini"


def load_config() -> configparser.ConfigParser:
    config = configparser.ConfigParser()
    config.read(CONFIG_PATH)
    return config


def resolve_path(value: str | Path | None, *, base: Path | None = None) -> Path:
    if value is None or str(value).strip() == "":
        return base if base is not None else PROJECT_ROOT

    path = Path(str(value)).expanduser()
    if not path.is_absolute():
        base_dir = base if base is not None else PROJECT_ROOT
        path = base_dir / path

    return path.resolve()


def get_dataset_root(dataset_name: str | None = None) -> Path:
    config = load_config()
    dataset_name = dataset_name or config["General"].get("dataset", "conference").strip()
    dataset_folder = config["Paths"].get("dataset_folder", "datasets")
    dataset_base = resolve_path(dataset_folder)

    if dataset_base.name == "datasets" or dataset_base.name == "dataset":
        return (dataset_base / dataset_name).resolve()

    if dataset_base.exists() and dataset_base.is_dir():
        if (dataset_base / dataset_name).exists():
            return (dataset_base / dataset_name).resolve()

    return (PROJECT_ROOT / "datasets" / dataset_name).resolve()


def get_dataset_dir(dataset_name: str | None = None) -> Path:
    return (get_dataset_root(dataset_name) / "ontologies").resolve()


def get_alignment_dir(dataset_name: str | None = None) -> Path:
    config = load_config()
    dataset_root = get_dataset_root(dataset_name)
    alignment_folder = config["Paths"].get("alignment_folder", "/alignments/")
    adjusted = alignment_folder.strip()

    if adjusted.startswith("/"):
        resolved = resolve_path(adjusted.lstrip("/"), base=PROJECT_ROOT)
    else:
        resolved = resolve_path(adjusted, base=dataset_root)

    if resolved.exists() and resolved.is_dir():
        return resolved.resolve()

    return (dataset_root / "alignments").resolve()


def get_model_path(model_name: str | None = None) -> Path:
    config = load_config()
    model_name = model_name or config["Paths"].get("save_model_path", "saved_models/conference.pt")
    return resolve_path(model_name, base=PROJECT_ROOT)
