"""Portable discovery of H&M sources and writable runtime artifacts."""

from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Optional
import os


DATASET_FILES = ("transactions_train.csv", "customers.csv", "articles.csv", "images")
KAGGLE_DATASET_RELATIVE_PATH = Path("input/competitions/h-and-m-personalized-fashion-recommendations")
DEFAULT_PRECOMPUTED_ROOT = Path(
    "/kaggle/input/notebooks/classichit/notebook9c33091b06/customer-value-segmentation-pipeline"
)


@dataclass(frozen=True)
class RuntimeContext:
    """Resolved paths for one portable pipeline execution."""

    runtime_name: str
    project_root: Path
    raw_data_root: Path
    precomputed_root: Path | None
    runtime_mode: str
    runtime_root: Path
    processed_root: Path
    feature_root: Path
    aggregate_root: Path
    artifact_root: Path


def discover_runtime(
    project_root: Optional[Path] = None,
    environ: Optional[Mapping[str, str]] = None,
    kaggle_root: Path = Path("/kaggle"),
) -> RuntimeContext:
    """Resolve source priority and create only writable runtime directories."""
    environment = os.environ if environ is None else environ
    root = (Path.cwd() if project_root is None else Path(project_root)).resolve()
    explicit = environment.get("HM_RAW_DATA_DIR")
    precomputed_root = _resolve_precomputed_root(environment)
    kaggle_candidate = Path(kaggle_root) / KAGGLE_DATASET_RELATIVE_PATH
    local_candidate = root / "data" / "raw" / "h-and-m"
    if explicit:
        raw_data_root = Path(explicit).expanduser().resolve()
        if not _is_dataset_root(raw_data_root):
            raise ValueError(_missing_dataset_message())
        runtime_name = "kaggle" if _is_kaggle_path(raw_data_root, kaggle_root) else "local"
    elif _is_dataset_root(kaggle_candidate):
        raw_data_root = kaggle_candidate
        runtime_name = "kaggle"
    elif _is_dataset_root(local_candidate):
        raw_data_root = local_candidate
        runtime_name = "local"
    else:
        raise ValueError(_missing_dataset_message())

    configured_runtime = environment.get("HM_RUNTIME_DIR")
    runtime_root = (
        Path(configured_runtime).expanduser().resolve()
        if configured_runtime
        else (Path(kaggle_root) / "working" / "hm-customer-value" if runtime_name == "kaggle" else root / "data" / "runtime")
    )
    processed_root = runtime_root / "processed"
    feature_root = runtime_root / "features" / "product_images"
    aggregate_root = runtime_root / "aggregates"
    artifact_root = runtime_root / "artifacts"
    for path in (processed_root, feature_root, aggregate_root, artifact_root):
        path.mkdir(parents=True, exist_ok=True)
    return RuntimeContext(
        runtime_name=runtime_name,
        project_root=root,
        raw_data_root=raw_data_root,
        precomputed_root=precomputed_root,
        runtime_mode="precomputed" if precomputed_root is not None else runtime_name,
        runtime_root=runtime_root,
        processed_root=processed_root,
        feature_root=feature_root,
        aggregate_root=aggregate_root,
        artifact_root=artifact_root,
    )


def _is_dataset_root(path: Path) -> bool:
    return path.is_dir() and all((path / name).exists() for name in DATASET_FILES)


def _resolve_precomputed_root(environment: Mapping[str, str]) -> Path | None:
    explicit = environment.get("HM_PRECOMPUTED_DIR")
    candidate = Path(explicit).expanduser() if explicit else DEFAULT_PRECOMPUTED_ROOT
    return candidate.resolve() if candidate.is_dir() else None


def _is_kaggle_path(path: Path, kaggle_root: Path) -> bool:
    try:
        path.resolve().relative_to(Path(kaggle_root).resolve())
        return True
    except ValueError:
        return False


def _missing_dataset_message() -> str:
    return (
        "H&M dataset was not found.\n\nProvide it using one of:\n\n"
        "1. HM_RAW_DATA_DIR=/path/to/h-and-m\n"
        "2. Attach the H&M competition dataset in Kaggle\n"
        "3. Place it under data/raw/h-and-m"
    )
