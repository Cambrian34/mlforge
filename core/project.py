from __future__ import annotations
import json
from pathlib import Path
from core.model_spec import ModelSpec


PROJECT_FILENAME = "mlforge.json"


def save_project(spec: ModelSpec, directory: Path | str) -> Path:
    """Serialize a ModelSpec to mlforge.json inside directory."""
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / PROJECT_FILENAME
    with open(path, "w", encoding="utf-8") as f:
        json.dump(spec.to_dict(), f, indent=2)
    return path


def load_project(path: Path | str) -> ModelSpec:
    """Load a ModelSpec from a mlforge.json file."""
    path = Path(path)
    if path.is_dir():
        path = path / PROJECT_FILENAME
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    return ModelSpec.from_dict(data)
