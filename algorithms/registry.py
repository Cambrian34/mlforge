from __future__ import annotations
import json
from pathlib import Path

from algorithms.catalog import PLACEHOLDER_ALGORITHMS

DEFINITIONS_DIR = Path(__file__).parent / "definitions"


class AlgorithmRegistry:
    """Loads algorithm metadata from JSON definition files.

    Usage::

        registry = AlgorithmRegistry()
        algo = registry.get("random_forest_classifier")
    """

    def __init__(self, definitions_dir: Path = DEFINITIONS_DIR):
        self._definitions_dir = Path(definitions_dir)
        self._algorithms: dict[str, dict] = {}
        self._load()

    def _load(self) -> None:
        for json_file in sorted(self._definitions_dir.glob("*.json")):
            with open(json_file, encoding="utf-8") as f:
                data = json.load(f)
            algo_id = data["id"]
            self._algorithms[algo_id] = data

        for algo in PLACEHOLDER_ALGORITHMS:
            self._algorithms.setdefault(algo["id"], algo)

    def get(self, algo_id: str) -> dict | None:
        """Return metadata for *algo_id*, or None if not found."""
        return self._algorithms.get(algo_id)

    def all(self) -> list[dict]:
        """Return all algorithm metadata dicts, sorted by name."""
        return sorted(self._algorithms.values(), key=lambda a: a["name"])

    def by_task(self, task: str) -> list[dict]:
        """Return algorithms that support *task* ('classification' or 'regression')."""
        return [a for a in self.all() if task in a.get("tasks", [])]

    def by_framework(self, framework: str) -> list[dict]:
        """Return algorithms for a specific framework."""
        return [a for a in self.all() if a.get("framework") == framework]

    def ids(self) -> list[str]:
        return list(self._algorithms.keys())
