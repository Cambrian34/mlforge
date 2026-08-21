from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any


@dataclass
class DatasetConfig:
    """Configuration for dataset loading."""

    source_type: str = "csv"  # csv only for v0.1
    file_path: str = ""  # path to CSV file
    input_columns: list[str] = field(default_factory=list)  # feature columns
    target_column: str = ""  # label/target column
    train_split: float = 0.8  # fraction for training


@dataclass
class TrainingConfig:
    """Configuration for training (sklearn = mostly unused in v0.1)."""

    test_size: float = 0.2
    random_state: int = 42
    cross_validate: bool = False
    cv_folds: int = 5


@dataclass
class ModelSpec:
    """Central Intermediate Representation. The GUI produces this; the generator consumes it."""

    project_name: str = "MyProject"
    framework: str = "sklearn"  # sklearn only for v0.1
    task: str = "classification"  # classification | regression
    algorithm_id: str = ""  # e.g. random_forest_classifier
    parameters: dict[str, Any] = field(default_factory=dict)
    dataset: DatasetConfig = field(default_factory=DatasetConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    mlforge_version: str = "0.1.0"

    def to_dict(self) -> dict:
        """Serialize to a JSON-serialisable dict."""
        import dataclasses

        return dataclasses.asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> ModelSpec:
        """Deserialise from a dict (e.g. loaded from mlforge.json)."""
        dataset = DatasetConfig(**data.pop("dataset", {}))
        training = TrainingConfig(**data.pop("training", {}))
        return cls(dataset=dataset, training=training, **data)
