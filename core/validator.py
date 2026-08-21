from __future__ import annotations
from dataclasses import dataclass
from enum import Enum
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from core.model_spec import ModelSpec


class Severity(Enum):
    ERROR = "error"
    WARNING = "warning"
    INFO = "info"


@dataclass
class ValidationIssue:
    severity: Severity
    field: str        # which field/parameter the issue is about
    message: str

    def __str__(self) -> str:
        return f"[{self.severity.value.upper()}] {self.field}: {self.message}"


class ValidationResult:
    def __init__(self, issues: list[ValidationIssue]):
        self.issues = issues

    @property
    def is_valid(self) -> bool:
        return not any(i.severity == Severity.ERROR for i in self.issues)

    @property
    def errors(self) -> list[ValidationIssue]:
        return [i for i in self.issues if i.severity == Severity.ERROR]

    @property
    def warnings(self) -> list[ValidationIssue]:
        return [i for i in self.issues if i.severity == Severity.WARNING]

    def __str__(self) -> str:
        if not self.issues:
            return "✓ Architecture valid"
        return "  ".join(str(i) for i in self.issues)


class Validator:
    """Validates a ModelSpec before code generation."""

    CLASSIFICATION_ONLY = {"logistic_regression", "gaussian_naive_bayes"}
    REGRESSION_ONLY = {"linear_regression", "ridge_regression", "lasso_regression", "elastic_net"}
    BOTH_TASKS = {"random_forest", "knn", "svm", "decision_tree", "gradient_boosting"}

    def validate(self, spec: "ModelSpec", registry=None) -> ValidationResult:
        issues: list[ValidationIssue] = []
        issues.extend(self._check_required_fields(spec))
        issues.extend(self._check_task_algorithm_compatibility(spec))
        issues.extend(self._check_dataset(spec))
        if registry is not None:
            issues.extend(self._check_parameters(spec, registry))
        return ValidationResult(issues)

    def _check_required_fields(self, spec: "ModelSpec") -> list[ValidationIssue]:
        issues = []
        if not spec.project_name.strip():
            issues.append(ValidationIssue(Severity.ERROR, "project_name", "Project name cannot be empty."))
        if not spec.algorithm_id:
            issues.append(ValidationIssue(Severity.ERROR, "algorithm_id", "No algorithm selected."))
        return issues

    def _check_task_algorithm_compatibility(self, spec: "ModelSpec") -> list[ValidationIssue]:
        issues = []
        alg = spec.algorithm_id
        task = spec.task
        if task == "regression" and alg in self.CLASSIFICATION_ONLY:
            issues.append(ValidationIssue(
                Severity.ERROR, "task",f"Algorithm '{alg}' only supports classification, not regression."
            ))
        if task == "classification" and alg in self.REGRESSION_ONLY:
            issues.append(ValidationIssue(
                Severity.ERROR, "task",f"Algorithm '{alg}' only supports regression, not classification."
            ))
        return issues

    def _check_dataset(self, spec: "ModelSpec") -> list[ValidationIssue]:
        issues = []
        ds = spec.dataset
        if not ds.input_columns:
            issues.append(ValidationIssue(Severity.WARNING, "dataset.input_columns","No input columns specified — dataset loader will use all columns except target."))
        if not ds.target_column:
            issues.append(ValidationIssue(Severity.WARNING, "dataset.target_column", "No target column specified."))
        if ds.target_column and ds.target_column in ds.input_columns:
            issues.append(ValidationIssue(Severity.ERROR, "dataset.target_column",f"Target column '{ds.target_column}' must not also be an input column."))
        if not (0.0 < ds.train_split < 1.0):
            issues.append(ValidationIssue(Severity.ERROR, "dataset.train_split","Train split must be between 0 and 1 (exclusive)."))
        return issues

    def _check_parameters(self, spec: "ModelSpec", registry) -> list[ValidationIssue]:
        """Check parameter types and ranges against the algorithm registry metadata."""
        issues = []
        algo_meta = registry.get(spec.algorithm_id)
        if algo_meta is None:
            issues.append(ValidationIssue(Severity.ERROR, "algorithm_id",f"Unknown algorithm '{spec.algorithm_id}'."))
            return issues

        param_defs = algo_meta.get("parameters", {})
        for param_name, param_def in param_defs.items():
            value = spec.parameters.get(param_name)
            if value is None and not param_def.get("nullable", False):
                # Treat as using default
                continue
            if value is None:
                continue
            param_type = param_def.get("type")
            if param_type == "integer":
                if not isinstance(value, int):
                    issues.append(ValidationIssue(Severity.ERROR, f"parameters.{param_name}",f"'{param_name}' must be an integer."))
                else:
                    mn = param_def.get("min")
                    mx = param_def.get("max")
                    if mn is not None and value < mn:
                        issues.append(ValidationIssue(Severity.ERROR, f"parameters.{param_name}",f"'{param_name}' must be >= {mn}."))
                    if mx is not None and value > mx:
                        issues.append(ValidationIssue(Severity.ERROR, f"parameters.{param_name}",f"'{param_name}' must be <= {mx}."))
            elif param_type == "float":
                if not isinstance(value, (int, float)):
                    issues.append(ValidationIssue(Severity.ERROR, f"parameters.{param_name}",f"'{param_name}' must be a number."))
                else:
                    mn = param_def.get("min")
                    mx = param_def.get("max")
                    if mn is not None and value < mn:
                        issues.append(ValidationIssue(Severity.ERROR, f"parameters.{param_name}",f"'{param_name}' must be >= {mn}."))
                    if mx is not None and value > mx:
                        issues.append(ValidationIssue(Severity.ERROR, f"parameters.{param_name}",f"'{param_name}' must be <= {mx}."))
            elif param_type == "enum":
                allowed = param_def.get("values", [])
                if value not in allowed:
                    issues.append(ValidationIssue(Severity.ERROR, f"parameters.{param_name}",f"'{param_name}' must be one of {allowed}."))
        return issues
