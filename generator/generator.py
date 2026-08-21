from __future__ import annotations
import zipfile
from pathlib import Path
from typing import TYPE_CHECKING

from generator.template_engine import TemplateEngine

if TYPE_CHECKING:
    from core.model_spec import ModelSpec


class CodeGenerator:
    """Generates a complete project from a ModelSpec.
    
    Usage::

        gen = CodeGenerator()
        files = gen.generate(spec)         # dict[filename -> code string]
        gen.write(spec, output_dir)        # write to disk
        gen.write_zip(spec, zip_path)      # write to ZIP
    """

    # Files to generate (output name -> template path)
    SKLEARN_FILES = {
        "src/model.py":        "sklearn/model.py.jinja",
        "src/dataset.py":      "sklearn/dataset.py.jinja",
        "src/train.py":        "sklearn/train.py.jinja",
        "src/evaluate.py":     "sklearn/evaluate.py.jinja",
        "src/config.py":       "sklearn/config.py.jinja",
        "requirements.txt":    "sklearn/requirements.txt.jinja",
    }

    def __init__(self):
        self._engine = TemplateEngine()

    # Build the Jinja2 template context from a ModelSpec
    def _build_context(self, spec: "ModelSpec") -> dict:
        """Build the Jinja2 template context from a ModelSpec."""
        # Determine the correct sklearn class name based on task
        algorithm_id = spec.algorithm_id
        task = spec.task

        # Map algorithm_id + task to sklearn class
        sklearn_class_map = {
            ("linear_regression", "regression"):         "LinearRegression",
            ("ridge_regression", "regression"):          "Ridge",
            ("lasso_regression", "regression"):          "Lasso",
            ("elastic_net", "regression"):               "ElasticNet",
            ("logistic_regression", "classification"):   "LogisticRegression",
            ("random_forest", "classification"):          "RandomForestClassifier",
            ("random_forest", "regression"):              "RandomForestRegressor",
            ("svm", "classification"):                    "SVC",
            ("svm", "regression"):                        "SVR",
            ("knn", "classification"):                    "KNeighborsClassifier",
            ("knn", "regression"):                        "KNeighborsRegressor",
            ("decision_tree", "classification"):          "DecisionTreeClassifier",
            ("decision_tree", "regression"):              "DecisionTreeRegressor",
            ("gradient_boosting", "classification"):      "GradientBoostingClassifier",
            ("gradient_boosting", "regression"):          "GradientBoostingRegressor",
            ("gaussian_naive_bayes", "classification"):   "GaussianNB",
        }

        sklearn_import_map = {
            "LinearRegression":               "from sklearn.linear_model import LinearRegression",
            "Ridge":                          "from sklearn.linear_model import Ridge",
            "Lasso":                          "from sklearn.linear_model import Lasso",
            "ElasticNet":                     "from sklearn.linear_model import ElasticNet",
            "LogisticRegression":             "from sklearn.linear_model import LogisticRegression",
            "RandomForestClassifier":         "from sklearn.ensemble import RandomForestClassifier",
            "RandomForestRegressor":          "from sklearn.ensemble import RandomForestRegressor",
            "SVC":                            "from sklearn.svm import SVC",
            "SVR":                            "from sklearn.svm import SVR",
            "KNeighborsClassifier":           "from sklearn.neighbors import KNeighborsClassifier",
            "KNeighborsRegressor":            "from sklearn.neighbors import KNeighborsRegressor",
            "DecisionTreeClassifier":         "from sklearn.tree import DecisionTreeClassifier",
            "DecisionTreeRegressor":          "from sklearn.tree import DecisionTreeRegressor",
            "GradientBoostingClassifier":     "from sklearn.ensemble import GradientBoostingClassifier",
            "GradientBoostingRegressor":      "from sklearn.ensemble import GradientBoostingRegressor",
            "GaussianNB":                     "from sklearn.naive_bayes import GaussianNB",
        }

        sklearn_class = sklearn_class_map.get((algorithm_id, task), "UnknownEstimator")
        sklearn_import = sklearn_import_map.get(sklearn_class, "")

        if sklearn_class == "UnknownEstimator":
            # Placeholder support: keep the generated project runnable by producing a
            # clear, explicit stub instead of a broken import.
            sklearn_import = "" 

        # Build clean parameter dict, filtering out task-specific params that don't apply
        params = dict(spec.parameters)
        # Remove task-filtered params
        if task == "classification":
            params.pop("criterion_regression", None)
        elif task == "regression":
            params.pop("criterion_classification", None)

        # Rename task-specific params to their canonical names
        if "criterion_classification" in params:
            params["criterion"] = params.pop("criterion_classification")
        if "criterion_regression" in params:
            params["criterion"] = params.pop("criterion_regression")

        # Filter out None values (let sklearn use defaults)
        params = {k: v for k, v in params.items() if v is not None}

        return {
            "project_name": spec.project_name,
            "project_slug": spec.project_name.replace(" ", "_").lower(),
            "framework": spec.framework,
            "task": task,
            "algorithm_id": algorithm_id,
            "sklearn_class": sklearn_class,
            "sklearn_import": sklearn_import,
            "parameters": params,
            "dataset": spec.dataset,
            "training": spec.training,
            "mlforge_version": spec.mlforge_version,
        }

    # Generate all files and return a dict mapping filename -> rendered content
    def generate(self, spec: "ModelSpec") -> dict[str, str]:
        """Generate all files. Returns a dict mapping filename -> rendered content."""
        context = self._build_context(spec)
        output: dict[str, str] = {}
        for out_name, template_path in self.SKLEARN_FILES.items():
            output[out_name] = self._engine.render(template_path, context)
        # Also add a README
        output["README.md"] = self._render_readme(context)
        return output

    # generate README.md content
    def _render_readme(self, context: dict) -> str:
        lines = [
            f"# {context['project_name']}",
            "",
            f"Generated by **MLForge v{context['mlforge_version']}**.",
            "",
            "## Setup",
            "",
            "```bash",
            "pip install -r requirements.txt",
            "```",
            "",
            "## Train",
            "",
            "```bash",
            "python src/train.py",
            "```",
            "",
            "## Evaluate",
            "",
            "```bash",
            "python src/evaluate.py",
            "```",
        ]
        return "\n".join(lines) + "\n"

    def write(self, spec: "ModelSpec", output_dir: Path | str) -> list[Path]:
        """Write all generated files to *output_dir*."""
        output_dir = Path(output_dir)
        files = self.generate(spec)
        written: list[Path] = []
        for rel_path, content in files.items():
            dest = output_dir / rel_path
            dest.parent.mkdir(parents=True, exist_ok=True)
            dest.write_text(content, encoding="utf-8")
            written.append(dest)
        # Also save mlforge.json
        from core.project import save_project
        written.append(save_project(spec, output_dir))
        return written

    def write_zip(self, spec: "ModelSpec", zip_path: Path | str) -> Path:
        """Write all generated files to a ZIP archive."""
        zip_path = Path(zip_path)
        files = self.generate(spec)
        with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
            for rel_path, content in files.items():
                zf.writestr(rel_path, content)
            # Save mlforge.json inside the zip too
            import json
            zf.writestr("mlforge.json", json.dumps(spec.to_dict(), indent=2))
        return zip_path
