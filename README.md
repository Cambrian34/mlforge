# MLForge

> **Visual machine-learning code generator — deterministic templates, no AI.**

Configure → Validate → Generate → Run

---

## What is MLForge?

MLForge lets you visually configure machine-learning models and automatically generates runnable Python source code. Every generated program comes from deterministic Jinja2 templates with no generative AI.


---

## Setup

### 1. Install dependencies

```bash
cd mlforge
pip install -e ".[dev]"
```

Or without dev tools:

```bash
pip install -e .
```

### 2. Run the application

```bash
python app/main.py
```

Or, if installed:

```bash
mlforge
```

---

## Running tests- in progress

```bash
pytest
```

---

## Project structure
'''
In progress
```

---

## Architecture

```
GUI → ModelSpec (IR) → Validator → Code Generator → Generated Project
```

The GUI never generates code directly. It builds a `ModelSpec` (a Python dataclass / JSON intermediate representation), which the validator checks and the generator renders through Jinja2 templates.

This makes the system **predictable, reproducible, testable**, and easy to extend with new algorithms (add a JSON definition + no GUI changes required).

---

## Adding a new algorithm- still being worked on

1. Create `algorithms/definitions/<your_algorithm>.json` following the existing schema.
2. Add a template path reference (can reuse `sklearn/model.py.jinja`).
3. Update `generator/generator.py`'s `sklearn_class_map` with the new ID → class name mapping.

---
