from __future__ import annotations
from pathlib import Path
import jinja2

TEMPLATES_DIR = Path(__file__).parent / "templates"


class TemplateEngine:
    """ Wrapper around Jinja2 for rendering MLForge templates."""

    def __init__(self, templates_dir: Path = TEMPLATES_DIR):
        """"
        Initialize the template engine with the templates directory.
        """
        self._env = jinja2.Environment(
            loader=jinja2.FileSystemLoader(str(templates_dir)),
            trim_blocks=True,
            lstrip_blocks=True,
            keep_trailing_newline=True,
            undefined=jinja2.StrictUndefined,
        )

    def render(self, template_path: str, context: dict) -> str:
        """Render a template file with *context*. *template_path* is relative to templates dir."""
        template = self._env.get_template(template_path)
        return template.render(**context)
