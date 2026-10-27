"""Sphinx configuration for GenerativeModelProtocols.py."""

from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

project = "GenerativeModelProtocols.py"
author = "Jose Romero"
copyright = "2026, Jose Romero"
release = "0.1.0"

extensions = [
    "myst_parser",
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
    "sphinxcontrib.bibtex",
]

source_suffix = {
    ".rst": "restructuredtext",
    ".md": "markdown",
}
myst_enable_extensions = ["amsmath", "dollarmath"]
bibtex_bibfiles = ["refs.bib"]
bibtex_default_style = "unsrt"
bibtex_reference_style = "label"
autosummary_generate = True
autodoc_typehints = "description"
autodoc_member_order = "bysource"
autodoc_mock_imports = ["juliacall"]
templates_path = ["_templates"]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]

html_theme = "sphinx_rtd_theme"
html_theme_options = {
    "navigation_depth": 1,
    "titles_only": True,
}
html_static_path = ["_static"]
html_css_files = ["custom.css"]
