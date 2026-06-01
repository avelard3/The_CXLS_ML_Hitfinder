"""Sphinx configuration for The CXLS ML Hitfinder documentation."""

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
sys.path.insert(0, str(SRC))

project = "The CXLS ML Hitfinder"
author = "The CXLS ML Hitfinder contributors"
copyright = "2026, The CXLS ML Hitfinder contributors"

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
]

autosummary_generate = True
autodoc_default_options = {
    "members": True,
    "undoc-members": True,
    "show-inheritance": True,
}
autodoc_typehints = "description"
autodoc_mock_imports = [
    "h5py",
    "matplotlib",
    "numpy",
    "optuna",
    "plotly",
    "sklearn",
    "scipy",
    "torch",
    "torchvision",
]

napoleon_google_docstring = True
napoleon_numpy_docstring = True
napoleon_include_init_with_doc = True

templates_path = ["_templates"]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]

html_theme = "alabaster"
html_static_path = ["_static"]
