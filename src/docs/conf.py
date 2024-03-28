import os
import sys

# enable autodoc to load local modules
sys.path.insert(0, os.path.abspath("."))
sys.path.insert(0, os.path.abspath(".."))

project = "Initiation au ML"
copyright = ""
author = "Romain Tavenard"
extensions = ["sphinx.ext.autodoc", "sphinx.ext.intersphinx", 'sphinx.ext.napoleon']
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]
templates_path = ["_templates"]
html_theme = "pydata_sphinx_theme"
# html_static_path = ["_static"]
intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None)
}
html_theme_options = {
    "nosidebar": True,
    "logo": {
        "text": "Initiation au ML"
    }
}
html_sidebars = {
  "*": [],
}
language = "fr"
add_module_names = False