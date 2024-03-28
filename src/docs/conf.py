import os
import sys

# enable autodoc to load local modules
sys.path.insert(0, os.path.abspath("."))
sys.path.insert(0, os.path.abspath(".."))

project = "Initiation au ML"
copyright = ""
author = "Romain Tavenard"
extensions = ["sphinx.ext.autodoc", 
              "sphinx.ext.intersphinx", 
              'sphinx.ext.napoleon',
              "sphinx_copybutton"]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]
templates_path = ["_templates"]
html_theme = "pydata_sphinx_theme"
html_static_path = ["_static"]
html_css_files = [
    'css/custom.css',
]
intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None)
}
html_theme_options = {
    "nosidebar": True,
    "logo": {
        "text": "Initiation au ML (L1 MIASHS)"
    },
    "show_toc_level": 3,
    "show_prev_next": False
}
html_sidebars = {
  "*": [],
}
html_show_sourcelink = False

copybutton_exclude = '.linenos, .gp, .go'
copybutton_copy_empty_lines = False

language = "fr"
add_module_names = False