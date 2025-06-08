# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

import os
import sys
sys.path.insert(0, os.path.abspath(".."))

project = 'TTFEMesh'
copyright = '2025, Mazen Ali'
author = 'Mazen Ali'
release = '0.1.0'

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
    "sphinx.ext.linkcode",
    "sphinx.ext.githubpages",
    "sphinx.ext.intersphinx",
    "myst_parser",
    "nbsphinx",
    "sphinx_copybutton",
]

templates_path = ['_templates']
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = 'sphinx_rtd_theme'
html_static_path = ['_static']

# Theme options
html_theme_options = {
    'navigation_depth': 2,
    'titles_only': True,
    'logo_only': False,
    'style_nav_header_background': '#2980B9',
    'collapse_navigation': False,
    'sticky_navigation': True,
    'includehidden': False,
}

# GitHub configuration
html_context = {
    'display_github': True,
    'github_user': 'MazenAli',
    'github_repo': 'TT-FEMesh',
    'github_banner': True,
    'github_button': True,
    'github_type': 'star',
    'github_count': True,
    'github_url': 'https://github.com/MazenAli/TT-FEMesh',
    'github_version': 'main',  # or your default branch name
    'conf_py_path': '/docs/',  # path in the checkout to the docs root
}

autodoc_member_order = "bysource"
autodoc_default_options = {
    "members": True,
    "undoc-members": True,
    "show-inheritance": True,
}

# Enable source code links
viewcode_follow_imported_members = True
viewcode_enable_epub = True

# -- Intersphinx configuration -----------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/extensions/intersphinx.html

intersphinx_mapping = {
    'python': ('https://docs.python.org/3', None),
    'numpy': ('https://numpy.org/doc/stable/', None),
    'torch': ('https://pytorch.org/docs/stable/', None),
}

# -- Napoleon settings ------------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/extensions/napoleon.html

napoleon_google_docstring = True
napoleon_numpy_docstring = True
napoleon_include_init_with_doc = True
napoleon_include_private_with_doc = False
napoleon_include_special_with_doc = True
napoleon_use_admonition_for_examples = True
napoleon_use_admonition_for_notes = True
napoleon_use_admonition_for_references = True
napoleon_use_ivar = True
napoleon_use_param = True
napoleon_use_rtype = True
napoleon_type_aliases = None

# -- MyST settings ---------------------------------------------------------
# https://myst-parser.readthedocs.io/en/latest/configuration.html

myst_enable_extensions = [
    'amsmath',
    'colon_fence',
    'deflist',
    'dollarmath',
    'html_image',
    'html_admonition',
    'replacements',
    'smartquotes',
    'substitution',
    'tasklist',
] 


def linkcode_resolve(domain, info):
    if domain != "py":
        return None
    if not info["module"]:
        return None
    filename = info["module"].replace(".", "/")
    return f"https://github.com/MazenAli/TT-FEMesh/blob/main/{filename}.py"