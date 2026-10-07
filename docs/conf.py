"""
Sphinx configuration for the slv docs.

The full list of settings: https://www.sphinx-doc.org/en/master/usage/configuration.html
"""

from importlib.metadata import version as package_version

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = "Salt Lake Valley py"
copyright = "2026, James Mineau"
author = "James Mineau"
release = package_version("slv")  # from git tags, via setuptools-scm
version = release
# Builds from main (and local builds) are "dev"; release builds are their version.
version_match = "dev" if (".dev" in release or "+" in release) else release

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
    "sphinx.ext.intersphinx",
    "sphinx_autodoc_typehints",
    "sphinx_copybutton",
]

templates_path = ["_templates"]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]

# fips and matplotlib annotate with names they import only for type checking; those
# forward references cannot be resolved and are not ours to fix.
suppress_warnings = ["sphinx_autodoc_typehints.forward_reference"]

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = "pydata_sphinx_theme"
html_title = f"slv {version_match}"
html_static_path = ["_static"]
html_css_files = ["custom.css"]

html_theme_options = {
    "github_url": "https://github.com/jmineau/slv",
    "show_toc_level": 2,
    "navbar_align": "left",
    "navbar_end": ["version-switcher", "theme-switcher", "navbar-icon-links"],
    # The version dropdown. The Documentation workflow publishes dev/ (main),
    # one folder per release and stable/, and writes switcher.json listing them.
    "switcher": {
        "json_url": "https://jmineau.github.io/slv/switcher.json",
        "version_match": version_match,
    },
    "check_switcher": False,  # switcher.json exists only on the deployed site
    "show_version_warning_banner": True,  # point old versions at the latest
}

# -- Extension configuration -------------------------------------------------

# Napoleon settings
napoleon_google_docstring = False
napoleon_numpy_docstring = True
napoleon_include_init_with_doc = True
napoleon_include_private_with_doc = False
napoleon_include_special_with_doc = True
napoleon_use_admonition_for_examples = False
napoleon_use_admonition_for_notes = False
napoleon_use_admonition_for_references = False
napoleon_use_ivar = False
napoleon_use_param = True
napoleon_use_rtype = True
napoleon_preprocess_types = False
napoleon_type_aliases = None
napoleon_attr_annotations = True

# Autodoc settings
autodoc_default_options = {
    "members": True,
    "member-order": "bysource",
    "special-members": "__init__",
    "undoc-members": True,
    "exclude-members": "__weakref__",
}

# Autosummary settings
autosummary_generate = True

# Intersphinx settings
intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    # Assuming the project uses numpy and pandas; modify as needed
    "numpy": ("https://numpy.org/doc/stable/", None),
    "pandas": ("https://pandas.pydata.org/docs/", None),
}
