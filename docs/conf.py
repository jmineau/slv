"""
Sphinx configuration for the slv docs.

The full list of settings: https://www.sphinx-doc.org/en/master/usage/configuration.html
"""

import sys
from importlib.metadata import version as package_version
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "_ext"))

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
    "matplotlib.sphinxext.plot_directive",  # `.. plot::` runs code, shows the figure
    "IPython.sphinxext.ipython_directive",  # `.. ipython::` runs code in an .rst page
    "IPython.sphinxext.ipython_console_highlighting",  # colors its In/Out prompts
    "sphinx_autodoc_typehints",
    "sphinx_copybutton",
    "api_pages",  # _ext/api_pages.py: class pages with member tables
]

templates_path = ["_templates"]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]

# fips and matplotlib annotate with names they import only for type checking; those
# forward references cannot be resolved and are not ours to fix.
suppress_warnings = ["sphinx_autodoc_typehints.forward_reference"]

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = "pydata_sphinx_theme"
html_title = f"{project} {version_match}"  # the site title (tab, search); not the full dev version
html_static_path = ["_static"]
html_css_files = ["custom.css"]

# Optional branding: drop files in docs/_static/ and the site uses them.
#   logo.svg (logo_dark.svg for the dark theme), favicon.ico, social-preview.png
# The preview image is the picture Slack, Discord and the like show for a link;
# they need a full URL, so it is read from the main branch on GitHub.
_static = Path(__file__).parent / "_static"
if (_static / "favicon.ico").exists():
    html_favicon = "_static/favicon.ico"
html_context = {
    "og_description": "Salt Lake Valley python modules",
    "og_image": (
        "https://raw.githubusercontent.com/jmineau/slv/main/docs/_static/social-preview.png"
        if (_static / "social-preview.png").exists()
        else ""
    ),
}
html_theme_options = {
    "github_url": "https://github.com/jmineau/slv",
    "show_toc_level": 2,
    "navbar_align": "left",
    # The navbar shows the name alone (it defaults to html_title); the version
    # switcher already says which version this is.
    # Text only here; the image is added below when docs/_static/logo.svg exists.
    "logo": {"text": project},
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

if (_static / "logo.svg").exists():
    html_theme_options["logo"]["image_light"] = "_static/logo.svg"
    _dark = "logo_dark.svg" if (_static / "logo_dark.svg").exists() else "logo.svg"
    html_theme_options["logo"]["image_dark"] = f"_static/{_dark}"

# -- Extensions ----------------------------------------------------------------

# Napoleon settings
napoleon_google_docstring = False
napoleon_numpy_docstring = True
napoleon_include_init_with_doc = True
napoleon_include_private_with_doc = False
napoleon_include_special_with_doc = True
napoleon_use_admonition_for_examples = False
napoleon_use_admonition_for_notes = False
napoleon_use_admonition_for_references = False
napoleon_use_ivar = True  # what a class page's tables leave in "Attributes"
napoleon_use_param = True
napoleon_use_rtype = True
napoleon_preprocess_types = False
napoleon_type_aliases = None
napoleon_attr_annotations = True

# Autodoc settings
# A page per module, class, function, and class member, as in pandas'
# reference. The page templates are in _templates/autosummary/.
autodoc_default_options = {
    "member-order": "bysource",
}

# Autosummary settings
autosummary_generate = True

# Examples run when the docs build, so they show real output and fail the build
# when they break. A `.. plot::` directive (in a docstring's Examples section or
# on any page) shows its code and the figure it draws.
plot_include_source = True
plot_html_show_source_link = False
plot_html_show_formats = False
plot_formats = [("png", 100)]

# In an .rst page, `.. ipython:: python` runs its code and shows each line with
# its output; a block that raises or warns fails the build. Its figures (a line
# `@savefig name.png` above the plotting call) are saved under docs/_build.
ipython_savefig_dir = "_build/savefig"

# Intersphinx settings
intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    # Assuming the project uses numpy and pandas; modify as needed
    "numpy": ("https://numpy.org/doc/stable/", None),
    "pandas": ("https://pandas.pydata.org/docs/", None),
}
