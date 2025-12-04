# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = 'lightcone_io'
copyright = '2025, John Helly'
author = 'John Helly'

from importlib.metadata import version
release = version('lightcone_io')
version = '.'.join(release.split('.')[:2])

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    'sphinx.ext.autodoc',
    'sphinx_autodoc_typehints',
    'sphinx.ext.intersphinx',
]

templates_path = ['_templates']
exclude_patterns = []

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = 'sphinx_rtd_theme'
html_static_path = ['_static']

# API documentation ordering
autodoc_member_order = 'bysource'

# This allows linking to sphinx docs of other projects
intersphinx_mapping = dict(
    unyt=("https://unyt.readthedocs.io/en/stable/", None),
    h5py=("https://docs.h5py.org/en/latest/", None),
    hdfstream=("https://hdfstream-python.readthedocs.io/en/latest/", None),
)
