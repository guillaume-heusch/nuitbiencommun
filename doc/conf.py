import os

# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = 'nuitbiencommun'
copyright = '2024, Guillaume Heusch'
author = 'Guillaume Heusch'

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    'sphinx.ext.todo',
    'sphinx.ext.coverage',
    'sphinx.ext.ifconfig',
    'sphinx.ext.autodoc',
    'sphinx.ext.autosummary',
    'sphinx.ext.doctest',
    'sphinx.ext.graphviz',
    'sphinx.ext.intersphinx',
    'sphinx.ext.napoleon',
    'sphinx.ext.viewcode',
    'sphinx.ext.mathjax',
    ]

# Generates auto-summary automatically
autosummary_generate = True

# Create numbers on figures with captions
numfig = True

# If we are on OSX, the 'dvipng' path maybe different
dvipng_osx = '/opt/local/libexec/texlive/binaries/dvipng'
if os.path.exists(dvipng_osx): pngmath_dvipng = dvipng_osx


templates_path = ['_templates']

# The suffix of source filenames.
source_suffix = '.rst'


exclude_patterns = ["links.rst"]

# The name of the Pygments (syntax highlighting) style to use.
pygments_style = 'sphinx'

# -- LaTeX ---------------------------------------------------------------------
latex_documents = [
 ('index', 'index.tex', u'CDS', u'Guillaume Heusch', 'manual'),
]




# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output


import sphinx_rtd_theme
html_theme = 'sphinx_rtd_theme'

# Theme options are theme-specific and customize the look and feel of a theme
# further.  For a list of options available for each theme, see the
# documentation.
#html_theme_options = {}

# Add any paths that contain custom themes here, relative to this directory.
html_theme_path = [sphinx_rtd_theme.get_html_theme_path()]


todo_include_todos = True

# Default processing flags for sphinx
autoclass_content = 'class'
autodoc_member_order = 'bysource'
autodoc_default_flags = [
  'members',
  'undoc-members',
  'show-inheritance',
  ]

