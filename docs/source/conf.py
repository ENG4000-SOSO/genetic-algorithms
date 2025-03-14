# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html


import os
import logging
import logging.config
import sys
sys.path.insert(0, os.path.abspath(os.path.join('..', '..', 'src')))
logging.config.fileConfig('../../logging_config.ini')


# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = 'Satellite Operations Services Optimizer 2025'
copyright = '2025, Daniel Di Giovanni, Dwumah Anokye, Jasleen Kaur, Amir Ibrahim, Vikramjeet Singh, Darick Mendes'
author = 'Daniel Di Giovanni, Dwumah Anokye, Jasleen Kaur, Amir Ibrahim, Vikramjeet Singh, Darick Mendes'

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.napoleon'
]

templates_path = ['_templates']
exclude_patterns = []

autodoc_member_order = 'bysource'

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = 'sphinx_rtd_theme'
html_static_path = ['_static']
