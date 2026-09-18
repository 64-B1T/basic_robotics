# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html
import os
import shutil
import sys

sys.path.insert(0, os.path.abspath('..'))

# The project README lives at the repo root (so it renders on GitHub/PyPI);
# copy it into the source tree on every build so it can be included as a docs
# page alongside the handwritten .rst content.
shutil.copyfile(
    os.path.join(os.path.dirname(__file__), '..', 'README.md'),
    os.path.join(os.path.dirname(__file__), 'readme.md'),
)

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = 'basic_robotics'
copyright = '2026, William Chapin'
author = 'William Chapin'

version = '1.0.2'
release = '1.0.2'

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.napoleon',
    'sphinx.ext.viewcode',
    'sphinx.ext.intersphinx',
    'myst_parser',
]

source_suffix = {
    '.rst': 'restructuredtext',
    '.md': 'markdown',
}

# Google-style (Args:/Returns:) docstrings throughout the codebase are parsed
# by napoleon; NumPy-style sections are left off since the project doesn't use them.
napoleon_google_docstring = True
napoleon_numpy_docstring = False
napoleon_include_init_with_doc = True
napoleon_use_rtype = False

autodoc_default_options = {
    'members': True,
    'undoc-members': True,
    'show-inheritance': True,
}
autodoc_member_order = 'bysource'

# Optional/hardware-specific dependencies that a plain docs build shouldn't
# require installing (ROS bridges are already import-guarded at runtime, but
# autodoc still imports the module itself to read its docstrings/signatures).
autodoc_mock_imports = ['rospy', 'rclpy', 'std_msgs', 'opcua', 'flask', 'flask_cors']

intersphinx_mapping = {
    'python': ('https://docs.python.org/3', None),
    'numpy': ('https://numpy.org/doc/stable/', None),
    'scipy': ('https://docs.scipy.org/doc/scipy/', None),
}

templates_path = ['_templates']
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']

language = 'en'

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = 'sphinx_rtd_theme'
html_static_path = ['_static']
