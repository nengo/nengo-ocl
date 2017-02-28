# -*- coding: utf-8 -*-
#
# This file is execfile()d with the current directory set to its
# containing dir.

import os

import nengo_ocl
import sphinx_rtd_theme

extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.autosummary',
    'sphinx.ext.intersphinx',
    'sphinx.ext.mathjax',
    'sphinx.ext.todo',
    'sphinx.ext.viewcode',
    'numpydoc',
]

# -- sphinx.ext.autodoc
autoclass_content = 'both'  # class and __init__ docstrings are concatenated
autodoc_default_flags = ['members']
autodoc_member_order = 'bysource'  # default is alphabetical

# -- sphinx.ext.intersphinx
intersphinx_mapping = {
    'nengo': ('http://pythonhosted.org/nengo/', None),
    'numpy': ('https://docs.scipy.org/doc/numpy', None),
    'pyopencl': ('https://documen.tician.de/pyopencl/', None),
    'scipy': ('https://docs.scipy.org/doc/scipy/reference', None),
}

# -- sphinx.ext.todo
todo_include_todos = True
# -- numpydoc config
numpydoc_show_class_members = False

# -- sphinx
exclude_patterns = ['_build']
source_suffix = '.rst'
source_encoding = 'utf-8'
master_doc = 'index'

# Need to include https Mathjax path for sphinx < v1.3
mathjax_path = ("https://cdn.mathjax.org/mathjax/latest/MathJax.js"
                "?config=TeX-AMS-MML_HTMLorMML")

project = u'Nengo OpenCL'
authors = u'Nengo team'
copyright = u'2017, Nengo team'
version = '.'.join(nengo_ocl.__version__.split('.')[:2])  # Short X.Y version
release = nengo_ocl.__version__  # Full version, with tags
pygments_style = 'default'

# -- Options for HTML output --------------------------------------------------

html_theme = 'sphinx_rtd_theme'
html_theme_path = [sphinx_rtd_theme.get_html_theme_path()]
html_title = "Nengo OpenCL {0} docs".format(release)
html_static_path = ['_static']
html_context = {
    'css_files': [os.path.join('_static', 'custom.css')],
}
html_use_smartypants = True
htmlhelp_basename = 'NengoOCLdoc'
html_last_updated_fmt = ''  # Suppress 'Last updated on:' timestamp
html_show_sphinx = False

# -- Options for LaTeX output -------------------------------------------------

latex_elements = {
    'papersize': 'letterpaper',
    'pointsize': '11pt',
    # 'preamble': '',
}

latex_documents = [
    # (source start file, target, title, author, documentclass [howto/manual])
    ('index', 'nengo_ocl.tex', html_title, authors, 'manual'),
]

# -- Options for manual page output -------------------------------------------

man_pages = [
    # (source start file, name, description, authors, manual section).
    ('index', 'nengo_ocl', html_title, [authors], 1)
]

# -- Options for Texinfo output -----------------------------------------------

texinfo_documents = [
    # (source start file, target, title, author, dir menu entry,
    #  description, category)
    ('index', 'nengo_ocl', html_title, authors, 'Nengo OpenCL',
     'Accelerated Nengo simulations with OpenCL', 'Miscellaneous'),
]
