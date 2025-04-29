# Configuration file for the Sphinx documentation builder.
# Started by copy-pasting from readthedocs' tutorial;
#   some of the stuff in here is probably not necessary.
#   Feel free to mess around with it.

import os
import sys

# from primesw import __version__
# from primesw import DocstringInfo

# Source code dir relative to this file;
# used by predictor.rst's autosummary
sys.path.insert(0, os.path.abspath(os.path.join('..', '..', 'prime_lib')))


# -- Project information

project = 'primesw'
release = '0.4.0' #__version__
version = '0.4.0' #__version__
copyright = '2025, primesw Developers'


# -- General configuration

extensions = [
    'sphinx.ext.duration',
    'sphinx.ext.doctest',
    'sphinx.ext.autodoc',
    'sphinx.ext.autosummary',
    'sphinx.ext.intersphinx',
    'sphinx.ext.viewcode',  # Add a link to the Python source code for classes, functions etc.
]

autosummary_generate = True # Turn on sphinx.ext.autosummary
autoclass_content = "both"  # Add __init__ doc (ie. params) to class summaries

intersphinx_mapping = {
    'python': ('https://docs.python.org/3/', None),
    'sphinx': ('https://www.sphinx-doc.org/en/master/', None),
}
intersphinx_disabled_domains = ['std']

templates_path = ['_templates']

# -- Options for HTML output

#html_theme = "pydata_sphinx_theme"  # <-- compiles >4x slower than sphinx_rtd_theme.
html_theme = 'sphinx_rtd_theme'


''' --------------------- reformat pc docstrings --------------------- '''

# def reformat_pc_docstrings(app, what, name, obj, options, lines0):
#     '''modify docstring formatting to be decent even if not written to respect rst standards.
#     Many PlasmaCalcs docstrings don't respect sphinx / python standards,
#         preferring instead to be more readable directly in the source code.
#         (Also, PlasmaCalcs didn't have online docs set up for its first ~2 years,
#             so nobody was checking if it respected sphinx standards...)
#     Thankfully, sphinx provides "autodoc-process-docstring" hook to modify docstrings.
#     That's when this function gets called (see sphinx docs on that hook for more details).

#     app: the sphinx application object
#     what: type of obj: 'module', 'class', 'function', 'method', 'attribute', 'exception'
#     name: fully qualified name of obj
#     obj: the obj itself
#     options: options given to the sphinx directive
#     lines: lines of the docstring. Edit in-place to modify the docstring.

#     E.g. we want functions like this to not look ugly when rendered by sphinx:
#         def f(x,y,z,t, **kw):
#             """one line summary.
#             Longer description, but we don't want to destroy the line breaks;
#                 we also want to keep any indents like this one.

#             x: int. description about x
#             y: None or any value
#                 description about y
#                 extends to multiple lines
#             z: str   # has no description here
#             t: bool.
#                 description about t
#                 extends to multiple lines
#                     and sometimes includes sub-indents on those lines!
#             additional kwargs go to ...

#             returns something.
#             """
#             ...  # code for f goes here.

#     [TODO] spend more time fiddling with this function, to make into more "official" format:
#         e.g. :param p: for params, :returns: for return info...
#     '''
#     if len(lines0)<=1 or all(len(l.strip())==0 for l in lines0):
#         return  # don't mess with anything if only 1 line, or empty.
#     else:
#         di = DocstringInfo('\n'.join(lines0))
#         newlines = di.to_sphinx_lines()
#         lines0[:] = newlines

# def setup(app):
#     app.connect("autodoc-process-docstring", reformat_pc_docstrings)
