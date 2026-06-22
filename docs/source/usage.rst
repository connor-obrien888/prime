Usage
=====

``primesw`` can load three different models by default by specifying keywords to ``primesw.load()``.
PRIME for solar wind prediction (``primesw.load('PRIME')``), PRIME-SH for magnetosheath prediction (``primesw.load('PRIME-SH')``), and PRIME-PS for plasmasheet prediction (``primesw.load('PRIME-PS')``).
The models are all contained in `SWRegressor` objects and have shared usage, so it's best to start with looking at the `SWRegressor` class.

.. toctree::
   swregressor
   sh
   ps