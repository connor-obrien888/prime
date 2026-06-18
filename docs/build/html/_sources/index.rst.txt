Welcome to PRIME's documentation!
=======================================

``primesw`` is an implementation of the Probabilistic Regressor for Input to the Magnetosphere Estimation (PRIME) architecture.
This package provides an API to access the L1-to-Earth solar wind propagation algorithm PRIME, the L1-to-magnetosheath propagation algorithm PRIME-SH, and the L1-to-plasmasheet propagation algorithm PRIME-PS.
For details on the algorithm development, see the `paper <https://www.frontiersin.org/articles/10.3389/fspas.2023.1250779/full>`_.
For details on the magnetosheath prediction algorithm PRIME-SH, see the `other paper <https://agupubs.onlinelibrary.wiley.com/doi/10.1029/2024JH000235>`_.
Install using pip:

.. code-block:: console

   pip install primesw

You can also find the latest version on `GitHub <https://github.com/connor-obrien888/prime>`_.

.. note::

   This project is still under active development. This note was last updated on June 17, 2026.

Why Use PRIME?
--------------
The solar wind propagation algorithm PRIME is a probabilistic recurrent neural network trained to predict the solar wind conditions just upstream of Earth's bow shock using measurements of the solar wind at the first Earth-Sun Lagrange point (L1). PRIME is capable of predicting:

- Solar wind flow velocity vector (km/s) in GSE coordinates
- Interplanetary magnetic field vector (nT) in GSM coordinates
- Solar wind plasma number density (cm^-3)

PRIME's predictions consist of a mean and a variance defining a Gaussian probability distribution for each parameter. 
Users can use these distributions to define confidence intervals, error bars, or another measure of uncertainty suited for the user's task.
PRIME's probability distributions are reliable to within 3.5% on average, and the means of the distribution are shown to be more accurate predictors of the solar wind than the outputs of other common solar wind propagation algorithms (see the `paper <https://www.frontiersin.org/articles/10.3389/fspas.2023.1250779/full>`_ for more details).

This package also includes the magnetosheath and plasmasheet preidction algorithms PRIME-SH and PRIME-PS.
These algorithms also predict a mean and a variance defining a Gaussian probability distribution for each parameter in the regions they predict.
Since they were trained on different MMS data, they are capable of predicting the perp-to-B and parallel-to-B temperature in the magnetosheath (PRIME-SH), and the electron and ion temperatures in the plasmasheet (PRIME-PS).

Predicting with PRIME
---------------------

Making predictions with PRIME using the ``primesw`` package is done using the ``primesw.SWRegressor`` class. Use the ``primesw.load()`` function to instantiate the model, being sure to specify which model you desire:

.. code-block:: python

   import primesw as psw
   propagator = psw.load('PRIME')
   propagator.predict_ts(start = '2020-01-01 00:00:00', stop = '2020-01-02 00:00:00')


``primesw.SWRegressor`` objects can also make predictions at locations other than Earth's bow shock nose by passing a location given in Geocentric Solar Ecliptic coordinates:

.. code-block:: python

   propagator.predict_ts(start = '2020-01-01 00:00:00', stop = '2020-01-02 00:00:00', pos = [13, 5, 0])


Users can also generate synthetic L1 solar wind data using the ``primesw.SWRegressor.build_synth_input()`` method in order to use PRIME to study solar wind propagtion in a theoretical sense.
It is not recommended to use PRIME to predict the solar wind in areas it was not trained on, or using inputs outside the range of solar wind conditions it was trained on.

Citation
--------
If you make use of PRIME, please cite it:

.. code-block:: console

	@article{obrien_prime_2023,
		title = {{PRIME}: a probabilistic neural network approach to solar wind propagation from {L1}},
		volume = {10},
		issn = {2296-987X},
		shorttitle = {{PRIME}},
		url = {https://www.frontiersin.org/articles/10.3389/fspas.2023.1250779/full},
		doi = {10.3389/fspas.2023.1250779},
		urldate = {2023-11-13},
		journal = {Frontiers in Astronomy and Space Sciences},
		author = {O’Brien, Connor and Walsh, Brian M. and Zou, Ying and Tasnim, Samira and Zhang, Huaming and Sibeck, David Gary},
		month = sep,
		year = {2023}
		}

If you make use of PRIME-SH, please cite it:

.. code-block:: console

	@article{obrien_primesh_2024,
		title = {{PRIME}‐{SH}: {A} {Data}‐{Driven} {Probabilistic} {Model} of {Earth}'s {Magnetosheath}},
		volume = {1},
		issn = {2993-5210, 2993-5210},
		shorttitle = {{PRIME}‐{SH}},
		url = {https://agupubs.onlinelibrary.wiley.com/doi/10.1029/2024JH000235},
		doi = {10.1029/2024JH000235},
		language = {en},
		number = {3},
		urldate = {2024-07-30},
		journal = {Journal of Geophysical Research: Machine Learning and Computation},
		author = {O’Brien, C. and Walsh, B. M. and Zou, Y. and Qudsi, R. and Tasnim, S. and Zhang, H. and Sibeck, D. G.},
		month = sep,
		year = {2024},
		pages = {e2024JH000235},
		}


Contents
--------

``primesw`` can load three different models by default by specifying keywords to ``primesw.load()``.
PRIME for solar wind prediction (``primesw.load('PRIME')``), PRIME-SH for magnetosheath prediction (``primesw.load('PRIME-SH')``), and PRIME-PS for plasmasheet prediction (``primesw.load('PRIME-PS')``).
Their usage is slightly different, please check each section for caveats.

.. toctree::

	install
	usage