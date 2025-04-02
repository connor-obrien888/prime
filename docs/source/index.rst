Welcome to PlasmaCalcs's documentation!
=======================================

`primesw` is an implementation of the Probabilistic Regressor for Input to the Magnetosphere Estimation (PRIME) L1-to-Earth solar wind propagation algorithm.
For details on the algorithm development, see the `paper <https://www.frontiersin.org/articles/10.3389/fspas.2023.1250779/full>`_.
Install using pip:

.. code-block:: console

   pip install primesw

You can also find the latest version on `GitHub <https://github.com/connor-obrien888/prime>`_.

.. note::

   This project is still under active development. This note was last updated on April 2, 2025.

Why Use PRIME?
-------
PRIME is a probabilistic recurrent neural network trained to predict the solar wind conditions just upstream of Earth's bow shock using measurements of the solar wind at the first Earth-Sun Lagrange point (L1). PRIME is capable of predicting:
- Solar wind flow velocity vector (km/s) in GSE coordinates
- Interplanetary magnetic field vector (nT) in GSM coordinates
- Solar wind plasma number density (cm^-3)

PRIME's predictions consist of a mean and a variance defining a Gaussian probability distribution for each parameter. 
Users can use these distributions to define confidence intervals, error bars, or another measure of uncertainty suited for the user's task.
PRIME's probability distributions are reliable to within 3.5% on average, and the means of the distribution are shown to be more accurate predictors of the solar wind than the outputs of other common solar wind propagation algorithms (see the [paper](https://www.frontiersin.org/articles/10.3389/fspas.2023.1250779/full) for more details).

Predicting with PRIME
----------

Making predictions with PRIME using the `primesw` package is done using the `primesw.prime` class. `primesw.prime` objects wrap an instance of PRIME that can be used to predict the solar wind conditions at Earth's bow shock nose given a time range:

.. code-block:: console

   import primesw as psw
   propagator = psw.prime()
   propagator.predict(start = '2020-01-01 00:00:00', stop = '2020-01-02 00:00:00')


`primesw.prime` objects can also make predictions at locations other than Earth's bow shock nose by passing a location given in Geocentric Solar Ecliptic coordinates:

.. code-block:: console

   propagator.predict(start = '2020-01-01 00:00:00', stop = '2020-01-02 00:00:00', pos = [13, 5, 0])


Users can also generate synthetic L1 solar wind data using the `primesw.prime.build_synth_input` method in order to use PRIME to study solar wind propagtion in a theoretical sense. It is not recommended to use PRIME to predict the solar wind in areas it was not trained on, or using inputs outside the range of solar wind conditions it was trained on.

Citation
--------
If you make use of PRIME, please cite it:
```
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
	year = {2023},
	pages = {1250779},
}
```

Contents
--------

**primesw** has two submodules, one for solar wind prediction (**prime**) and one for magnetosheath prediction (**primesh**).

.. toctree::

   usage
   api