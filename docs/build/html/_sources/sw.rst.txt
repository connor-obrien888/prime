Solar Wind Prediction
=====================

To predict the solar wind, ``primesw`` includes the ``prime`` class.
This class loads a pretrained implemetation of the PRIME algorithm, and includes several methods to help users make predictions on grids and/or with synthetic data.
It is recommended to instantiate ``prime`` objects in their default configuration:

.. code-block:: python

    import primesw as psw
    propagator = psw.prime()


The ``prime`` class method ``prime.predict()`` is the way that most users will interface with PRIME.
To generate solar wind predictions from Wind spacecraft data, specify ``start`` and ``stop`` times for the desired prediction.
``start`` and ``stop`` are strings with format ``'YYYY-MM-DD HH:MM:SS'``.

.. code-block:: python

    import primesw as psw
    propagator = psw.prime()
    propagator.predict(start = '2020-01-01 00:00:00', stop = '2020-01-02 00:00:00')

If using data from an L1 monitor to make predictions, pass the input data using ``input`` argument.
If ``input`` is specified, ``start`` and ``stop`` should not be (and vice versa).
``input`` is also useful for making predicitons from synthetic solar wind data (see ``prime.build_synth_input``).
For instance, one can predict what the solar wind at the bow shock nose would be if the solar wind flow at L1 was 700km/s:

.. code-block:: python

    import primesw as psw
    propagator = psw.prime()
    propagator.predict(input = propagator.build_synth_input(vx=-700))

By default, predictions are made at the average location of the nose of Earth's bow shock 13.25 Earth Radii upstream on the Geocentric Solar Ecliptic (GSE) x-axis.
One can also specify a position to propagate to besides the default by specifying ``pos``:

.. code-block:: python

    import primesw as psw
    propagator = psw.prime()
    propagator.predict(start = '2020-01-01 00:00:00', stop = '2020-01-02 00:00:00', pos = [13.25, 5, 0])

All positions are in GSE coordinates with units of Earth Radii.
It is not recommended to make predictions outside of the region PRIME was trained on (within 30 Earth radii of the Earth on the dayside).

.. autoclass:: primesw.prime
    :members: