Plasmasheet Prediction
======================

The `SWRegressor` class can also load the plasmasheet prediction algorithm PRIME-PS.
You can do so automatically by specifying the keyword `'PRIME-PS'` to `primesw.load()`:

.. code-block:: python

    import primesw as psw
    primeps = psw.load('PRIME-PS')

To generate plasmasheet predictions from Wind spacecraft data, specify ``start`` and ``stop`` times for the desired prediction to the class method ``SWRegressor.predict_ts()``.
``start`` and ``stop`` are strings with format ``'YYYY-MM-DD HH:MM:SS'``.

.. code-block:: python

    import primesw as psw
    primeps = psw.load('PRIME-PS')
    primeps.predict_ts(start = '2020-01-01 00:00:00', stop = '2020-01-02 00:00:00')

If using data from an L1 monitor to make predictions, pass the input data using the ``in_data`` argument.
If ``in_data`` is specified, ``start`` and ``stop`` should not be (and vice versa).
``in_data`` is also useful for making predicitons from synthetic solar wind data (see ``SWRegressor.build_synth_input()``).
For instance, one can predict what the ion temperature in the middle plasmasheet would be given a solar wind speed of 700 km/s:

.. code-block:: python

    import primesw as psw
    primeps = psw.load('PRIME-PS')
    primeps.predict_ts(in_data = primesh.build_synth_input(vx=-700))

By default for PRIME-PS, predictions are made 15RE downstream of Earth on the Geocentric Solar Ecliptic (GSE) x-axis.
One can also specify a position to propagate to besides the default by specifying ``pos``:

.. code-block:: python

    import primesw as psw
    primeps = psw.load('PRIME-PS')
    primeps.predict_ts(start = '2020-01-01 00:00:00', stop = '2020-01-02 00:00:00', pos = [-15, 5, 0])

All positions are in GSE coordinates with units of Earth Radii.
It is not recommended to make predictions outside of the region PRIME-SH was trained on (within 10-30 Earth radii of the Earth on the nightside).