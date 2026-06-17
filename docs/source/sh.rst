Magnetosheath Prediction
========================

The `SWRegressor` class can also load the magnetosheath prediction algorithm PRIME-SH.
You can do so automatically by specifying the keyword `'PRIME-SH'` to `primesw.load()`:

.. code-block:: python

    import primesw as psw
    primesh = psw.load('PRIME-SH')

To generate magnetosheath predictions from Wind spacecraft data, specify ``start`` and ``stop`` times for the desired prediction to the class method ``SWRegressor.predict_ts()``.
``start`` and ``stop`` are strings with format ``'YYYY-MM-DD HH:MM:SS'``.

.. code-block:: python

    import primesw as psw
    primesh = psw.load('PRIME-SH')
    primesh.predict_ts(start = '2020-01-01 00:00:00', stop = '2020-01-02 00:00:00')

If using data from an L1 monitor to make predictions, pass the input data using the ``in_data`` argument.
If ``in_data`` is specified, ``start`` and ``stop`` should not be (and vice versa).
``in_data`` is also useful for making predicitons from synthetic solar wind data (see ``SWRegressor.build_synth_input()``).
For instance, one can predict what the solar wind at the magnetopause nose would be if the solar wind density at L1 is 20 ions per cc:

.. code-block:: python

    import primesw as psw
    primesh = psw.load('PRIME-SH')
    primesh.predict_ts(input = primesh.build_synth_input(n=20))

By default for PRIME-SH, predictions are made at the average middle of Earth's magnetosheath 12.25 Earth Radii upstream on the Geocentric Solar Ecliptic (GSE) x-axis.
One can also specify a position to propagate to besides the default by specifying ``pos``:

.. code-block:: python

    import primesw as psw
    primesh = psw.load('PRIME-SH')
    primesh.predict_ts(start = '2020-01-01 00:00:00', stop = '2020-01-02 00:00:00', pos = [11.25, 5, 0])

All positions are in GSE coordinates with units of Earth Radii.
It is not recommended to make predictions outside of the region PRIME-SH was trained on (within 30 Earth radii of the Earth on the dayside).