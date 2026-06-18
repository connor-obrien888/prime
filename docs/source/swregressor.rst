The SWRegressor Class and Solar Wind Prediction
===============================================

The ``primesw`` package is based around the ``SWRegressor`` class.
This class subclasses the Pytorch Lightning ``LightningModule``, and is flexible enough to load different model configurations that were trained to predict different plasma environments.
For instance, you can load a ``SWRegressor`` wrapping the solar wind prediction model PRIME by specifying the keyword ``'PRIME'`` to ``primesw.load()``:

.. code-block:: python

    import primesw as psw
    prime = psw.load('PRIME')

.. autofunction:: primesw.prime_torch.load

The class method ``SWRegressor.predict_ts()`` is the way that most users will interface with the model.
To generate predictions from Wind spacecraft data, specify ``start`` and ``stop`` times for the desired prediction.
``start`` and ``stop`` are strings with format ``'YYYY-MM-DD HH:MM:SS'``.

.. code-block:: python

    import primesw as psw
    prime = psw.load('PRIME')
    prime.predict_ts(start = '2020-01-01 00:00:00', stop = '2020-01-02 00:00:00')

If using data from an L1 monitor to make predictions, pass the input data using the ``in_data`` argument.
If ``in_data`` is specified, ``start`` and ``stop`` should not be (and vice versa).
``in_data`` is also useful for making predicitons from synthetic solar wind data (see ``SWRegressor.build_synth_input()``).
For instance, one can predict what the solar wind at the bow shock nose would be if the solar wind flow at L1 was 700km/s:

.. code-block:: python

    import primesw as psw
    prime = psw.load('PRIME')
    prime.predict_ts(in_data = prime.build_synth_input(vx=-700))

By default for PRIME, predictions are made at the average location of the nose of Earth's bow shock 13.25 Earth Radii upstream on the Geocentric Solar Ecliptic (GSE) x-axis.
One can also specify a position to propagate to besides the default by specifying ``pos``:

.. code-block:: python

    import primesw as psw
    prime = psw.load('PRIME')
    prime.predict_ts(start = '2020-01-01 00:00:00', stop = '2020-01-02 00:00:00', pos = [13.25, 5, 0])

All positions are in GSE coordinates with units of Earth Radii.
It is not recommended to make predictions outside of the region any given model was trained on.
For PRIME, that's within 30 Earth radii of the Earth on the dayside.

.. autoclass:: primesw.prime_torch.SWRegressor
    :members: