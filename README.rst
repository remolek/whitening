================
whitening
================

The package implements in Python with a sklearn-like interface the whitening methods:

- ZCA
- PCA
- Cholesky
- ZCA-cor
- PCA-cor

discussed in [KLS2018]_.


Usage
-----

.. code:: python

    from whitening import whiten
    import numpy as np
    X = np.random.random((10000, 15)) # data array
    trf = whiten().fit(X, method = "zca")
    X_whitened = trf.transform(X)
    X_reconstructed = trf.inverse_transform(X_whitened)
    assert(np.allclose(X, X_reconstructed)) # True


Installation
------------

.. code:: bash

    git clone https://github.com/remolek/whitening.git
    cd whitening; python setup.py install

Requirements
^^^^^^^^^^^^

- NumPy
- SciPy
- scikit-learn


Licence
-------
GPLv3

Authors
-------

'whitening' was rewritten by `Jeremi Ochab <jeremi.ochab@uj.edu.pl>`_

based on:

1. https://CRAN.R-project.org/package=whitening by Korbinian Strimmer, Takoua Jendoubi, Agnan Kessy, Alex Lewin
2. Python implementation https://gist.github.com/joelouismarino/ce239b5601fff2698895f48003f7464b by Joe Marino
3. sklearn interface from https://github.com/mwv/zca by Maarten Versteegh

.. [KLS2018] Kessy, Lewin, and Strimmer (2018) ``Optimal whitening and decorrelation'', https://doi.org/10.1080/00031305.2016.1277159.
