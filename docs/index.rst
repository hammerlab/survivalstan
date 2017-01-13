.. survivalstan documentation master file, created by
   sphinx-quickstart on Fri Jan  6 06:25:53 2017.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

survivalstan
========

Survivalstan is a library of Survival Models written in `Stan <http://mc-stan.org>`_ 

It also contains a number of utility functions helpful when doing survival analysis. 

Typical usage:
    
:: 
 
    import survivalstan
    # .. prep your data
    fit1 = survivalstan.fit_stan_survival_model(...)
    survivalstan.utils.plot_stan_summary([fit1], metric=’Rhat’) 
    survivalstan.utils.plot_coefs([fit1])

Features
--------

- Variety of standard survival models
    - Weibull, Exponential, and Gamma parameterization
    - A variety of semi-parametric and non-parametric baseline hazards
    - Supports time-varying-coefficients
    - Estimate time-varying effects
    - Varying-effects by group
- Extensible framework - bring your own Stan code, or edit the models provided
- Uses `pandas <https://pandas.pydata.org>`_ data frames & `patsy <https://pypi.python.org/pypi/patsy>`_ formulas
- Graphical posterior predictive checking
- Plot posterior estimates of key parameters using `seaborn <https://pypi.python.org/pypi/seaborn>`_
- Annotated posterior draws of parameter estimates, as `pandas <http://pandas.pydata.org>`_ dataframes
- Supports caching via as `stancache <http://github.com/jburos/stancache>`_ or `pystan-cache <https://github.com/paulkernfeld/pystan-cache>`_

Installation
------------

Install survivalstan via pip:

    pip install survivalstan

Contribute
----------

- Issue Tracker: github.com/hammerlab/survivalstan/issues
- Source Code: github.com/survivalstan/survivalstan

Support
-------

If you are having issues or questions, please let us know.

We can be reached by email (via github) or via https://gitter.im/survivalstan 

License
-------

The project is licensed under the Apache 2.0 license.


Contents:

.. toctree::
   :maxdepth: 2

   Usage.ipynb

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`

