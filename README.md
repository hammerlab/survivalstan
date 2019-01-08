[![Build Status](https://travis-ci.org/hammerlab/survivalstan.svg?branch=master)](https://travis-ci.org/hammerlab/survivalstan) 
[![Coverage Status](https://img.shields.io/coveralls/hammerlab/survivalstan.svg)](https://coveralls.io/github/hammerlab/survivalstan?branch=master)
[![PyPI version](https://img.shields.io/pypi/v/survivalstan.svg)](https://pypi.python.org/pypi/survivalstan)

survivalstan: Survival Models in Stan
===============================

author: Jacki Novik

Overview
--------

Library of Stan Models for Survival Analysis

Features:

* Variety of standard survival models
	- Weibull, Exponential, and Gamma parameterizations
	- PEM models with variety of baseline hazards
	- PEM model with varying-coefficients (by group)
	- PEM model with time-varying-effects
* Extensible framework - bring your own Stan code, or edit the models above
* Uses [pandas](http://pandas.pydata.org) data frames & [patsy](https://pypi.python.org/pypi/patsy) formulas
* Graphical posterior predictive checking (currently PEM models only)
* Plot posterior estimates of key parameters using [seaborn](https://pypi.python.org/pypi/seaborn)
* Annotate posterior draws of parameter estimates, format as [pandas](http://pandas.pydata.org) dataframes
* Works with extensions to [pystan](https://pystan.readthedocs.io/en/latest/), such as [stancache](http://github.com/jburos/stancache) or [pystan-cache](https://github.com/paulkernfeld/pystan-cache)

Support
-------

Documentation is available [online](http://jburos.github.io/survivalstan).

For help, please reach out to us on [gitter](https://gitter.im/survivalstan).

Installation / Usage
--------------------

Install using pip, as:

    $ pip install survivalstan


Or, you can clone the repo:

    $ git clone https://github.com/hammerlab/survivalstan.git
    $ pip install .

Contributing
------------

Please contribute to survivalstan development by letting us know if you encounter any [bugs](http://github.com/hammerlab/survivalstan/issues) or have specific [feature requests](http://github.com/hammerlab/survivalstan/issues).

In addition, we welcome contributions of:

* Stan code for survival models
* Worked examples, as jupyter notebooks or markdown documents

Usage examples
--------------

There are several examples included in the [example-notebooks](http://nbviewer.jupyter.org/github/hammerlab/survivalstan/tree/master/example-notebooks/), roughly one corresponding to each model.

If you are not sure where to start, [Test pem_survival_model with simulated data.ipynb](http://nbviewer.jupyter.org/github/hammerlab/survivalstan/blob/master/example-notebooks/Test%20pem_survival_model%20with%20simulated%20data.ipynb) contains the most explanatory text. Many of the other notebooks are sparse on explanation, but do illustrate variations on the different models.

For basic usage:

```
import survivalstan
import stanity
import seaborn as sb
import matplotlib.pyplot as plt
import statsmodels

## load flchain test data from R's `survival` package
dataset = statsmodels.datasets.get_rdataset(package = 'survival', dataname = 'flchain' )
d  = dataset.data.query('futime > 7')
d.reset_index(level = 0, inplace = True)

## e.g. fit Weibull survival model
testfit_wei = survivalstan.fit_stan_survival_model(
	model_cohort = 'Weibull model',
	model_code = survivalstan.models.weibull_survival_model,
	df = d,
	time_col = 'futime',
	event_col = 'death',
	formula = 'age + sex',
	iter = 3000,
	chains = 4,
	make_inits = survivalstan.make_weibull_survival_model_inits
	)

## coefplot for Weibull coefficient estimates
sb.boxplot(x = 'value', y = 'variable', data = testfit_wei['coefs'])

## or, use plot_coefs
survivalstan.utils.plot_coefs([testfit_wei])

## print summary of MCMC draws from posterior for each parameter
print(testfit_wei['fit'])


## e.g. fit Piecewise-exponential survival model 
dlong = survivalstan.prep_data_long_surv(d, time_col = 'futime', event_col = 'death')
testfit_pem = survivalstan.fit_stan_survival_model(
	model_cohort = 'PEM model',
	model_code = survivalstan.models.pem_survival_model,
	df = dlong,
	sample_col = 'index',
	timepoint_end_col = 'end_time',
	event_col = 'end_failure',
	formula = 'age + sex',
	iter = 3000,
	chains = 4,
	)

## print summary of MCMC draws from posterior for each parameter
print(testfit_pem['fit'])

## coefplot for PEM model results
sb.boxplot(x = 'value', y = 'variable', data = testfit_pem['coefs'])

## plot baseline hazard (only PEM models)
survivalstan.utils.plot_coefs([testfit_pem], element='baseline')

## posterior-predictive checking (only PEM models)
survivalstan.utils.plot_pp_survival([testfit_pem])

## e.g. compare models using PSIS-LOO
stanity.loo_compare(testfit_wei['loo'], testfit_pem['loo'])

## compare coefplots 
sb.boxplot(x = 'value', y = 'variable', hue = 'model_cohort',
    data = testfit_pem['coefs'].append(testfit_wei['coefs']))
plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)

## (or, use survivalstan.utils.plot_coefs)
survivalstan.utils.plot_coefs([testfit_wei, testfit_pem])

```


