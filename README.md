Stan Models
===============================

author: Jacki Novik

Overview
--------

Library of Stan Models for Computational Biology

Installation / Usage
--------------------

To install use pip:

    $ pip install stanmodels


Or clone the repo:

    $ git clone https://github.com/jburos/stanmodels.git
    $ python setup.py install
    
Contributing
------------

TBD

Example
-------

```
import stanmodels
import stanity
import seaborn as sb
import matplotlib.pyplot as plt

## load flchain test data from R's `survival` package
dataset = statsmodels.datasets.get_rdataset(package = 'survival', dataname = 'flchain' )
d  = dataset.data.query('futime > 7')
d.reset_index(level = 0, inplace = True)

## e.g. fit Weibull survival model
testfit_wei = stanmodels.fit_stan_survival_model(
	model_cohort = 'Weibull model',
	model_code = stanmodels.stan.weibull_survival_model,
	df = d,
	time_col = 'futime',
	event_col = 'death',
	formula = 'age + sex',
	iter = 3000,
	chains = 4,
	make_inits = stanmodels.make_weibull_survival_model_inits
	)

## coefplot for Weibull coefficient estimates
sb.boxplot(x = 'value', y = 'variable', data = testfit_wei['coefs'])


## e.g. fit Piecewise-exponential survival model 
dlong = stanmodels.prep_data_long_surv(d, time_col = 'futime', event_col = 'death')
testfit_pem = stanmodels.fit_stan_survival_model(
	model_cohort = 'PEM model',
	model_code = stanmodels.stan.pem_survival_model,
	df = dlong,
	sample_col = 'index',
	timepoint_end_col = 'end_time',
	event_col = 'end_failure',
	formula = 'age + sex',
	iter = 3000,
	chains = 4,
	)

## coefplot for PEM model results
sb.boxplot(x = 'value', y = 'variable', data = testfit_pem['coefs'])

## e.g. compare models using PSIS-LOO
stanity.loo_compare(testfit_wei['fit'], testfit_pem['fit'])

## compare coefplots 
sb.boxplot(x = 'value', y = 'variable', hue = 'model_cohort',
    data = testfit_pem['coefs'].append(testfit_wei['coefs']))
plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
```


