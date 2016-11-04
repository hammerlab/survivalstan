
import survivalstan
from stancache import stancache
import numpy as np
from functools import partial
from nose.tools import ok_
num_iter = 500
from .test_datasets import load_test_dataset, sim_test_dataset

model_code = survivalstan.models.exp_survival_model
make_inits = survivalstan.make_weibull_survival_model_inits

def test_model_sim(force=True, **kwargs):
    ''' Test weibull survival model on simulated dataset
    '''
    d = sim_test_dataset()
    testfit = survivalstan.fit_stan_survival_model(
        model_cohort = 'test model',
        model_code = model_code,
        df = d,
        time_col = 't',
        event_col = 'event',
        formula = '~ 1',
        iter = num_iter,
        chains = 2,
        FIT_FUN = partial(stancache.cached_stan_fit, force=force, **kwargs),
        seed = 9001,
        make_inits = make_inits,
        )
    ok_('fit' in testfit)
    ok_('coefs' in testfit)
    ok_('loo' in testfit)
    survivalstan.utils.plot_coefs([testfit])
    survivalstan.utils.plot_coefs([testfit], trans=np.exp)
    return(testfit)


def test_model(force=True, **kwargs):
    ''' Test survival model on test dataset
    '''
    d = load_test_dataset()
    testfit = survivalstan.fit_stan_survival_model(
        model_cohort = 'test model',
        model_code = model_code,
        df = d,
        time_col = 't',
        event_col = 'event',
        formula = 'age + sex',
        iter = num_iter,
        chains = 2,
        FIT_FUN = partial(stancache.cached_stan_fit, force=force, **kwargs),
        seed = 9001,
        make_inits = make_inits,
        )
    ok_('fit' in testfit)
    ok_('coefs' in testfit)
    ok_('loo' in testfit)
    survivalstan.utils.plot_coefs([testfit])
    survivalstan.utils.plot_coefs([testfit], trans=np.exp)
    return(testfit) 


def test_null_model(force=True, **kwargs):
    ''' Test NULL survival model on flchain dataset
    '''
    d = load_test_dataset()
    testfit = survivalstan.fit_stan_survival_model(
        model_cohort = 'test model',
        model_code = model_code,
        df = d,
        time_col = 't',
        event_col = 'event',
        formula = '~ 1',
        iter = num_iter,
        chains = 2,
        FIT_FUN = partial(stancache.cached_stan_fit, force=force, **kwargs),
        seed = 9001,
        make_inits = make_inits,
        )
    ok_('fit' in testfit)
    ok_('coefs' in testfit)
    ok_('loo' in testfit)
    survivalstan.utils.plot_coefs([testfit])
    survivalstan.utils.plot_coefs([testfit], trans=np.exp)
    return(testfit)

