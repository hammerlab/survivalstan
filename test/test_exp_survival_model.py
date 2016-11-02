
import survivalstan
from stancache import stancache
import numpy as np
from nose.tools import ok_
num_iter = 3000
from .test_datasets import load_test_dataset, sim_test_dataset

model_code = survivalstan.models.exp_survival_model
def test_model_sim(model_code = model_code, force=True, **kwargs):
    ''' Test survival model on simulated dataset
    '''
    d = sim_test_dataset()
    testfit = stancache.cached_stan_fit(survivalstan.fit_stan_survival_model,
        model_cohort = 'test model',
        model_code = model_code,
        df = d,
        time_col = 't',
        event_col = 'event',
        formula = '~ 1',
        iter = num_iter,
        chains = 2,
        seed = 9001,
        force = force,
        **kwargs
        )
    ok_('fit' in testfit)
    ok_('coefs' in testfit)
    ok_('loo' in testfit)
    return(testfit)

def test_model(model_code=model_code, force=True, **kwargs):
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
        seed = 9001
        )
    ok_('fit' in testfit)
    ok_('coefs' in testfit)
    ok_('loo' in testfit)
    return(testfit)	

def test_null_model(model_code=model_code, force=True, **kwargs):
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
        seed = 9001
        )
    ok_('fit' in testfit)
    ok_('coefs' in testfit)
    ok_('loo' in testfit)
    return(testfit)


def test_plot_coefs():
    ''' Test plot_coefs with exp survival model
    '''
    testfit = test_model_sim(force=False)
    survivalstan.utils.plot_coefs([testfit])


def test_plot_coefs_exp():
    ''' Test plot_coefs with exp survival model & np.exp transform
    '''
    testfit = test_model_sim(force=False, cache_only=True)
    survivalstan.utils.plot_coefs([testfit], trans=np.exp)


