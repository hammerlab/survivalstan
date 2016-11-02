
import survivalstan
from stancache import stancache
import numpy as np
from nose.tools import ok_
num_iter = 10000
from .test_datasets import load_test_dataset_long, sim_test_dataset_long

model_code = survivalstan.models.pem_survival_model
make_inits = None

def test_pem_model_sim(force=True, **kwargs):
    ''' Test weibull survival model on simulated dataset
    '''
    dlong = sim_test_dataset_long()
    testfit = stancache.cached_stan_fit(
        survivalstan.fit_stan_survival_model,
        model_cohort = 'test model',
        model_code = model_code,
        df = dlong,
        sample_col = 'index',
        timepoint_end_col = 'end_time',
        event_col = 'end_failure',
        formula = '~ 1',
        iter = num_iter,
        chains = 2,
        seed = 9001,
        force = force,
        make_inits = make_inits,
        **kwargs
        )
    ok_('fit' in testfit)
    ok_('coefs' in testfit)
    ok_('loo' in testfit)
    return(testfit)


def test_pem_model(force=True, **kwargs):
    ''' Test survival model on test dataset
    '''
    dlong = load_test_dataset_long()
    testfit = stancache.cached_stan_fit(
        survivalstan.fit_stan_survival_model,
        model_cohort = 'test model',
        model_code = model_code,
        df = dlong,
        sample_col = 'index',
        timepoint_end_col = 'end_time',
        event_col = 'end_failure',
        formula = 'age + sex',
        iter = num_iter,
        chains = 2,
        seed = 9001,
        force = force,
        make_inits = make_inits,
        **kwargs
        )
    ok_('fit' in testfit)
    ok_('coefs' in testfit)
    ok_('loo' in testfit)
    return(testfit) 

def test_pem_null_model(force=True, **kwargs):
    ''' Test NULL survival model on flchain dataset
    '''
    dlong = load_test_dataset_long()
    testfit = stancache.cached_stan_fit(
        survivalstan.fit_stan_survival_model,
        model_cohort = 'test model',
        model_code = model_code,
        df = dlong,
        sample_col = 'index',
        timepoint_end_col = 'end_time',
        event_col = 'end_failure',
        formula = '~ 1',
        iter = num_iter,
        chains = 2,
        seed = 9001,
        force = force,
        make_inits = make_inits,
        **kwargs
        )
    ok_('fit' in testfit)
    ok_('coefs' in testfit)
    ok_('loo' in testfit)
    return(testfit)


def test_plot_coefs():
    ''' Test plot_coefs
    '''
    testfit = test_pem_model_sim(force=False, cache_only=True)
    survivalstan.utils.plot_coefs([testfit])


def test_plot_coefs_exp():
    ''' Test plot_coefs with np.exp transform
    '''
    testfit = test_pem_model_sim(force=False, cache_only=True)
    survivalstan.utils.plot_coefs([testfit], trans=np.exp)

def test_plot_baseline_hazard():
    ''' Test plot_baseline_hazard
    '''
    testfit = test_pem_model_sim(force=False, cache_only=True)
    survivalstan.utils.plot_coefs([testfit], element='baseline')

