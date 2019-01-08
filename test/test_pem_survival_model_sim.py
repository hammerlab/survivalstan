from matplotlib import pyplot as plt
plt.switch_backend('Agg')
import survivalstan
from stancache import stancache
import numpy as np
from nose.tools import ok_
from functools import partial
num_iter = 500
from .test_datasets import load_test_dataset_long, sim_test_dataset_long

model_code = survivalstan.models.pem_survival_model
make_inits = None

def test_pem_model_sim():
    ''' Test weibull survival model on simulated dataset
    '''
    dlong = sim_test_dataset_long()
    testfit = survivalstan.fit_stan_survival_model(
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
        make_inits = make_inits,
        FIT_FUN = stancache.cached_stan_fit,
        )
    ok_('fit' in testfit)
    ok_('coefs' in testfit)
    ok_('loo' in testfit)
    survivalstan.utils.plot_coefs([testfit])
    survivalstan.utils.plot_coefs([testfit], trans=np.exp)
    survivalstan.utils.plot_coefs([testfit], element='baseline')
    return(testfit)


def test_pem_model_sim_with_formula():
    ''' Test pem survival model using `surv` formula syntax
    '''
    dlong = sim_test_dataset_long()
    testfit = survivalstan.fit_stan_survival_model(
        model_cohort = 'test model',
        model_code = model_code,
        df = dlong,
        formula = 'surv(event_status=end_failure, time=end_time, subject=index) ~ 1',
        iter = num_iter,
        chains = 2,
        seed = 9001,
        make_inits = make_inits,
        FIT_FUN = stancache.cached_stan_fit,
        )
    ok_('fit' in testfit)
    ok_('coefs' in testfit)
    ok_('loo' in testfit)
    survivalstan.utils.plot_coefs([testfit])
    survivalstan.utils.plot_coefs([testfit], trans=np.exp)
    survivalstan.utils.plot_coefs([testfit], element='baseline')
    return(testfit)

def test_pem_model_sim_covar():
    ''' Test weibull survival model on simulated dataset
    '''
    dlong = sim_test_dataset_long()
    testfit = survivalstan.fit_stan_survival_model(
        model_cohort = 'test model',
        model_code = model_code,
        df = dlong,
        sample_col = 'index',
        timepoint_end_col = 'end_time',
        event_col = 'end_failure',
        formula = '~ age + sex',
        iter = num_iter,
        chains = 2,
        seed = 9001,
        make_inits = make_inits,
        FIT_FUN = stancache.cached_stan_fit,
        )
    ok_('fit' in testfit)
    ok_('coefs' in testfit)
    ok_('loo' in testfit)
    survivalstan.utils.plot_coefs([testfit])
    survivalstan.utils.plot_coefs([testfit], trans=np.exp)
    survivalstan.utils.plot_coefs([testfit], element='baseline')
    return(testfit)

def test_pem_model_sim_covar_with_form():
    dlong = sim_test_dataset_long()
    testfit = survivalstan.fit_stan_survival_model(
        model_cohort = 'test model',
        model_code = model_code,
        df = dlong,
        formula = 'surv(event_status=end_failure, time=end_time, subject=index) ~ age + sex',
        iter = num_iter,
        chains = 2,
        seed = 9001,
        make_inits = make_inits,
        FIT_FUN = stancache.cached_stan_fit,
        )
    ok_('fit' in testfit)
    ok_('coefs' in testfit)
    ok_('loo' in testfit)
    survivalstan.utils.plot_coefs([testfit])
    survivalstan.utils.plot_coefs([testfit], trans=np.exp)
    survivalstan.utils.plot_coefs([testfit], element='baseline')
    return(testfit)
