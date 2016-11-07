
import matplotlib as mpl
mpl.use('Agg')
import survivalstan
from stancache import stancache
import numpy as np
from nose.tools import ok_
from functools import partial
num_iter = 500
from .test_datasets import load_test_dataset_long, sim_test_dataset_long, load_test_dataset

model_code = survivalstan.models.pem_survival_model
make_inits = None

def test_null_pem_model(**kwargs):
    ''' Test weibull survival model on simulated dataset
    '''
    d = load_test_dataset(n=20)
    dlong = survivalstan.prep_data_long_surv(df=d, time_col='t', event_col='event')
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
        **kwargs
        )
    ok_('fit' in testfit)
    ok_('coefs' in testfit)
    ok_('loo' in testfit)
    survivalstan.utils.plot_coefs([testfit])
    survivalstan.utils.plot_coefs([testfit], trans=np.exp)
    survivalstan.utils.plot_coefs([testfit], element='baseline')

    survivalstan.utils.plot_pp_survival([testfit])
    survivalstan.utils.plot_observed_survival(df=d, time_col='t', event_col='event')
    fitsum = survivalstan.utils.filter_stan_summary([testfit], pars='baseline')
    fitsum = survivalstan.utils.filter_stan_summary(testfit['fit'], remove_nan=True)
    survivalstan.utils.print_stan_summary([testfit], pars='lp__')
    survivalstan.utils.plot_stan_summary([testfit], pars='log_baseline_raw')
    return(testfit)


def test_pem_model(**kwargs):
    ''' Test weibull survival model on simulated dataset
    '''
    d = load_test_dataset(n=20)
    dlong = load_test_dataset_long()
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
        **kwargs
        )
    ok_('fit' in testfit)
    ok_('coefs' in testfit)
    ok_('loo' in testfit)
    survivalstan.utils.plot_coefs([testfit])
    survivalstan.utils.plot_coefs([testfit], trans=np.exp)
    survivalstan.utils.plot_coefs([testfit], element='baseline')
    survivalstan.utils.plot_pp_survival([testfit])
    survivalstan.utils.plot_observed_survival(df=d, time_col='t', event_col='event')
    return(testfit)

