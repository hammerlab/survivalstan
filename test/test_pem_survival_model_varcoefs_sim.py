
import matplotlib as mpl
mpl.use('Agg')
import survivalstan
from stancache import stancache
import numpy as np
from nose.tools import ok_
from functools import partial
num_iter = 500
from .test_datasets import load_test_dataset_long, sim_test_dataset_long

model_code = survivalstan.models.pem_survival_model_varying_coefs
make_inits = None

def test_pem_model(**kwargs):
    ''' Test survival model on test dataset
    '''
    dlong = sim_test_dataset_long()
    testfit = survivalstan.fit_stan_survival_model(
        model_cohort = 'test model',
        model_code = model_code,
        df = dlong,
        sample_col = 'index',
        timepoint_end_col = 'end_time',
        event_col = 'end_failure',
        group_col = 'sex',
        formula = '~ age',
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
    survivalstan.utils.plot_coefs([testfit], trans=np.exp, element='grp_coefs')
    survivalstan.utils.plot_coefs([testfit], element='baseline')
    return(testfit) 

def test_pem_null_model(force=True, **kwargs):
    ''' Test NULL survival model on flchain dataset
    '''
    dlong = sim_test_dataset_long()
    testfit = survivalstan.fit_stan_survival_model(
        model_cohort = 'test model',
        model_code = model_code,
        df = dlong,
        sample_col = 'index',
        timepoint_end_col = 'end_time',
        event_col = 'end_failure',
        group_col = 'sex',
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
    survivalstan.utils.plot_coefs([testfit], trans=np.exp, element='grp_coefs')
    survivalstan.utils.plot_coefs([testfit], element='baseline')
    return(testfit)


