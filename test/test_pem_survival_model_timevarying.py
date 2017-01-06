
import matplotlib as mpl
mpl.use('Agg')
import survivalstan
from stancache import stancache
import numpy as np
from nose.tools import ok_
from functools import partial
num_iter = 500
from .test_datasets import load_test_dataset_long, sim_test_dataset_long

model_code = survivalstan.models.pem_survival_model_timevarying
make_inits = None

def test_pem_model(**kwargs):
    ''' Test survival model on test dataset
    '''
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
    survivalstan.utils.plot_coefs([testfit], trans=np.exp, element='grp_coefs')
    survivalstan.utils.plot_coefs([testfit], element='baseline')
    survivalstan.utils.plot_coefs([testfit], element='beta_time')
    survivalstan.utils.plot_coefs([testfit], element='beta_time', trans=np.exp)
    survivalstan.utils.plot_pp_survival([testfit])
    survivalstan.utils.plot_time_betas(models=[testfit], y='beta', x='end_time', coefs=[testfit['x_names'][0]])
    survivalstan.utils.plot_time_betas(models=[testfit], y='exp(beta)')
    survivalstan.utils.plot_time_betas(models=[testfit], y='exp(beta)', ylim=[0, 4])
    first_beta = survivalstan.utils.extract_time_betas([testfit], coefs=[testfit['x_names'][0]])
    survivalstan.utils.plot_time_betas(df=first_beta, by=['coef'], y='beta', x='end_time')
    return(testfit) 

def test_pem_null_model(force=True, **kwargs):
    ''' Test NULL survival model on flchain dataset
    '''
    dlong = load_test_dataset_long()
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
    survivalstan.utils.plot_coefs([testfit], trans=np.exp, element='grp_coefs')
    survivalstan.utils.plot_coefs([testfit], element='baseline')
    survivalstan.utils.plot_coefs([testfit], element='beta_time')
    survivalstan.utils.plot_coefs([testfit], element='beta_time', trans=np.exp)
    survivalstan.utils.plot_pp_survival([testfit])
    survivalstan.utils.plot_time_betas(models=[testfit], y='beta', x='end_time', coefs=[testfit['x_names'][0]])
    survivalstan.utils.plot_time_betas(models=[testfit], y='exp(beta)')
    survivalstan.utils.plot_time_betas(models=[testfit], y='exp(beta)', ylim=[0, 4])
    first_beta = survivalstan.utils.extract_time_betas([testfit], coefs=[testfit['x_names'][0]])
    return(testfit)


