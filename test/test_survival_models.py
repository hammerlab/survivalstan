
import statsmodels
import survivalstan
import random
from nose.tools import ok_, nottest
random.seed(9001)
num_iter = 300

def load_test_dataset():
    ''' Load test dataset from R survival package
    '''
    dataset = statsmodels.datasets.get_rdataset(package = 'survival', dataname = 'flchain' )
    d  = dataset.data.query('futime > 7').sample(frac = 0.1)
    d.reset_index(level = 0, inplace = True)
    d.rename(columns={'futime': 't', 'death': 'event'}, inplace=True)
    return(d)

def sim_test_dataset():
    dataset = survivalstan.sim.sim_data_exp(N = 100, censor_time = 10, rate = 0.9)
    dataset.reset_index(level = 0, inplace = True)
    return(dataset)


def test_weibull_model_sim():
    ''' Test weibull survival model on simulated dataset
    '''
    d = sim_test_dataset()
    testfit = survivalstan.fit_stan_survival_model(
        model_cohort = 'test model',
        model_code = survivalstan.models.weibull_survival_model,
        df = d,
        time_col = 't',
        event_col = 'event',
        formula = '~ 1',
        iter = num_iter,
        chains = 2,
        make_inits = survivalstan.make_weibull_survival_model_inits,
        seed = 9001
        )
    ok_('fit' in testfit)
    ok_('coefs' in testfit)
    ok_('loo' in testfit)
    return(testfit)


def test_exp_model_sim():
    ''' Test exp survival model on simulated dataset
    '''
    d = sim_test_dataset()
    testfit = survivalstan.fit_stan_survival_model(
        model_cohort = 'test model',
        model_code = survivalstan.models.exp_survival_model,
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


def test_pem_model_sim():
    ''' Test PEM survival model on simulated dataset
    '''
    d = sim_test_dataset()
    dlong = survivalstan.prep_data_long_surv(d, time_col = 't', event_col = 'event')
    testfit = survivalstan.fit_stan_survival_model(
        model_cohort = 'test model',
        model_code = survivalstan.models.pem_survival_model,
        df = dlong,
        sample_col = 'index',
        timepoint_end_col = 'end_time',
        event_col = 'end_failure',
        formula = '~ 1',
        iter = num_iter,
        chains = 2,
        seed = 9001
        )
    ok_('fit' in testfit)
    ok_('coefs' in testfit)
    ok_('loo' in testfit)
    return(testfit)


def test_weibull_model():
    ''' Test Weibull survival model on flchain dataset
    '''
    d = load_test_dataset()
    testfit = survivalstan.fit_stan_survival_model(
        model_cohort = 'test model',
        model_code = survivalstan.models.weibull_survival_model,
        df = d,
        time_col = 't',
        event_col = 'event',
        formula = 'age + sex',
        iter = num_iter,
        chains = 2,
        make_inits = survivalstan.make_weibull_survival_model_inits,
        seed = 9001
        )
    ok_('fit' in testfit)
    ok_('coefs' in testfit)
    ok_('loo' in testfit)
    return(testfit)

@nottest
def test_null_weibull_model():
    ''' Test Weibull survival model on flchain dataset
    '''
    d = load_test_dataset()
    testfit = survivalstan.fit_stan_survival_model(
        model_cohort = 'test model',
        model_code = survivalstan.models.weibull_survival_model,
        df = d,
        time_col = 't',
        event_col = 'event',
        formula = '~ 1',
        iter = num_iter,
        chains = 2,
        make_inits = survivalstan.make_weibull_survival_model_inits,
        seed = 9001
        )
    ok_('fit' in testfit)
    ok_('coefs' in testfit)
    ok_('loo' in testfit)
    return(testfit)


@nottest
def test_pem_model():
    ''' Test PEM unstructured survival model on test dataset
    '''
    d = load_test_dataset()
    dlong = survivalstan.prep_data_long_surv(d, time_col = 't', event_col = 'event')
    testfit = survivalstan.fit_stan_survival_model(
        model_cohort = 'test model',
        model_code = survivalstan.models.pem_survival_model,
        df = dlong,
        sample_col = 'index',
        timepoint_end_col = 'end_time',
        event_col = 'end_failure',
        formula = 'age + sex',
        iter = num_iter,
        chains = 4,
        seed = 9001
        )
    ok_('fit' in testfit)
    ok_('coefs' in testfit)
    ok_('loo' in testfit)
    return(testfit)


## note: varcoef model has been deprecated
@nottest
def test_pem_varcoef_model():
    ''' Test varying-coef PEM survival model on test dataset
    '''
    d = load_test_dataset()
    dlong = survivalstan.prep_data_long_surv(d, time_col = 't', event_col = 'event')
    testfit = survivalstan.fit_stan_survival_model(
        model_cohort = 'pem varcoef model',
        model_code = survivalstan.models.pem_survival_model_varying_coefs,
        df = dlong,
        sample_col = 'index',
        timepoint_end_col = 'end_time',
        event_col = 'end_failure',
        group_col = 'chapter',
        formula = 'age + sex',
        iter = num_iter,
        chains = 4,
        seed = 9001
        )
    ok_('fit' in testfit)
    ok_('coefs' in testfit)
    ok_('loo' in testfit)
    return(testfit)

