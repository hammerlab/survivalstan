
import statsmodels
import survivalstan
import random
from nose.tools import ok_, nottest
random.seed(9001)

def load_test_dataset():
    ''' Load test dataset from R survival package
    '''
    dataset = statsmodels.datasets.get_rdataset(package = 'survival', dataname = 'flchain' )
    d  = dataset.data.query('futime > 7').sample(frac = 0.2)
    d.reset_index(level = 0, inplace = True)
    return(d)

def test_weibull_model():
    ''' Test Weibull survival model on test dataset
    '''
    d = load_test_dataset()
    testfit = survivalstan.fit_stan_survival_model(
        model_cohort = 'test model',
        model_code = survivalstan.models.weibull_survival_model,
        df = d,
        time_col = 'futime',
        event_col = 'death',
        formula = 'age + sex',
        iter = 1000,
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
    ''' Test Weibull survival model on test dataset
    '''
    d = load_test_dataset()
    dlong = survivalstan.prep_data_long_surv(d, time_col = 'futime', event_col = 'death')
    testfit = survivalstan.fit_stan_survival_model(
        model_cohort = 'test model',
        model_code = survivalstan.models.pem_survival_model,
        df = dlong,
        sample_col = 'index',
        timepoint_end_col = 'end_time',
        event_col = 'end_failure',
        formula = 'age + sex',
        iter = 1000,
        chains = 4,
        seed = 9001
        )
    ok_('fit' in testfit)
    ok_('coefs' in testfit)
    ok_('loo' in testfit)
    return(testfit)


@nottest
def test_pem_varcoef_model():
    ''' Test varying-coef PEM survival model on test dataset
    '''
    d = load_test_dataset()
    dlong = survivalstan.prep_data_long_surv(d, time_col = 'futime', event_col = 'death')
    testfit = survivalstan.fit_stan_survival_model(
        model_cohort = 'pem varcoef model',
        model_code = survivalstan.models.pem_survival_model_varying_coefs,
        df = dlong,
        sample_col = 'index',
        timepoint_end_col = 'end_time',
        event_col = 'end_failure',
        group_col = 'chapter',
        formula = 'age + sex',
        iter = 500,
        chains = 4,
        seed = 9001
        )
    ok_('fit' in testfit)
    ok_('coefs' in testfit)
    ok_('loo' in testfit)
    return(testfit)

