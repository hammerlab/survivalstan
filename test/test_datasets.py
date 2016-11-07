import statsmodels
import survivalstan
import random
random.seed(9001)

def load_test_dataset(n=50):
    ''' Load test dataset from R survival package
    '''
    dataset = statsmodels.datasets.get_rdataset(package='survival', dataname='flchain' )
    d = dataset.data.query('futime > 7').sample(n=n)
    d.reset_index(level=0, inplace=True)
    d.rename(columns={'futime': 't', 'death': 'event'}, inplace=True)
    return(d)


def sim_test_dataset(n=50):
    dataset = survivalstan.sim.sim_data_exp_correlated(N=n, censor_time=10)
    return(dataset)


def load_test_dataset_long(n=20):
    ''' Load test dataset from R survival package
    '''
    d = load_test_dataset(n=n)
    dlong = survivalstan.prep_data_long_surv(d, time_col='t', event_col='event')
    return dlong

def sim_test_dataset_long(n=20):
    d = sim_test_dataset(n=n)
    dlong = survivalstan.prep_data_long_surv(d, time_col='t', event_col='event')
    return dlong
