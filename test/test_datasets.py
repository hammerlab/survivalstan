import statsmodels
import survivalstan
import random
random.seed(9001)

def load_test_dataset():
    ''' Load test dataset from R survival package
    '''
    dataset = statsmodels.datasets.get_rdataset(package = 'survival', dataname = 'flchain' )
    d  = dataset.data.query('futime > 7').sample(frac = 0.5)
    d.reset_index(level = 0, inplace = True)
    d.rename(columns={'futime': 't', 'death': 'event'}, inplace=True)
    return(d)


def sim_test_dataset():
    dataset = survivalstan.sim.sim_data_exp(N = 100, censor_time = 10, rate = 0.9)
    dataset.reset_index(level = 0, inplace = True)
    return(dataset)


def load_test_dataset_long():
    ''' Load test dataset from R survival package
    '''
    d = load_test_dataset()
    dlong = survivalstan.prep_data_long_surv(d, time_col = 't', event_col = 'event')
    return dlong

def sim_test_dataset_long():
    d = sim_test_dataset()
    dlong = survivalstan.prep_data_long_surv(d, time_col = 't', event_col = 'event')
    return dlong
