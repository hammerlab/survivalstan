import survivalstan
from .test_formulas import _dict_keys_include
from nose.tools import ok_, eq_
import pandas as pd


def get_test_data(n=50):
    dataset = survivalstan.sim.sim_data_exp_correlated(N=n,
                                                       censor_time=10)
    dataset.rename(columns={'index': 'subject_id',
                            'event': 'event_value',
                            't': 'time',
                           }, inplace=True)
    return(dataset)

def get_alt_test_data(n=100):
    data = survivalstan.sim.sim_data_jointmodel(N=n)
    df = pd.merge(data['events'].query('event_name == "death"'),
                  data['covars'], on='subject_id')
    return(df)

def test_basic_SurvivalStanData(df=get_test_data(),
                                stan_data_keys=['event', 'y', 'x', 'M', 'N'],
                                **kwargs):
    ''' Test that SurvivalStanData works with old syntax
    '''
    ssdata = survivalstan.SurvivalStanData(formula = ' ~ 1', df=df,
                                         event_col='event_value',
                                           time_col='time',
                                           **kwargs)
    _dict_keys_include(ssdata.data, stan_data_keys)
    return(ssdata)

test_basic_SurvivalStanData(df=get_alt_test_data())

def test_basic_SurvivalStanData_with_sample(df=get_test_data()):
    ## note - also tests for safety against redundant id names
    ssdata = test_basic_SurvivalStanData(df=df,
                                        stan_data_keys=['event','t','t_obs','t_dur','T','S','s','M','N','x'],
                                        sample_col='subject_id')

def test_basic_SurvivalStanData_with_sample_and_group(df=get_test_data()):
    ## note - also tests for safety against redundant id names
    ssdata = test_basic_SurvivalStanData(df=df,
                                        stan_data_keys=['event','t','t_obs','t_dur','T','S','s','M','N','x','G','g'],
                                        sample_col='subject_id',
                                        group_col='sex')
    grp_ids = ssdata.get_group_names()
    eq_(len(grp_ids), 2)


