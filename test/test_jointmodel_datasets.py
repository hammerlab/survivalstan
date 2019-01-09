from matplotlib import pyplot as plt
plt.switch_backend('Agg')
import statsmodels
import survivalstan
import random
random.seed(9001)

import pandas as pd 

def sim_test_dataset_long(n=200):
    data = survivalstan.sim.sim_data_jointmodel(N=n)
    ldf = survivalstan.prep_data_long_surv(data['events'],
                                           event_col='event_value',
                                           event_name='event_name',
                                           time_col='time',
                                           sample_col='subject_id')
    ldf = pd.merge(ldf, data['covars'], on='subject_id', how='outer')
    return(ldf)

