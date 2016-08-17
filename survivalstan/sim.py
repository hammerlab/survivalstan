
"""

   Functions to simulate failure-time data 
   for testing & model checking purposes

"""

import numpy as np
import pandas as pd

def sim_data_exp(N, censor_time, rate):
    """
      simulate true lifetimes (t) according to exponential model

      Parameters
      -----------

        N: (int) number of observations 
        censor_time: (float) uniform censor time for each observation
		rate: (float, positive) hazard rate used to parameterize failure times

	  Returns
	  -------

	  pandas DataFrame with N observations, and 3 columns:
	    - true_t: "actual" simulated failure time
	    - t: observed failure/censor time, given censor_time
	    - event: boolean indicating if failure event was observed (TRUE)
	          or censored (FALSE)
    """
    sample_data = pd.DataFrame({
            'true_t': np.random.exponential((1/rate), size=N) 
            })
    ## censor observations at censor_time
    sample_data['t'] = np.minimum(sample_data['true_t'], censor_time)
    sample_data['event'] = sample_data['t'] >= sample_data['true_t']
    return(sample_data)

