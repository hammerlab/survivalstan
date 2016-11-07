
"""

   Functions to simulate failure-time data 
   for testing & model checking purposes

"""

import numpy as np
import pandas as pd
import patsy

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
    sample_data['sex'] = ['female' if np.random.uniform()>0.5 else 'male' for i in np.arange(N)]
    sample_data['age'] = np.random.poisson(55, N)
    sample_data['index'] = np.arange(N)
    return(sample_data)


def _make_sim_rate(df, form, coefs):
    rate_df = patsy.dmatrix(data=df, formula_like=form, return_type='matrix')
    return np.exp(np.dot(rate_df, coefs))


def sim_data_exp_correlated(N, censor_time, rate_form = '1 + age + sex', rate_coefs = [-3, 0.3, 0]):
    """
      simulate true lifetimes (t) according to exponential model

      Parameters
      -----------

        N: (int) number of observations 
        censor_time: (float) uniform censor time for each observation
        rate_form: names of variables to use when estimating rate. defaults to `'1 + age + sex'`
        rate_coefs: inputs to rate-calc (coefs used to estimate log-rate). defaults to `[-3, 0.3, 0]`


    Returns
    -------

    pandas DataFrame with N observations, and 3 columns:
      - true_t: "actual" simulated failure time
      - t: observed failure/censor time, given censor_time
      - event: boolean indicating if failure event was observed (TRUE)
            or censored (FALSE)
      - age: simulated age in years (poisson random variable, expectation = 55)
      - sex: simulated sex, as 'female' or 'male' (uniform 50/50 split)
      - rate: simulated rate value for each obs
    """
    sample_data = pd.DataFrame({
            'sex': ['female' if np.random.uniform()>0.5 else 'male' for i in np.arange(N)],
            'age': np.random.poisson(55, N),
            })

    sample_data['rate'] = _make_sim_rate(df=sample_data, form=rate_form, coefs=rate_coefs)
    ## censor observations at censor_time
    sample_data['true_t'] = sample_data['rate'].apply(lambda rate: np.random.exponential(1/rate))
    sample_data['t'] = np.minimum(sample_data['true_t'], censor_time)
    sample_data['event'] = sample_data['t'] >= sample_data['true_t']
    sample_data['index'] = np.arange(N)
    return(sample_data)

