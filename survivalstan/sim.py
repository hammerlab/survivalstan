
"""

   Functions to simulate failure-time data
   for testing & model checking purposes

"""

import numpy as np
import pandas as pd
import patsy

# -- generic/simple simulate data functions


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
    # censor observations at censor_time
    sample_data['t'] = np.minimum(sample_data['true_t'], censor_time)
    sample_data['event'] = sample_data['t'] >= sample_data['true_t']
    sample_data['sex'] = ['female' if np.random.uniform() > 0.5 else 'male'
                          for i in np.arange(N)]
    sample_data['age'] = np.random.poisson(55, N)
    sample_data['index'] = np.arange(N)
    return(sample_data)


def _make_sim_rate(df, form, coefs):
    rate_df = patsy.dmatrix(data=df, formula_like=form, return_type='matrix')
    return np.exp(np.dot(rate_df, coefs))


def sim_data_exp_correlated(N, censor_time,
                            rate_form='1 + age + sex',
                            rate_coefs=[-3, 0.3, 0]):
    """
      simulate true lifetimes (t) according to exponential model

      Parameters
      -----------

        N: (int) number of observations
        censor_time: (float) uniform censor time for each observation
        rate_form: names of variables to use when estimating rate.
                defaults to `'1 + age + sex'`
        rate_coefs: inputs to rate-calc (coefs used to estimate log-rate).
                defaults to `[-3, 0.3, 0]`


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
            'sex': ['female' if np.random.uniform() > 0.5 else 'male'
                    for i in np.arange(N)],
            'age': np.random.poisson(55, N),
            })

    sample_data['rate'] = _make_sim_rate(df=sample_data,
                                         form=rate_form,
                                         coefs=rate_coefs)
    # censor observations at censor_time
    sample_data['true_t'] = (sample_data['rate']
                             .apply(lambda rate:
                                    np.random.exponential(1/rate)))
    sample_data['t'] = np.minimum(sample_data['true_t'], censor_time)
    sample_data['event'] = sample_data['t'] >= sample_data['true_t']
    sample_data['index'] = np.arange(N)
    return(sample_data)

# -- simulate data for joint models


def _sim_jointmodel_inputs(N,
                           B_matrix=None,
                           p=0.5,
                           sigma_00=1.5,
                           sigma_01=-0.5,
                           sigma_10=-0.5,
                           sigma_11=0.8,
                           sigma_v=0.8,
                           meas_error_scale=1.25,
                           **kwargs):
    ''' Simulate parameter & covariate values for data-generating process,
        some of which are hard-coded. Hard-coded values can be overwritted by
        passing named kwargs to this function.
        Returns a dict of inputs for other sim-data functions in this module.
    '''
    # construct B_matrix if not provided
    if not B_matrix:
        B1_matrix = np.matrix([[np.power(sigma_00, 2), sigma_01],
                              [sigma_10, np.power(sigma_11, 2)]])
        B_matrix = np.zeros(shape=(3, 3))
        B_matrix[0:2, 0:2] = B1_matrix
        B_matrix[2, 2] = np.power(sigma_v, 2)

    # simulate subject-level parameters for random effects
    raneff = np.matrix(np.random.multivariate_normal(mean=(0, 0, 0),
                                                     cov=B_matrix,
                                                     size=N))
    b = raneff[:, 0:2]
    vi = raneff[:, 2]
    # measurement error for longitudinal data simulation

    def meas_error(n=1):
        return(np.random.normal(loc=0, scale=meas_error_scale, size=n))

    # simulate covariate values
    X = np.matrix(np.random.binomial(size=(N, 2), p=p, n=1))
    X_l = X[:, 0:1]  # covariate X for longitudinal submodel
    X_r = X[:, 0:1]  # covariate X for recurrent events submodel
    X_t = X[:, 1:2]  # covariate X for terminal event submodel
    params = {
        'X': X,
        'X_l': X_l,
        'X_r': X_r,
        'X_t': X_t,
        'meas_error': meas_error,
        'vi': vi,
        'b': b,
        'N': N,
        'lambda_t_0': 1.5,
        'lambda_r_0': 2.0,
        'alpha': 2.6,
        'B_t': np.matrix([0.1]),
        'B_r': np.matrix([0.5]),
        'B_l': np.matrix([3., 0.5, 0.5]).transpose(),
        'eta_t': np.matrix([0.5, 0.5]).transpose(),
        'eta_r': np.matrix([0.2, 0.2]).transpose(),
        'censor': 5.5,
        'meas_gap': 0.2,
        }
    if dict(**kwargs):
        params.update(dict(**kwargs))
    return(params)


def _sim_jointmodel_terminal_events(lambda_t_0,
                                    alpha,
                                    vi,
                                    X_t,
                                    B_t,
                                    b,
                                    eta_t,
                                    censor,
                                    **kwargs):
    ''' Simulate data for terminating events
        Returns a dataframe containing:
            - event_status (1:observed, 0:censor)
            - event_time (time to event)
            - index (subject_id)
    '''
    t_df = pd.DataFrame(data=np.random.exponential(lambda_t_0
                                                   * np.exp(alpha*vi
                                                            + X_t*B_t
                                                            + b*eta_t)),
                        columns=['death'])
    t_df['event_status'] = (t_df['death']
                            .apply(lambda x: 1 if x <= censor else 0))
    t_df['event_time'] = (t_df['death']
                          .apply(lambda x: x if x <= censor else censor))
    t_df.reset_index(inplace=True)
    return(t_df.loc[:, ['index', 'event_time', 'event_status']])


def _sim_jointmodel_recurrent_events(lambda_r_0, vi, X_r, B_r, b, eta_r,
                                     N, t_df, max_events=6, **kwargs):
    ''' Simulate data for recurrent events.
        Returns a data frame containing one obs for
            each recurrent event observed.
        Columns include:
            - index (subject_id)
            - recurrence_time (calendar time of recurrent event)
    '''
    r_df = pd.DataFrame(data=np.random.exponential(lambda_r_0
                                                   * np.exp(vi
                                                            + X_r*B_r
                                                            + b*eta_r),
                                                   size=(N, max_events)))
    r_df = r_df.cumsum(axis=1)
    r_df.reset_index(inplace=True)
    r_df2 = pd.melt(r_df, id_vars='index', value_name='recurrence_time')
    del r_df2['variable']
    r_df2 = pd.merge(r_df2, t_df, on='index', how='outer')
    r_df3 = r_df2.query('recurrence_time <= event_time').copy()
    return(r_df3.loc[:, ['index', 'recurrence_time']])


def _sim_jointmodel_longitudinal_biomarker(N, meas_gap, t_df, X_l, b,
                                           meas_error, B_l,
                                           left_censor_at=-0.4,
                                           max_visits=20,
                                           **kwargs):
    ''' Simulate data for longitudinal biomarker correlated with events
        Returns a data frame containing one obs
            for each longitudinal biomarker measure observed.
        Columns:
            - index (subject_id)
            - biomarker_time (time of biomarker measurement)
            - biomarker_value (measured value of biomarker)
    '''
    l_df = pd.DataFrame(np.ones(shape=(N, max_visits))*meas_gap)
    l_df = l_df.cumsum(axis=1)
    l_df.reset_index(inplace=True)
    l_df = pd.melt(l_df, id_vars=['index'], value_name='biomarker_time')
    del l_df['variable']
    l_df = pd.merge(l_df, t_df, on='index', how='outer')
    l_df = l_df.query('biomarker_time <= event_time').copy()

    def _sim_biomarker_values(row):
        index = int(row['index'])
        time = row['biomarker_time']
        X_x = np.matrix([1, time, X_l[index, 0]])
        b_x = b[index, :]
        x_x = np.matrix([1, time]).transpose()  # pylint: disable=E1111
        epsilon = meas_error()
        value = X_x*B_l + b_x*x_x + epsilon
        if len(value) != 1:
            print('Warning')
        else:
            return(float(value))
    l_df['biomarker_value'] = l_df.apply(_sim_biomarker_values, axis=1)
    if left_censor_at:
        l_df['biomarker_value'] = (l_df['biomarker_value']
                                   .apply(lambda x: x if x >= left_censor_at
                                          else left_censor_at))
    return(l_df.loc[:, ['index', 'biomarker_time', 'biomarker_value']])


def _prep_jointmodel_covariate_data(X, **kwargs):
    ''' Helper function to prepare subject-level covariate values (X)
            as a dataframe.
        Returns a dataframe containing one record per subject.
        Columns:
            - index (subject_id)
            - X1 (first simulated covariate)
            - X2 (second simulated covariate)
    '''
    X_df = pd.DataFrame(X, columns=['X1', 'X2'])
    X_df.reset_index(inplace=True)
    return(X_df)


def sim_data_jointmodel(N, p=0.5, **kwargs):
    ''' Simulate data for joint model
        Returns
        ---------
        Dictionary of 4 ojects:
        - params: parameter values used to simulate data
        - covars: dataframe of covariates per subject_id
        - events: dataframe of multiple-event data, per subject_id
        - biomarker: dataframe of longitudinal biomarker values simulated
    '''
    params = _sim_jointmodel_inputs(N=N, p=p, **kwargs)
    x_df = _prep_jointmodel_covariate_data(**params)
    t_df = _sim_jointmodel_terminal_events(**params)
    r_df = _sim_jointmodel_recurrent_events(t_df=t_df, **params)
    l_df = _sim_jointmodel_longitudinal_biomarker(t_df=t_df, **params)
    # prepare data in survivalstan format
    t_df.rename(columns={'event_status': 'event_value',
                         'event_time': 'time',
                         'index': 'subject_id'},
                inplace=True)
    t_df['event_name'] = 'death'

    r_df.rename(columns={'recurrence_time': 'time',
                         'index': 'subject_id'},
                inplace=True)
    r_df['event_value'] = 1
    r_df['event_name'] = 'new_lesion'

    x_df.rename(columns={'index': 'subject_id'}, inplace=True)
    l_df.rename(columns={'index': 'subject_id'}, inplace=True)
    events = pd.concat([t_df, r_df])
    return dict(params=params, covars=x_df, events=events, biomarker=l_df)
