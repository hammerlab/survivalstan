
import matplotlib as mpl
mpl.use('Agg')
import survivalstan
from stancache import stancache
import numpy as np
from functools import partial
from nose.tools import ok_
num_iter = 500
from .test_datasets import sim_test_dataset

model_code = '''
functions {
    int count_value(vector a, real val) {
        int s;
        s = 0;
        for (i in 1:num_elements(a)) 
            if (a[i] == val) 
                s = s + 1;
        return s;
    }

  // Defines the log survival
  real surv_gamma_lpdf (vector t, vector d, real shape, vector rate, int num_cens, int num_obs) {
    vector[2] log_lik;
    int idx_obs[num_obs];
    int idx_cens[num_cens];
    real prob;
    int i_cens;
    int i_obs;
    i_cens = 1;
    i_obs = 1;
    for (i in 1:num_elements(t)) {
        if (d[i] == 1) {
            idx_obs[i_obs] = i;
            i_obs = i_obs+1;
        }
        else {
            idx_cens[i_cens] = i;
            i_cens = i_cens+1;
        }
    }
    print(idx_obs);
    log_lik[1] = gamma_lpdf(t[idx_obs] | shape, rate[idx_obs]);
    log_lik[2] = gamma_lccdf(t[idx_cens] | shape, rate[idx_cens]);
    prob = sum(log_lik);
    return prob;
  }
}
data {
  int N;                            // number of observations
  vector<lower=0>[N] y;             // observed times
  vector<lower=0,upper=1>[N] event; // censoring indicator (1=observed, 0=censored)
  int M;                            // number of covariates
  matrix[N, M] x;                   // matrix of covariates (with n rows and H columns)
}
transformed data {
  int num_cens;
  int num_obs;
  num_obs = count_value(event, 1);
  num_cens = N - num_obs;
}
parameters {
  vector[M] beta;         // Coefficients in the linear predictor (including intercept)
  real<lower=0> alpha;    // shape parameter
}
transformed parameters {
  vector[N] linpred;
  vector[N] mu;
  linpred = x*beta;
  mu = exp(linpred);
}
model {
  alpha ~ gamma(0.01,0.01);
  beta ~ normal(0,5);
  y ~ surv_gamma(event, alpha, mu, num_cens, num_obs);
}
'''
make_inits = None

def test_null_model(**kwargs):
    ''' Test weibull survival model on simulated dataset
    '''
    d = sim_test_dataset()
    testfit = survivalstan.fit_stan_survival_model(
        model_cohort = 'test model',
        model_code = model_code,
        df = d,
        time_col = 't',
        event_col = 'event',
        formula = '~ 1',
        iter = num_iter,
        chains = 2,
        seed = 9001,
        make_inits = make_inits,
        FIT_FUN = stancache.cached_stan_fit,
        drop_intercept = False,
        **kwargs
        )
    ok_('fit' in testfit)
    ok_('coefs' in testfit)
    survivalstan.utils.plot_coefs([testfit])
    survivalstan.utils.plot_coefs([testfit], trans=np.exp)
    return(testfit)

def test_model(**kwargs):
    ''' Test weibull survival model on simulated dataset
    '''
    d = sim_test_dataset()
    testfit = survivalstan.fit_stan_survival_model(
        model_cohort = 'test model',
        model_code = model_code,
        df = d,
        time_col = 't',
        event_col = 'event',
        formula = '~ age + sex',
        iter = num_iter,
        chains = 2,
        seed = 9001,
        make_inits = make_inits,
        FIT_FUN = stancache.cached_stan_fit,
        drop_intercept = False,
        **kwargs
        )
    ok_('fit' in testfit)
    ok_('coefs' in testfit)
    survivalstan.utils.plot_coefs([testfit])
    survivalstan.utils.plot_coefs([testfit], trans=np.exp)
    return(testfit)


