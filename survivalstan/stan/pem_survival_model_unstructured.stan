/*  Variable naming:
 // dimensions
 N          = total number of observations (length of data)
 S          = number of sample ids
 T          = max timepoint (number of timepoint ids)
 M          = number of covariates
 
 // main data matrix (per observed timepoint*record)
 s          = sample id for each obs
 t          = timepoint id for each obs
 event      = integer indicating if there was an event at time t for sample s
 x          = matrix of real-valued covariates at time t for sample n [N, X]
 
 // timepoint-specific data (per timepoint, ordered by timepoint id)
 t_obs      = observed time since origin for each timepoint id (end of period)
 t_dur      = duration of each timepoint period (first diff of t_obs)
 
*/
// Jacqueline Buros Novik <jackinovik@gmail.com>

data {
  // dimensions
  int<lower=1> N;
  int<lower=1> S;
  int<lower=1> T;
  int<lower=0> M;
  
  // data matrix
  int<lower=1, upper=N> s[N];     // sample id
  int<lower=1, upper=T> t[N];     // timepoint id
  int<lower=0, upper=1> event[N]; // 1: event, 0:censor
  matrix[N, M] x;                 // explanatory vars
  
  // timepoint data
  vector<lower=0>[T] t_obs;
  vector<lower=0>[T] t_dur;
}
transformed data {
  vector[T] log_t_dur;  // log-duration for each timepoint
  
  log_t_dur = log(t_obs);
}
parameters {
  vector[T] log_baseline_raw; // unstructured baseline hazard for each timepoint t
  vector[M] beta;         // beta for each covariate
  real<lower=0> baseline_sigma;
  real log_baseline_mu;
}
transformed parameters {
  vector[N] log_hazard;
  vector[T] log_baseline;     // unstructured baseline hazard for each timepoint t
  
  log_baseline = log_baseline_raw + log_t_dur;
  
  for (n in 1:N) {
    log_hazard[n] = log_baseline_mu + log_baseline[t[n]] + x[n,]*beta;
  }
}
model {
  beta ~ cauchy(0, 2);
  event ~ poisson_log(log_hazard);
  log_baseline_mu ~ normal(0, 1);
  baseline_sigma ~ normal(0, 1);
  log_baseline_raw ~ normal(0, baseline_sigma);
}
generated quantities {
  real log_lik[N];
  //int yhat_uncens[N];
  vector[T] baseline_raw;
  
  // compute raw baseline hazard, for summary/plotting
  baseline_raw = exp(log_baseline_raw);
  
  // prepare yhat_uncens & log_lik
  for (n in 1:N) {
      //yhat_uncens[n] = poisson_log_rng(log_hazard[n]);
      log_lik[n] = poisson_log_log(event[n], log_hazard[n]);
  }
}