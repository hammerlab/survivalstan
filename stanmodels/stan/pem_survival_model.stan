/*  Variable naming:
 // dimensions
 N          = total number of observations (length of data)
 S          = number of sample ids
 T          = max timepoint (number of timepoint ids)
 M          = number of covariates
 
 // data
 s          = sample id for each obs
 t          = timepoint id for each obs
 event      = integer indicating if there was an event at time t for sample s
 x          = matrix of real-valued covariates at time t for sample n [N, X]
 obs_t      = observed end time for interval for timepoint for that obs
 
*/
// Jacqueline Buros Novik <jackinovik@gmail.com>

data {
  int<lower=1> N;
  int<lower=1> S;
  int<lower=1> T;
  int<lower=0> M;
  int<lower=1, upper=N> s[N];     // sample id
  int<lower=1, upper=T> t[N];     // timepoint id
  int<lower=0, upper=1> event[N]; // 1: event, 0:censor
  matrix[N, M] x;                 // explanatory vars
  real<lower=0> obs_t[N];         // observed end time for each obs
}
transformed data {
  real t_dur[T];  // duration for each timepoint
  real t_obs[T];  // observed end time for each timepoint
  real c;
  real r;
  
  // baseline hazard params (fixed)
  c <- 0.001; 
  r <- 0.1;
  
  // capture observation time for each timepoint id t
  for (i in 1:N) {
      // assume these are constant per id across samples
      t_obs[t[i]] <- obs_t[i];  
  }
  
  // duration of each timepoint
  // duration at first timepoint = t_obs[1] ( implicit t0 = 0 )
  t_dur[1] <- t_obs[1];
  for (i in 2:T) {
      t_dur[i] <- t_obs[i] - t_obs[i-1];
  }
}
parameters {
  vector<lower=0>[T] baseline; // unstructured baseline hazard for each timepoint t
  vector[M] beta; // beta for each covariate
}
transformed parameters {
  vector<lower=0>[N] hazard;
  
  for (n in 1:N) {
    hazard[n] <- exp(x[n,]*beta)*baseline[t[n]];
  }
}
model {
  for (i in 1:T) {
      baseline[i] ~ gamma(r * t_dur[i] * c, c);
  }
  beta ~ cauchy(0, 2);
  event ~ poisson(hazard);
}
generated quantities {
  real log_lik[N];
  
  for (n in 1:N) {
      log_lik[n] <- poisson_log(event[n], hazard[n]);
  }
}