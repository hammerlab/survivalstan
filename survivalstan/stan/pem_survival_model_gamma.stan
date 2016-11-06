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
  real<lower=0> t_dur[T];
  real<lower=0> t_obs[T];
}
transformed data {
  real c_unit;
  real r_unit;
  int n_trans[S, T];  
  
  // scale for baseline hazard params (fixed)
  c_unit = 0.001; 
  r_unit = 0.1;
  
  // n_trans used to map each sample*timepoint to n (used in gen quantities)
  // map each patient/timepoint combination to n values
  for (n in 1:N) {
      n_trans[s[n], t[n]] = n;
  }

  // fill in missing values with n for max t for that patient
  // ie assume "last observed" state applies forward (may be problematic for TVC)
  // this allows us to predict failure times >= observed survival times
  for (samp in 1:S) {
      int last_value;
      last_value = 0;
      for (tp in 1:T) {
          // manual says ints are initialized to neg values
          // so <=0 is a shorthand for "unassigned"
          if (n_trans[samp, tp] <= 0 && last_value != 0) {
              n_trans[samp, tp] = last_value;
          } else {
              last_value = n_trans[samp, tp];
          }
      }
  }
}
parameters {
  vector<lower=0>[T] baseline; // unstructured baseline hazard for each timepoint t
  vector[M] beta; // beta for each covariate
  real<lower=0> c_raw;
  real<lower=0> r_raw;
}
transformed parameters {
  vector[N] log_hazard;
  vector[T] log_baseline;
  real<lower=0> c;
  real<lower=0> r;
  
  
  log_baseline = log(baseline);
  
  r = r_unit*r_raw;
  c = c_unit*c_raw;
  
  for (n in 1:N) {
    log_hazard[n] = x[n,]*beta + log_baseline[t[n]];
  }
}
model {
  for (i in 1:T) {
      baseline[i] ~ gamma(r * t_dur[i] * c, c);
  }
  beta ~ cauchy(0, 2);
  event ~ poisson_log(log_hazard);
  c_raw ~ normal(0, 1);
  r_raw ~ normal(0, 1);
}
generated quantities {
  real log_lik[N];
  int y_hat_mat[S, T]; // ppcheck for each S*T combination
  real y_hat_time[S];       // predicted failure time for each sample
  int y_hat_event[S];      // predicted event (0:censor, 1:event)
  
  // log-likelihood, for loo
  for (n in 1:N) {
      log_lik[n] = poisson_log_lpmf(event[n] | log_hazard[n]);
  }
  
  // posterior predicted values
  for (samp in 1:S) {
      int sample_alive;
      sample_alive = 1;
      for (tp in 1:T) {
        if (sample_alive == 1) {
              real log_haz;
              int n;
              int pred_y;
              
              // determine predicted value of y
              n = n_trans[samp, tp];
              log_haz = x[n,]*beta + log_baseline[tp];
              if (log_haz < log(pow(2, 30))) 
                  pred_y = poisson_log_rng(log_haz);
              else
                  pred_y = 9; 
              
              // mark this patient as ineligible for future tps
              // note: deliberately make 9s ineligible 
              if (pred_y >= 1) {
                  sample_alive = 0;
                  y_hat_time[samp] = t_obs[tp];
                  y_hat_event[samp] = 1;
              }
              
              // save predicted value of y to matrix
              y_hat_mat[samp, tp] = pred_y;
          }
          else if (sample_alive == 0) {
              y_hat_mat[samp, tp] = 9;
          } 
      } // end per-timepoint loop
      
      // if patient still alive at max
      // 
      if (sample_alive == 1) {
          y_hat_time[samp] = t_obs[T];
          y_hat_event[samp] = 0;
      }
  } // end per-sample loop
}
