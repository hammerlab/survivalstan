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
  int n_trans[S, T];  
  
  log_t_dur = log(t_dur);

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
  vector[T] log_baseline_raw; // unstructured baseline hazard for each timepoint t
  vector[M] beta;         // beta for each covariate
  real<lower=0> baseline_sigma;
  real log_baseline_mu;
}
transformed parameters {
  vector[N] log_hazard;
  vector[T] log_baseline;     // unstructured baseline hazard for each timepoint t
  
  log_baseline = log_baseline_mu + log_baseline_raw + log_t_dur;
  
  for (n in 1:N) {
    log_hazard[n] = log_baseline[t[n]] + x[n,]*beta;
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
  vector[T] baseline;
  real y_hat_time[S];      // predicted failure time for each sample
  int y_hat_event[S];      // predicted event (0:censor, 1:event)
  
  // compute raw baseline hazard, for summary/plotting
  baseline = exp(log_baseline_mu + log_baseline_raw);
  
  // prepare log_lik for loo-psis
  for (n in 1:N) {
      log_lik[n] = poisson_log_log(event[n], log_hazard[n]);
  }

  // posterior predicted values
  for (samp in 1:S) {
      int sample_alive;
      sample_alive = 1;
      for (tp in 1:T) {
        if (sample_alive == 1) {
              int n;
              int pred_y;
              real log_haz;
              
              // determine predicted value of this sample's hazard
              n = n_trans[samp, tp];
              log_haz = log_baseline[tp] + x[n,] * beta;
              
              // now, make posterior prediction of an event at this tp
              if (log_haz < log(pow(2, 30))) 
                  pred_y = poisson_log_rng(log_haz);
              else
                  pred_y = 9; 
              
              // summarize survival time (observed) for this pt
              if (pred_y >= 1) {
                  // mark this patient as ineligible for future tps
                  // note: deliberately treat 9s as events 
                  sample_alive = 0;
                  y_hat_time[samp] = t_obs[tp];
                  y_hat_event[samp] = 1;
              }
              
          }
      } // end per-timepoint loop
      
      // if patient still alive at max
      if (sample_alive == 1) {
          y_hat_time[samp] = t_obs[T];
          y_hat_event[samp] = 0;
      }
  } // end per-sample loop  
}
