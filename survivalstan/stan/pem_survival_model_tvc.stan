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

functions {
  matrix spline(vector x, int N, int H, vector xi, int P) {
    matrix[N, H + P] b_x;         // expanded predictors
    for (n in 1:N) {
        for (p in 1:P) {
            b_x[n,p] <- pow(x[n],p-1);  // x[n]^(p-1)
        }
        for (h in 1:H)
          b_x[n, h + P] <- fmax(0, pow(x[n] - xi[h],P-1)); 
    }
    return b_x;
  }
}
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
  vector[T] log_t_dur;
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
  vector[T] log_baseline_raw;    // unstructured baseline hazard for each timepoint t
  real<lower=0> baseline_sigma;
  real log_baseline_mu;
  
  vector[M] beta; // beta-intercept
  vector<lower=0>[M] beta_time_sigma;
  vector[T-1] raw_beta_time_deltas[M]; // for each coefficient
                                       // change in coefficient value from previous time
}
transformed parameters {
  vector[N] log_hazard;
  vector[T] log_baseline;
  vector[T] beta_time[M];
  vector[T] beta_time_deltas[M];

  // adjust baseline hazard for duration of each period
  log_baseline = log_baseline_raw + log_t_dur;
  
  // compute timepoint-specific betas 
  // offsets from previous time
  for (coef in 1:M) {
      beta_time_deltas[coef][1] = 0;
      for (time in 2:T) {
          beta_time_deltas[coef][time] = raw_beta_time_deltas[coef][time-1];
      }
  }
  
  // coefficients for each timepoint T
  for (coef in 1:M) {
      beta_time[coef] = beta[coef] + cumulative_sum(beta_time_deltas[coef]);
  }

  // compute log-hazard for each obs
  for (n in 1:N) {
    real log_linpred;
    log_linpred <- 0;
    for (coef in 1:M) {
      // for now, handle each coef separately
      // (to be sure we pull out the "right" beta..)
      log_linpred = log_linpred + x[n, coef] * beta_time[coef][t[n]]; 
    }
    log_hazard[n] = log_baseline_mu + log_baseline[t[n]] + log_linpred;
  }
}
model {
  // priors on time-varying coefficients
  for (m in 1:M) {
    raw_beta_time_deltas[m][1] ~ normal(0, 100);
    for(i in 2:(T-1)){
        raw_beta_time_deltas[m][i] ~ normal(raw_beta_time_deltas[m][i-1], beta_time_sigma[m]);
    }
  }
  beta_time_sigma ~ cauchy(0, 1);
  beta ~ cauchy(0, 1);
  
  // priors on baseline hazard
  log_baseline_mu ~ normal(0, 1);
  baseline_sigma ~ normal(0, 1);
  log_baseline_raw[1] ~ normal(0, 1);
  for (i in 2:T) {
      log_baseline_raw[i] ~ normal(log_baseline_raw[i-1], baseline_sigma);
  }
  
  // model
  event ~ poisson_log(log_hazard);
}
generated quantities {
  real log_lik[N];
  vector[T] baseline;
  int y_hat_mat[S, T];     // ppcheck for each S*T combination
  real y_hat_time[S];      // predicted failure time for each sample
  int y_hat_event[S];      // predicted event (0:censor, 1:event)
  
  // compute raw baseline hazard, for summary/plotting
  baseline = exp(log_baseline_raw);
  
  // log_likelihood for loo-psis
  for (n in 1:N) {
      log_lik[n] <- poisson_log_lpmf(event[n] | log_hazard[n]);
  }
  
  // posterior predicted values
  for (samp in 1:S) {
      int sample_alive;
      sample_alive = 1;
      for (tp in 1:T) {
        if (sample_alive == 1) {
              int n;
              int pred_y;
              real log_linpred;
              real log_haz;
              
              // determine predicted value of y
              n = n_trans[samp, tp];
              
              // (borrow code from above to calc linpred)
              // but use sim tp not t[n] 
              log_linpred = 0;
              for (coef in 1:M) {
                  // for now, handle each coef separately
                  // (to be sure we pull out the "right" beta..)
                  log_linpred = log_linpred + x[n, coef] * beta_time[coef][tp]; 
              }
              log_haz = log_baseline_mu + log_baseline[tp] + log_linpred;
              
              // now, make posterior prediction 
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
