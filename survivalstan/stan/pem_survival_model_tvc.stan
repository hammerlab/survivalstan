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
  int<lower=1> N;
  int<lower=1> S;
  int<lower=1> T;
  int<lower=0> M;
  int<lower=1, upper=N> s[N];     // sample id
  int<lower=1, upper=T> t[N];     // timepoint id
  int<lower=0, upper=1> event[N]; // 1: event, 0:censor
  matrix[N, M] x;                 // explanatory vars
  real<lower=0> obs_t[N];         // observed end time for each obs
  int<lower=1> H;                 // number of knots (fixed)
  int<lower=0> power;             // power of spline (1:linear, 2:quad, 3:cubic)
}
transformed data {
  vector<lower=0>[T] t_dur;  // duration for each timepoint
  vector<lower=0>[T] t_obs;  // observed end time for each timepoint
  int<lower=1> P;
  vector[H+1] xi_prior;
  
  for (h in 1:(H+1)) {
      xi_prior[h] <- 1;
  }
  
  P <- 1+power;
  
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
  vector<lower=0>[T] baseline_raw;    // unstructured baseline hazard for each timepoint t
  vector[H+P] beta_time_spline[M];    // time-spline coefficients for each beta
  real<lower=0> baseline_sigma;
  real<lower=0> baseline_loc;
  simplex[H+1] xi_proportions; // time-segment proportions
}
transformed parameters {
  vector<lower=0>[N] hazard;
  vector<lower=0>[T] baseline;
  matrix[T, H+P] time_spline;
  vector[T] beta_time[M];
  positive_ordered[H] est_xi;          // locations of knots
  
  est_xi <- cumulative_sum(xi_proportions[1:H])*max(t_obs);
  
  // coefficients for each timepoint T
  time_spline <- spline(t_obs, T, H, est_xi, P);
  for (m in 1:M) {
      beta_time[m] <- time_spline*beta_time_spline[m];
  }
  
  // adjust baseline hazard for duration of each timepoint T
  for (i in 1:T) { 
    baseline[i] <- baseline_raw[i] * t_dur[i];
  }
  
  // linear predictor / hazard for each obs N
  for (n in 1:N) {
    real linpred;
    linpred <- 0;
    for (m in 1:M) {
      // for now, handle each M separately
      // (to be sure we pull out the "right" beta.. )
      linpred <- linpred + x[n, m] * beta_time[m][t[n]]; 
    }
    hazard[n] <- exp(linpred) * baseline[t[n]]; 
  }
}
model {
  xi_proportions ~ dirichlet(xi_prior);
  for (m in 1:M) {
    beta_time_spline[m] ~ normal(0, 1);
  }
  baseline_loc ~ normal(0, 1);
  baseline_raw[1] ~ normal(baseline_loc, 1);
  for (i in 2:T) {
    baseline_raw[i] ~ normal(baseline_raw[i-1], baseline_sigma);
  }
  baseline_sigma ~ normal(0.2, 0.01);
  event ~ poisson(hazard);
}
generated quantities {
  real log_lik[N];
  real beta[M];
  
  for (m in 1:M) {
    beta[m] <- beta_time_spline[m][1];
  }
  
  for (n in 1:N) {
    log_lik[n] <- poisson_log(event[n], hazard[n]);
  }
}