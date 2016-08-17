functions {
  vector sqrt_vec(vector x) {
    vector[dims(x)[1]] res;

    for (m in 1:dims(x)[1]){
      res[m] = sqrt(x[m]);
    }

    return res;
  }

  vector bg_prior_lp(real r_global, vector r_local) {
    r_global ~ normal(0.0, 10.0);
    r_local ~ inv_chi_square(1.0);

    return r_global * sqrt_vec(r_local);
  }
}
data {
  // dimensions
  int<lower=0> N;             // number of observations
  int<lower=1> M;             // number of predictors
  
  // observations
  matrix[N, M] x;             // predictors for observation n
  vector[N] y;                // time for observation n
  vector[N] event;            // event status (1:event, 0:censor) for obs n
}
parameters {
  real<lower=0> tau_s_raw;
  vector<lower=0>[M] tau_raw;
  vector[M] beta_raw;
  real alpha;
}
transformed parameters {
  vector[M] beta;
  vector[N] lp;

  beta = bg_prior_lp(tau_s_raw, tau_raw) .* beta_raw;
  for (n in 1:N) {
    lp[n] = exp(dot_product(x[n], beta));
  }
}
model {
  // priors
  target += normal_lpdf(beta_raw | 0.0, 1.0);
  target += normal_lpdf(alpha | 0.0, 1.0);

  // likelihood
  for (n in 1:N) {
      if (event[n]==1)
          target += exponential_lpdf(y[n] | (lp[n] * alpha));
      else
          target += exponential_lccdf(y[n] | (lp[n] * alpha));
  }
}
generated quantities {
  vector[N] yhat_uncens;
  vector[N] log_lik;
  
  for (n in 1:N) {
      yhat_uncens[n] = exponential_rng((lp[n] * alpha));
      if (event[n]==1) {
          log_lik[n] = exponential_log(y[n], (lp[n] * alpha));
      } else {
          log_lik[n] = exponential_ccdf_log(y[n], (lp[n] * alpha));
      }
  }
}