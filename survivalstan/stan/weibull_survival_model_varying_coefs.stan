functions {
  vector sqrt_vec(vector x) {
    vector[dims(x)[1]] res;

    for (m in 1:dims(x)[1]){
      res[m] <- sqrt(x[m]);
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
  int<lower=1> G;             // number of groups
  
  // observations
  matrix[N, M] x;             // predictors for observation n
  vector[N] y;                // time for observation n
  vector[N] event;            // event status (1:event, 0:censor) for obs n
  int<lower=0, upper=G> g[N]; // group indicator for observation n
}
transformed data {
  real<lower=0> tau_mu;
  real<lower=0> tau_al;
  tau_mu <- 10.0;
  tau_al <- 10.0;
}
parameters {
  real<lower=0> tau_s_raw;
  vector<lower=0>[M] tau_raw;
  vector[M] beta;
  vector<lower=0>[M] beta_sigma;
  real alpha_raw;
  
  // beta_raw & mu vary by group G
  matrix[M, G] grp_beta_raw;
  vector[G] mu;
}

transformed parameters {
  matrix[M, G] grp_beta;
  real alpha;
  vector[N] lp;

  for (grp in 1:G) {
    grp_beta[,grp] <- bg_prior_lp(tau_s_raw, tau_raw) .* grp_beta_raw[,grp];
  }
  alpha <- exp(tau_al * alpha_raw);
  for (n in 1:N) {
    lp[n] <- mu[g[n]] + dot_product(x[n], grp_beta[,g[n]]);
  }
}
model {
  // priors
  alpha_raw ~ normal(0.0, 1.0);
  beta ~ normal(0.0, 1.0);
  beta_sigma ~ cauchy(0, 1);
  for (grp in 1:G) {
      grp_beta_raw[,grp] ~ normal(beta, beta_sigma);
      mu[grp] ~ normal(0.0, tau_mu);
  }
  
  // likelihood
  for (n in 1:N) {
    if (event[n]==1)
      y[n] ~ weibull(alpha, exp(-(lp[n])/alpha));
    else
      increment_log_prob(weibull_ccdf_log(y[n], alpha, exp(-(lp[n])/alpha)));
  }
}
generated quantities {
  vector[N] yhat_uncens;
  vector[N] log_lik;
  
  for (n in 1:N) {
    yhat_uncens[n] <- weibull_rng(alpha, exp(-(lp[n])/alpha));
    if (event[n]==1) {
      log_lik[n] <- weibull_log(y[n], alpha, exp(-(lp[n])/alpha));
    } else {
      log_lik[n] <- weibull_ccdf_log(y[n], alpha, exp(-(lp[n])/alpha));
    }
  }
}