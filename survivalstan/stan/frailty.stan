/*  Variable naming:
 obs       = observed
 cen       = (right) censored
 N         = number of samples
 tau       = scale parameter
 G         = grups for example  different cohorts
 mu        = shared frailty var
 nu        = degrees of freedom for the half-t priors for the global and scale shrinkage
 Z         = matrix of well-stablished covariates
 P         = matrix of new (genomic) covariates
 y         = observed outcome for example
 

author: Carlos Traynor
Heavily inspired and acknowledged to Tomi Peltola and Jackie Buros
Also the scale_global parameter must be tuned from the imput data file following the premises described in this article by Piironen and Vehtari
https://arxiv.org/pdf/1610.05559.pdf
*/
functions {
  vector sqrt_vec(vector x) {
    vector[dims(x)[1]] res;

    for (m in 1:dims(x)[1]){
      res[m] = sqrt(x[m]);
    }
    return res;
  }

  vector b_prior_lp(real r_global, vector r_local) {
    r_global ~ normal(0.0, 10.0);
    r_local ~ inv_chi_square(1.0);
    return r_global * sqrt_vec(r_local);
  }
  vector hs_prior_lp(real r1_global, real r2_global, vector r1_local, vector r2_local, real nu_local, real nu_global, real scale_global) {
    //half-t preior for lambdas
    r1_local ~ normal(0.0, 1.0);
    r2_local ~ inv_gamma(0.5 * nu_local, 0.5 * nu_local);
    //half-t prior for tau
    r1_global ~ normal(0.0, scale_global);
    r2_global ~ inv_gamma(0.5 * nu_global, 0.5 * nu_global);

    return (r1_global * sqrt(r2_global)) * r1_local .* sqrt_vec(r2_local);
  }
}

data {
  int<lower=0> Nobs;
  int<lower=0> Ncen;
  int<lower=0> G; //intClust
  vector[Nobs] yobs;
  vector[Ncen] ycen;
  int Gobs[Nobs]; 
  int Gcen[Ncen]; 
  int<lower=0> M;
  matrix[Nobs, M] Zobs;
  matrix[Ncen, M] Zcen;
  int<lower=0> P;
  matrix[Nobs, P] Pobs;
  matrix[Ncen, P] Pcen;
  real scale_global;
}

transformed data {
  real<lower=0> tau_al;
  real<lower=0> tau_xi;
  int<lower=0> N;
  int<lower=0> nu_global;
  real<lower=0> nu_local;
  
  tau_al = 5.0;
  tau_xi = 1.0;
  N = Nobs + Ncen;
  nu_global = 1;
  nu_local = 1;

}

parameters {
  real alpha_raw;
    
  real<lower=0> tau_s_b_raw;
  vector<lower=0>[M] tau_b_raw;
  vector[M] beta_b_raw;
  
  real<lower=0> tau_s1_p_raw;
  real<lower=0> tau_s2_p_raw;
  vector<lower=0>[P] tau_1_p_raw;
  vector<lower=0>[P] tau_2_p_raw;
  vector[P] beta_p_raw;

  vector[G] mu;
  real<lower=0> tau_mu;
  real xi;
}

transformed parameters {
  real<lower=0> alpha;
  vector[M] beta_b;
  vector[P] beta_p;
  
  alpha = exp(tau_al * alpha_raw);
  beta_b = b_prior_lp(tau_s_b_raw, tau_b_raw) .* beta_b_raw;
  beta_p = hs_prior_lp(tau_s1_p_raw, tau_s2_p_raw, tau_1_p_raw, tau_2_p_raw, nu_local, nu_global, scale_global) .* beta_p_raw; 
  
}
model {
  //priors
  alpha_raw ~ normal(0.0, 1.0);
  
  beta_b_raw ~ normal(0.0, 1.0);
  beta_p_raw ~ normal(0.0, 1.0); 
  
  mu ~ normal(0 , tau_mu);
  tau_mu ~ gamma(2, .1);
  xi ~ normal(0, tau_xi);
  
  //model
  yobs ~ weibull(alpha, exp(-( mu[Gobs] + Pobs * beta_p + Zobs * beta_b +xi)/alpha));
  target += weibull_lccdf(ycen | alpha, exp(-( mu[Gcen] + Pcen * beta_p + Zcen * beta_b + xi )/alpha));
}
generated quantities {
  vector[N] yhat_uncens;
  vector[N] log_lik;
  
  //log likelihood
  for (n in 1:Nobs){
    log_lik[n] = weibull_lpdf(yobs[n] | alpha, exp(-(xi + mu[Gobs[n]] + Pobs[n,] * beta_p + Zobs[n,] * beta_b)/alpha));
  }
  for (n in 1:Ncen){
    log_lik[Nobs + n] = weibull_lccdf(ycen[n]| alpha, exp(-(xi + mu[Gcen[n]] + Pcen[n,] * beta_p + Zcen[n,] * beta_b)/alpha));
  }
  //yhat uncens calculation
  for (n in 1:Nobs){
    yhat_uncens[n] = weibull_rng(alpha, exp(-(xi + mu[Gobs[n]] + Pobs[n,] * beta_p + Zobs[n,] * beta_b)/alpha));
  }
  for (n in 1:Ncen){
    yhat_uncens[Nobs + n] = weibull_rng(alpha, exp(-(xi + mu[Gcen[n]] + Pcen[n,] * beta_p + Zcen[n,] * beta_b)/alpha));
  }
  
}

