/***********************************************************************************************************************/
/* 
 * Adapted from https://github.com/ermeel86/paramaetricsurvivalmodelsinstan/blob/master/royston_parmar_3_qr.stan
 * Written by Eren M. Elci
 */
data {
    int<lower=1> N;
    int<lower=1> S;                                                 // number of basis splines
    int<lower=1> M;                                                 // number of covariates
    matrix[N, M] Q;                                                // Q-transf. of design matrix
    matrix[M, M] R;
    vector[N] log_times;                                            // x=log(t)
    int<lower=0, upper=1> surv_status[N];                           // 1: survival_event, 0: censor
    matrix[N, S] basis_evals;
    matrix[N, S] deriv_basis_evals;
}
transformed data {
    int<lower=0> N_uncensored = N - sum(status);                    // number of uncensored data points
    int<lower=0> N_censored = sum(status);                          // number of censored data points
    int<lower=1, upper=N> id_cens[N_censored];
    int<lower=1, upper=N> id_uncens[N_uncensored];
    matrix[N_censored, M] Q_censored;                               // Q-transf. of design matrix (censored)
    matrix[N_uncensored, M] Q_uncensored;                           // Q-transf. of design matrix (uncensored)
    vector[N_censored] log_times_censored;                          // x=log(t) in the paper (censored)
    vector[N_uncensored] log_times_uncensored;                      // x=log(t) in the paper (uncensored)
    matrix[N_censored, S] basis_evals_censored;                      // ispline basis matrix (censored)
    matrix[N_uncensored, S] basis_evals_uncensored;                  // ispline basis matrix (uncensored)
    matrix[N_uncensored, S] deriv_basis_evals_uncensored;            // derivatives of isplines matrix (uncensored)
    
    // get ids for censored & uncensored obs
    {
        int loc_unc = 1;  // location within uncensored id-vector
        int loc_cens = 1; // location within censored id-vector
        for (i in 1:N) {
            if (surv_status[i] == 1) {
                id_uncens[loc_unc] = i;
                loc_unc = loc_unc + 1;
            } else if (surv_status[i] == 0) {
                id_cens[loc_cens] = i;
                loc_cens = loc_cens + 1;
            }
        }
    }
    
    // populate censored & uncensored objects using indices
    Q_censored = Q[id_cens, ];
    Q_uncensored = Q[id_uncens, ];
    log_times_censored = log_times[id_cens];
    log_times_uncensored = log_times[id_uncens];
    basis_evals_uncensored = basis_evals[id_uncens, ];
    basis_evals_censored = basis_evals[id_cens, ];
    deriv_basis_evals_uncensored = deriv_basis_evals[id_uncens, ];
}
/************************************************************************************************************************/
parameters {
    vector<lower=0>[S] gammas;                                      // regression coefficients for splines
    vector[M] betas_tr;                                            // regression coefficients for covariates
    real gamma_intercept;                                           // \gamma_0 in the paper
    real<lower=0> gamma1;
}
/************************************************************************************************************************/
transformed parameters {
    vector[M] betas = R \ betas_tr;
}
/************************************************************************************************************************/
model {
    vector[N_censored] etas_censored;
    vector[N_uncensored] etas_uncensored;
    gamma1 ~ normal(1,.2);
    gammas ~ normal(0, 2);
    betas ~ normal(0,1);
    gamma_intercept ~ normal(0,5);
    
    etas_censored = Q_censored*betas_tr + basis_evals_censored*gammas  + gamma_intercept + gamma1*log_times_censored;
    etas_uncensored = Q_uncensored*betas_tr + basis_evals_uncensored*gammas  + gamma_intercept + gamma1*log_times_uncensored;
    
    target += -exp(etas_censored);
    target += etas_uncensored - exp(etas_uncensored) - log_times_uncensored + log(deriv_basis_evals_uncensored*gammas + gamma1);
}
