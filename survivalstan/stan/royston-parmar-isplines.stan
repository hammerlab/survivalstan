/***********************************************************************************************************************/
/* 
 * Adapted from https://github.com/ermeel86/paramaetricsurvivalmodelsinstan/blob/master/royston_parmar_3_qr.stan
 * by Eren M. Elci
 */
functions {
    int count_values(int[] x, int val) {
        int n = 0;
        for (i in 1:num_elements(x)) 
            if (x[i] == val) 
                n = n + 1;
        return n;
    }

    int[] get_ids_where(int[] status, int value) {
        int N = num_elements(status);
        int ids[count_values(status, value)]; 
        int loc = 1;
        for (i in 1:N) {
            if (status[i] == value) {
                ids[loc] = i;
                loc = loc + 1;
            }
        }
        return ids;
    }
}
data {
    int<lower=1> N;                                                 // number of obs
    int<lower=1> S;                                                 // number of basis splines
    int<lower=1> M;                                                 // number of covariates
    matrix[N, M] Q;                                                 // Q-transf. of design matrix
    matrix[M, M] R;
    vector[N] log_times;                                            // x=log(t)
    int<lower=0, upper=1> surv_status[N];                           // 1: survival_event, 0: censor
    matrix[N, S] basis_evals;
    matrix[N, S] deriv_basis_evals;
}
transformed data {
    int<lower=0> N_uncensored = N - sum(surv_status);                    // number of uncensored data points
    int<lower=0> N_censored = sum(surv_status);                          // number of censored data points
    int<lower=1, upper=N> id_cens[N_censored] = get_ids_where(surv_status, 0);              // ids where surv_status == 0
    int<lower=1, upper=N> id_uncens[N_uncensored] = get_ids_where(surv_status, 1);          // ids where surv_status == 1
    matrix[N_censored, M] Q_censored = Q[id_cens, ];                                       // Q-transf. of design matrix (censored)
    matrix[N_uncensored, M] Q_uncensored = Q[id_uncens, ];                                 // Q-transf. of design matrix (uncensored)
    vector[N_censored] log_times_censored = log_times[id_cens];                            // x=log(t) in the paper (censored)
    vector[N_uncensored] log_times_uncensored = log_times[id_uncens];                      // x=log(t) in the paper (uncensored)
    matrix[N_censored, S] basis_evals_censored = basis_evals[id_cens, ];                   // ispline basis matrix (censored)
    matrix[N_uncensored, S] basis_evals_uncensored = basis_evals[id_uncens, ];             // ispline basis matrix (uncensored)
    matrix[N_uncensored, S] deriv_basis_evals_uncensored = deriv_basis_evals[id_uncens,];  // derivatives of isplines matrix (uncensored)
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
