## -- simulate data for complex hazards
## basically a port from sambrilleman/simsurv

import logging
import numpy as np
import random
from scipy.optimize import brentq as root

logging.basicConfig()
logger = logging.getLogger(__name__)

class RootFnAtLimitError(ValueError):
    pass

class RootFnBeyondLimitError(ValueError):
    pass

def simsurv(dist,
            x,
            betas = None,
            lambdas = None,
            gammas = None,
            tde = None,
            tdefunction= None,
            mixture = False,
            pmix = 0.5,
            hazard = None,
            loghazard = None,
            idvar = None,
            ids = None,
            nodes = 15,
            maxt = None,
            interval = c(1E-8, 500),
            seed = None,
            **kwargs
            ):
    ## default inputs
    if not seed:
        seed = random.getrandbits(k)
    random.seed(seed)
    if dist not in ['weibull', 'exponential', 'gompertz']:
        raise ValueError('Distribution must be one of: weibull, exponential or gompertz.')
    if any([i > 0 for i in interval]):
        raise ValueError('Interval limits must be positive (>0)')
    if maxt and interval[1] <= maxt:
        raise ValueError('Upper bound of interval must be greater than maxt.')
    if hazard & loghazard:
        raise ValueError('Both hazard & loghazard cannot be specified.')
    hazard_args = c('t', 'x', 'betas')
    if hazard:
        ## TODO test that hazard is a function
        ## TODO confirm that hazard takes hazard_args
        logger.debug('Not confirming state of `hazard` param.')
    if loghazard:
        ## TODO test that loghazard is a functions
        ## TODO confirm that hazard takes hazard_args
        logger.debug('Not confirming state of `loghazard` param.')
    x = _validate_x(x)
    if betas:
        betas = _validate_betas(betas)
        if not _confirm_names_match(x.columns, names(betas)):
            raise ValueError('the named elements of `betas` should exist in `x`.')
    if tde:
        tde = _validate_tde(tde)
        if not _confirm_names_match(x.columns, names(tds)):
            raise ValueError('the named elements of `tde` should exist in `x`.')
    if (ids is None) != (idvar is None):
        raise ValueError('Both ids & idvar (or neither) must be provided.')
    if ids:
        N = len(ids)
        if len(set(ids)) != len(ids):
            raise ValueError('The ids provided are not unique.')
    else:
        N = len(x.index)
        ids = np.arange(N)
    if maxt and maxt <= 0:
        raise ValueError('maxt must be positive.')


    ## get survival-times according to user inputs
    if dist and not tde:
        survival = _get_survival_f(dist=dist, lambdas=lambdas, gammas=gammas,
                                mixture=mixture, pmix=pmix)
        survival_times = [_get_survival_time(i, x=x, betas=betas, idvar=idvar, survival=survival)
                           for i in ids]
    elif dist and tde:
        raise NotImplementedError('time-dependent effect simulation not yet implemented.')
    else:
        raise NotImplementedError('User-provided hazard or log-hazard not yet implemented.')

    ## truncate survival times at maxt
    if maxt:
        d = survival_times < maxt
        survival_times = survival_times * d + maxt * (not d)
    else:
        d = np.ones(N)

    # construct dataframe containing survival times
    df = pd.DataFrame({'id': ids,
                       'eventtime': survival_times,
                       'status': d,
                       })
    if x:
        df.merge(x, on=id, how='left')
    reurn df



def _get_survival_time(i, x=x, betas=betas, idvar=idvar, survival=survival):
    x_i = x.loc[x[idvar] == i,:]
    betas_i = betas.loc[betas[idvar] == i,:]
    u_i = random.uniform(1)
    test_response = _rootfn_surv(interval[1], survival=survival, x=x_i,
                                 betas=betas_i, u=u_i, **kwargs)
    if test_response is None:
        raise RootFnAtLimitError('Estimated survival at limit is None for id {}'.format(id))
    elif test_response > 0: # b/c survival fx always decreasing
        if maxt:
            ## beyond max-time, return upper bound
            ## will be censored anyways
            return interval[1]
        else:
            raise RootFnBeyondLimitError('Survival time is beyond evaluable interval. Need to increase interval or set maxt.')
    else:
        return root(_rootfn_surv, survival=survival, x=x_i, betas=betas_i,
                    u=u_i, , interval=interval, **kwargs)

def _get_survival_f(dist,
                    lambdas,
                    gammas,
                    mixture,
                    pmix
                    ):
    _validate_lambdas(lambdas=lambdas, dist=dist, mixture=mixture)
    _validate_gammas(gammas=gammas, dist=dist, mixture=mixture)
    if pmix < 0 or pmix > 1:
        raise ValueError('pmix must be in interval [0, 1].')
    if dist == 'weibull' and not mixture:
        surv =  _surv_weibull(lambdas=lambdas, gammas=gammas)
    elif dist == 'weibull' and mixture:
        surv =  _surv_weibull(lambdas=lambdas, gammas=gammas, pmix=pmix)
    elif dist == 'gompertz' and not mixture:
        surv =  _surv_gompertz(lambdas=lambdas, gammas=gammas)
    elif dist == 'gompertz' and mixture:
        surv =  _surv_gompertz(lambdas=lambdas, gammas=gammas, pmix=pmix)




def _surv_gompertz(lambdas, gammas, pmix=1):
    def surv(t, x, beta):
        if betas:
            eta = sum([betas[nm] * x[nm] for nm in list(betas.columns)])
        else:
            eta = 0
        basesurv1 = np.exp(-1 * lambdas[0] / gammas[0] * (np.exp(gammas[0] * t) - 1))
        if pmix != 1:
            basesurv2 = np.exp(-1 * lambdas[1] / gammas[1] * (np.exp(gammas[1] * t) - 1))
        else:
            basesurv2 = 0
        return pow(pmix * basesurv1 + (1 - pmix) * basesurv2, np.exp(eta))
    return surv


def _surv_weibull(lambdas, gammas, pmix=1):
    def surv(t, x, beta):
        if betas:
            eta = sum([betas[nm] * x[nm] for nm in list(betas.columns)])
        else:
            eta = 0
        basesurv1 = np.exp(-1 * lambdas[0]) * pow(t, gammas[0])
        if pmix != 1:
            basesurv2 = np.exp(-1 * lambdas[1]) * pow(t, gammas[1])
        else:
            basesurv2 = 0
        return pow(pmix * basesurv1 + (1 - pmix) * basesurv2, np.exp(eta))
    return surv


def _surv_gompertz(lambdas, gammas):


def _rootfn_surv(t, survival, x, betas, u, **kwargs):
    pass

def _validate_lambdas(lambdas, dist, mixture):
    return True

def _validate_gammas(gammas, dist, mixture):
    return True

def _validate_tde(tde):
    return tde

def _confirm_names_match(x_names, names):
    return all([nm in list(x_names) for nm in list(names)])


def _validate_betas(betas):
    return betas

def _validate_x(x):
    return x









def _sim_mixture_weibull_S(t,
                         gamma1,
                         gamma2,
                         lambda1,
                         lambda2,
                         p):
    '''
    Simulate data for complex hazards, as mixture of weibull distributions.

    Returns Survival at time t, given parameter values

    Based on:
    ## The use of restricted cubic splines to approximate complex hazard functions in the analysis of time-to-event data: a simulation study
    ## Mark J. Rutherford, Michael J. Crowther & Paul C. Lambert Journal of Statistical Computation and Simulation Vol. 85 , Iss. 4, 2015

    '''
    S = p * np.exp(pow(t, gamma1) * -1 * lambda1) + (1 - p) * np.exp(pow(t, gamma2) * -1 * lambda2)
    return S

def _sim_mixture_weibull_h(t,
                           gamma1,
                           gamma2,
                           lambda1,
                           lambda2,
                           p,
                           X=None,
                           beta=None):
    h = ((lambda1 * gamma1 * pow(t, gamma1-1) * p * np.exp(-1 * lambda1 * pow(t, gamma1)) + lambda2 * gamma2 * pow(t, gamma2-1) * (1 - p)* np.exp(-1 * lambda2 * pow(t, gamma2)))
         / (p * np.exp(-1 * lambda1 * pow(t, gamma1)) + (1 - p) * np.exp(-1 * lambda2 * pow(t, gamma2)) ))
    if X and beta:
        ## proportional hazards
        h = h * np.exp(X*beta)
    return h

def _sim_mixture_weibull_S2(t,
                           stepsize=0.01,
                           **kwargs):
    h = _sim_mixture_weibull_h(t=np.arange(start=stepsize, stop=t, step=stepsize), **kwargs)
    return np.exp( -1 * sum(h * stepsize))
