"""Independent finite-difference and scalar-optimizer checks of likelihood code."""
import numpy as np
from scipy.optimize import minimize
from scipy.optimize._numdiff import approx_derivative
from scipy.special import log_ndtr
from choice_model import complete_edges
from estimation import collect_pair_data, fit_pairs, gaussian_likelihood_objective
from experiment_utils import ROOT, save_json, provenance


def main():
    rng=np.random.default_rng(59123); n,r=12,2
    mu=rng.normal(0,.3,n); x=rng.normal(0,.6,(n,r)); d=rng.uniform(.4,1,n); edges=complete_edges(n)
    y,N,t=collect_pair_data(mu,x,d,edges,100,'adaptive',rng)
    theta=np.r_[mu,x.ravel(),d]
    objective=gaussian_likelihood_objective(n,r,edges,y,N,t)
    gradient=objective(theta)[1]
    numerical=approx_derivative(lambda v:objective(v)[0],theta,method='3-point').ravel()
    error=float(np.max(np.abs(gradient-numerical)))
    if error>=1e-7:
        raise AssertionError(error)
    delta,D,_,diagnostics=fit_pairs(y,N,t)
    gaps=[]
    for k in range(12):
        def single(v):
            eta=v[0]+v[1]*t[k]
            return -np.sum(y[k]*log_ndtr(eta)+(N[k]-y[k])*log_ndtr(-eta))
        a,b=delta[k]/np.sqrt(D[k]),1/np.sqrt(D[k])
        optimum=minimize(single,[0,1],method='L-BFGS-B',bounds=[(-12,12),(.05,10)],options={'ftol':1e-13,'gtol':1e-8})
        gap=float(single([a,b])-optimum.fun); gaps.append(gap)
        if gap>=1e-6:
            raise AssertionError(gap)
    save_json(ROOT/'output/estimation_validation.json',{'provenance':provenance('validate_estimation.py','estimation.py','choice_model.py'),
        'joint_gradient_max_error':error,'pair_likelihood_gaps_against_independent_optimizer':gaps,
        'pair_diagnostics':diagnostics,'status':'passed'})
    print('Likelihood checks passed:',error,max(gaps))


if __name__=='__main__':
    main()
