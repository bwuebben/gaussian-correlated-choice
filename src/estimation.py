"""Finite-sample binary estimation. All bounds/regularization are explicit."""
import numpy as np
from scipy.optimize import minimize
from scipy.special import expit, log_ndtr, ndtr, ndtri
from sparse_recovery import edge_distances


def fit_pairs(counts, totals, treatments, iterations=80):
    """Vectorized bounded probit likelihood using Fisher scoring and line search.

    Bounds: intercept [-12,12], positive treatment slope [0.05,10]. They prevent
    infinite separated-data fits and must be reported as part of the estimator.
    """
    y = np.asarray(counts, dtype=float)
    totals = np.broadcast_to(np.asarray(totals, dtype=float), y.shape)
    treatment = np.broadcast_to(np.asarray(treatments, dtype=float), y.shape)
    z = ndtri((y+0.5)/(totals+1))
    centered_t = treatment - treatment.mean(axis=1)[:, None]
    b = np.clip(np.sum(centered_t*z, axis=1)/np.sum(centered_t**2, axis=1), 0.05, 10)
    a = np.clip(z.mean(axis=1)-b*treatment.mean(axis=1), -12, 12)

    def likelihood(intercept, slope):
        eta = intercept[:, None] + slope[:, None]*treatment
        return np.sum(y*log_ndtr(eta) + (totals-y)*log_ndtr(-eta), axis=1)

    current = likelihood(a, b)
    last_change = np.ones(len(a))
    for _ in range(iterations):
        eta = a[:, None] + b[:, None]*treatment
        lp = -0.5*eta**2-0.5*np.log(2*np.pi)
        derivative = y*np.exp(lp-log_ndtr(eta))-(totals-y)*np.exp(lp-log_ndtr(-eta))
        information = totals*np.exp(2*lp-log_ndtr(eta)-log_ndtr(-eta))
        i00 = information.sum(axis=1)+1e-10
        i01 = (information*treatment).sum(axis=1)
        i11 = (information*treatment**2).sum(axis=1)+1e-10
        determinant = np.maximum(i00*i11-i01**2, 1e-12)
        g0, g1 = derivative.sum(axis=1), (derivative*treatment).sum(axis=1)
        da = (i11*g0-i01*g1)/determinant
        db = (i00*g1-i01*g0)/determinant
        step = np.minimum(1, 2/np.maximum(np.max(np.abs(da[:, None]+db[:, None]*treatment), axis=1), 1e-12))
        for _ in range(20):
            new_a, new_b = np.clip(a+step*da, -12, 12), np.clip(b+step*db, 0.05, 10)
            trial = likelihood(new_a, new_b)
            bad = trial < current-1e-10
            if not np.any(bad):
                break
            step[bad] *= 0.5
        last_change = np.maximum(np.abs(new_a-a), np.abs(new_b-b))
        a, b, current = new_a, new_b, trial
        if np.max(last_change) < 1e-7:
            break
    # Fisher scoring with box projection can stall near a boundary. Finish
    # those pairs with a separate bounded convex likelihood optimization.
    fallback_indices = np.flatnonzero(last_change > 1e-6)
    fallback_failures = 0
    for index in fallback_indices:
        def objective(parameter):
            eta = parameter[0]+parameter[1]*treatment[index]
            lp = -0.5*eta**2-0.5*np.log(2*np.pi)
            normalizer = totals[index].sum()
            loss = -np.sum(y[index]*log_ndtr(eta)+(totals[index]-y[index])*log_ndtr(-eta))/normalizer
            derivative = -(y[index]*np.exp(lp-log_ndtr(eta))-(totals[index]-y[index])*np.exp(lp-log_ndtr(-eta)))/normalizer
            return float(loss), np.array([derivative.sum(), np.sum(derivative*treatment[index])])
        result = minimize(objective, [a[index],b[index]], method="L-BFGS-B",jac=True,
                          bounds=[(-12,12),(.05,10)], options={"ftol":1e-14,"gtol":1e-9,"maxiter":1000})
        a[index], b[index] = result.x
        fallback_failures += int(not result.success)
    delta, distance = a/b, 1/b**2
    # Expected-information delta-method variance of D=b^{-2}.
    eta = a[:, None]+b[:, None]*treatment
    lp = -0.5*eta**2-0.5*np.log(2*np.pi)
    information = totals*np.exp(2*lp-log_ndtr(eta)-log_ndtr(-eta))
    i00, i01, i11 = information.sum(1), (information*treatment).sum(1), (information*treatment**2).sum(1)
    variance_d = 4*i00/(b**6*np.maximum(i00*i11-i01**2, 1e-12))
    diagnostics = {"bounded_fit_count": int(np.sum((np.abs(a)>11.999)|(b<0.05001)|(b>9.999))),
                   "fallback_optimization_count": int(len(fallback_indices)),
                   "fallback_failure_count": int(fallback_failures),
                   "intercept_bounds": [-12, 12], "slope_bounds": [0.05, 10]}
    return delta, distance, variance_d, diagnostics


def collect_pair_data(mean, x, d, edges, total_per_treatment, design, rng):
    delta = mean[edges[:, 0]]-mean[edges[:, 1]]
    scale = np.sqrt(edge_distances(x, d, edges))
    n = total_per_treatment
    fixed = np.broadcast_to(np.array([-1.5, 1.5]), (len(edges), 2)).copy()
    if design == "fixed":
        treatments, totals = fixed, np.full((len(edges), 2), n)
        counts = rng.binomial(totals, ndtr((delta[:, None]+treatments)/scale[:, None]))
    elif design == "oracle":
        treatments = -delta[:, None]+scale[:, None]*np.array([-1.13810128, 1.13810128])
        totals = np.full((len(edges), 2), n)
        counts = rng.binomial(totals, ndtr((delta[:, None]+treatments)/scale[:, None]))
    elif design == "adaptive":
        pilot = max(10, int(0.2*n))
        pilot_counts = rng.binomial(pilot, ndtr((delta[:, None]+fixed)/scale[:, None]))
        pilot_delta, pilot_d, _, _ = fit_pairs(pilot_counts, pilot, fixed)
        main_t = np.clip(-pilot_delta[:, None]+np.sqrt(pilot_d[:, None])*np.array([-1.13810128, 1.13810128]), -8, 8)
        main_counts = rng.binomial(n-pilot, ndtr((delta[:, None]+main_t)/scale[:, None]))
        treatments = np.column_stack((fixed, main_t))
        counts = np.column_stack((pilot_counts, main_counts))
        totals = np.broadcast_to(np.array([pilot, pilot, n-pilot, n-pilot]), counts.shape).copy()
    else:
        raise ValueError(design)
    return counts, totals, treatments


def gaussian_likelihood_objective(n, rank, edges, counts, totals, treatments):
    i, j = edges.T
    normalizer = totals.sum()
    def objective(parameters):
        mean = parameters[:n]
        x = parameters[n:n+n*rank].reshape(n, rank)
        d = parameters[n+n*rank:]
        difference = x[i]-x[j]
        variance = np.sum(difference**2, axis=1)+d[i]+d[j]
        scale = np.sqrt(variance)
        eta = (mean[i, None]-mean[j, None]+treatments)/scale[:, None]
        lp = -0.5*eta**2-0.5*np.log(2*np.pi)
        ll = counts*log_ndtr(eta)+(totals-counts)*log_ndtr(-eta)
        derivative = -(counts*np.exp(lp-log_ndtr(eta))-(totals-counts)*np.exp(lp-log_ndtr(-eta)))/normalizer
        mean_coefficient = derivative.sum(axis=1)/scale
        variance_coefficient = np.sum(-0.5*derivative*eta, axis=1)/variance
        grad_mean = np.bincount(i, weights=mean_coefficient, minlength=n)-np.bincount(j, weights=mean_coefficient, minlength=n)
        grad_x = np.column_stack([
            np.bincount(i, weights=2*variance_coefficient*difference[:, k], minlength=n)
            -np.bincount(j, weights=2*variance_coefficient*difference[:, k], minlength=n)
            for k in range(rank)]) if rank else np.empty((n, 0))
        grad_d = np.bincount(i, weights=variance_coefficient, minlength=n)+np.bincount(j, weights=variance_coefficient, minlength=n)
        return float(-ll.sum()/normalizer), np.r_[grad_mean, grad_x.ravel(), grad_d]
    return objective


def fit_gaussian_likelihood(mean, x, d, edges, counts, totals, treatments, maxiter=500):
    n, rank = x.shape
    initial = np.r_[mean, x.ravel(), np.maximum(d, 0.02)]
    result = minimize(gaussian_likelihood_objective(n, rank, edges, counts, totals, treatments),
                      initial, method="L-BFGS-B", jac=True,
                      bounds=[(None, None)]*(n+n*rank)+[(0.02, None)]*n,
                      options={"maxiter": maxiter, "ftol": 1e-12, "gtol": 1e-7, "maxcor": 20})
    fit_mean = result.x[:n]; fit_mean -= fit_mean.mean()
    fit_x = result.x[n:n+n*rank].reshape(n, rank); fit_x -= fit_x.mean(axis=0)
    return fit_mean, fit_x, result.x[n+n*rank:], {
        "optimizer_success": bool(result.success), "message": str(result.message),
        "iterations": int(result.nit), "mean_negative_log_likelihood": float(result.fun), "diagonal_floor": 0.02}


def fit_logit(n, edges, counts, totals, treatments):
    i, j = edges.T
    normalizer = totals.sum()
    def objective(parameters):
        utility, beta = parameters[:n], parameters[-1]
        eta = utility[i, None]-utility[j, None]+beta*treatments
        loss = np.sum(totals*np.logaddexp(0, eta)-counts*eta)/normalizer
        derivative = (totals*expit(eta)-counts)/normalizer
        coefficient = derivative.sum(1)
        grad = np.bincount(i, weights=coefficient, minlength=n)-np.bincount(j, weights=coefficient, minlength=n)
        return float(loss), np.r_[grad, np.sum(derivative*treatments)]
    result = minimize(objective, np.r_[np.zeros(n), 1.0], method="L-BFGS-B", jac=True,
                      bounds=[(None, None)]*n+[(0.01, 10)], options={"ftol": 1e-12, "gtol": 1e-8})
    return result.x[:n], {"optimizer_success": bool(result.success), "treatment_slope": float(result.x[-1]),
                           "mean_negative_log_likelihood": float(result.fun)}
