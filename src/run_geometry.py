"""Quadrature Laplacian algebra, computational costs, and an allocation example."""
from time import perf_counter
import numpy as np
from scipy.optimize import brentq
from scipy.special import log_ndtr
from scipy.stats import multivariate_normal
from scipy.sparse.linalg import cg, LinearOperator
from choice_model import winner_probabilities
from experiment_utils import ROOT, provenance, save_json


def quadrature(mean, scale, points, left=-9.0, right=9.0):
    nodes, weights = np.polynomial.legendre.leggauss(points)
    nodes = left+(nodes+1)*(right-left)/2
    weights *= (right-left)/2
    z = (nodes[None, :]-mean[:, None])/scale[:, None]
    log_cdf = log_ndtr(z)
    log_density = -0.5*z**2-0.5*np.log(2*np.pi)-np.log(scale[:, None])
    log_field = log_cdf.sum(axis=0)
    log_hazard = log_density-log_cdf
    h = np.exp(0.5*(np.log(weights)+log_field)[None, :]+log_hazard)
    probabilities = np.exp((np.log(weights)+log_field)[None, :]+log_hazard).sum(axis=1)
    return h, probabilities


def reduced_solve(h, right):
    diagonal = h@h.sum(axis=0)
    small = h.T@(h/diagonal[:, None])
    null = h.sum(axis=0); null /= np.linalg.norm(null)
    rhs = h.T@(right/diagonal)
    z = np.linalg.solve(np.eye(h.shape[1])-small+np.outer(null, null), rhs)
    solution = (right+h@z)/diagonal
    solution -= solution.mean()
    return solution, diagonal


def main():
    evidence = provenance("run_geometry.py", "choice_model.py", "experiment_utils.py")
    rng = np.random.default_rng(61000)
    # Compare the symmetric quadrature of the continuum Hessian with the
    # finite-difference derivative of the raw fixed-grid forward quadrature.
    mean = rng.normal(0, 0.4, 12); scale = rng.uniform(0.8, 1.2, 12)
    derivative_rows = []
    for points in (17, 33, 65, 129):
        h, p = quadrature(mean, scale, points)
        diagonal = h@h.sum(0)
        laplacian = np.diag(diagonal)-h@h.T
        jacobian = np.empty((12, 12))
        for k in range(12):
            shift = np.eye(12)[k]*1e-5
            jacobian[:, k] = (quadrature(mean+shift, scale, points)[1]-quadrature(mean-shift, scale, points)[1])/(2e-5)
        right = rng.normal(size=12); right -= right.mean()
        solution, _ = reduced_solve(h, right)
        residual = np.linalg.norm(laplacian@solution-right)/np.linalg.norm(right)
        if residual>1e-8:
            raise AssertionError("reduced solve failed")
        normalized = laplacian/np.sqrt(diagonal[:, None]*diagonal[None, :])
        derivative_rows.append({"quadrature_points": points, "probability_mass": float(p.sum()),
            "hessian_vs_raw_grid_derivative_relative": float(np.linalg.norm(laplacian-jacobian)/np.linalg.norm(jacobian)),
            "solve_relative_residual": float(residual), "minimum_normalized_eigenvalue": float(np.linalg.eigvalsh(normalized).min()),
            "d0_vs_jacobi_diagonal_relative": float(np.linalg.norm(diagonal-np.diag(laplacian))/np.linalg.norm(np.diag(laplacian)))})

    # Matrix benchmark uses strictly positive synthetic Gram factors, isolating
    # linear algebra. It is not a million-alternative probability benchmark.
    benchmark = []
    for n, width in [(100,16),(1000,16),(10000,16),(100000,16),(1000000,16),
                     (1000,32),(10000,32),(100000,32),(1000,64),(10000,64),(100000,64)]:
        h = rng.lognormal(0, 1.0, (n, width))/np.sqrt(n*width)
        right = rng.normal(size=n); right -= right.mean()
        start = perf_counter(); solution, diagonal = reduced_solve(h, right); reduced_time = perf_counter()-start
        residual = np.linalg.norm(diagonal*solution-h@(h.T@solution)-right)/np.linalg.norm(right)
        iterations = [0]
        def callback(_):
            iterations[0] += 1
        operator = LinearOperator((n,n), matvec=lambda v: diagonal*v-h@(h.T@v)+np.full(n,v.mean()))
        preconditioner = LinearOperator((n,n), matvec=lambda v: v/diagonal)
        start = perf_counter()
        cg_solution, info = cg(operator, right, M=preconditioner, rtol=1e-10, maxiter=500, callback=callback)
        cg_time = perf_counter()-start
        cg_solution -= cg_solution.mean()
        if residual>1e-8 or info != 0:
            raise AssertionError("benchmark solve failed")
        row = {"n":n, "quadrature_rank_parameter":width, "gram_storage_bytes":int(h.nbytes),
               "reduced_seconds_including_small_matrix_formation":reduced_time, "cg_seconds":cg_time,
               "cg_iterations":iterations[0], "reduced_relative_residual":float(residual),
               "solution_relative_difference":float(np.linalg.norm(solution-cg_solution)/np.linalg.norm(solution)),
               "data_kind":"synthetic positive Gram factor, not evaluated choice probabilities"}
        benchmark.append(row)
        save_json(ROOT/"output"/"geometry_partial.json", {"provenance":evidence,"derivatives":derivative_rows,"benchmarks":benchmark})
        print(f"Laplacian n={n}, M={width}: reduced {reduced_time:.3f}s, CG {cg_time:.3f}s, {iterations[0]} iterations",flush=True)

    # Genuine correlated Gaussian probabilities for an illustrative allocation.
    mu = np.array([0.8,0.65,0.2,0.1,-0.1])
    v = np.array([[1,0],[.95,.05],[0,1],[.05,.9],[-.4,-.3]])
    covariance = v@v.T+np.diag([.4,.4,.5,.5,.7])
    menu = np.arange(5)
    def probabilities(tax, seed=63100):
        adjusted = mu.copy(); adjusted[0] -= tax
        result=[]
        for winner in range(5):
            others=np.delete(menu,winner)
            contrast_cov=(covariance[np.ix_(others,others)]-covariance[others,winner][:,None]
                          -covariance[winner,others][None,:]+covariance[winner,winner])
            result.append(multivariate_normal.cdf(np.zeros(4),mean=adjusted[others]-adjusted[winner],
                          cov=contrast_cov,maxpts=4194304,abseps=1e-8,releps=1e-8,
                          rng=np.random.default_rng(seed+winner)))
        return np.array(result)
    baseline=probabilities(0)
    cap=0.1
    tax=brentq(lambda z:probabilities(z)[0]-cap,0,10,xtol=1e-8)
    projected=probabilities(tax)
    kl_projected=baseline.copy(); kl_projected[0]=cap; kl_projected[1:] *= (1-cap)/(1-baseline[0])
    independent=probabilities(tax,seed=64100)
    integration_gap=float(np.max(np.abs(projected-independent)))
    if integration_gap>5e-7:
        raise AssertionError("independent allocation integration discrepancy")
    if abs(projected[0]-cap)>5e-7:
        raise AssertionError("allocation constraint inaccurate")
    allocation={"mean":mu.tolist(),"covariance":covariance.tolist(),"baseline":baseline.tolist(),
                "gaussian_bregman_projection":projected.tolist(),"kl_projection":kl_projected.tolist(),
                "cap":cap,"latent_utility_tax":float(tax),
                "requested_orthant_absolute_tolerance":1e-8,"independent_seed_probability_gap":integration_gap,
                "scope":"Interior probability allocation with one cap; no asset-return or investment-performance claim."}
    # Exact categorical Fisher identity on a simplex tangent vector.
    p=np.array([.1,.2,.3,.4]); tangent=np.array([.3,-.2,.1,-.2])
    l=np.diag(p)-np.outer(p,p)
    fisher_gap=abs(tangent@np.linalg.pinv(l)@tangent-np.sum(tangent**2/p))
    if fisher_gap>1e-12:
        raise AssertionError("Fisher identity failed")
    save_json(ROOT/"output"/"geometry.json", {"provenance":evidence,"derivatives":derivative_rows,
        "benchmarks":benchmark,"allocation":allocation,"categorical_fisher_identity_error":float(fisher_gap),"status":"passed"})


if __name__ == "__main__":
    main()
