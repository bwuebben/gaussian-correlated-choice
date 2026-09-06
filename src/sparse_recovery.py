"""Sparse variance-distance reconstruction; theorem algorithm and numerical fits."""
import numpy as np
from scipy.optimize import minimize
from scipy.sparse import coo_matrix
from scipy.sparse.linalg import lsqr, eigsh


def edge_distances(loading, diagonal, edges):
    i, j = edges.T
    return np.sum((loading[i] - loading[j])**2, axis=1) + diagonal[i] + diagonal[j]


def anchor_edges(n, group_size):
    k = 3 * group_size
    if n < k:
        raise ValueError("three disjoint anchor groups must fit in the universe")
    i, j = np.triu_indices(k, 1)
    anchors = np.column_stack((i, j))
    rest = np.column_stack((np.repeat(np.arange(k, n), k), np.tile(np.arange(k), n-k)))
    return np.sort(np.vstack((anchors, rest)), axis=1)


def anchor_table(n, group_size, edges, distances):
    k = 3 * group_size
    table = np.full((n, k), np.nan)
    i, j = edges.T
    first = i < k
    second = j < k
    table[j[first], i[first]] = distances[first]
    table[i[second], j[second]] = distances[second]
    table[np.arange(k), np.arange(k)] = 0.0
    if not np.all(np.isfinite(table)):
        raise ValueError("anchor design has missing measurements")
    return table


def rank_pseudoinverse(matrix, rank):
    u, s, vt = np.linalg.svd(matrix, full_matrices=False)
    if s[rank-1] <= np.finfo(float).eps * max(matrix.shape) * s[0]:
        raise ValueError("anchor cross matrix is rank deficient")
    return (vt[:rank].T / s[:rank]) @ u[:, :rank].T, s


def reconstruct_anchors(n, rank, edges, distances, group_size=None, noisy=False):
    """Three disjoint groups cancel node variances in rectangular differences.

    Minimal group size r+1 is the proved inverse/Cholesky construction. Larger
    groups use rank-r pseudoinverses and are an overdetermined extension. In
    noisy mode PSD/eigenvalue and nonnegative-variance regularization are
    explicitly reported; no claim of exact inversion is made for that mode.
    """
    h = group_size or rank + 1
    table = anchor_table(n, h, edges, distances)
    groups = [np.arange(h) + g*h for g in range(3)]
    a, b, c = groups

    def cross(left, right):
        block = table[np.ix_(left, right)]
        return -0.5 * (block[1:, 1:] - block[1:, :1] - block[:1, 1:] + block[0, 0])

    ab, ac, bc = cross(a, b), cross(a, c), cross(b, c)
    bc_inverse, singular = rank_pseudoinverse(bc, rank)
    gram_raw = ab @ bc_inverse.T @ ac.T
    gram = (gram_raw + gram_raw.T) / 2
    values, vectors = np.linalg.eigh(gram)
    retained = values[-rank:]
    relative_asymmetry = np.linalg.norm(gram_raw - gram_raw.T) / max(np.linalg.norm(gram_raw), 1e-15)
    floor = max(np.max(np.abs(values)), 1.0) * 1e-8
    if not noisy and retained.min() <= 0:
        raise ValueError("reconstructed anchor Gram matrix is not positive definite on its rank")
    if h == rank + 1 and not noisy:
        basis_a = np.linalg.cholesky(gram)
    else:
        basis_a = vectors[:, -rank:] * np.sqrt(np.maximum(retained, floor))
    loading = np.zeros((n, rank))
    outside_a = np.arange(h, n)
    # Inner products (v_i-v_b0) A^T.
    cross_i = -0.5 * (table[np.ix_(outside_a, a[1:])] - table[outside_a, a[0]][:, None]
                     - table[b[0], a[1:]][None, :] + table[b[0], a[0]])
    loading[outside_a] = np.linalg.lstsq(basis_a, cross_i.T, rcond=None)[0].T
    basis_b = loading[b[1:]] - loading[b[0]]
    cross_a = -0.5 * (table[np.ix_(a, b[1:])] - table[a, b[0]][:, None]
                     - table[c[0], b[1:]][None, :] + table[c[0], b[0]])
    loading[a] = loading[c[0]] + np.linalg.lstsq(basis_b, cross_a.T, rcond=None)[0].T
    loading -= loading.mean(axis=0)
    residual = distances - np.sum((loading[edges[:, 0]] - loading[edges[:, 1]])**2, axis=1)
    if noisy:
        incidence = coo_matrix((np.ones(2*len(edges)),
                                (np.repeat(np.arange(len(edges)), 2), edges.ravel())), shape=(len(edges), n)).tocsr()
        diagonal = lsqr(incidence, residual, atol=1e-11, btol=1e-11)[0]
    else:
        # Triangle residuals identify d_0 and then a star identifies every d_i.
        def residual_pair(i, j):
            return table[i, j] - np.sum((loading[i] - loading[j])**2)
        d0 = (residual_pair(0, 1) + residual_pair(0, 2) - residual_pair(1, 2)) / 2
        diagonal = table[:, 0] - np.sum((loading - loading[0])**2, axis=1) - d0
        diagonal[0] = d0
    diagnostics = {"anchor_cross_condition": float(singular[0]/singular[rank-1]),
                   "anchor_gram_asymmetry": float(relative_asymmetry),
                   "minimum_retained_gram_eigenvalue": float(retained.min()),
                   "negative_diagonal_count_before_projection": int(np.sum(diagonal < 0)),
                   "minimum_diagonal_before_projection": float(diagonal.min()),
                   "regularized": bool(noisy)}
    if noisy:
        diagonal = np.maximum(diagonal, 1e-4)
    return loading, diagonal, diagnostics


def spectral_initialization(n, rank, edges, distances):
    observed = np.zeros((n, n))
    i, j = edges.T
    probability = len(edges) / (n*(n-1)/2)
    observed[i, j] = observed[j, i] = distances / probability
    gram = -0.5 * (observed - observed.mean(axis=0)[None, :] - observed.mean(axis=1)[:, None] + observed.mean())
    values, vectors = eigsh(gram, k=rank, which="LA", v0=np.ones(n) + np.arange(n)/n, tol=1e-7)
    return vectors * np.sqrt(np.maximum(values, 1e-6))


def fit_stress(n, rank, edges, distances, initial=None, diagonal_initial=None,
               estimate_diagonal=False, weights=None, maxiter=1000):
    """Observed-edge least squares, optimized numerically without a convergence theorem."""
    distances = np.asarray(distances)
    loading = spectral_initialization(n, rank, edges, distances) if initial is None else np.array(initial)
    diagonal = np.full(n, max(np.median(distances)*0.1, 0.01)) if diagonal_initial is None else np.maximum(diagonal_initial, 0)
    scale = max(np.mean(distances), 1e-8)
    weights = np.ones(len(edges)) if weights is None else np.asarray(weights)/np.mean(weights)
    i, j = edges.T

    def objective(parameter):
        x = parameter[:n*rank].reshape(n, rank)
        d = parameter[n*rank:] if estimate_diagonal else np.zeros(n)
        difference = x[i] - x[j]
        residual = np.sum(difference**2, axis=1) + d[i] + d[j] - distances
        coefficient = weights * residual / (len(edges)*scale**2)
        gradient_x = np.column_stack([
            np.bincount(i, weights=2*coefficient*difference[:, k], minlength=n)
            - np.bincount(j, weights=2*coefficient*difference[:, k], minlength=n)
            for k in range(rank)])
        gradient = gradient_x.ravel()
        if estimate_diagonal:
            gradient = np.r_[gradient, np.bincount(i, weights=coefficient, minlength=n)
                            + np.bincount(j, weights=coefficient, minlength=n)]
        return float(0.5*np.mean(weights*residual**2)/scale**2), gradient

    parameter = loading.ravel()
    bounds = None
    if estimate_diagonal:
        parameter = np.r_[parameter, diagonal]
        bounds = [(None, None)]*(n*rank) + [(0.0, None)]*n
    result = minimize(objective, parameter, method="L-BFGS-B", jac=True, bounds=bounds,
                      options={"maxiter": maxiter, "ftol": 1e-15, "gtol": 1e-9, "maxcor": 20})
    x = result.x[:n*rank].reshape(n, rank)
    x = x - x.mean(axis=0)
    d = result.x[n*rank:] if estimate_diagonal else np.zeros(n)
    return x, d, {"optimizer_success": bool(result.success), "optimizer_message": str(result.message),
                   "iterations": int(result.nit), "scaled_loss": float(result.fun)}
