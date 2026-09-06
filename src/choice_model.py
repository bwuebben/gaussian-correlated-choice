"""Population identification for additive Gaussian random utility.

Utilities use the maximum convention. Treatments are actual, known utility
increments. These routines do not estimate probabilities from finite samples
and do not complete missing pairwise variances.
"""

import numpy as np
from scipy.special import ndtr, ndtri
from scipy.stats import multivariate_normal


def complete_edges(n):
    return np.column_stack(np.triu_indices(n, 1))


def checked_edges(n, edges):
    edges = np.asarray(edges)
    if edges.ndim != 2 or edges.shape[1] != 2:
        raise ValueError("edges must have shape (m, 2)")
    if not np.issubdtype(edges.dtype, np.integer):
        raise ValueError("edge indices must be integers")
    if np.any(edges < 0) or np.any(edges >= n):
        raise ValueError("edge index out of range")
    if np.any(edges[:, 0] == edges[:, 1]):
        raise ValueError("self comparisons are not allowed")
    if len(np.unique(np.sort(edges, axis=1), axis=0)) != len(edges):
        raise ValueError("duplicate unordered pair")
    return edges


def pair_responses(mean, covariance, edges, treatments):
    """Return population binary probabilities for two treatments per pair."""
    mean = np.asarray(mean, dtype=float)
    covariance = np.asarray(covariance, dtype=float)
    edges = checked_edges(len(mean), edges)
    if covariance.shape != (len(mean), len(mean)):
        raise ValueError("covariance shape does not match mean")
    i, j = edges.T
    variance = covariance[i, i] + covariance[j, j] - 2 * covariance[i, j]
    if np.any(variance <= 0):
        raise ValueError("pairwise contrasts must have positive variance")
    treatment = np.broadcast_to(np.asarray(treatments, dtype=float), (len(edges), 2))
    return ndtr((mean[i, None] - mean[j, None] + treatment) / np.sqrt(variance[:, None]))


def identify_pairs(probabilities, treatments):
    """Invert two interior population probabilities; reject saturated inputs.

    Clipping probabilities would disguise nonidentification/numerical loss and
    is deliberately not performed. Finite-sample counts require a separate MLE.
    """
    probability = np.asarray(probabilities, dtype=float)
    if probability.ndim != 2 or probability.shape[1] != 2:
        raise ValueError("probabilities must have shape (m, 2)")
    if np.any(~np.isfinite(probability)) or np.any((probability <= 0) | (probability >= 1)):
        raise ValueError("probabilities must be finite and strictly between zero and one")
    treatment = np.broadcast_to(np.asarray(treatments, dtype=float), probability.shape)
    if np.any(~np.isfinite(treatment)):
        raise ValueError("treatments must be finite")
    dt = treatment[:, 1] - treatment[:, 0]
    z = ndtri(probability)
    dz = z[:, 1] - z[:, 0]
    if np.any(dt == 0) or np.any(dz == 0):
        raise ValueError("distinct treatment levels and responses are required")
    scale = dt / dz
    if np.any(scale <= 0):
        raise ValueError("responses must increase with the known utility shift")
    delta = scale * z[:, 0] - treatment[:, 0]
    return delta, scale**2


def reconstruct_means(n, edges, differences):
    """Centered least-squares means, requiring a connected observation graph."""
    edges = checked_edges(n, edges)
    differences = np.asarray(differences, dtype=float)
    if differences.shape != (len(edges),) or np.any(~np.isfinite(differences)):
        raise ValueError("one finite mean difference per edge is required")
    incidence = np.zeros((len(edges), n))
    incidence[np.arange(len(edges)), edges[:, 0]] = 1
    incidence[np.arange(len(edges)), edges[:, 1]] = -1
    design = np.vstack((incidence, np.ones(n)))
    mean, _, rank, _ = np.linalg.lstsq(design, np.r_[differences, 0.0], rcond=None)
    if rank != n:
        raise ValueError("experiment graph must be connected")
    return mean


def reconstruct_covariance(n, edges, variances):
    """Double-center a COMPLETE variance-distance matrix.

    The result is not automatically projected to the PSD cone. Negative
    eigenvalues in noisy data are an observable model/estimation diagnostic.
    """
    edges = checked_edges(n, edges)
    variances = np.asarray(variances, dtype=float)
    if len(edges) != n * (n - 1) // 2:
        raise ValueError("complete pair coverage is required; sparse completion is not implemented")
    if variances.shape != (len(edges),) or np.any(~np.isfinite(variances)):
        raise ValueError("one finite variance per edge is required")
    if np.any(variances < 0):
        raise ValueError("variances cannot be negative")
    distance = np.zeros((n, n))
    i, j = edges.T
    distance[i, j] = distance[j, i] = variances
    centered = distance - distance.mean(axis=0)[None, :] - distance.mean(axis=1)[:, None] + distance.mean()
    return -0.5 * centered


def covariance_root(covariance):
    """PSD factor, dropping only eigenvalues within numerical roundoff."""
    covariance = np.asarray(covariance, dtype=float)
    if not np.allclose(covariance, covariance.T, rtol=1e-12, atol=1e-12):
        raise ValueError("covariance must be symmetric")
    values, vectors = np.linalg.eigh(covariance)
    tolerance = 100 * np.finfo(float).eps * len(values) * max(1.0, np.max(np.abs(values)))
    if values.min() < -tolerance:
        raise ValueError("covariance is not positive semidefinite")
    keep = values > tolerance
    return vectors[:, keep] * np.sqrt(values[keep])


def winner_probabilities(mean, covariance, menu, seed, maxpts=16384):
    """Gaussian orthant integration, with explicit, reproducible integration RNG.

    Returns raw integrals without renormalization. The integration tolerances
    are requested accuracies, not certified error bounds.
    """
    menu = np.asarray(menu, dtype=int)
    mean = np.asarray(mean)[menu]
    covariance = np.asarray(covariance)[np.ix_(menu, menu)]
    if len(menu) == 1:
        return np.ones(1)
    result = []
    for winner in range(len(menu)):
        others = np.delete(np.arange(len(menu)), winner)
        # X_j = U_j - U_winner; winner event is X <= 0.
        contrast_mean = mean[others] - mean[winner]
        contrast_cov = (
            covariance[np.ix_(others, others)]
            - covariance[others, winner][:, None]
            - covariance[winner, others][None, :]
            + covariance[winner, winner]
        )
        result.append(multivariate_normal.cdf(
            np.zeros(len(others)), mean=contrast_mean, cov=contrast_cov,
            allow_singular=True, maxpts=maxpts, abseps=1e-5, releps=1e-5,
            rng=np.random.default_rng(seed + winner),
        ))
    return np.asarray(result)
