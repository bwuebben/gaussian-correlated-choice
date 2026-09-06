"""Local geometry of the two specified three-alternative Gaussian contests.

Uses exact Gaussian conditioning for tie weights and independent numerical
differentiation of winner probabilities. All inputs are synthetic.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
from scipy.integrate import quad
from scipy.optimize import root
from scipy.special import ndtr

from examples import phi, probabilities, write_table

HERE = Path(__file__).resolve().parents[1]


def tie_weight(mu, covariance, i, j):
    """Density of U_i-U_j at zero times conditional probability of beating k."""
    k = next(k for k in range(3) if k not in (i, j))
    a = np.eye(3)[i] - np.eye(3)[j]
    c = np.eye(3)[i] - np.eye(3)[k]
    variance = a @ covariance @ a
    cross = c @ covariance @ a
    conditional_mean = c @ mu - cross / variance * (a @ mu)
    conditional_variance = c @ covariance @ c - cross**2 / variance
    assert variance > 0 and conditional_variance > 0
    return float(phi((a @ mu) / np.sqrt(variance)) / np.sqrt(variance)
                 * ndtr(conditional_mean / np.sqrt(conditional_variance)))


def laplacian(mu, covariance):
    weights = np.zeros((3, 3))
    for i in range(3):
        for j in range(3):
            if i != j:
                weights[i, j] = tie_weight(mu, covariance, i, j)
    assert np.max(np.abs(weights - weights.T)) < 1e-13
    return np.diag(weights.sum(axis=1)) - weights


def centered_inverse(target, covariance, initial):
    def unpack(x):
        return np.r_[x, -sum(x)]
    solved = root(lambda x: probabilities(unpack(x), covariance)[:2] - target[:2],
                  initial[:2], tol=1e-10)
    mu = unpack(solved.x)
    assert np.max(np.abs(probabilities(mu, covariance) - target)) < 1e-10
    return mu


def analyze(row):
    mu, covariance = np.array(row['mean']), np.array(row['covariance'])
    p = probabilities(mu, covariance)
    L = laplacian(mu, covariance)
    inverse = np.linalg.pinv(L, hermitian=True)
    eigenvalues, eigenvectors = np.linalg.eigh(L)
    assert eigenvalues[1] > 0
    assert np.max(np.abs(L.sum(axis=1))) < 1e-13
    step = 1e-4
    numerical = np.column_stack([
        (probabilities(mu + step*e, covariance) - probabilities(mu - step*e, covariance)) / (2*step)
        for e in np.eye(3)])
    jacobian_error = float(np.max(np.abs(numerical - L)))
    assert jacobian_error < 2e-8

    v = np.array([1., -1., 0.])
    transfer_derivative = (
        centered_inverse(p + step*v, covariance, mu)
        - centered_inverse(p - step*v, covariance, mu)) / (2*step)
    inverse_error = float(np.max(np.abs(transfer_derivative - inverse @ v)))
    assert inverse_error < 2e-6
    resistance = float(v @ inverse @ v)
    fiedler = eigenvectors[:, 1]
    assert abs(np.linalg.norm(inverse @ fiedler) - 1/eigenvalues[1]) < 1e-12

    # Integrate dF = mu(p).dp along a probability segment independently of L.
    bregman = []
    for delta in (0.01, 0.005, 0.0025):
        divergence = delta * quad(
            lambda t: float((centered_inverse(p + delta*t*v, covariance, mu)-mu) @ v),
            0., 1., epsabs=1e-10, epsrel=1e-9)[0]
        quadratic = 0.5 * delta**2 * resistance
        bregman.append({'delta': delta, 'divergence': divergence,
                        'quadratic_approximation': quadratic,
                        'relative_difference': abs(divergence/quadratic - 1)})
    assert bregman[-1]['relative_difference'] < bregman[0]['relative_difference']
    assert bregman[-1]['relative_difference'] < 0.01

    b = 10.
    q = b*mu
    market_numerical = np.column_stack([
        (probabilities((q+step*e)/b, covariance) - probabilities((q-step*e)/b, covariance))/(2*step)
        for e in np.eye(3)])
    market_error = float(np.max(np.abs(market_numerical - L/b)))
    assert market_error < 2e-9
    return {'mean': mu.tolist(), 'covariance': covariance.tolist(),
            'probabilities': p.tolist(), 'laplacian': L.tolist(),
            'pseudoinverse': inverse.tolist(), 'eigenvalues': eigenvalues.tolist(),
            'fiedler_vector': fiedler.tolist(), 'R_AB': resistance,
            'inverse_amplification': float(1/eigenvalues[1]),
            'checks': {'jacobian_max_error': jacobian_error,
                       'inverse_derivative_max_error': inverse_error,
                       'market_price_jacobian_max_error': market_error,
                       'bregman_expansion': bregman}}


def algebra_checks():
    p = np.array([.5, .3, .2])
    v = np.array([.7, -.2, -.5])
    fisher_errors = []
    for scale in (1., 2.5):
        L = (np.diag(p) - np.outer(p, p)) / scale
        error = abs(v @ np.linalg.pinv(L, hermitian=True) @ v - scale*np.sum(v*v/p))
        assert error < 1e-12
        fisher_errors.append(float(error))
    # Nonsymmetric matrices make the inverse-transpose order consequential.
    X = np.array([[1., 2.], [0., 1.]])
    Y = np.array([[2., 0.], [1., 3.]])
    Z = np.array([[1., -1.], [2., 1.]])
    R = np.array([[2., 1.], [0., 1.]])
    KAB, KAH, KBH = X@Y.T, X@Z.T, Y@Z.T
    changed_X, changed_Y = X@R, Y@np.linalg.inv(R).T
    assert np.allclose(changed_X@changed_Y.T, KAB)
    assert not np.allclose(changed_X@changed_X.T, X@X.T)
    recovered = KAB @ np.linalg.inv(KBH).T @ KAH.T
    anchor_error = float(np.max(np.abs(recovered - X@X.T)))
    assert anchor_error < 1e-12
    return {'fisher_scale_errors': fisher_errors, 'three_group_metric_max_error': anchor_error}


def main():
    source = HERE/'output'/'examples.json'
    inputs = json.loads(source.read_text())
    models = {name: analyze(row) for name, row in inputs['inverse'].items()}
    paths = [Path(__file__), HERE/'src'/'examples.py', source]
    report = {'data_kind': 'synthetic', 'models': models, 'algebra_checks': algebra_checks(),
              'source_sha256': {str(p.relative_to(HERE.parent)): hashlib.sha256(p.read_bytes()).hexdigest()
                                for p in paths}, 'status': 'passed'}
    (HERE/'output'/'geometry.json').write_text(json.dumps(report, indent=2, allow_nan=False)+'\n')
    write_table('geometry', 'lrrrrr',
        [['Model', '$w_{AB}$', '$w_{AC}$', '$w_{BC}$', '$R_{AB}$', '$1/\\lambda_2$'], ['\\midrule']]
        + [[name.capitalize()]+[f'{x:.4f}' for x in
            (-np.array(row['laplacian'])[0, 1], -np.array(row['laplacian'])[0, 2],
             -np.array(row['laplacian'])[1, 2], row['R_AB'], row['inverse_amplification'])]
           for name, row in models.items()])
    print(json.dumps({'status': 'passed', 'models': {
        name: {'R_AB': row['R_AB'], 'inverse_amplification': row['inverse_amplification'],
               'jacobian_error': row['checks']['jacobian_max_error']}
        for name, row in models.items()}}, indent=2))


if __name__ == '__main__':
    main()
