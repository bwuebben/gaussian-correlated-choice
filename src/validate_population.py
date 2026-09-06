"""Experiments A/B and boundary checks. All observations here are synthetic.

Run from the project root:
    .venv/bin/python src/validate_population.py
"""

import argparse
import hashlib
import json
import platform
from pathlib import Path

import numpy as np
import scipy
from scipy.optimize import minimize_scalar
from scipy.special import log_ndtr, ndtr

from choice_model import (
    complete_edges, covariance_root, identify_pairs, pair_responses,
    reconstruct_covariance, reconstruct_means, winner_probabilities,
)

ROOT = Path(__file__).resolve().parents[1]


def require(condition, message):
    if not condition:
        raise AssertionError(message)


def relative_error(estimate, truth):
    return float(np.linalg.norm(estimate - truth) / max(np.linalg.norm(truth), 1e-15))


def factor_model(seed, n=50, rank=3):
    rng = np.random.default_rng(seed)
    mean = rng.normal(0, 0.3, n)
    loading = rng.normal(0, 0.7, (n, rank))
    diagonal = rng.uniform(0.4, 1.2, n)
    covariance = loading @ loading.T + np.diag(diagonal)
    return mean, loading, diagonal, covariance


def recover(mean, covariance, treatment=(-1.5, 1.5)):
    edges = complete_edges(len(mean))
    probability = pair_responses(mean, covariance, edges, treatment)
    delta, variance = identify_pairs(probability, treatment)
    return (
        reconstruct_means(len(mean), edges, delta),
        reconstruct_covariance(len(mean), edges, variance),
        probability,
    )


def algebra_checks(seed):
    rows = []
    for index in range(5):
        mean, loading, diagonal, covariance = factor_model(seed + index)
        models = {"factor_plus_diagonal": covariance}
        if index == 0:
            rng = np.random.default_rng(seed + 100)
            dense = rng.normal(size=(len(mean), len(mean))) / np.sqrt(len(mean))
            models["unrestricted_positive_definite"] = dense @ dense.T + np.eye(len(mean))
            models["independent"] = np.diag(diagonal)
            # Pure-factor contrasts can be arbitrarily narrow. The separate
            # smaller example below uses fixed treatments in a nonsaturated regime.
        for name, sigma in models.items():
            recovered_mean, recovered_cov, probability = recover(mean, sigma)
            projection = np.eye(len(mean)) - np.ones((len(mean), len(mean))) / len(mean)
            true_cov = projection @ sigma @ projection
            mean_error = relative_error(recovered_mean, mean - mean.mean())
            cov_error = relative_error(recovered_cov, true_cov)
            require(max(mean_error, cov_error) < 1e-10, f"population recovery failed: {name}")
            rows.append({
                "model": name, "seed": seed + index, "n": len(mean),
                "mean_relative_error": mean_error, "covariance_relative_error": cov_error,
                "minimum_observed_probability": float(probability.min()),
                "maximum_observed_probability": float(probability.max()),
                "minimum_recovered_eigenvalue": float(np.linalg.eigvalsh(recovered_cov).min()),
            })
    # A rank-one model with all pairwise variances positive, but singular
    # multiway contrast covariance. Reconstruction does not require full rank.
    loading = np.arange(6, dtype=float)[:, None]
    mean = 0.2 * loading[:, 0]
    sigma = loading @ loading.T
    recovered_mean, recovered_cov, _ = recover(mean, sigma, (-0.5, 0.5))
    projection = np.eye(6) - np.ones((6, 6)) / 6
    rows.append({
        "model": "singular_pure_factor", "n": 6,
        "mean_relative_error": relative_error(recovered_mean, projection @ mean),
        "covariance_relative_error": relative_error(recovered_cov, projection @ sigma @ projection),
    })
    require(max(rows[-1]["mean_relative_error"], rows[-1]["covariance_relative_error"]) < 1e-10,
            "singular reconstruction failed")
    # Here U_i = i (0.2 + Z): only the two endpoints can win. This gives
    # an analytic probability check for a singular, six-alternative model.
    endpoint_probability = np.zeros(6)
    endpoint_probability[[0, -1]] = [ndtr(-0.2), ndtr(0.2)]
    singular_probability = winner_probabilities(recovered_mean, recovered_cov, np.arange(6), seed)
    singular_gap = float(np.max(np.abs(singular_probability - endpoint_probability)))
    require(singular_gap < 1e-8, "singular endpoint choice probabilities failed")
    rows[-1]["analytic_menu_probability_error"] = singular_gap
    return rows


def boundary_checks(seed):
    mean, loading, diagonal, covariance = factor_model(seed, n=8)
    edges = complete_edges(8)
    treatment = np.array([-1.5, 1.5])
    probability = pair_responses(mean, covariance, edges, treatment)
    delta, variance = identify_pairs(probability, treatment)
    checks = []

    def rejected(name, operation):
        try:
            operation()
        except ValueError:
            checks.append(name)
        else:
            raise AssertionError(f"invalid input was accepted: {name}")

    rejected("saturated probabilities", lambda: identify_pairs(np.array([[0.0, 1.0]]), treatment))
    rejected("equal treatment levels", lambda: identify_pairs(probability, [0.0, 0.0]))
    rejected("decreasing utility response", lambda: identify_pairs(probability[:, ::-1], treatment))
    rejected("disconnected mean graph", lambda: reconstruct_means(8, edges[:1], delta[:1]))
    rejected("missing covariance edges", lambda: reconstruct_covariance(8, edges[:-1], variance[:-1]))
    rejected("duplicate covariance edge", lambda: reconstruct_covariance(8, np.vstack((edges[:-1], edges[0])), variance))
    rejected("zero variance contrast", lambda: pair_responses(np.zeros(2), np.ones((2, 2)), np.array([[0, 1]]), treatment))
    rejected("indefinite covariance", lambda: covariance_root(np.diag([1.0, -0.1])))

    # Edge orientation should not affect the recovered parameters.
    reversed_probability = pair_responses(mean, covariance, edges[:, ::-1], treatment)
    reversed_delta, reversed_variance = identify_pairs(reversed_probability, treatment)
    require(np.allclose(reversed_delta, -delta, atol=1e-12), "edge reversal changed mean contrast")
    require(np.allclose(reversed_variance, variance, atol=1e-12), "edge reversal changed variance")
    # A path suffices for means, even though it does not suffice for covariance.
    path = np.column_stack((np.arange(7), np.arange(1, 8)))
    path_mean = reconstruct_means(8, path, mean[:-1] - mean[1:])
    require(relative_error(path_mean, mean - mean.mean()) < 1e-12, "connected-path mean recovery failed")

    # Add a common random variable correlated with the original factors.
    # This tests the full covariance equivalence, not just adding c 11^T.
    h = np.array([0.3, -0.2, 0.4])
    changed_loading = loading + h
    changed_cov = changed_loading @ changed_loading.T + np.diag(diagonal)
    changed_probability = pair_responses(mean + 3.7, changed_cov, edges, treatment)
    equivalence_error = float(np.max(np.abs(changed_probability - probability)))
    require(equivalence_error < 1e-12, "common correlated shock changed pair responses")
    covariance_delta = changed_cov - covariance
    require(np.ptp(covariance_delta) > 0.1, "equivalence example accidentally reduced to c 11^T")

    scale_errors = []
    for beta in (2.5, -2.5):
        scaled_probability = pair_responses(mean, covariance, edges, beta * treatment)
        # The sign is observable from response monotonicity. Positive scale
        # normalization uses |beta|, avoiding reversal of utility ordering.
        scale_delta, scale_variance = identify_pairs(scaled_probability, np.sign(beta) * treatment)
        scale_errors.append({
            "beta": beta,
            "mean_error": relative_error(scale_delta, (mean[edges[:, 0]] - mean[edges[:, 1]]) / abs(beta)),
            "variance_error": relative_error(scale_variance, variance / beta**2),
        })
        require(max(scale_errors[-1]["mean_error"], scale_errors[-1]["variance_error"]) < 1e-10,
                "common treatment scale normalization failed")
    return {"expected_rejections": checks, "edge_reversal": "passed", "connected_path_means": "passed",
            "correlated_common_shock_probability_error": equivalence_error,
            "common_unknown_coefficient": scale_errors}


def path_counterexample():
    edges = np.array([[0, 1], [1, 2]])
    first = np.array([0.0, 1.0, 3.0])
    second = np.array([0.0, 1.0, -1.0])
    covariance_a = np.outer(first, first) + 0.5 * np.eye(3)
    covariance_b = np.outer(second, second) + 0.5 * np.eye(3)
    mean = np.zeros(3)
    qa = pair_responses(mean, covariance_a, edges, [-1.0, 1.0])
    qb = pair_responses(mean, covariance_b, edges, [-1.0, 1.0])
    require(np.array_equal(qa, qb), "path example does not agree on observed pairs")

    def zero_mean_three_way(covariance):
        distances = np.diag(covariance)[:, None] + np.diag(covariance)[None, :] - 2 * covariance
        probability = []
        for i in range(3):
            j, k = [x for x in range(3) if x != i]
            correlation = (distances[i, j] + distances[i, k] - distances[j, k]) / (2 * np.sqrt(distances[i, j] * distances[i, k]))
            probability.append(0.25 + np.arcsin(correlation) / (2 * np.pi))
        return distances, np.array(probability)

    distance_a, probability_a = zero_mean_three_way(covariance_a)
    distance_b, probability_b = zero_mean_three_way(covariance_b)
    for covariance, probability in ((covariance_a, probability_a), (covariance_b, probability_b)):
        numerical = winner_probabilities(mean, covariance, [0, 1, 2], 500, maxpts=131072)
        require(np.max(np.abs(numerical - probability)) < 1e-4, "trivariate analytic/CDF disagreement")
        require(abs(probability.sum() - 1) < 1e-12, "trivariate probabilities do not sum to one")
    gap = float(np.max(np.abs(probability_a - probability_b)))
    require(gap > 0.1, "sparse counterexample has no substantial choice difference")
    return {"factor_loadings_a": first.tolist(), "factor_loadings_b": second.tolist(),
            "idiosyncratic_variances": [0.5] * 3, "observed_edges_zero_based": edges.tolist(),
            "distance_a": distance_a.tolist(), "distance_b": distance_b.tolist(),
            "observed_pair_probability_max_gap": float(np.max(np.abs(qa - qb))),
            "three_way_probabilities_a": probability_a.tolist(),
            "three_way_probabilities_b": probability_b.tolist(), "max_choice_probability_gap": gap}


def optimal_symmetric_design():
    # Log determinant, up to constants independent of a, for the equal-weight
    # design at standardized indices +/- a. No global design theorem is asserted.
    def log_determinant(a):
        log_phi = -0.5 * a**2 - 0.5 * np.log(2 * np.pi)
        return 2 * np.log(a) + 2 * (2 * log_phi - log_ndtr(a) - log_ndtr(-a))

    result = minimize_scalar(lambda a: -log_determinant(a), bounds=(1e-5, 8),
                             method="bounded", options={"xatol": 1e-12})
    require(result.success and abs(result.x - 1.138) < 0.001, "symmetric design numerical check failed")
    grid = np.linspace(0.001, 8, 10000)
    require(log_determinant(result.x) >= np.max(log_determinant(grid)) - 1e-10, "design optimizer missed grid maximum")
    return {"standardized_index": float(result.x), "probabilities": [float(ndtr(-result.x)), float(ndtr(result.x))],
            "scope": "numerical optimization within equal-weight symmetric two-point designs only"}


def menu_checks(seed, sample_count):
    mean, loading, diagonal, covariance = factor_model(seed)
    recovered_mean, recovered_cov, _ = recover(mean, covariance)
    rng = np.random.default_rng(seed + 200)
    menus = [np.sort(rng.choice(len(mean), size, replace=False)) for size in (3, 5, 10, 20, 50)]
    rows = []
    for number, menu in enumerate(menus):
        integration_seed = seed + 300 + 100 * number
        reference = winner_probabilities(mean, covariance, menu, integration_seed)
        recovered = winner_probabilities(recovered_mean, recovered_cov, menu, integration_seed)
        gap = float(np.max(np.abs(reference - recovered)))
        require(gap < 1e-9, f"paired orthant consistency failed for menu size {len(menu)}")
        rows.append({"size": len(menu), "alternatives_zero_based": menu.tolist(),
                     "orthant_reference": reference.tolist(), "orthant_reconstructed": recovered.tolist(),
                     "orthant_max_gap_shared_rng": gap,
                     "orthant_reference_sum": float(reference.sum()),
                     "orthant_reconstructed_sum": float(recovered.sum())})
        print(f"  Orthant consistency: menu {len(menu)}, max gap {gap:.3g}", flush=True)

    # Independent simulation paths: original factors plus independent noise,
    # versus an eigendecomposition of the recovered contrast covariance alone.
    reference_rng = np.random.default_rng(seed + 1000)
    reconstructed_rng = np.random.default_rng(seed + 2000)
    root = covariance_root(recovered_cov)
    reference_counts = [np.zeros(len(menu), dtype=np.int64) for menu in menus]
    reconstructed_counts = [np.zeros(len(menu), dtype=np.int64) for menu in menus]
    remaining = sample_count
    while remaining:
        batch = min(8192, remaining)
        truth = mean + reference_rng.standard_normal((batch, loading.shape[1])) @ loading.T
        truth += reference_rng.standard_normal((batch, len(mean))) * np.sqrt(diagonal)
        reconstruction = recovered_mean + reconstructed_rng.standard_normal((batch, root.shape[1])) @ root.T
        for index, menu in enumerate(menus):
            reference_counts[index] += np.bincount(np.argmax(truth[:, menu], axis=1), minlength=len(menu))
            reconstructed_counts[index] += np.bincount(np.argmax(reconstruction[:, menu], axis=1), minlength=len(menu))
        remaining -= batch

    # For each coordinate, Hoeffding applied to the difference of two independent
    # N-sample means gives 2 exp(-N epsilon^2). A union bound over all reported
    # coordinates does not require independence between menus.
    family_alpha = 1e-6
    coordinates = sum(map(len, menus))
    simultaneous_bound = float(np.sqrt(np.log(2 * coordinates / family_alpha) / sample_count))
    # A separate consistency comparison ties the orthant code to actual
    # utility maximization. The additional integration allowance is a numerical
    # diagnostic tolerance, not a certified bound for SciPy's integrals.
    cdf_mc_tolerance = simultaneous_bound / np.sqrt(2) + 1e-3
    for row, actual_count, predicted_count in zip(rows, reference_counts, reconstructed_counts):
        actual, predicted = actual_count / sample_count, predicted_count / sample_count
        gap = float(np.max(np.abs(actual - predicted)))
        require(gap < simultaneous_bound, f"independent Monte Carlo disagreement for menu {row['size']}")
        cdf_mc_gap = float(np.max(np.abs(actual - np.array(row["orthant_reference"]))))
        require(cdf_mc_gap < cdf_mc_tolerance, f"orthant/utility simulation disagreement for menu {row['size']}")
        row.update({"mc_reference_counts": actual_count.tolist(), "mc_reconstructed_counts": predicted_count.tolist(),
                    "mc_max_coordinate_gap": gap, "mc_total_variation_between_estimates": float(np.abs(actual - predicted).sum() / 2),
                    "orthant_vs_reference_mc_max_gap": cdf_mc_gap})
    return {"samples_per_model": sample_count, "family_error_probability": family_alpha,
            "simultaneous_coordinate_bound": simultaneous_bound,
            "orthant_vs_mc_diagnostic_tolerance": float(cdf_mc_tolerance),
            "note": "Shared-RNG orthant agreement is an algebraic consistency diagnostic; independent MC is separate. Neither is a proof or an empirical validation.",
            "menus": rows}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=20260905)
    parser.add_argument("--samples", type=int, default=262144)
    parser.add_argument("--output", type=Path, default=ROOT / "output" / "population_validation.json")
    args = parser.parse_args()
    if args.samples <= 0:
        parser.error("--samples must be positive")
    report = {"status": "running", "data_kind": "synthetic population probabilities and synthetic Gaussian draws",
              "seed": args.seed, "python": platform.python_version(), "numpy": np.__version__, "scipy": scipy.__version__,
              "source_sha256": {path.name: hashlib.sha256(path.read_bytes()).hexdigest() for path in sorted((ROOT / "src").glob("*.py"))}}
    print("Checking full-pair reconstruction and assumption boundaries...", flush=True)
    report["algebra"] = algebra_checks(args.seed)
    report["boundaries"] = boundary_checks(args.seed)
    report["sparse_path_counterexample"] = path_counterexample()
    report["symmetric_two_point_design"] = optimal_symmetric_design()
    print("Checking unseen menus and independent Monte Carlo...", flush=True)
    report["menu_validation"] = menu_checks(args.seed, args.samples)
    report["status"] = "passed"
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(f"PASS: {len(report['algebra'])} reconstruction cases; 5 menu sizes; results: {args.output}", flush=True)


if __name__ == "__main__":
    main()
