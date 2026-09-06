"""Experiments C, E, F: finite pair counts and genuinely held-out menus."""
import argparse
import hashlib
from time import perf_counter
import numpy as np
from scipy.special import softmax
from choice_model import complete_edges, reconstruct_covariance, reconstruct_means
from estimation import collect_pair_data, fit_pairs, fit_gaussian_likelihood, fit_logit
from experiment_utils import ROOT, cached_case, covariance_error, provenance, relative_error, save_json
from sparse_recovery import anchor_edges, edge_distances, fit_stress, reconstruct_anchors


def simulated_probabilities(mean, x, d, menus, seed, draws=65536):
    rng = np.random.default_rng(seed)
    count = [np.zeros(len(menu), dtype=np.int64) for menu in menus]
    for start in range(0, draws, 8192):
        size = min(8192, draws-start)
        utilities = mean+rng.standard_normal((size, x.shape[1]))@x.T
        utilities += rng.standard_normal((size, len(mean)))*np.sqrt(np.maximum(d, 0))
        for index, menu in enumerate(menus):
            count[index] += np.bincount(np.argmax(utilities[:, menu], axis=1), minlength=len(menu))
    return [values/draws for values in count]


def menu_evaluation(mean, x, d, models, logit, seed):
    rng = np.random.default_rng(seed)
    menus = [np.sort(rng.choice(len(mean), size, replace=False)) for size in (5, 20, len(mean))]
    parent = menus[1]
    removed_local = int(np.argmax(mean[parent]))
    menus.append(np.delete(parent, removed_local))
    truth = simulated_probabilities(mean, x, d, menus, seed+1)
    estimates = {name: simulated_probabilities(*model, menus, seed+index+100) for index, (name, model) in enumerate(models.items())}
    estimates["logit"] = [softmax(logit[menu]) for menu in menus]
    # An independent oracle simulation estimates the Monte Carlo comparison floor.
    estimates["oracle_mc"] = simulated_probabilities(mean, x, d, menus, seed+900)
    result = {}
    for name, probability in estimates.items():
        tv = [float(np.abs(p-q).sum()/2) for p, q in zip(probability, truth)]
        clipped = [np.maximum(p, 0.5/65536) for p in probability]
        clipped = [p/p.sum() for p in clipped]
        kl = [float(np.sum(q[q>0]*np.log(q[q>0]/p[q>0]))) for p, q in zip(clipped, truth)]
        true_diversion = (truth[3]-np.delete(truth[1], removed_local))/truth[1][removed_local]
        estimated_diversion = (probability[3]-np.delete(probability[1], removed_local))/max(probability[1][removed_local], 1/65536)
        result[name] = {"tv_by_menu": tv, "kl_by_menu_smoothed": kl,
                        "diversion_rmse": float(np.sqrt(np.mean((estimated_diversion-true_diversion)**2))),
                        "probabilities": [p.tolist() for p in probability]}
    return {"menu_sizes": list(map(len, menus)), "menus_zero_based": [m.tolist() for m in menus],
            "removed_alternative_zero_based": int(parent[removed_local]), "draws_per_model": 65536,
            "truth_mc": [p.tolist() for p in truth], "models": result,
            "interpretation": "All fits use only pairwise training outcomes. Menu and deletion outcomes are synthetic held-out evaluations."}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--quick", action="store_true")
    args = parser.parse_args()
    evidence = provenance("run_finite.py", "estimation.py", "sparse_recovery.py", "choice_model.py", "experiment_utils.py")
    configurations = [{"n": 100, "rank": 3, "per_treatment": count, "design": design,
                        "graph": graph, "seed": 41000+seed}
                      for graph in ("anchors", "complete")
                      for count in (50, 100, 250, 500, 1000)
                      for design in ("fixed", "adaptive", "oracle") for seed in range(3)]
    if args.quick:
        configurations = [configurations[0]]
    for cfg in configurations:
        label = f"{cfg['graph']}_{cfg['design']}_N{cfg['per_treatment']}_s{cfg['seed']}"
        target = ROOT/"output"/"finite"/f"{label}.json"
        if cached_case(target, cfg, evidence):
            continue
        n, rank = cfg["n"], cfg["rank"]
        rng = np.random.default_rng(cfg["seed"])
        mean, x, d = rng.normal(0, 0.4, n), rng.normal(0, 0.8, (n, rank)), rng.uniform(0.4, 1, n)
        h = 2*rank+2
        edges = anchor_edges(n, h) if cfg["graph"] == "anchors" else complete_edges(n)
        start = perf_counter()
        counts, totals, treatments = collect_pair_data(mean, x, d, edges, cfg["per_treatment"], cfg["design"], rng)
        delta, distances, variance_d, pair_diagnostics = fit_pairs(counts, totals, treatments)
        fitted_mean = reconstruct_means(n, edges, delta)
        weights = np.clip(1/np.maximum(variance_d, 1e-6), 0.01, 1000)
        initial_x, initial_d, anchor_diagnostics = reconstruct_anchors(n, rank, edges, distances, group_size=h, noisy=True)
        two_x, two_d, stress_diagnostics = fit_stress(n, rank, edges, distances, initial=initial_x,
                    diagonal_initial=initial_d, estimate_diagonal=True, weights=weights, maxiter=1200)
        # A second initialization is selected by observed loss alone, never by
        # the true covariance or held-out menu outcomes.
        alternative_x, alternative_d, alternative_diag = fit_stress(n, rank, edges, distances,
                    estimate_diagonal=True, weights=weights, maxiter=1200)
        selected_start = "anchors"
        if alternative_diag["scaled_loss"] < stress_diagnostics["scaled_loss"]:
            two_x, two_d, stress_diagnostics = alternative_x, alternative_d, alternative_diag
            selected_start = "spectral"
        joint_mean, joint_x, joint_d, joint_diagnostics = fit_gaussian_likelihood(fitted_mean, two_x, two_d,
                                                        edges, counts, totals, treatments)
        ind_mean, ind_x, ind_d, independent_diagnostics = fit_gaussian_likelihood(fitted_mean, np.empty((n, 0)),
                        np.full(n, np.median(distances)/2), edges, counts, totals, treatments)
        logit, logit_diagnostics = fit_logit(n, edges, counts, totals, treatments)
        estimates = {"two_stage": (fitted_mean, two_x, two_d), "joint": (joint_mean, joint_x, joint_d),
                     "independent_gaussian": (ind_mean, ind_x, ind_d)}
        evaluation = menu_evaluation(mean, x, d, estimates, logit, cfg["seed"]+10000)
        result = {"configuration": cfg, "provenance": evidence, "observed_pairs": len(edges),
                  "total_binary_observations": int(totals.sum()), "training_counts_sha256": hashlib.sha256(counts.tobytes()).hexdigest(),
                  "pair_variance_relative_error": relative_error(distances, edge_distances(x, d, edges)),
                  "pair_mean_rmse": float(np.sqrt(np.mean((delta-(mean[edges[:, 0]]-mean[edges[:, 1]]))**2))),
                  "parameter_errors": {name: {"covariance_relative": covariance_error(xx, dd, x, d),
                          "mean_rmse": float(np.sqrt(np.mean((mu-(mean-mean.mean()))**2)))} for name, (mu, xx, dd) in estimates.items()},
                  "diagnostics": {"pairs": pair_diagnostics, "initial_anchor": anchor_diagnostics, "stress": stress_diagnostics,
                                  "selected_stress_initialization": selected_start, "joint": joint_diagnostics,
                                  "independent": independent_diagnostics, "logit": logit_diagnostics},
                  "menu_evaluation": evaluation, "seconds": perf_counter()-start}
        if cfg["graph"] == "complete":
            raw = reconstruct_covariance(n, edges, distances)
            eigen, vectors = np.linalg.eigh(raw)
            projected = (vectors*np.maximum(eigen, 0))@vectors.T
            true_cov = x@x.T+np.diag(d)
            true_cov = true_cov-true_cov.mean(0)[None, :]-true_cov.mean(1)[:, None]+true_cov.mean()
            result["dense_covariance"] = {"raw_relative_error": relative_error(raw, true_cov),
                                           "psd_projected_relative_error": relative_error(projected, true_cov),
                                           "negative_eigenvalues": int(np.sum(eigen < -1e-10))}
        save_json(target, result)
        print(f"{label}: joint covariance error={result['parameter_errors']['joint']['covariance_relative']:.3f}, "
              f"menu20 TV={evaluation['models']['joint']['tv_by_menu'][1]:.3f}, {result['seconds']:.1f}s", flush=True)


if __name__ == "__main__":
    main()
