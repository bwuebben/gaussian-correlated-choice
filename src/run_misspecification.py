"""Experiment G: seven explicitly specified violations plus a Gaussian control."""
import numpy as np
from scipy.special import ndtr
from scipy.stats import t as student
from choice_model import complete_edges, identify_pairs, reconstruct_covariance, reconstruct_means
from experiment_utils import ROOT, cached_case, provenance, save_json
from run_finite import simulated_probabilities
from sparse_recovery import edge_distances

SCENARIOS = ("gaussian", "student_t", "mixture", "menu_covariance", "context_effect",
             "treatment_variance", "pair_treatment_coefficient", "nonadditive")


def main():
    evidence = provenance("run_misspecification.py", "run_finite.py", "choice_model.py", "sparse_recovery.py", "experiment_utils.py")
    for seed in (51000, 51001, 51002):
        rng = np.random.default_rng(seed)
        n, rank = 30, 2
        mean, x, d = rng.normal(0, 0.5, n), rng.normal(0, 0.8, (n, rank)), rng.uniform(0.4, 1, n)
        mixture_shift = rng.normal(0, 1.1, n)
        edges = complete_edges(n); i, j = edges.T
        delta = mean[i]-mean[j]
        scale = np.sqrt(edge_distances(x, d, edges))
        beta = 0.5+1.5/(1+np.exp(-(x[i, 0]+x[j, 0])))
        menus = [np.sort(rng.choice(n, size, replace=False)) for size in (5, 10, 20, 30)]
        nonlinear_rng = np.random.default_rng(seed+500)
        epsilon = nonlinear_rng.standard_normal((131072, rank))@x.T + nonlinear_rng.standard_normal((131072, n))*np.sqrt(d)
        for scenario in SCENARIOS:
            cfg = {"scenario": scenario, "seed": seed, "n": n, "rank": rank}
            target = ROOT/"output"/"misspecification"/f"{scenario}_{seed}.json"
            if cached_case(target, cfg, evidence):
                continue
            def response(treatment):
                treatment = np.broadcast_to(np.asarray(treatment), (len(edges), len(treatment)))
                eta = (delta[:, None]+treatment)/scale[:, None]
                if scenario == "student_t":
                    return student.cdf(eta/np.sqrt(3/5), df=5)
                if scenario == "mixture":
                    shift = (mixture_shift[i]-mixture_shift[j])[:, None]/scale[:, None]
                    return (ndtr(eta+shift)+ndtr(eta-shift))/2
                if scenario == "treatment_variance":
                    return ndtr(eta/np.sqrt(1+0.4*treatment**2))
                if scenario == "pair_treatment_coefficient":
                    return ndtr((delta[:, None]+beta[:, None]*treatment)/scale[:, None])
                if scenario == "nonadditive":
                    result = np.empty_like(treatment)
                    for edge, (a, b) in enumerate(edges):
                        other = mean[b]*np.exp(0.35*epsilon[:, b])+epsilon[:, b]
                        for level in range(treatment.shape[1]):
                            own = (mean[a]+treatment[edge, level])*np.exp(0.35*epsilon[:, a])+epsilon[:, a]
                            result[edge, level] = np.mean(own>other)
                    return result
                return ndtr(eta)
            q = response([-1.5, 1.5])
            recovered_delta, distance = identify_pairs(q, [-1.5, 1.5])
            mu = reconstruct_means(n, edges, recovered_delta)
            raw = reconstruct_covariance(n, edges, distance)
            values, vectors = np.linalg.eigh(raw)
            keep = values > 1e-10
            fit_x = vectors[:, keep]*np.sqrt(values[keep])
            fitted_cov = fit_x@fit_x.T
            fitted_pair_scale = np.sqrt(np.diag(fitted_cov)[i]+np.diag(fitted_cov)[j]-2*fitted_cov[i, j])
            held_t = np.array([-0.75, 0.0, 0.75])
            q_test = response(held_t)
            q_pred = ndtr((mu[i, None]-mu[j, None]+held_t)/fitted_pair_scale[:, None])
            curve_rmse = float(np.sqrt(np.mean((q_pred-q_test)**2)))
            prediction = simulated_probabilities(mu, fit_x, np.zeros(n), menus, seed+1000, draws=131072)
            truth_counts = [np.zeros(len(menu), dtype=np.int64) for menu in menus]
            truth_rng = np.random.default_rng(seed+2000)
            for start in range(0, 131072, 8192):
                error = truth_rng.standard_normal((8192, rank))@x.T + truth_rng.standard_normal((8192, n))*np.sqrt(d)
                if scenario == "student_t":
                    error *= np.sqrt(3/truth_rng.chisquare(5, (8192, 1)))
                if scenario == "mixture":
                    error += truth_rng.choice([-1, 1], (8192, 1))*mixture_shift
                if scenario == "menu_covariance":
                    error = truth_rng.standard_normal((8192, rank))@(1.8*x.T) + truth_rng.standard_normal((8192, n))*np.sqrt(d)
                utilities = mean+error
                if scenario == "nonadditive":
                    utilities = mean*np.exp(0.35*error)+error
                for index, menu in enumerate(menus):
                    selected = utilities[:, menu].copy()
                    if scenario == "context_effect":
                        dist = np.sum((x[menu, None, :]-x[None, menu, :])**2, axis=2)
                        np.fill_diagonal(dist, np.inf)
                        selected += 1.5*np.exp(-np.min(dist, axis=1)/0.5)
                    truth_counts[index] += np.bincount(np.argmax(selected, axis=1), minlength=len(menu))
            actual = [count/131072 for count in truth_counts]
            result = {"configuration": cfg, "provenance": evidence,
                      "curve_estimation": "Monte Carlo, 131072 draws" if scenario == "nonadditive" else "population formula",
                      "heldout_curve_rmse": curve_rmse,
                      "mean_cycle_rmse": float(np.sqrt(np.mean((mu[i]-mu[j]-recovered_delta)**2))),
                      "negative_covariance_eigenvalues": int(np.sum(values < -1e-8)),
                      "menu_sizes": list(map(len, menus)),
                      "menu_tv": [float(np.abs(p-q).sum()/2) for p,q in zip(prediction, actual)],
                      "predicted_probabilities": [p.tolist() for p in prediction],
                      "actual_probabilities": [p.tolist() for p in actual],
                      "note": "PSD projection is explicit; violations need not be individually identifiable from discrepancies."}
            save_json(target, result)
            print(f"{scenario} seed {seed}: curve RMSE {curve_rmse:.4f}; menu20 TV {result['menu_tv'][2]:.4f}", flush=True)


if __name__ == "__main__":
    main()
