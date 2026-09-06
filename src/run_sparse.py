"""Resumable pure-factor random graphs and factor-plus-diagonal anchor designs."""
import argparse
from time import perf_counter
import numpy as np
from choice_model import complete_edges
from experiment_utils import ROOT, cached_case, covariance_error, provenance, relative_error, save_json
from sparse_recovery import anchor_edges, edge_distances, fit_stress, reconstruct_anchors


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=["anchors", "random", "all"], default="all")
    args = parser.parse_args()
    evidence = provenance("run_sparse.py", "sparse_recovery.py", "experiment_utils.py", "choice_model.py")
    configurations = []
    if args.stage in ("anchors", "all"):
        for n in (50, 200, 1000):
            for rank in (1, 2, 5, 10):
                for seed in range(3):
                    configurations.append({"kind": "anchors", "n": n, "rank": rank, "seed": 31000+seed})
    if args.stage in ("random", "all"):
        for n in (200, 500, 1000):
            for rank in (2, 5, 10):
                for c in (0.5, 1.0, 2.0, 4.0, 8.0):
                    for seed in range(3):
                        configurations.append({"kind": "random", "n": n, "rank": rank, "c": c, "seed": 32000+seed})
        for c in (0.05, 0.1, 0.25):
            for seed in range(3):
                configurations.append({"kind": "random", "n": 200, "rank": 2, "c": c, "seed": 32000+seed})
    for cfg in configurations:
        n, rank = cfg["n"], cfg["rank"]
        label = f"{cfg['kind']}_n{n}_r{rank}_s{cfg['seed']}" + (f"_c{cfg['c']}" if "c" in cfg else "")
        target = ROOT / "output" / "sparse" / f"{label}.json"
        if cached_case(target, cfg, evidence):
            continue
        rng = np.random.default_rng(cfg["seed"] + n + 10*rank)
        x = rng.normal(0, 1/np.sqrt(rank), (n, rank))
        d = rng.uniform(0.3, 1.0, n) if cfg["kind"] == "anchors" else np.zeros(n)
        all_edges = complete_edges(n)
        if cfg["kind"] == "anchors":
            edges = anchor_edges(n, rank+1)
        else:
            count = min(len(all_edges), max(1, int(np.ceil(cfg["c"]*n*rank*np.log(n)))))
            edges = all_edges[rng.choice(len(all_edges), count, replace=False)]
        distance = edge_distances(x, d, edges)
        start = perf_counter()
        if cfg["kind"] == "anchors":
            estimated_x, estimated_d, diagnostics = reconstruct_anchors(n, rank, edges, distance)
        else:
            estimated_x, estimated_d, diagnostics = fit_stress(n, rank, edges, distance, maxiter=800)
        duration = perf_counter()-start
        observed_error = relative_error(edge_distances(estimated_x, estimated_d, edges), distance)
        available = np.ones((n, n), dtype=bool)
        available[edges[:, 0], edges[:, 1]] = available[edges[:, 1], edges[:, 0]] = False
        heldout = all_edges[available[all_edges[:, 0], all_edges[:, 1]]]
        if len(heldout) > 20000:
            heldout = heldout[rng.choice(len(heldout), 20000, replace=False)]
        holdout_error = relative_error(edge_distances(estimated_x, estimated_d, heldout), edge_distances(x, d, heldout)) if len(heldout) else None
        result = {"configuration": cfg, "provenance": evidence, "observed_pairs": len(edges),
                  "total_possible_pairs": len(all_edges), "heldout_pairs_evaluated": len(heldout),
                  "observed_relative_error": observed_error, "heldout_relative_error": holdout_error,
                  "covariance_relative_error": covariance_error(estimated_x, estimated_d, x, d),
                  "diagonal_relative_error": relative_error(estimated_d, d) if cfg["kind"] == "anchors" else None,
                  "seconds": duration, "diagnostics": diagnostics}
        save_json(target, result)
        print(f"{label}: m={len(edges)}, covariance error={result['covariance_relative_error']:.3g}, {duration:.2f}s", flush=True)


if __name__ == "__main__":
    main()
