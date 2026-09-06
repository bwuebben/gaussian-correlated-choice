"""Linear-storage population anchor reconstruction and conditioning stress."""
from time import perf_counter
import numpy as np
from scipy.special import ndtr
from choice_model import identify_pairs
from experiment_utils import ROOT, cached_case, provenance, relative_error, save_json
from sparse_recovery import anchor_edges, edge_distances, reconstruct_anchors


def main():
    evidence=provenance("run_scaling.py","sparse_recovery.py","choice_model.py","experiment_utils.py")
    for n in (1000,10000,100000,1000000):
        cfg={"n":n,"rank":3,"seed":71000}
        target=ROOT/"output"/"scaling"/f"anchor_n{n}.json"
        if cached_case(target,cfg,evidence):
            continue
        rng=np.random.default_rng(71000)
        x=rng.normal(0,.6,(n,3)); d=rng.uniform(.4,1,n); mean=rng.normal(0,.3,n)
        start=perf_counter(); edges=anchor_edges(n,4)
        distances=np.empty(len(edges)); deltas=np.empty(len(edges))
        for begin in range(0,len(edges),100000):
            chunk=edges[begin:begin+100000]
            actual=edge_distances(x,d,chunk)
            delta=mean[chunk[:,0]]-mean[chunk[:,1]]
            q=ndtr((delta[:,None]+np.array([-1.5,1.5]))/np.sqrt(actual[:,None]))
            recovered_delta,recovered_d=identify_pairs(q,[-1.5,1.5])
            distances[begin:begin+len(chunk)]=recovered_d
            deltas[begin:begin+len(chunk)]=recovered_delta
        response_time=perf_counter()-start
        start=perf_counter(); fit_x,fit_d,diagnostics=reconstruct_anchors(n,3,edges,distances)
        mu=np.zeros(n); star=edges[:,0]==0; mu[edges[star,1]]=-deltas[star]; mu-=mu.mean()
        reconstruction_time=perf_counter()-start
        test=rng.integers(0,n,(20000,2)); test=test[test[:,0]!=test[:,1]]
        distance_error=relative_error(edge_distances(fit_x,fit_d,test),edge_distances(x,d,test))
        diagonal_error=relative_error(fit_d,d)
        if distance_error>1e-7 or diagonal_error>1e-7:
            raise AssertionError("population scaling reconstruction failed")
        save_json(target,{"configuration":cfg,"provenance":evidence,"observed_pairs":len(edges),
            "population_response_evaluation_seconds":response_time,"reconstruction_seconds":reconstruction_time,
            "distance_relative_error_20000_sampled_pairs":distance_error,"diagonal_relative_error":diagonal_error,
            "mean_relative_error":relative_error(mu,mean-mean.mean()),"diagnostics":diagnostics,
            "scope":"Population reconstruction only; no finite respondent sample or million-way integration.",
            "stored_edges_bytes":int(edges.nbytes)})
        print(f"Anchor n={n}, {len(edges)} pairs: responses {response_time:.2f}s, inverse {reconstruction_time:.2f}s, error {distance_error:.2g}",flush=True)

    rows=[]
    for flattening in (1,.1,.01,.001):
        for perturbation in (0,1e-9,1e-6):
            rng=np.random.default_rng(72000)
            n,r=100,2; x=rng.normal(size=(n,r)); x[:9,1]*=flattening
            d=rng.uniform(.4,1,n); edges=anchor_edges(n,3)
            exact=edge_distances(x,d,edges); observed=exact+perturbation*rng.uniform(-1,1,len(edges))
            row={"anchor_flattening":flattening,"distance_max_perturbation":perturbation}
            try:
                fit_x,fit_d,diagnostics=reconstruct_anchors(n,r,edges,observed)
                test=np.array([[i,j] for i in range(9,n) for j in range(i+1,n)])
                row.update({"heldout_relative_error":relative_error(edge_distances(fit_x,fit_d,test),edge_distances(x,d,test)),
                            "diagnostics":diagnostics,"status":"returned"})
            except ValueError as exc:
                row.update({"status":"rejected","reason":str(exc)})
            rows.append(row)
    save_json(ROOT/"output"/"conditioning.json",{"provenance":evidence,"cases":rows})


if __name__=="__main__":
    main()
