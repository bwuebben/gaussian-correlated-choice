"""Remaining outline checks: noisy pure factors, lifted completion, rankings."""
import hashlib
import itertools
import numpy as np
from scipy.linalg import eigh
from scipy.special import softmax
from choice_model import complete_edges,reconstruct_covariance,reconstruct_means
from estimation import collect_pair_data,fit_pairs,fit_gaussian_likelihood,fit_logit
from experiment_utils import ROOT,cached_case,covariance_error,provenance,relative_error,save_json
from sparse_recovery import anchor_edges,edge_distances,fit_stress,reconstruct_anchors
from run_finite import simulated_probabilities


def lifted_completion(n,rank,edges,distances,maxiter=1000):
    # Rank-(r+2) symmetric completion of M_ij=b_i+b_j-2 v_i.v_j,
    # whose diagonal 2d_i is unobserved. This relaxation does not enforce
    # the factor-plus-diagonal form or positivity; project contrast C to PSD.
    mask=np.zeros((n,n),bool);mask[edges[:,0],edges[:,1]]=True;mask|=mask.T
    observed=np.zeros((n,n));observed[edges[:,0],edges[:,1]]=distances;observed+=observed.T
    matrix=observed.copy(); change=np.inf
    for iteration in range(maxiter):
        filled=np.where(mask,observed,matrix)
        eigen,vectors=eigh(filled,check_finite=False)
        keep=np.argsort(np.abs(eigen))[-(rank+2):]
        updated=(vectors[:,keep]*eigen[keep])@vectors[:,keep].T
        change=np.linalg.norm(updated-matrix)/max(np.linalg.norm(matrix),1e-12)
        matrix=updated
        if change<1e-8:break
    distance=matrix.copy();np.fill_diagonal(distance,0)
    centered=distance-distance.mean(0)[None,:]-distance.mean(1)[:,None]+distance.mean()
    raw=-centered/2;eigen,vectors=eigh(raw);keep=eigen>1e-10
    x=vectors[:,keep]*np.sqrt(eigen[keep])
    return x,np.zeros(n),{'iterations':iteration+1,'relative_step':float(change),'step_converged':bool(change<1e-8),'observed_relative_residual':float(np.linalg.norm((matrix-observed)[mask])/np.linalg.norm(observed[mask])),'negative_covariance_eigenvalues':int(np.sum(eigen<-1e-10)),'method':'alternating observed-entry and symmetric rank-(r+2) projections; final contrast PSD projection'}


def ranking_probability(model,menu,seed,draws=131072,logit=None):
    if draws <= 0:
        raise ValueError("draws must be positive")
    rng=np.random.default_rng(seed); size=len(menu);count=np.zeros(size**size,dtype=np.int64)
    for start in range(0,draws,8192):
        batch=min(8192,draws-start)
        if logit is None:
            mean,x,d=model
            utility=mean[menu]+rng.normal(size=(batch,x.shape[1]))@x[menu].T+rng.normal(size=(batch,size))*np.sqrt(d[menu])
        else:
            utility=logit[menu]+rng.gumbel(size=(batch,size))
        order=np.argsort(-utility,axis=1)
        code=order@(size**np.arange(size))
        count+=np.bincount(code,minlength=size**size)
    codes=np.array(list(itertools.permutations(range(size))))@(size**np.arange(size))
    return count[codes]/draws


def main():
    evidence=provenance('run_supplement.py','run_finite.py','estimation.py','sparse_recovery.py','choice_model.py','experiment_utils.py')
    for n in (200,):
     for rank in (2,5,10):
      for count in (250,1000):
       for seed in (81000,81001,81002):
        cfg={'kind':'noisy_pure_factor','n':n,'rank':rank,'N':count,'c':2,'seed':seed}
        target=ROOT/'output'/'supplement'/f'pure_n{n}_r{rank}_N{count}_s{seed}.json'
        if cached_case(target,cfg,evidence):continue
        rng=np.random.default_rng(seed+rank); x=rng.normal(0,1/np.sqrt(rank),(n,rank));d=np.zeros(n);mu=rng.normal(0,.4,n)
        all_edges=complete_edges(n);m=min(len(all_edges),int(np.ceil(2*n*rank*np.log(n))));edges=all_edges[rng.choice(len(all_edges),m,replace=False)]
        counts,totals,t=collect_pair_data(mu,x,d,edges,count,'fixed',rng)
        delta,distance,variance,pairdiag=fit_pairs(counts,totals,t)
        weights=np.clip(1/np.maximum(variance,1e-6),.01,1000)
        xx,dd,diag=fit_stress(n,rank,edges,distance,weights=weights,maxiter=1200)
        save_json(target,{'configuration':cfg,'provenance':evidence,'observed_pairs':m,'pair_variance_relative_error':relative_error(distance,edge_distances(x,d,edges)),'covariance_relative_error':covariance_error(xx,dd,x,d),'diagnostics':diag,'pair_diagnostics':pairdiag})
        print('pure',rank,count,seed,'error',covariance_error(xx,dd,x,d),flush=True)
    for graph in ('anchors','complete'):
     for count in (250,1000):
      for seed in (41000,41001,41002):
        cfg={'kind':'completion_and_rankings','graph':graph,'N':count,'design':'adaptive','seed':seed,'n':100,'rank':3}
        target=ROOT/'output'/'supplement'/f'comparison_{graph}_N{count}_s{seed}.json'
        if cached_case(target,cfg,evidence):continue
        n,rank=100,3;rng=np.random.default_rng(seed)
        mu,x,d=rng.normal(0,.4,n),rng.normal(0,.8,(n,rank)),rng.uniform(.4,1,n)
        edges=anchor_edges(n,8) if graph=='anchors' else complete_edges(n)
        counts,totals,t=collect_pair_data(mu,x,d,edges,count,'adaptive',rng)
        original=__import__('json').loads((ROOT/'output'/'finite'/f'{graph}_adaptive_N{count}_s{seed}.json').read_text())
        assert hashlib.sha256(counts.tobytes()).hexdigest()==original['training_counts_sha256']
        delta,distance,variance,pairdiag=fit_pairs(counts,totals,t);fmu=reconstruct_means(n,edges,delta)
        xx,dd,adiag=reconstruct_anchors(n,rank,edges,distance,group_size=8,noisy=True); weights=np.clip(1/np.maximum(variance,1e-6),.01,1000)
        xx,dd,sdiag=fit_stress(n,rank,edges,distance,initial=xx,diagonal_initial=dd,estimate_diagonal=True,weights=weights,maxiter=1200)
        altx,altd,altdiag=fit_stress(n,rank,edges,distance,estimate_diagonal=True,weights=weights,maxiter=1200)
        if altdiag['scaled_loss']<sdiag['scaled_loss']:xx,dd,sdiag=altx,altd,altdiag
        jointmu,jointx,jointd,jdiag=fit_gaussian_likelihood(fmu,xx,dd,edges,counts,totals,t)
        imu,ix,idd,idiag=fit_gaussian_likelihood(fmu,np.empty((n,0)),np.full(n,np.median(distance)/2),edges,counts,totals,t)
        logit,ldiag=fit_logit(n,edges,counts,totals,t)
        lx,ld,ldiagn=lifted_completion(n,rank,edges,distance)
        models={'joint':(jointmu,jointx,jointd),'two_stage':(fmu,xx,dd),'independent_gaussian':(imu,ix,idd),'lifted':(fmu,lx,ld)}
        # Same counts and identical fits as main study, independently checked.
        assert abs(covariance_error(jointx,jointd,x,d)-original['parameter_errors']['joint']['covariance_relative'])<1e-8
        menu=np.sort(np.random.default_rng(seed+100000).choice(n,4,replace=False))
        true_rank=ranking_probability((mu,x,d),menu,seed+200000)
        ranks={name:ranking_probability(model,menu,seed+201000+j) for j,(name,model) in enumerate(models.items())}
        ranks['logit']=ranking_probability(None,menu,seed+202000,logit=logit)
        ranks['oracle_mc']=ranking_probability((mu,x,d),menu,seed+203000)
        rankresult={name:{'tv':float(np.abs(p-true_rank).sum()/2),'probabilities':p.tolist()} for name,p in ranks.items()}
        menus=[np.asarray(original['menu_evaluation']['menus_zero_based'][1])]
        lp=simulated_probabilities(fmu,lx,ld,menus,seed+204000)[0];truth=np.array(original['menu_evaluation']['truth_mc'][1])
        metrics={}
        for name,item in original['menu_evaluation']['models'].items():
            prob=np.array(item['probabilities'][1]);clipped=np.maximum(prob,.5/65536);clipped/=clipped.sum()
            metrics[name]={'cross_entropy20':float(-np.sum(truth[truth>0]*np.log(clipped[truth>0]))),'maximum_choice_probability_error20':float(np.max(np.abs(prob-truth))),'top_true_item_probability_error20':float(abs(prob[np.argmax(truth)]-max(truth)))}
        save_json(target,{'configuration':cfg,'provenance':evidence,'lifted_covariance_relative_error':covariance_error(lx,ld,x,d),'lifted_menu20_tv':float(np.abs(lp-truth).sum()/2),'lifted_diagnostics':ldiagn,'ranking_menu_zero_based':menu.tolist(),'ranking_draws_per_model':131072,'true_ranking_probabilities':true_rank.tolist(),'ranking':rankresult,'additional_choice_metrics':metrics,'fitted_parameters':{name:{'mean':a.tolist(),'loadings':b.tolist(),'diagonal':c.tolist()} for name,(a,b,c) in models.items()}})
        print('comparison',graph,count,seed,'lifted',covariance_error(lx,ld,x,d),'ranking',rankresult['joint']['tv'],flush=True)

if __name__=='__main__':main()
