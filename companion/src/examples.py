"""Reproducible synthetic examples; no market data or investment backtest.

Run from project root: .venv/bin/python companion/src/examples.py
Three-way probabilities, inverse calibration, inquiry values, and market-maker
costs use deterministic one-dimensional integration or closed formulas. Larger
menu probabilities use Gaussian orthant integration checked with a second seed.
"""
import hashlib
import itertools
import json
import sys
from pathlib import Path
import numpy as np
from scipy.integrate import quad
from scipy.optimize import root
from scipy.special import ndtr
from scipy.stats import multivariate_normal
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parents[1]
ROOT = HERE.parent
sys.path.insert(0, str(ROOT / "src"))
from sparse_recovery import anchor_edges, edge_distances, reconstruct_anchors

SQRT2PI = np.sqrt(2*np.pi)

def phi(z):
    return np.exp(-z*z/2)/SQRT2PI

def bivariate_cdf(upper, covariance):
    scale = np.sqrt(np.diag(covariance))
    u = upper/scale
    rho = covariance[0, 1]/np.prod(scale)
    return quad(lambda z: phi(z)*ndtr((u[1]-rho*z)/np.sqrt(1-rho*rho)),
                -np.inf, u[0], epsabs=2e-12, epsrel=2e-12)[0]

def probabilities(mu, cov, seed=7000):
    n = len(mu)
    ans = []
    for i in range(n):
        js = np.delete(np.arange(n), i)
        contrast = np.eye(n)[js]-np.eye(n)[i]
        cc = contrast@cov@contrast.T
        upper = mu[i]-mu[js]
        if n == 2:
            ans.append(ndtr(upper[0]/np.sqrt(cc[0, 0])))
        elif n == 3:
            ans.append(bivariate_cdf(upper, cc))
        else:
            ans.append(multivariate_normal.cdf(upper, cov=cc, maxpts=4194304,
                       abseps=1e-8, releps=1e-8, rng=np.random.default_rng(seed+i)))
    ans = np.array(ans)
    assert abs(ans.sum()-1) < 3e-7
    return ans

def inverse_three(target, covariance):
    def unpack(x):
        return np.r_[x, -sum(x)]
    sol = root(lambda x: probabilities(unpack(x), covariance)[:2]-target[:2], [0.3, 0])
    assert sol.success
    mu = unpack(sol.x)
    p = probabilities(mu, covariance)
    assert np.max(np.abs(p-target)) < 1e-10
    rankings = {}
    for order in itertools.permutations(range(3)):
        # Consecutive contrasts are negative precisely for the stated order.
        contrast = np.eye(3)[list(order[1:])]-np.eye(3)[list(order[:-1])]
        rankings[''.join('ABC'[j] for j in order)] = bivariate_cdf(-contrast@mu, contrast@covariance@contrast.T)
    assert abs(sum(rankings.values())-1) < 1e-10
    return {'mean': mu.tolist(), 'covariance': covariance.tolist(), 'full_menu': p.tolist(),
            'A_beats_B': float(probabilities(mu[:2], covariance[:2, :2])[0]),
            'rankings': rankings, 'A_B_top_two_any_order': rankings['ABC']+rankings['BAC']}

def surplus_iid(mu):
    # E max_i(mu_i+Z_i) from its upper and lower tails, integrated on R+.
    return quad(lambda x: 1-np.prod(ndtr(x-mu))-np.prod(ndtr(-x-mu)),
                0, np.inf, epsabs=1e-11, epsrel=1e-11)[0]

def expected_max_pair(mu, cov, i, j):
    s = np.sqrt(cov[i, i]+cov[j, j]-2*cov[i, j])
    z = (mu[i]-mu[j])/s
    return mu[i]*ndtr(z)+mu[j]*ndtr(-z)+s*phi(z)

def write_table(name, columns, rows):
    text = '\\begin{tabular}{'+columns+'}\n\\toprule\n'
    text += ' \\\\\n'.join(' & '.join(str(x) for x in row) for row in rows)
    text += ' \\\\\n\\bottomrule\n\\end{tabular}\n'
    (HERE/'generated'/f'{name}.tex').write_text(text.replace('\\midrule \\\\', '\\midrule'))

def main():
    for folder in ('output', 'generated', 'figures'):
        (HERE/folder).mkdir(exist_ok=True)
    target = np.array([.5, .3, .2])
    inverse = {'independent': inverse_three(target, np.eye(3)),
               'correlated': inverse_three(target, np.array([[1,.8,0],[.8,1,0],[0,0,1]]))}
    write_table('inverse', 'lrrrrrr',
        [['Model', '$\\mu_A$', '$\\mu_B$', '$\\mu_C$', '$P(A>B)$', '$P(ABC)$', '$P(\\{A,B\\}\\text{ top two})$'],
         ['\\midrule']]+[[name.capitalize()]+[f'{x:.4f}' for x in row['mean']]+
             [f"{row['A_beats_B']:.4f}", f"{row['rankings']['ABC']:.4f}", f"{row['A_B_top_two_any_order']:.4f}"] for name,row in inverse.items()])

    b = 10.0
    origin = b*surplus_iid(np.zeros(3))
    assert abs(origin-30/(2*np.sqrt(np.pi))) < 1e-10
    market=[]
    for q in ([0,0,0], [10,0,0], [20,0,0], [20,10,0]):
        q = np.array(q, dtype=float)
        p = probabilities(q/b, np.eye(3))
        cost = b*surplus_iid(q/b)-origin
        assert q.max()-cost <= origin+1e-10
        market.append({'q':q.tolist(), 'prices':p.tolist(), 'revenue_from_origin':cost,
                       'worst_outcome_loss_at_q':q.max()-cost})
    # Independent finite-difference check of the cost derivative, all coordinates.
    probe = np.array([.5, -.2, .1]); step=1e-4
    gradient = np.array([(surplus_iid(probe+step*np.eye(3)[i])-surplus_iid(probe-step*np.eye(3)[i]))/(2*step) for i in range(3)])
    derivative_gap = float(np.max(np.abs(gradient-probabilities(probe, np.eye(3)))))
    assert derivative_gap < 1e-8
    write_table('market', 'lrrrrr', [['Shares $(q_A,q_B,q_C)$', '$p_A$', '$p_B$', '$p_C$', 'Revenue', 'Max. loss'], ['\\midrule']]+
       [['$(' + ','.join(str(int(v)) for v in row['q']) + ')$']+[f'{p:.4f}' for p in row['prices']]+
        [f"{row['revenue_from_origin']:.4f}", f"{row['worst_outcome_loss_at_q']:.4f}"] for row in market])

    geometry_path = ROOT/'output'/'geometry.json'
    allocation = json.loads(geometry_path.read_text())['allocation']
    mu = np.array(allocation['mean']); cov = np.array(allocation['covariance'])
    baseline = probabilities(mu, cov)
    keep = np.arange(1, 5)
    deletion = probabilities(mu[keep], cov[np.ix_(keep, keep)])
    deletion_check = probabilities(mu[keep], cov[np.ix_(keep, keep)], seed=8000)
    integration_gap = float(np.max(np.abs(deletion-deletion_check)))
    assert integration_gap < 5e-7
    renormalized = baseline[keep]/(1-baseline[0])
    pair_rows=[]
    for i,j in itertools.combinations(range(5),2):
        value=expected_max_pair(mu,cov,i,j)
        pair_rows.append({'pair':'ABCDE'[i]+'ABCDE'[j], 'expected_best_utility':float(value),
                          'net_of_two_inquiries':float(value-.24)})
    singles=[{'provider':'ABCDE'[i], 'net_of_one_inquiry':float(mu[i]-.12)} for i in range(5)]
    inquiry_best=max(pair_rows,key=lambda x:x['net_of_two_inquiries'])
    assert inquiry_best['net_of_two_inquiries'] > max(row['net_of_one_inquiry'] for row in singles)
    write_table('providers', 'lrrrrr', [['Provider', 'Full menu', 'Delete A', 'Renormalize', 'Cap A', 'KL cap'], ['\\midrule']]+
      [['ABCDE'[i], f'{baseline[i]:.4f}', '--' if i==0 else f'{deletion[i-1]:.4f}',
        '--' if i==0 else f'{renormalized[i-1]:.4f}', f"{allocation['gaussian_bregman_projection'][i]:.4f}",
        f"{allocation['kl_projection'][i]:.4f}"] for i in range(5)])
    write_table('inquiry', 'lrr', [['Contacted providers','Expected best utility','Net of inquiry cost'], ['\\midrule']]+
      [[row['pair'], f"{row['expected_best_utility']:.4f}", f"{row['net_of_two_inquiries']:.4f}"] for row in pair_rows])

    # Eight alternatives, six one-dimensional anchors. All input data are printed.
    v = np.array([0,1,2,4,-1,2,3,-2.], dtype=float)[:, None]
    d = np.array([.2,.3,.4,.5,.6,.7,.8,.9])
    means = np.array([.4,.3,.2,.1,0,-.1,-.2,-.3])
    edges = anchor_edges(8,2); distances=edge_distances(v,d,edges)
    vr, dr, diag = reconstruct_anchors(8,1,edges,distances)
    vc=v-v.mean(0); gram_error=float(np.max(np.abs(vr@vr.T-vc@vc.T)))
    diagonal_error=float(np.max(np.abs(dr-d)))
    assert max(gram_error, diagonal_error) < 1e-10
    pair_index=np.where(np.all(edges==[0,1],axis=1))[0][0]
    response=ndtr((means[0]-means[1]+np.array([-1.,1.]))/np.sqrt(distances[pair_index]))
    write_table('anchors', 'lrrrrrrrr', [['Alternative']+list('ABCDEFGH'), ['\\midrule'],
        ['$v_i$']+[f'{x:.1f}' for x in v.ravel()], ['$d_i$']+[f'{x:.1f}' for x in d],
        ['$\\mu_i$']+[f'{x:.1f}' for x in means]])

    sigma1=np.eye(2); sigma2=np.array([[4.,2.],[2.,2.]])
    contrast=np.array([1.,-1.]); assert contrast@sigma1@contrast==contrast@sigma2@contrast==2
    assert min(np.linalg.eigvalsh(sigma2)) > 0
    x=np.linspace(0,1,201); w=np.column_stack((x,1-x))
    variances=[np.einsum('ni,ij,nj->n',w,s,w) for s in (sigma1,sigma2)]

    plt.rcParams.update({'font.family':'serif','font.size':10,'axes.spines.top':False,'axes.spines.right':False,
                         'pdf.fonttype':42,'ps.fonttype':42})
    fig, axes = plt.subplots(1,2,figsize=(9,3.3), layout='constrained')
    labels=['Independent','Correlated']
    vals=[inverse[k]['A_B_top_two_any_order'] for k in inverse]
    axes[0].bar(labels,vals,color=['#38678c','#c87539'],width=.55)
    axes[0].axhline(.5*.3/.5+.3*.5/.7,color='black',ls='--',lw=1,label='Sequential logit ranking')
    axes[0].set_ylim(0,1); axes[0].set_ylabel('Probability that A and B finish top two'); axes[0].legend(fontsize=8,loc='upper left')
    axes[0].set_title('Same full-menu probabilities (0.5, 0.3, 0.2)')
    for i,y in enumerate(variances):
        axes[1].plot(x,y,label=f'Covariance {i+1}',color=['#38678c','#c87539'][i])
    axes[1].set_xlabel('Weight in asset A');axes[1].set_ylabel('Portfolio variance')
    axes[1].set_title('Same pairwise law, different portfolio risk'); axes[1].legend(fontsize=8)
    fig.savefig(HERE/'figures'/'identification.pdf'); fig.savefig(HERE/'figures'/'identification.png',dpi=180);plt.close(fig)
    source_paths=[Path(__file__), ROOT/'src'/'sparse_recovery.py', geometry_path]
    result={'scope':'Synthetic models and deterministic mathematical illustrations; no observed market data.',
        'source_sha256':{str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in source_paths},
        'inverse':inverse, 'market':{'b':b, 'worst_loss_bound':origin, 'cost_gradient_check':derivative_gap, 'rows':market},
        'providers':{'mean':mu.tolist(),'covariance':cov.tolist(),'baseline':baseline.tolist(), 'delete_A':deletion.tolist(),
                     'renormalize':renormalized.tolist(),'second_seed_gap':integration_gap,'inquiry_pairs':pair_rows,
                     'inquiry_singles':singles,'best_pair':inquiry_best, 'allocation':allocation},
        'anchors':{'v':v.ravel().tolist(),'d':d.tolist(),'mu':means.tolist(),'pairs':len(edges),
                   'q_AB_minus1_plus1':response.tolist(),'gram_error':gram_error,'diagonal_error':diagonal_error,
                   'diagnostics':diag},
        'portfolio_counterexample':{'covariance1':sigma1.tolist(),'covariance2':sigma2.tolist(),
                    'contrast_variance':2.,'minimum_variance_weight_A':[.5,0.]},'status':'passed'}
    (HERE/'output'/'examples.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({'status':'passed','inverse':inverse,'market':result['market'],
                      'best_inquiry_pair':inquiry_best,'anchor_responses':response.tolist(),
                      'integration_gap':integration_gap},indent=2))

if __name__ == '__main__':
    main()
