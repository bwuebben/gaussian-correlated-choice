"""Generate manuscript tables and figures from saved synthetic experiments."""
import hashlib
import json
import platform
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import NullFormatter
from experiment_utils import ROOT, save_json

OUT=ROOT/'paper'/'generated'; OUT.mkdir(parents=True,exist_ok=True)
FIG=ROOT/'paper'/'figures'; FIG.mkdir(parents=True,exist_ok=True)
def read(name): return json.loads((ROOT/'output'/name).read_text())
def group(name): return [json.loads(p.read_text()) for p in sorted((ROOT/'output'/name).glob('*.json'))]
def table(name,columns,heading,rows):
    text='\\begin{tabular}{'+columns+'}\n\\toprule\n'+heading+' \\\\\n\\midrule\n'
    text+='\n'.join(' & '.join(map(str,row))+r' \\' for row in rows)
    (OUT/(name+'.tex')).write_text(text+'\n\\bottomrule\n\\end{tabular}\n')
def avg(rows,fn): return float(np.mean([fn(x) for x in rows]))
def fmt(x): return f'{x:.3f}'
def sci(x): return f'${x:.1e}$'.replace('e-0',r'\times10^{-').replace('e+0',r'\times10^{').replace('e-',r'\times10^{-').replace('e+',r'\times10^{').replace('$}', '$')[:-1]+'}$' if x else '$0$'
# Reject stale results before publishing any aggregate.
manifest={}
for p in sorted((ROOT/'output').rglob('*.json')):
    if p.name in ('results_summary.json','integrity.json'): continue
    a=json.loads(p.read_text())
    for name,h in a.get('provenance',{}).get('source_sha256',{}).items():
        assert hashlib.sha256((ROOT/'src'/name).read_bytes()).hexdigest()==h,(p,name)
    manifest[str(p.relative_to(ROOT))]=hashlib.sha256(p.read_bytes()).hexdigest()
finite=group('finite'); sparse=group('sparse'); misspec=group('misspecification'); scaling=group('scaling'); geometry=read('geometry.json')
assert len(finite)==90 and len(sparse)==180 and len(misspec)==24 and len(scaling)==4
summary={'data_kind':'synthetic','result_counts':{'finite':90,'anchor':36,'random':144,'misspecification':24,'scaling':4},'platform':platform.platform(),'input_sha256':manifest}
summary['finite_optimizer_failures']={k:sum(not a['diagnostics'][k]['optimizer_success'] for a in finite) for k in ('joint','independent','logit','stress')}
summary['pair_fallback_count']=sum(a['diagnostics']['pairs']['fallback_optimization_count'] for a in finite)
summary['pair_fallback_failures']=sum(a['diagnostics']['pairs']['fallback_failure_count'] for a in finite)
anchors=[a for a in sparse if a['configuration']['kind']=='anchors']; random=[a for a in sparse if a['configuration']['kind']=='random']
summary['anchor_max_covariance_error']=max(a['covariance_relative_error'] for a in anchors)
summary['random_successes']=sum(a['covariance_relative_error']<1e-4 for a in random)
table('anchors','rrrr',r'$n$ & $r$ & Pairs & Maximum relative error',[(n,r,next(a['observed_pairs'] for a in anchors if a['configuration']['n']==n and a['configuration']['rank']==r),sci(max(a['covariance_relative_error'] for a in anchors if a['configuration']['n']==n and a['configuration']['rank']==r))) for n in (50,200,1000) for r in (1,2,5,10)])
rows=[]; summary['finite_aggregates']=[]
for graph in ('anchors','complete'):
 for design in ('fixed','adaptive','oracle'):
  for count in (50,100,250,500,1000):
   selected=[a for a in finite if a['configuration']['graph']==graph and a['configuration']['design']==design and a['configuration']['per_treatment']==count]
   record={'graph':graph,'design':design,'N':count,'covariance':avg(selected,lambda a:a['parameter_errors']['joint']['covariance_relative'])}
   for model in ('joint','two_stage','independent_gaussian','logit','oracle_mc'):
    record[model+'_tv20']=avg(selected,lambda a:a['menu_evaluation']['models'][model]['tv_by_menu'][1])
    record[model+'_tv100']=avg(selected,lambda a:a['menu_evaluation']['models'][model]['tv_by_menu'][2])
    record[model+'_diversion']=avg(selected,lambda a:a['menu_evaluation']['models'][model]['diversion_rmse'])
   record['joint_tv20_range']=[min(a['menu_evaluation']['models']['joint']['tv_by_menu'][1] for a in selected),max(a['menu_evaluation']['models']['joint']['tv_by_menu'][1] for a in selected)]
   summary['finite_aggregates'].append(record)
   if design=='adaptive' and count in (50,250,1000):
    rows.append([graph.capitalize(),count,fmt(record['covariance'])]+[fmt(record[k+'_tv20']) for k in ('two_stage','joint','independent_gaussian','logit')]+[fmt(record['joint_tv100'])])
table('finite','lrrrrrrr',r'Design & $N$ & $C$ error & Two-stage & Joint & Indep. & Logit & Joint, 100',rows)
table('diversion','lrrrr',r'Design & $N$ & Joint & Indep. & Logit',[[a['graph'].capitalize(),a['N']]+[fmt(a[k+'_diversion']) for k in ('joint','independent_gaussian','logit')] for a in summary['finite_aggregates'] if a['design']=='adaptive' and a['N'] in (50,250,1000)])
labels={'gaussian':'Gaussian control','student_t':r'Student $t_5$','mixture':'Gaussian mixture','menu_covariance':'Menu-dependent covariance','context_effect':'Context-dependent means','treatment_variance':'Treatment-dependent variance','pair_treatment_coefficient':'Pair-specific treatment scale','nonadditive':'Nonadditive utility'}
rows=[]; summary['misspecification']=[]
for key,label in labels.items():
 selected=[a for a in misspec if a['configuration']['scenario']==key]
 record={'scenario':key,'curve_rmse':avg(selected,lambda a:a['heldout_curve_rmse']),'tv20':avg(selected,lambda a:a['menu_tv'][2]),'tv20_min':min(a['menu_tv'][2] for a in selected),'tv20_max':max(a['menu_tv'][2] for a in selected)}
 summary['misspecification'].append(record)
 rows.append([label,fmt(record['curve_rmse']),fmt(record['tv20']),f"{record['tv20_min']:.3f}--{record['tv20_max']:.3f}"])
table('misspecification','lrrr',r'Data-generating model & Pair RMSE & Menu TV & Range',rows)
table('scaling','rrrrr',r'$n$ & Pairs & Response (s) & Inverse (s) & Distance error',[[f"{a['configuration']['n']:,}",f"{a['observed_pairs']:,}",f"{a['population_response_evaluation_seconds']:.4f}",f"{a['reconstruction_seconds']:.4f}",sci(a['distance_relative_error_20000_sampled_pairs'])] for a in sorted(scaling,key=lambda a:a['configuration']['n'])])
table('quadrature','rrrr',r'Nodes & Total probability & Derivative discrepancy & Solve residual',[[a['quadrature_points'],f"{a['probability_mass']:.10f}",sci(a['hessian_vs_raw_grid_derivative_relative']),sci(a['solve_relative_residual'])] for a in geometry['derivatives']])
a=geometry['allocation']; table('allocation','rrrr',r'Alternative & Baseline & Gaussian projection & KL projection',[[i+1]+[f'{a[k][i]:.5f}' for k in ('baseline','gaussian_bregman_projection','kl_projection')] for i in range(5)])
table('linear_algebra','rrrrr',r'$n$ & $M$ & Reduced (s) & CG (s) & CG steps',[[f"{a['n']:,}",a['quadrature_rank_parameter'],f"{a['reduced_seconds_including_small_matrix_formation']:.5f}",f"{a['cg_seconds']:.5f}",a['cg_iterations']] for a in geometry['benchmarks'] if a['n']>=100000])
plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False,'pdf.fonttype':42})
fig,axes=plt.subplots(1,2,figsize=(8,3.3),sharey=True)
for ax,graph in zip(axes,('anchors','complete')):
 for design,color in zip(('fixed','adaptive','oracle'),('#2c7bb6','#d95f02','#388347')):
  a=[a for a in summary['finite_aggregates'] if a['graph']==graph and a['design']==design]
  xx=[a['N'] for a in a]; yy=[a['joint_tv20'] for a in a]
  ax.plot(xx,yy,'o-',label=design.capitalize(),color=color,markersize=4)
  if design=='adaptive': ax.fill_between(xx,[a['joint_tv20_range'][0] for a in a],[a['joint_tv20_range'][1] for a in a],color=color,alpha=.14)
 ax.axhline(avg(finite,lambda a:a['menu_evaluation']['models']['oracle_mc']['tv_by_menu'][1]),linestyle=':',color='gray',label='MC comparison floor')
 ax.set_xscale('log'); ax.set_xticks([50,100,250,500,1000],[50,100,250,500,1000]); ax.set_xlabel('Half the total observations per pair, N'); ax.set_title(graph.capitalize()); ax.grid(alpha=.15)
axes[0].set_ylabel('Mean total variation, 20 alternatives'); axes[1].legend(fontsize=8);fig.tight_layout();fig.savefig(FIG/'finite.pdf');plt.close(fig)
fig,axes=plt.subplots(1,3,figsize=(9,3),sharey=True)
for ax,n in zip(axes,(200,500,1000)):
 for r,color in zip((2,5,10),('#2c7bb6','#d95f02','#388347')):
  cc=sorted(set(a['configuration']['c'] for a in random if a['configuration']['n']==n and a['configuration']['rank']==r))
  yy=[np.mean([a['covariance_relative_error']<1e-4 for a in random if a['configuration']['n']==n and a['configuration']['rank']==r and a['configuration']['c']==c]) for c in cc]
  ax.plot(cc,yy,'o-',label=f'r = {r}',color=color,markersize=4)
 ax.set_xscale('log'); ax.set_xticks(([.05,.1,.5,1,2,4,8] if n==200 else [.5,1,2,4,8]), (['.05','.1','.5','1','2','4','8'] if n==200 else ['.5','1','2','4','8'])); ax.xaxis.set_minor_formatter(NullFormatter()); ax.tick_params(axis='x',labelsize=8); ax.set_xlabel('Pair-count multiplier c'); ax.set_title(f'n = {n}');ax.set_ylim(-.06,1.06);ax.grid(alpha=.15)
axes[0].set_ylabel('Fraction with relative error < 10⁻⁴');axes[-1].legend();fig.tight_layout();fig.savefig(FIG/'random.pdf');plt.close(fig)
supplement=group('supplement')
assert len(supplement)==30
summary['result_counts']['supplement']=30
summary['supplement_aggregates']=[]
rows=[]
for graph in ('anchors','complete'):
 for count in (250,1000):
  selected=[a for a in supplement if a['configuration']['kind']=='completion_and_rankings' and a['configuration']['graph']==graph and a['configuration']['N']==count]
  rec={'graph':graph,'N':count,'lifted_covariance':avg(selected,lambda a:a['lifted_covariance_relative_error']),'lifted_tv20':avg(selected,lambda a:a['lifted_menu20_tv'])}
  for model in ('joint','two_stage','independent_gaussian','logit','oracle_mc'):
   rec[model+'_ranking_tv']=avg(selected,lambda a:a['ranking'][model]['tv'])
  summary['supplement_aggregates'].append(rec)
  rows.append([graph.capitalize(),count,fmt(rec['lifted_covariance']),fmt(rec['lifted_tv20'])]+[fmt(rec[k+'_ranking_tv']) for k in ('joint','independent_gaussian','logit','oracle_mc')])
table('supplement','lrrrrrrr',r'Graph & $N$ & Lifted $C$ & Lifted TV & Joint rank & Indep. rank & Logit rank & MC rank',rows)
summary['supplement_lifted_step_failures']=sum(not a['lifted_diagnostics']['step_converged'] for a in supplement if a['configuration']['kind']=='completion_and_rankings')
summary['noisy_pure_factor']=[{'rank':r,'N':count,'covariance':avg([a for a in supplement if a['configuration']['kind']=='noisy_pure_factor' and a['configuration']['rank']==r and a['configuration']['N']==count],lambda a:a['covariance_relative_error'])} for r in (2,5,10) for count in (250,1000)]
save_json(ROOT/'output'/'results_summary.json',summary)
print('Generated nine tables, two figures, and source-checked result summary.')
