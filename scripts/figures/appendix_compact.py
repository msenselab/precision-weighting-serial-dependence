"""Compact appendix figures from frozen inputs only: no fitting, sampling or new tests.

Canonical run (with the project's v3 optional numerical dependencies available):
  .venv/bin/python scripts/figures/appendix_compact.py

Each plotted coordinate is exported to sdu.paths.FIGURE_DATA / 'compact_figures'.
The original printed tables are retained separately in FIGURE_DATA / 'precompact_tables'.
"""
from pathlib import Path
import os,json,hashlib,argparse,re
# Set the author-verified v3 configuration before importing model/report modules.
for key,value in dict(GKF_SCALE='lin',GKF_LIK='gh',GKF_CORE='reduced',GKF_BOUNDS='v3',GKF_INIT='gs',GKF_NOISELESS='1',GKF_QDIAG='1',GKF_READOUT='1',GKF_NOISE_TREND='0',GKF_EFFENC='0').items():os.environ[key]=value
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from scipy.stats import norm
ROOT=Path(__file__).resolve().parents[2]
import sys
sys.path.insert(0,str(ROOT/'scripts/model'))
sys.path.insert(0,str(ROOT/'src'))
from sdu import paths as SP
FIG=SP.FIGURES;DATA=SP.FIGURE_DATA/'compact_figures'
BEH=ROOT/'results/behavior';MOD=ROOT/'results/model/minimal_qdiag'
LOW='#E8503A';HIGH='#4BB8D3';EDGE_LOW='#872C20';EDGE_HIGH='#215664'
EXP={'Exp 1':dict(color='#303030',marker='o',ls='-',mfc='#303030'), 'Exp 2':dict(color='#777777',marker='s',ls='--',mfc='white')}
COH={'High':dict(color=HIGH,marker='o',ls='-',edge=EDGE_HIGH),'Low':dict(color=LOW,marker='^',ls='--',edge=EDGE_LOW)}
SOURCE_HASHES={};OUTPUTS=[]
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':9,'axes.titlesize':10,'axes.labelsize':9,'xtick.labelsize':8,'ytick.labelsize':8,'legend.fontsize':8,'axes.spines.top':False,'axes.spines.right':False,'axes.linewidth':.65,'pdf.fonttype':42,'ps.fonttype':42,'savefig.facecolor':'white','figure.facecolor':'white'})
def read(path):
 p=Path(path);SOURCE_HASHES[str(p.relative_to(ROOT))]=hashlib.sha256(p.read_bytes()).hexdigest();return pd.read_csv(p)
def export(name,frame):
 DATA.mkdir(parents=True,exist_ok=True);frame.to_csv(DATA/(name+'.csv'),index=False)
def save(fig,name):
 FIG.mkdir(parents=True,exist_ok=True)
 for ext in ['png','pdf']:fig.savefig(FIG/(name+'.'+ext),dpi=260,bbox_inches='tight',pad_inches=.05)
 OUTPUTS.append(name);plt.close(fig)
def clean(ax,zero=True):
 if zero:ax.axvline(0,color='#9C9C9C',lw=.7,zorder=0)
 ax.grid(axis='x',color='#ECECEC',lw=.6);ax.set_axisbelow(True)
def title(ax,s):ax.set_title(s,loc='left',fontweight='bold',pad=8)
def legend_exp(fig,y=1.02):
 h=[Line2D([],[],color=v['color'],marker=v['marker'],mfc=v['mfc'],linestyle='none',label=k,ms=4.5) for k,v in EXP.items()];fig.legend(handles=h,loc='upper center',bbox_to_anchor=(.56,y),ncol=2,frameon=False)
def normal_ci(frame,b='beta',se='se'):
 f=frame.copy();f['ci_low']=f[b]-norm.ppf(.975)*f[se];f['ci_high']=f[b]+norm.ppf(.975)*f[se];return f
def coef_forest(ax,frame,keys,labels,heading):
 y=np.arange(len(keys))[::-1]
 for exp,v in EXP.items():
  f=frame[frame.experiment==exp].set_index('term').loc[keys];off=.13 if exp=='Exp 1' else -.13
  ax.errorbar(f.beta,y+off,xerr=np.vstack([f.beta-f.ci_low,f.ci_high-f.beta]),fmt=v['marker'],ms=4.3,color=v['color'],mfc=v['mfc'],mec=v['color'],lw=1.15,capsize=2)
 ax.set_yticks(y,labels);ax.set_ylim(-.55,len(keys)-.45);clean(ax);title(ax,heading)
def behavior_forests():
 p=read(BEH/'manuscript_ready/primary_dual_lmm_parameters.csv');p=p[p.parameter.isin(['curDur_c:current_low_coherence','preDur_c:current_low_coherence','curDur_c:same_transition','preDur_c:same_transition'])].copy();p['experiment']=p.dataset.map({'manuscript_exp1_dynamic':'Exp 1','manuscript_exp2_fixed':'Exp 2'});assert p.experiment.notna().all()
 p['term']=p.parameter.str.replace(':same_transition',':moderator',regex=False).str.replace(':current_low_coherence',':moderator',regex=False);p=normal_ci(p)
 c=read(BEH/'analysis/four_cells/trial_lmm_contrasts.csv');c['experiment']=c.experiment.map({'Exp1_dynamic':'Exp 1','Exp2_fixed':'Exp 2'});c=c.rename(columns={'contrast':'term'})
 export('figA1_primary_moderators',p);export('figA1_transition_contrasts',c)
 fig,axs=plt.subplots(2,2,figsize=(7.15,4.75),layout='constrained');fig.set_constrained_layout_pads(w_pad=.04,h_pad=.07,wspace=.14,hspace=.10)
 coef_forest(axs[0,0],p,['curDur_c:moderator','preDur_c:moderator'],['Current duration\n× moderator','Previous duration\n× moderator'],'A  Primary moderators')
 coef_forest(axs[0,1],c,['HH','HL','LH','LL'],['H → H','H → L','L → H','L → L'],'B  Transition-specific slopes')
 coef_forest(axs[1,0],c,['current_L_minus_H','previous_H_minus_L','same_minus_switch'],['Current L − H','Previous H − L','Same − switch'],'C  Planned transition contrasts')
 coef_forest(axs[1,1],c,['LL_minus_other_three','HL_minus_LL'],['L → L − other three','H → L − L → L'],'D  Post-hoc contrasts')
 axs[0,0].set_xlabel('Slope interaction (s/s)');axs[0,1].set_xlabel('Previous-duration slope (s/s)')
 for ax in axs[1]:ax.set_xlabel('Slope contrast (s/s)')
 legend_exp(fig,1.065);save(fig,'figA1_behavioral_forests')
def response_history():
 a=read(BEH/'analysis/response_history/lmm_parameters.csv');a=a[a.parameter.str.startswith('response_error_lag')].copy();a['experiment']=a.experiment.map({'Exp1_dynamic':'Exp 1','Exp2_fixed':'Exp 2'});a['lag']=a.parameter.str.extract(r'(\d+)$').astype(int);a=normal_ci(a);export('figA2_response_lags',a)
 b=read(BEH/'analysis/response_history/coherence_interaction_lmm.csv');b=b[b.term.isin(['preDur_c:low','response_error_lag1:low'])].copy();b['experiment']=b.experiment.map({'Exp1_dynamic':'Exp 1','Exp2_fixed':'Exp 2'});b=b.rename(columns={'b':'beta'});b=normal_ci(b);export('figA2_coherence_interactions',b)
 fig,axs=plt.subplots(1,2,figsize=(7.15,2.85),layout='constrained',gridspec_kw={'width_ratios':[1,1.15]})
 for exp,v in EXP.items():
  f=a[a.experiment==exp].sort_values('lag');off=-.045 if exp=='Exp 1' else .045
  axs[0].errorbar(f.lag+off,f.beta,yerr=np.vstack([f.beta-f.ci_low,f.ci_high-f.beta]),color=v['color'],marker=v['marker'],mfc=v['mfc'],ls=v['ls'],capsize=2,ms=4,lw=1)
 axs[0].axhline(0,color='#9C9C9C',lw=.7);axs[0].set(xticks=[1,2,3,4,5],xlabel='Previous response-error lag',ylabel='Conditional slope (s/s)');title(axs[0],'A  Residual response history')
 coef_forest(axs[1],b,['preDur_c:low','response_error_lag1:low'],['Previous duration\n× coherence','Lag-1 response error\n× coherence'],'B  Low − High interactions');axs[1].set_xlabel('Slope interaction (s/s)')
 legend_exp(fig,1.10);save(fig,'figA2_response_history_compact')
def behavioral_checks():
 shifts=read(BEH/'analysis/current_coherence_mean_shift/shift_by_duration.csv');export('figA3_duration_shift',shifts)
 disp=read(BEH/'analysis/dispersion/summary.csv');disp['experiment']=disp.experiment.map({'Exp1_dynamic':'Exp 1','Exp2_fixed':'Exp 2'});export('figA3_variability',disp)
 bf=read(BEH/'analysis/behavior_extra/ordering_bayes_factors.csv');bf=bf[(bf.prior_tau==.10)&(bf.delta==.020)&(bf.convention=='meaningful_gap')].copy();assert len(bf)==8;export('figA3_ordering_BF',bf)
 fig=plt.figure(figsize=(7.15,4.65),layout='constrained');gs=fig.add_gridspec(2,2,height_ratios=[1.15,1]);a=fig.add_subplot(gs[0,0]);b=fig.add_subplot(gs[0,1]);c=fig.add_subplot(gs[1,:])
 for exp,v in EXP.items():
  f=shifts[shifts.experiment==exp].sort_values('duration_s');a.errorbar(f.duration_s,f.low_minus_high_s,yerr=f.sem_s,marker=v['marker'],color=v['color'],mfc=v['mfc'],ls=v['ls'],capsize=2,ms=4,lw=1)
 a.axhline(0,color='#aaa',lw=.7);a.set(xlabel='Current duration (s)',ylabel='Low − High response (s)',xticks=[.8,1,1.2,1.4,1.6]);title(a,'A  Mean shift across duration')
 for j,(exp,v) in enumerate(EXP.items()):
  r=disp[disp.experiment==exp].iloc[0];b.errorbar(r.low_minus_high_sd,1-j,xerr=[[r.low_minus_high_sd-r.ci95_low],[r.ci95_high-r.low_minus_high_sd]],fmt=v['marker'],color=v['color'],mfc=v['mfc'],ms=4.5,capsize=2)
 b.set_yticks([1,0],['Exp 1','Exp 2']);b.set_ylim(-.55,1.55);b.set_xlabel('Low − High within-duration SD (s)');clean(b);title(b,'B  Conditional variability')
 keys=['reliability','current','reset','flat'];labels=['Static reliability weighting','Current coherence','L → L elevation','Approximately equal']
 for exp,v in EXP.items():
  f=bf[bf.experiment==exp].set_index('pattern').loc[keys];c.scatter(f.BF_vs_unconstrained,np.arange(4)[::-1]+(.11 if exp=='Exp 1' else -.11),marker=v['marker'],facecolors=v['mfc'],edgecolors=v['color'],s=24,zorder=3)
 c.axvline(1,color='#888',lw=.8,ls=':');c.set_xscale('log');c.set_yticks(np.arange(4)[::-1],labels);c.set_ylim(-.45,3.45);c.set_xlabel('Bayes factor vs unrestricted means (log scale)');c.grid(axis='x',color='#eee');title(c,'C  Explicit ordering predictions')
 legend_exp(fig,1.06);save(fig,'figA3_behavioral_checks')
def model_forest():
 from sdu.reports import v3 as R
 from sdu.model import fitting as FF
 import fit_minimal_space as M
 ro=pd.concat([read(p) for p in sorted((MOD/'readout_space').glob('*.csv'))],ignore_index=True)
 fits=R.Fits(ro,{e:FF.load(e) for e in R.EXPS},dict(M.space()))
 read(MOD/'readout_space_screening.csv');t,_=R.load_screening(MOD/'readout_space_screening.csv')
 full=R.c_level_comparison(fits,t,{2:'C3-R-S0-B2',1:'C3-R-S3-B2'},c_levels=[f'C{i}' for i in range(1,16)]);full['C']=full.model.str.split('-').str[0]
 old=read(SP.FIGURE_DATA/'precompact_tables/Table_B2.csv');rows=[]
 for j in range(15):
  row=dict(C=f'C{j+1}',k=int(old.iloc[j,2]),noise=old.iloc[j,1])
  for exp,acol,gcol in [('Exp 1',3,4),('Exp 2',5,6)]:
   value=full[(full.C==row['C'])&(full.experiment==exp)].iloc[0]
   assert abs(value.dAIC-float(old.iloc[j,acol]))<.00501
   expected=[float(v) for v in old.iloc[j,gcol].split(' / ')];assert np.allclose([value.Kx_Low,value.Kx_High],expected,atol=.000501)
   row[exp+' ΔAIC']=value.dAIC;row[exp+' gain Low']=value.Kx_Low;row[exp+' gain High']=value.Kx_High
  rows.append(row)
 raw=pd.DataFrame(rows);export('figB1_C_level_comparison',raw);export('figB1_frozen_model_details',full)
 y=np.arange(15)[::-1];fig,axs=plt.subplots(1,3,figsize=(7.15,4.85),sharey=True,layout='constrained',gridspec_kw={'width_ratios':[1.12,1,1]})
 for exp,v in EXP.items():axs[0].plot(raw[exp+' ΔAIC'],y+(.12 if exp=='Exp 1' else -.12),linestyle='none',marker=v['marker'],color=v['color'],mfc=v['mfc'],ms=4)
 axs[0].set_yticks(y,[f'C{i+1}  {raw.iloc[i].noise}  (k={int(raw.iloc[i].k)})' for i in range(15)]);axs[0].set_xlabel('ΔAIC (lower is better)');title(axs[0],'A  Relative fit');clean(axs[0]);axs[0].set_xlim(left=-.10)
 for ax,exp,letter in zip(axs[1:],['Exp 1','Exp 2'],['B','C']):
  vals=raw[[exp+' gain Low',exp+' gain High']].to_numpy(float);ax.hlines(y,vals.min(axis=1),vals.max(axis=1),color='#B0B0B0',lw=1)
  for idx,coh in enumerate(['Low','High']):
   v=COH[coh];ax.scatter(vals[:,idx],y+(-.09 if coh=='Low' else .09),marker=v['marker'],s=22,color=v['color'],edgecolors=v['edge'],lw=.55,zorder=3)
  ax.set_xlabel('Mean Kalman gain');title(ax,letter+'  '+exp+' gain');ax.set_xlim(.48,.64);ax.set_xticks([.48,.52,.56,.60,.64]);ax.grid(axis='x',color='#eee')
 for ax in axs:ax.axhspan(11.55,12.45,color='#F0F0F0',zorder=-1);ax.set_ylim(-.65,14.65);ax.tick_params(axis='y',length=0)
 handles=[Line2D([],[],marker=v['marker'],color=v['color'],mfc=v['mfc'],ls='none',label=k,ms=4) for k,v in EXP.items()]+[Line2D([],[],marker=COH[k]['marker'],color=COH[k]['color'],markeredgecolor=COH[k]['edge'],ls='none',label=k,ms=5) for k in ['High','Low']]
 fig.legend(handles=handles,loc='upper center',bbox_to_anchor=(.62,1.035),ncol=4,frameon=False);save(fig,'figB1_model_comparison_forest')
def ppc_overlay():
 d=read(MOD/'readout_space_ppc/duration_curves.csv');assert d.n_cohorts.eq(100).all();export('figB2_duration_PPC',d)
 fig,axes=plt.subplots(3,2,figsize=(7.15,5.50),sharex=True,sharey='row',layout='constrained')
 for i,(panel,ylabel,xlabel) in enumerate([('central tendency','Reproduction error (s)','Current duration (s)'),('serial dependence','Residual reproduction error (s)','Previous duration (s)'),('response variability','Within-duration\nresidual SD (s)','Current duration (s)')]):
  # The source uses the exact label below; no unit/model remapping is implicit.
  valid=[p for p in d.panel.unique() if p==panel]
  if not valid and panel=='within-duration SD':panel=next(p for p in d.panel.unique() if 'SD' in p or 'sd' in p.lower())
  for j,exp in enumerate(['Exp 1','Exp 2']):
   ax=axes[i,j]
   for coh in ['High','Low']:
    v=COH[coh];f=d[(d.experiment==exp)&(d.panel==panel)&(d.coherence==coh)].sort_values('duration');assert len(f)==5,(panel,exp,coh)
    x=f.duration.to_numpy();ax.fill_between(x,f.lo,f.hi,color=v['color'],alpha=.13,lw=0)
    ax.plot(x,f.simulated,color=v['edge'],ls=v['ls'],lw=1.45)
    # Identical x values: no horizontal jitter/distortion of duration is introduced.
    ax.errorbar(x,f.observed,yerr=f.observed_sem,fmt=v['marker'],mfc='white',mec=v['edge'],ecolor=v['edge'],capsize=2,ms=4.3,elinewidth=.8)
   if i<2:ax.axhline(0,color='#999',lw=.6,ls=':')
   ax.set_xticks([.8,1,1.2,1.4,1.6]);ax.tick_params(labelbottom=True,labelleft=True);ax.set_xlabel(xlabel);ax.grid(color='#ededed',lw=.55)
   if j==0:ax.set_ylabel(ylabel)
   letter='ABCDEF'[2*i+j];title(ax,letter+'  '+(exp+' · '+('C3-S0-B2' if j==0 else 'C3-S3-B2') if i==0 else panel.capitalize()))
 h=[Line2D([],[],color=v['edge'],marker=v['marker'],mfc='white',ls=v['ls'],label=k,ms=4.5) for k,v in COH.items()];fig.legend(handles=h,loc='upper center',bbox_to_anchor=(.56,1.08),ncol=2,frameon=False)
 save(fig,'figB2_duration_PPC_overlay')
def recovery_overview():
 d=read(MOD/'recovery_v3_restart_20261113/parameter_summary.csv')
 order=['beta','wuH','lq1','lq2','lam','lq3','d0','ad0','lsde','aq3']
 d=d[(d.design=='fitted')&d.parameter.isin(order)].copy();assert len(d)==20;export('figB3_parameter_recovery',d)
 labels=[r'$\phi_L$',r'$\phi_H$',r'$\sigma_{q_1}$',r'$\sigma_{q_2}$',r'$\lambda$',r'$\sigma_{q_3}$',r'$d_0$',r'$\Delta d_0$',r'$\sigma_e$',r'$\rho_{q_3}$']
 fig,axs=plt.subplots(1,2,figsize=(7.15,4.3),sharey=True,layout='constrained')
 for ax,exp,letter in zip(axs,['Exp 1','Exp 2'],['A','B']):
  e=2 if exp=='Exp 1' else 1;f=d[d.exp==e].copy();assert len(f)==10
  f=f.set_index('parameter').loc[order];y=np.arange(10)[::-1]
  ax.hlines(y,f.r,f.r_mean,color='#bbb',lw=1)
  ax.plot(f.r,y,'o',color='#303030',ms=4,label='Single-cohort median')
  ax.plot(f.r_mean,y,'^',mfc='white',mec='#303030',ms=4.7,ls='none',label='20-replicate average')
  ax.set_xlim(-.08,1.03);ax.set_xticks([0,.25,.5,.75,1]);ax.set_yticks(y,labels);ax.tick_params(labelleft=True);ax.set_xlabel('Generating–recovered correlation');clean(ax);title(ax,letter+'  '+exp)
 axs[0].set_ylim(-.6,9.6);h,l=axs[0].get_legend_handles_labels();fig.legend(h,l,loc='upper center',bbox_to_anchor=(.57,1.04),ncol=2,frameon=False)
 save(fig,'figB3_parameter_recovery_overview')
def effect_recovery():
 r=read(MOD/'recovery_v3_restart_20261113/recovery_long.csv');r=r[(r.design=='fitted')&r.parameter.isin(['delta_phi','delta_weight'])];rows=[]
 assert r.unit.nunique()==880
 for (e,rep),f in r.groupby(['exp','rep']):
  for kind,column in [('true','true'),('recovered','fitted')]:
   rows.append(dict(exp=e,rep=rep,kind=kind,delta_phi=f[f.parameter=='delta_phi'][column].mean(),delta_weight=f[f.parameter=='delta_weight'][column].mean()))
 d=pd.DataFrame(rows);assert len(d)==80;export('figB4_cohort_effect_recovery',d)
 fig,axs=plt.subplots(1,2,figsize=(7.15,2.9),layout='constrained')
 for ax,measure,letter,label in zip(axs,['delta_phi','delta_weight'],['A','B'],[r'$\phi_L-\phi_H$',r'$\Delta w$ from $w_i=(1-K_i^{(x)})\phi_i$']):
  for j,(exp,v) in enumerate(EXP.items()):
   e=2 if exp=='Exp 1' else 1;f=d[(d.exp==e)&(d.kind=='recovered')].sort_values('rep');true=d[(d.exp==e)&(d.kind=='true')][measure].iloc[0]
   # Fixed evenly spaced offsets only separate the 20 frozen cohort estimates.
   ax.scatter(j+np.linspace(-.13,.13,len(f)),f[measure],s=14,marker=v['marker'],facecolors=v['mfc'],edgecolors=v['color'],lw=.65,alpha=.8)
   ax.hlines(true,j-.20,j+.20,color='#111',lw=2,zorder=4)
   correct=int(((f[measure]>0)==(true>0)).sum());ax.annotate(f'{correct}/{len(f)}',xy=(j,.985),xycoords=('data','axes fraction'),ha='center',va='top',fontsize=8)
  # Reserve headroom so correct-direction counts never cover the largest recovery.
  lo,hi=ax.get_ylim();ax.set_ylim(lo,hi+.16*(hi-lo))
  ax.axhline(0,color='#999',lw=.7,ls=':');ax.set_xticks([0,1],['Exp 1','Exp 2']);ax.set_ylabel('Cohort-mean Low − High difference');title(ax,letter+'  '+label)
 save(fig,'figB4_cohort_effect_recovery')
def model_profiles():
 cohorts=read(MOD/'coherence_evidence_v3_20261113/switch_cohorts.csv');summary=read(MOD/'coherence_evidence_v3_20261113/profile_phi_high/summary.csv');export('figB5_switch_cohorts',cohorts);export('figB5_profile_summary',summary)
 fig,axs=plt.subplots(1,3,figsize=(7.15,2.95),layout='constrained',gridspec_kw={'width_ratios':[1.15,1,1]})
 screen=read(MOD/'readout_space_screening.csv');observed_rows=screen[(screen.exp==1)&screen.model.isin(['C3-R-S0-B2','C3-R-S3-B2'])].set_index('model');observed=float(observed_rows.loc['C3-R-S3-B2','AIC']-observed_rows.loc['C3-R-S0-B2','AIC']);assert round(observed,2)==.56
 tail_counts={g:int((f.delta_AIC_S3_minus_S0>=observed).sum()) for g,f in cohorts.groupby('generator')};assert tail_counts=={'S0':83,'S3':2}
 for gen,ls,mark,color in [('S0','-','o','#303030'),('S3','--','^','#777777')]:
  vals=np.sort(cohorts[cohorts.generator==gen].delta_AIC_S3_minus_S0);axs[0].plot(vals,np.arange(1,len(vals)+1)/len(vals),ls=ls,marker=mark,markevery=12,ms=3.5,color=color,lw=1.2,label=gen+' generator')
 axs[0].axvline(observed,color='#111',ls=':',lw=1);axs[0].axvline(0,color='#ccc',lw=.5);axs[0].set(xlabel='AIC(S3) − AIC(S0)',ylabel='Cumulative cohort fraction',ylim=(0,1.03));title(axs[0],'A  S0/S3 cohort recovery');axs[0].legend(frameon=False,fontsize=7,loc='lower right');axs[0].text(.97,.50,'Observed: 0.56\n≥ observed: 83% / 2%',transform=axs[0].transAxes,va='top',ha='right',fontsize=7.5,bbox=dict(facecolor='white',edgecolor='none',alpha=.9,pad=1))
 profiles=read(MOD/'coherence_evidence_v3_20261113/profile_phi_high/all_curves.csv');export('figB5_profile_curves',profiles)
 for ax,e,exp,letter in zip(axs[1:],[2,1],['Exp 1','Exp 2'],['B','C']):
  sub=profiles[profiles.exp==e];assert sub.Sub.nunique()==22
  for _,f in sub.groupby('Sub'):
   f=f.sort_values('phi_high');ax.plot(f.phi_high,f.loss,color='#858585',alpha=.40,lw=.65)
  regular=sub[np.isclose(sub.phi_high*10,np.round(sub.phi_high*10))].copy();regular['grid']=regular.phi_high.round(1)
  med=regular.drop_duplicates(['Sub','grid']).groupby('grid').loss.median();ax.plot(med.index,med.values,color='#222',lw=1.5)
  ax.axhline(3.841459,color='#222',lw=.9,ls='--');ax.set(xlabel=r'High continuity $\phi_H$',xlim=(0,1),ylim=(0,20));title(ax,letter+'  '+exp+' profiles');ax.text(.04,.96,('4/22' if e==2 else '16/22')+' span full grid',transform=ax.transAxes,va='top',fontsize=7.5)
 axs[1].set_ylabel(r'$2\Delta\mathrm{LL}$')
 save(fig,'figB5_model_recovery_profiles')
def gs_transfer():
 from sdu.reports import glasauer_shi as R
 # Existing frozen fits, deterministic observer predictions only; never call fitting routines.
 for p in [MOD/'glasauer_shi/v3/fits_separate.csv',MOD/'glasauer_shi/two_state_fits.csv']:read(p)
 ps=R.participant_statistics(R.Data());export('figC1_frozen_transfer_statistics',ps)
 fig,axs=plt.subplots(1,2,figsize=(7.15,2.8),layout='constrained')
 models=[('data','#222','o','-'),(R.THREE,'#376B99','s','--'),(R.TWO,'#B46F28','^','-.')];x=np.arange(2)
 for ax,measure,letter in zip(axs,['CT','SD'],['A','B']):
  for model,color,mark,ls in models:
   f=ps[ps.model==model].groupby('condition')[measure];mu=f.mean().reindex(R.CONDITIONS);sem=f.sem().reindex(R.CONDITIONS)
   ax.errorbar(x,mu,yerr=sem,color=color,marker=mark,mfc='white' if model!='data' else color,ls=ls,capsize=2,ms=4.5,lw=1.2,label={'data':'Observed',R.THREE:'3SKF',R.TWO:'Two-state'}[model])
  ax.axhline(0,color='#aaa',lw=.65);ax.set_xticks(x,['Randomized\n(fitting sequence)','Random walk\n(frozen transfer)']);ax.set_xlim(-.2,1.2);ax.set_ylabel(measure+' (mean ± SEM)');title(ax,letter+'  '+('Central tendency' if measure=='CT' else 'Serial dependence'))
 h,l=axs[0].get_legend_handles_labels();fig.legend(h,l,loc='upper center',bbox_to_anchor=(.56,1.09),ncol=3,frameon=False);save(fig,'figC1_transfer_compact')
def main():
 DATA.mkdir(parents=True,exist_ok=True)
 for fn in [behavior_forests,response_history,behavioral_checks,model_forest,ppc_overlay,recovery_overview,effect_recovery,model_profiles,gs_transfer]:
  print('Generating',fn.__name__,flush=True);fn()
 manifest=dict(source_hashes=SOURCE_HASHES,figures=OUTPUTS,method='Frozen-source visualization; no refits, simulations or new significance tests',coherence_encoding={'High':'cyan, solid line, circle','Low':'red, dashed line, triangle'},experiments_encoding={'Exp 1':'dark filled circle','Exp 2':'gray open square'})
 (DATA/'manifest.json').write_text(json.dumps(manifest,indent=2));print(json.dumps(dict(figures=len(OUTPUTS),source_files=len(SOURCE_HASHES)),indent=2))
if __name__=='__main__':main()
