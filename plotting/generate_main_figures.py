#!/usr/bin/env python3
"""Shared plotting functions for the manuscript figures.

Public datasets already follow manuscript display order:
  - Experiment 1 = dynamic coherence = data/experiment1/E1.pkl
  - Experiment 2 = fixed coherence   = data/experiment2/E2.pkl
"""

import os
import sys
import json
import pickle
from pathlib import Path
from itertools import combinations

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns

from scipy import stats
from scipy.stats import sem, ttest_rel, ttest_ind, ttest_1samp, pearsonr
from scipy.optimize import curve_fit, least_squares

import statsmodels.formula.api as smf
import statsmodels.api as sm
from statsmodels.stats.stattools import durbin_watson
# pingouin was previously imported but is not used in this figure script.

import warnings
warnings.filterwarnings('ignore')

# Add this directory to path for the shared plotting config.
sys.path.insert(0, os.path.dirname(__file__))

from plot_config import (
    COLORS, TRANSITION_ORDER, TRANSITION_COLORS, TRANSITION_PALETTE,
    COHERENCE_TRANSITION_LABELS,
    set_nature_style, despine, add_significance_stars
)

# Default figure output directory (callers override FIG_DIR before plotting).
FIG_DIR = Path(__file__).resolve().parents[1] / 'figures'
FIG_DIR.mkdir(exist_ok=True)

def save_figure(fig, stem, aliases=()):
    """Save PNG/PDF outputs for the canonical stem and manuscript aliases."""
    for name in (stem, *aliases):
        fig.savefig(FIG_DIR / f'{name}.png', dpi=300, bbox_inches='tight')
        fig.savefig(FIG_DIR / f'{name}.pdf', bbox_inches='tight')


# Set random seed for reproducibility
np.random.seed(42)

# Apply Nature style
set_nature_style()

# Small horizontal jitter per TransitionType so overlapping errorbars remain
# readable on continuous x axes (Figures 2 and 3 A/B). Total spread across the
# four transitions is ~22% of the 0.2 s sampling step, small enough to keep
# the continuous-axis reading intact.
TRANSITION_X_JITTER = {
    t: (i - (len(TRANSITION_ORDER) - 1) / 2) * 0.015
    for i, t in enumerate(TRANSITION_ORDER)
}


# ============================================================
# DATA LOADING  (experiment swap happens here)
# ============================================================

def load_data():
    """Load and prepare the public datasets in manuscript display order."""
    root = Path(__file__).resolve().parents[1]

    # Manuscript Experiment 1 = dynamic coherence.
    df1 = pd.read_pickle(root / "data" / "experiment1" / "E1.pkl")
    df1 = df1[df1["is_outlier"] == False].copy()

    # Manuscript Experiment 2 = fixed coherence.
    df2 = pd.read_pickle(root / "data" / "experiment2" / "E2.pkl")
    df2 = df2[df2["is_outlier"] == False].copy()

    # Add grouping variables
    df1['SameSwitch'] = df1['TransitionType'].map(
        {'HH': 'Same', 'LL': 'Same', 'HL': 'Switch', 'LH': 'Switch'})
    df2['SameSwitch'] = df2['TransitionType'].map(
        {'HH': 'Same', 'LL': 'Same', 'HL': 'Switch', 'LH': 'Switch'})
    # Dynamic Coherence experiment has curCoherence column
    df1['curCoherenceLevel'] = df1['curCoherence'].map({0.3: 'Low', 0.7: 'High'})

    # Decision Carryover: mark each trial as 'Long' or 'Short'
    for df in [df1, df2]:
        df_sorted = df.sort_values(['subID', 'trial_num']).copy()
        sub_means = df_sorted.groupby('subID')['rpr'].transform('mean')
        df_sorted['ResponseType'] = np.where(df_sorted['rpr'] > sub_means, 'Long', 'Short')
        df_sorted['priorResponseType'] = df_sorted.groupby('subID')['ResponseType'].shift(1)
        df.update(df_sorted)
        # Ensure columns are present
        for col in ['ResponseType', 'priorResponseType']:
            if col not in df.columns:
                df[col] = df_sorted[col]

    # Re-sort to make sure the update stuck
    df1 = df1.sort_values(['subID', 'trial_num']).copy()
    df2 = df2.sort_values(['subID', 'trial_num']).copy()

    # Recompute ResponseType/priorResponseType cleanly
    for df in [df1, df2]:
        sub_means = df.groupby('subID')['rpr'].transform('mean')
        df['ResponseType'] = np.where(df['rpr'] > sub_means, 'Long', 'Short')
        df['priorResponseType'] = df.groupby('subID')['ResponseType'].shift(1)

    print(f"Experiment 1 (Dynamic Coherence): {len(df1)} trials, {df1['subID'].nunique()} subjects")
    print(f"Experiment 2 (Fixed Coherence):   {len(df2)} trials, {df2['subID'].nunique()} subjects")

    return df1, df2


# ============================================================
# SDI computation  (used by fig3, fig5)
# ============================================================

def compute_sdi_dataframes(df1, df2):
    """Compute controlled previous-duration slopes for both experiments.

    Each participant × transition cell is fit with current and previous duration
    simultaneously, so the displayed serial-dependence slope is not confounded
    with the current-duration central-tendency component.
    """

    # Experiment 1 (Dynamic Coherence)
    sdi_list_1 = []
    for (sub_id, trans_type), group in df1.groupby(['subID', 'TransitionType']):
        if len(group) >= 5:
            X = sm.add_constant(group[['curDurc', 'preDur1backc']])
            y = group['curBias']
            model = sm.OLS(y, X).fit()
            slope = model.params['preDur1backc']
            sdi_list_1.append({
                'subID': sub_id,
                'TransitionType': trans_type,
                'SDI': slope
            })

    df_sdi_1 = pd.DataFrame(sdi_list_1)
    # Exp 1 (Dynamic Coherence) -- group SDI by coherence level
    df_sdi_1['curCoherenceLevel'] = df_sdi_1['TransitionType'].map(
        {'HH': 'Low', 'HL': 'High', 'LH': 'Low', 'LL': 'High'}
    )
    df_sdi_1['SameSwitch'] = df_sdi_1['TransitionType'].map(
        {'HH': 'Same', 'LL': 'Same', 'HL': 'Switch', 'LH': 'Switch'}
    )

    # Experiment 2 (Fixed Coherence)
    sdi_list_2 = []
    for (sub_id, trans_type), group in df2.groupby(['subID', 'TransitionType']):
        if len(group) >= 5:
            X = sm.add_constant(group[['curDurc', 'preDur1backc']])
            y = group['curBias']
            model = sm.OLS(y, X).fit()
            slope = model.params['preDur1backc']
            sdi_list_2.append({
                'subID': sub_id,
                'TransitionType': trans_type,
                'SDI': slope
            })

    df_sdi_2 = pd.DataFrame(sdi_list_2)
    df_sdi_2['SameSwitch'] = df_sdi_2['TransitionType'].map(
        {'HH': 'Same', 'LL': 'Same', 'HL': 'Switch', 'LH': 'Switch'}
    )

    print(f"Exp 1 SDI: {len(df_sdi_1)} observations")
    print(f"Exp 2 SDI: {len(df_sdi_2)} observations")

    return df_sdi_1, df_sdi_2


# ============================================================
# CTI computation  (used by supplementary figures in notebook 1)
# ============================================================

def compute_cti_dataframes(df1, df2):
    """Compute Central Tendency Index for both experiments (absolute values)."""

    cti_list_1 = []
    for (sub_id, trans_type), group in df1.groupby(['subID', 'TransitionType']):
        if len(group) >= 5:
            X = sm.add_constant(group['curDur'])
            y = group['curBias']
            model = sm.OLS(y, X).fit()
            cti_list_1.append({
                'subID': sub_id,
                'TransitionType': trans_type,
                'CTI': np.abs(model.params['curDur']),
                'intercept': model.params['const']
            })

    df_cti_1 = pd.DataFrame(cti_list_1)
    df_cti_1['priorCoh'] = df_cti_1['TransitionType'].str[0].map({'H': 'Low', 'L': 'High'})
    df_cti_1['curCoh'] = df_cti_1['TransitionType'].str[1].map({'H': 'Low', 'L': 'High'})

    cti_list_2 = []
    for (sub_id, trans_type), group in df2.groupby(['subID', 'TransitionType']):
        if len(group) >= 5:
            X = sm.add_constant(group['curDur'])
            y = group['curBias']
            model = sm.OLS(y, X).fit()
            cti_list_2.append({
                'subID': sub_id,
                'TransitionType': trans_type,
                'CTI': np.abs(model.params['curDur']),
                'intercept': model.params['const']
            })

    df_cti_2 = pd.DataFrame(cti_list_2)
    df_cti_2['priorCoh'] = df_cti_2['TransitionType'].str[0].map({'H': 'Low', 'L': 'High'})
    df_cti_2['curCoh'] = df_cti_2['TransitionType'].str[1].map({'H': 'Low', 'L': 'High'})

    print("CTI computed for supplementary figures (absolute values)")
    return df_cti_1, df_cti_2


# ============================================================
# FIGURE 2: Central Tendency
# ============================================================

def plot_fig2_central_tendency(df1, df2):
    """Figure 2: Central Tendency -- Combined Exp 1 & 2."""
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.5))

    # Panel A: Experiment 1 (Dynamic Coherence)
    ax = axes[0]
    mBiasDurPar1 = df1.groupby(['subID', 'TransitionType', 'curDur'])['curBias'].mean().reset_index()
    mBiasDurAll1 = mBiasDurPar1.groupby(['TransitionType', 'curDur'])['curBias'].agg(['mean', 'sem']).reset_index()
    mBiasDurAll1.columns = ['TransitionType', 'curDur', 'mBias', 'seBias']

    for trans in TRANSITION_ORDER:
        sub_df = mBiasDurAll1[mBiasDurAll1['TransitionType'] == trans]
        ax.errorbar(sub_df['curDur'] + TRANSITION_X_JITTER[trans], sub_df['mBias'], yerr=sub_df['seBias'],
                    label=COHERENCE_TRANSITION_LABELS[trans], marker='o', linestyle='-', capsize=3,
                    color=COLORS[trans], markersize=6, lw=1.5)

    ax.axhline(0, color='gray', ls='--', lw=1, alpha=0.5)
    ax.set_xticks(sorted(df1['curDur'].unique()))
    ax.set_xlabel('Current Duration (s)')
    ax.set_ylabel('Bias (s)')
    ax.set_title('A: Experiment 1', loc='left', fontweight='bold', fontsize=14, pad=8)
    ax.legend(title='Coherence transition', frameon=False, fontsize=12)
    despine(ax)

    # Panel B: Experiment 2 (Fixed Coherence)
    ax = axes[1]
    mBiasDurPar2 = df2.groupby(['subID', 'TransitionType', 'curDur'])['curBias'].mean().reset_index()
    mBiasDurAll2 = mBiasDurPar2.groupby(['TransitionType', 'curDur'])['curBias'].agg(['mean', 'sem']).reset_index()
    mBiasDurAll2.columns = ['TransitionType', 'curDur', 'mBias', 'seBias']

    for trans in TRANSITION_ORDER:
        sub_df = mBiasDurAll2[mBiasDurAll2['TransitionType'] == trans]
        ax.errorbar(sub_df['curDur'] + TRANSITION_X_JITTER[trans], sub_df['mBias'], yerr=sub_df['seBias'],
                    label=COHERENCE_TRANSITION_LABELS[trans], marker='o', linestyle='-', capsize=3,
                    color=COLORS[trans], markersize=6, lw=1.5)

    ax.axhline(0, color='gray', ls='--', lw=1, alpha=0.5)
    ax.set_xticks(sorted(df2['curDur'].unique()))
    ax.set_xlabel('Current Duration (s)')
    ax.set_ylabel('Bias (s)')
    ax.set_title('B: Experiment 2', loc='left', fontweight='bold', fontsize=14, pad=8)
    ax.legend(title='Coherence transition', frameon=False, fontsize=12)
    despine(ax)

    # Shared y-limits across both panels for comparability
    for a in axes:
        a.set_ylim(-0.25, 0.30)

    plt.tight_layout()
    save_figure(fig, 'fig2_central_tendency', aliases=('fig2_central_tendency_biases',))
    plt.close()
    print('Saved fig2_central_tendency')


# ============================================================
# FIGURE 3: Serial Dependence (4 subplots)
# ============================================================

def plot_fig3_serial_dependence(df1, df2, df_sdi_1, df_sdi_2):
    """Figure 3: Serial Dependence -- 4 subplots.

    Panels A/B: line plots of Bias ~ previous duration by transition type.
    Panel C: Exp 1 (Dynamic Coherence) SDI boxplots by curCoherenceLevel.
    Panel D: Exp 2 (Fixed Coherence) SDI boxplots by SameSwitch.
    """
    fig, axes = plt.subplots(2, 2, figsize=(10, 8))

    # Panel A: Exp 1 SD line plot
    ax = axes[0, 0]
    mBias_byPre1 = df1.groupby(['subID', 'TransitionType', 'preDur1back'])['curBias'].mean().reset_index()
    mBias_byPre_all1 = mBias_byPre1.groupby(['TransitionType', 'preDur1back']).agg(
        mBias=('curBias', 'mean'),
        seBias=('curBias', 'sem')
    ).reset_index()

    for trans in TRANSITION_ORDER:
        subdf = mBias_byPre_all1[mBias_byPre_all1['TransitionType'] == trans]
        ax.errorbar(subdf['preDur1back'] + TRANSITION_X_JITTER[trans], subdf['mBias'], yerr=subdf['seBias'],
                    label=COHERENCE_TRANSITION_LABELS[trans], marker='o', linestyle='-', capsize=3,
                    color=COLORS[trans], markersize=6, lw=1.5)

    ax.axhline(0, color='gray', linestyle='--', alpha=0.5)
    ax.set_xlabel('Previous Duration (s)')
    ax.set_ylabel('Bias (s)')
    ax.set_title('A: Experiment 1', loc='left', fontweight='bold', fontsize=14, pad=8)
    ax.legend(title='Coherence transition', frameon=False, fontsize=12)
    ax.set_ylim(-0.15, 0.2)
    despine(ax)

    # Panel B: Exp 2 SD line plot
    ax = axes[0, 1]
    mBias_byPre2 = df2.groupby(['subID', 'TransitionType', 'preDur1back'])['curBias'].mean().reset_index()
    mBias_byPre_all2 = mBias_byPre2.groupby(['TransitionType', 'preDur1back']).agg(
        mBias=('curBias', 'mean'),
        seBias=('curBias', 'sem')
    ).reset_index()

    for trans in TRANSITION_ORDER:
        subdf = mBias_byPre_all2[mBias_byPre_all2['TransitionType'] == trans]
        ax.errorbar(subdf['preDur1back'] + TRANSITION_X_JITTER[trans], subdf['mBias'], yerr=subdf['seBias'],
                    label=COHERENCE_TRANSITION_LABELS[trans], marker='o', linestyle='-', capsize=3,
                    color=COLORS[trans], markersize=6, lw=1.5)

    ax.axhline(0, color='gray', linestyle='--', alpha=0.5)
    ax.set_xlabel('Previous Duration (s)')
    ax.set_ylabel('Bias (s)')
    ax.set_title('B: Experiment 2', loc='left', fontweight='bold', fontsize=14, pad=8)
    ax.legend(title='Coherence transition', frameon=False, fontsize=12)
    ax.set_ylim(-0.15, 0.2)
    despine(ax)

    # Panel C: Exp 1 (Dynamic Coherence) SDI boxplots -- by Coherence
    ax = axes[1, 0]
    order_coh = ['Low', 'High']
    palette_coh = {'High': COLORS['high'], 'Low': COLORS['low']}

    sns.boxplot(data=df_sdi_1, x='curCoherenceLevel', y='SDI', order=order_coh,
                palette=palette_coh, ax=ax, width=0.5, showfliers=False)
    ax.set_xticklabels(['Low coherence', 'High coherence'])
    ax.axhline(0, color='gray', linestyle='--', alpha=0.5)
    ax.set_xlabel('Current coherence')
    ax.set_ylabel('Serial Dependence')
    ax.set_title('C: Experiment 1 — Current coherence', loc='left', fontweight='bold', fontsize=14, pad=8)
    despine(ax)

    # Add significance test
    low_coh_sdi = df_sdi_1[df_sdi_1['curCoherenceLevel'] == 'Low']['SDI']
    high_coh_sdi = df_sdi_1[df_sdi_1['curCoherenceLevel'] == 'High']['SDI']
    t_stat, p_val = ttest_ind(low_coh_sdi, high_coh_sdi)
    if p_val < 0.05:
        # Annotate inside shared y range so the bracket is visible after ylim is set
        add_significance_stars(ax, 0, 1, 0.37, p_val)

    # Panel D: Exp 2 (Fixed Coherence) SDI boxplots -- by Same vs. Switch
    ax = axes[1, 1]
    order_ss = ['Same', 'Switch']
    palette_ss = {'Same': COLORS['same'], 'Switch': COLORS['switch']}

    sns.boxplot(data=df_sdi_2, x='SameSwitch', y='SDI', order=order_ss,
                palette=palette_ss, ax=ax, width=0.5, showfliers=False)
    ax.axhline(0, color='gray', linestyle='--', alpha=0.5)
    ax.set_xlabel('Transition')
    ax.set_ylabel('Serial Dependence')
    ax.set_title('D: Experiment 2 - Same vs. Switch', loc='left', fontweight='bold', fontsize=14, pad=8)
    despine(ax)

    # Add significance test
    same_sdi = df_sdi_2[df_sdi_2['SameSwitch'] == 'Same']['SDI']
    switch_sdi = df_sdi_2[df_sdi_2['SameSwitch'] == 'Switch']['SDI']
    t_stat, p_val = ttest_ind(same_sdi, switch_sdi)
    if p_val < 0.05:
        add_significance_stars(ax, 0, 1, 0.37, p_val)
    else:
        # Plan: dual-interaction LMM p = .207 -- annotate n.s. for transparency
        ax.plot([0, 0, 1, 1], [0.37, 0.39, 0.39, 0.37], lw=0.8, color='black')
        ax.text(0.5, 0.39, 'n.s.', ha='center', va='bottom', fontsize=11)

    # Shared y-limits across panels C and D
    for a in (axes[1, 0], axes[1, 1]):
        a.set_ylim(-0.25, 0.42)

    plt.tight_layout()
    save_figure(fig, 'fig3_serial_dependence', aliases=('fig3_serial_dependence_effects',))
    plt.close()
    print(f"Saved fig3_serial_dependence")


# ============================================================
# FIGURE 4: Sanity Check  (n-3 to n+2 analysis)
# ============================================================

def plot_fig4_sanity_check(df1, df2):
    """Figure 4: Sanity Check -- n-3 to n+2 analysis.

    Panel A: Experiment 1 (Dynamic Coherence) grouped by curCoherenceLevel.
    Panel B: Experiment 2 (Fixed Coherence) grouped by lag-specific Same/Switch
             status: current trial vs the lagged/future trial shown on the x-axis.
    """

    def prepare_multilag_dataframe(df_exp):
        """Create lag/future duration and coherence columns for Figure 4."""
        df_sorted = df_exp.sort_values(['subID', 'trial_num']).copy()

        for lag in range(1, 4):
            df_sorted[f'preDur{lag}back'] = df_sorted.groupby('subID')['curDur'].shift(lag)
            df_sorted[f'preCoherence{lag}back'] = df_sorted.groupby('subID')['curCoherence'].shift(lag)

        for fut in range(1, 3):
            df_sorted[f'postDur{fut}'] = df_sorted.groupby('subID')['curDur'].shift(-fut)
            df_sorted[f'postCoherence{fut}'] = df_sorted.groupby('subID')['curCoherence'].shift(-fut)

        return df_sorted

    # Variables in past-on-left / future-on-right order: n-3, n-2, n-1, n+1, n+2
    lag_vars = ['preDur3back', 'preDur2back', 'preDur1back', 'postDur1', 'postDur2']
    lag_labels = ['n-3', 'n-2', 'n-1', 'n+1', 'n+2']

    def summarize_lag_slopes(df_sorted, group_cols_by_lag, group_values):
        """Compute participant-level lag slopes, then mean/SEM/significance by lag-specific group."""
        results = {group: {'mean': [], 'sem': [], 'stars': []} for group in group_values}
        n_tests = len(lag_vars)

        for var, group_col in zip(lag_vars, group_cols_by_lag):
            for group in group_values:
                group_data = df_sorted[df_sorted[group_col] == group]
                slopes = []
                for sub_id in group_data['subID'].unique():
                    sub_df = group_data[group_data['subID'] == sub_id].dropna(subset=[var, 'curBias'])
                    if len(sub_df) >= 5:
                        slope = np.polyfit(sub_df[var], sub_df['curBias'], 1)[0]
                        slopes.append(slope)

                mean_slope = np.mean(slopes) if slopes else 0
                sem_slope = sem(slopes) if len(slopes) > 1 else 0

                if len(slopes) >= 5:
                    t_stat, p_val = ttest_1samp(slopes, 0)
                    p_bonf = min(p_val * n_tests, 1.0)
                    star = '***' if p_bonf < 0.001 else '**' if p_bonf < 0.01 else '*' if p_bonf < 0.05 else ''
                else:
                    star = ''

                results[group]['mean'].append(mean_slope)
                results[group]['sem'].append(sem_slope)
                results[group]['stars'].append(star)

        return results, lag_labels

    # Panel A: Exp 1 (Dynamic Coherence) -- grouped by current coherence for every lag
    df1_lag = prepare_multilag_dataframe(df1)
    results_exp1, lag_labels = summarize_lag_slopes(
        df1_lag, ['curCoherenceLevel'] * len(lag_vars), ['Low', 'High']
    )

    # Panel B: Exp 2 (Fixed Coherence) -- Same/Switch recomputed for each lag.
    # For n-i bars: compare current coherence with coherence on trial n-i.
    # For n+i sanity bars: compare current coherence with coherence on trial n+i.
    df2_lag = prepare_multilag_dataframe(df2)
    lag_coherence_cols = ['preCoherence3back', 'preCoherence2back', 'preCoherence1back', 'postCoherence1', 'postCoherence2']
    same_switch_cols = []
    for label, coh_col in zip(lag_labels, lag_coherence_cols):
        col = f'SameSwitch_{label.replace("+", "plus").replace("-", "minus")}'
        df2_lag[col] = pd.Series(np.nan, index=df2_lag.index, dtype=object)
        valid = df2_lag[coh_col].notna()
        df2_lag.loc[valid, col] = np.where(
            df2_lag.loc[valid, 'curCoherence'].eq(df2_lag.loc[valid, coh_col]),
            'Same',
            'Switch'
        )
        same_switch_cols.append(col)

    results_exp2, _ = summarize_lag_slopes(df2_lag, same_switch_cols, ['Same', 'Switch'])

    fig, axes = plt.subplots(1, 2, figsize=(12, 6))

    x = np.arange(len(lag_labels))
    bar_width = 0.35

    # Panel A: Experiment 1 - Coherence
    ax = axes[0]
    colors_exp1 = {'High': COLORS['high'], 'Low': COLORS['low']}
    labels_exp1 = {'Low': 'Current low coherence', 'High': 'Current high coherence'}

    for i, group in enumerate(['Low', 'High']):
        offset = bar_width * (i - 0.5)
        bars = ax.bar(x + offset, results_exp1[group]['mean'], bar_width,
                      yerr=results_exp1[group]['sem'], capsize=3,
                      label=labels_exp1[group], color=colors_exp1[group], alpha=0.8)

        # Add significance stars
        for xi, (mean_val, star) in enumerate(zip(results_exp1[group]['mean'], results_exp1[group]['stars'])):
            if star:
                y_pos = mean_val + results_exp1[group]['sem'][xi] + 0.005
                ax.text(xi + offset, y_pos, star, ha='center', va='bottom', fontsize=14)

    ax.axhline(0, color='black', linestyle='--', alpha=0.5)
    ax.set_xticks(x)
    ax.set_xticklabels(lag_labels, fontsize=13)
    ax.set_xlabel('Lag', fontsize=14)
    ax.set_ylabel('Serial Dependence', fontsize=14)
    ax.tick_params(axis='y', labelsize=13)
    ax.legend(frameon=False, fontsize=13, loc='upper center', ncol=2,
              bbox_to_anchor=(0.5, 1.16), columnspacing=1.2, handletextpad=0.6)
    ax.set_title('A: Experiment 1', loc='left', fontweight='bold', fontsize=16, pad=8)
    despine(ax)

    # Panel B: Experiment 2 - Same vs Switch
    ax = axes[1]
    colors_exp2 = {'Same': COLORS['same'], 'Switch': COLORS['switch']}
    labels_exp2 = {'Same': 'Current vs lag: same', 'Switch': 'Current vs lag: switch'}

    for i, group in enumerate(['Same', 'Switch']):
        offset = bar_width * (i - 0.5)
        bars = ax.bar(x + offset, results_exp2[group]['mean'], bar_width,
                      yerr=results_exp2[group]['sem'], capsize=3,
                      label=labels_exp2[group], color=colors_exp2[group], alpha=0.8)

        # Add significance stars
        for xi, (mean_val, star) in enumerate(zip(results_exp2[group]['mean'], results_exp2[group]['stars'])):
            if star:
                y_pos = mean_val + results_exp2[group]['sem'][xi] + 0.005
                ax.text(xi + offset, y_pos, star, ha='center', va='bottom', fontsize=14)

    ax.axhline(0, color='black', linestyle='--', alpha=0.5)
    ax.set_xticks(x)
    ax.set_xticklabels(lag_labels, fontsize=13)
    ax.set_xlabel('Lag', fontsize=14)
    ax.set_ylabel('Serial Dependence', fontsize=14)
    ax.tick_params(axis='y', labelsize=13)
    ax.legend(frameon=False, fontsize=13, loc='upper center', ncol=2,
              bbox_to_anchor=(0.5, 1.16), columnspacing=1.2, handletextpad=0.6)
    ax.set_title('B: Experiment 2', loc='left', fontweight='bold', fontsize=16, pad=8)
    despine(ax)

    # Shared y-limits across both panels; keep headroom above stars and below negative error bars.
    for a in axes:
        a.set_ylim(-0.05, 0.22)

    plt.tight_layout()
    save_figure(fig, 'fig4_sanity_check', aliases=('fig4_temporal_window',))
    plt.close()
    print(f"Saved fig4_sanity_check")


# ============================================================
# FIGURE 5: Cross-Experiment Comparison
# ============================================================

def plot_fig5_cross_experiment(df1, df2, df_sdi_1, df_sdi_2):
    """Figure 5: Cross-Experiment Comparison (SDI by Same/Switch and coherence).

    X-axis: Experiment (Exp 1, Exp 2). Within each experiment, two adjacent bars
    compare the condition pair (Low vs High current coherence in A; Same vs Switch in B).
    Significance brackets sit directly above the pair of bars they compare,
    so each bracket is unambiguously a within-experiment paired test.
    """

    # Ensure grouping columns
    df_sdi_1['SameSwitch'] = df_sdi_1['TransitionType'].map(
        {'HH': 'Same', 'LL': 'Same', 'HL': 'Switch', 'LH': 'Switch'}
    )
    df_sdi_2['SameSwitch'] = df_sdi_2['TransitionType'].map(
        {'HH': 'Same', 'LL': 'Same', 'HL': 'Switch', 'LH': 'Switch'}
    )
    df_sdi_1['curCoh'] = df_sdi_1['TransitionType'].str[1].map({'H': 'Low', 'L': 'High'})
    df_sdi_2['curCoh'] = df_sdi_2['TransitionType'].str[1].map({'H': 'Low', 'L': 'High'})

    def add_sig_bracket(ax, x1, x2, y, p_val, h=0.01, lw=1.2):
        sig = '***' if p_val < 0.001 else '**' if p_val < 0.01 else '*' if p_val < 0.05 else 'n.s.'
        ax.plot([x1, x1, x2, x2], [y, y + h, y + h, y], lw=lw, c='black')
        ax.text((x1 + x2) / 2, y + h, sig, ha='center', va='bottom', fontsize=14)

    fig, axes = plt.subplots(1, 2, figsize=(11, 6))
    width = 0.35
    x_exp = np.arange(2)  # 0 = Exp 1, 1 = Exp 2

    # Within-experiment paired t-tests
    def paired_t(df, col, level_lo, level_hi):
        a = df[df[col] == level_lo].set_index('subID')['SDI']
        b = df[df[col] == level_hi].set_index('subID')['SDI']
        common = a.index.intersection(b.index)
        return ttest_rel(a.loc[common], b.loc[common])

    # ---- Panel A: Low vs High current coherence within each experiment ----
    ax = axes[0]
    sdi_coh_1 = df_sdi_1.groupby(['subID', 'curCoh'])['SDI'].mean().reset_index()
    sdi_coh_2 = df_sdi_2.groupby(['subID', 'curCoh'])['SDI'].mean().reset_index()
    conds_a = ['Low', 'High']
    colors_a = {'Low': COLORS['low'], 'High': COLORS['high']}

    means_a = {}
    sems_a = {}
    for j, exp_df in enumerate([sdi_coh_1, sdi_coh_2]):
        means_a[j] = {}
        sems_a[j] = {}
        for i, coh in enumerate(conds_a):
            d = exp_df[exp_df['curCoh'] == coh]['SDI']
            m, s = d.mean(), sem(d)
            means_a[j][coh] = m
            sems_a[j][coh] = s
            ax.bar(j + width * (i - 0.5), m, width, yerr=s, capsize=4,
                   color=colors_a[coh], alpha=0.85,
                   label=f'{coh} coherence' if j == 0 else None)

    ax.axhline(0, color='gray', linestyle='--', alpha=0.5)
    ax.set_xticks(x_exp)
    ax.set_xticklabels(['Exp 1', 'Exp 2'], fontsize=13)
    ax.set_ylabel('Serial Dependence', fontsize=14)
    ax.tick_params(axis='y', labelsize=13)
    ax.legend(title='Collapsed', frameon=False, fontsize=13,
              title_fontsize=12, loc='upper right')
    ax.text(0.02, 0.98, 'A', transform=ax.transAxes, ha='left', va='top',
            fontsize=16, fontweight='bold')
    despine(ax)

    t_a1, p_a1 = paired_t(sdi_coh_1, 'curCoh', 'Low', 'High')
    t_a2, p_a2 = paired_t(sdi_coh_2, 'curCoh', 'Low', 'High')

    for j, p_val in enumerate([p_a1, p_a2]):
        if p_val < 0.05:
            top = max(
                means_a[j]['Low'] + sems_a[j]['Low'],
                means_a[j]['High'] + sems_a[j]['High']
            ) + 0.015
            add_sig_bracket(ax, j - 0.175, j + 0.175, top, p_val)

    print(f"Panel A: Exp1 Low vs High t={t_a1:.3f}, p={p_a1:.4f}; "
          f"Exp2 t={t_a2:.3f}, p={p_a2:.4f}")

    # ---- Panel B: Same vs Switch within each experiment ----
    ax = axes[1]
    sdi_ss_1 = df_sdi_1.groupby(['subID', 'SameSwitch'])['SDI'].mean().reset_index()
    sdi_ss_2 = df_sdi_2.groupby(['subID', 'SameSwitch'])['SDI'].mean().reset_index()
    conds_b = ['Same', 'Switch']
    colors_b = {'Same': COLORS['same'], 'Switch': COLORS['switch']}

    means_b = {}
    sems_b = {}
    for j, exp_df in enumerate([sdi_ss_1, sdi_ss_2]):
        means_b[j] = {}
        sems_b[j] = {}
        for i, cond in enumerate(conds_b):
            d = exp_df[exp_df['SameSwitch'] == cond]['SDI']
            m, s = d.mean(), sem(d)
            means_b[j][cond] = m
            sems_b[j][cond] = s
            ax.bar(j + width * (i - 0.5), m, width, yerr=s, capsize=4,
                   color=colors_b[cond], alpha=0.85,
                   label=cond if j == 0 else None)

    ax.axhline(0, color='gray', linestyle='--', alpha=0.5)
    ax.set_xticks(x_exp)
    ax.set_xticklabels(['Exp 1', 'Exp 2'], fontsize=13)
    ax.set_ylabel('Serial Dependence', fontsize=14)
    ax.tick_params(axis='y', labelsize=13)
    ax.legend(title='Collapsed', frameon=False, fontsize=13,
              title_fontsize=12, loc='upper right')
    ax.text(0.02, 0.98, 'B', transform=ax.transAxes, ha='left', va='top',
            fontsize=16, fontweight='bold')
    despine(ax)

    t_b1, p_b1 = paired_t(sdi_ss_1, 'SameSwitch', 'Same', 'Switch')
    t_b2, p_b2 = paired_t(sdi_ss_2, 'SameSwitch', 'Same', 'Switch')

    for j, p_val in enumerate([p_b1, p_b2]):
        if p_val < 0.05:
            top = max(
                means_b[j]['Same'] + sems_b[j]['Same'],
                means_b[j]['Switch'] + sems_b[j]['Switch']
            ) + 0.015
            add_sig_bracket(ax, j - 0.175, j + 0.175, top, p_val)

    print(f"Panel B: Exp1 Same vs Switch t={t_b1:.3f}, p={p_b1:.4f}; "
          f"Exp2 t={t_b2:.3f}, p={p_b2:.4f}")

    # Shared y-limits across A and B
    for a in axes:
        a.set_ylim(0, 0.20)

    plt.tight_layout()
    save_figure(fig, 'fig5_cross_experiment', aliases=('fig5_cross_experiment_serial_dependence',))
    plt.close()
    print(f"\nSaved fig5_cross_experiment")


# ============================================================
# FIGURE C1 (was fig7): Model Comparison
# ============================================================

def plot_figC1_model_comparison(results_df, C_AXIS, S_AXIS, B_AXIS, compare_axes):
    """Figure C1: Model Comparison -- Axis Comparison."""
    fig, axes = plt.subplots(2, 3, figsize=(16, 10))

    for row, exp_num in enumerate([1, 2]):
        comparison = compare_axes(results_df, exp_num=exp_num)

        # C-axis (Coherence Modulation)
        ax = axes[row, 0]
        c_data = comparison['by_c'].sort_values('AIC_mean')
        colors = ['#2ca02c' if 'Q1' in C_AXIS[c]['name'] else '#1f77b4' for c in c_data.index]
        ax.barh(range(len(c_data)), c_data['AIC_mean'], color=colors, alpha=0.7)
        ax.set_yticks(range(len(c_data)))
        ax.set_yticklabels([f"{c} ({C_AXIS[c]['name']})" for c in c_data.index], fontsize=12)
        ax.set_xlabel('Mean AIC')
        ax.set_title(f'Exp {exp_num}: C-Axis (Coherence)', fontweight='bold')
        ax.invert_yaxis()
        despine(ax)

        # S-axis (Switch Effect)
        ax = axes[row, 1]
        s_data = comparison['by_s'].sort_values('AIC_mean')
        ax.barh(range(len(s_data)), s_data['AIC_mean'], color='#ff7f0e', alpha=0.7)
        ax.set_yticks(range(len(s_data)))
        ax.set_yticklabels([f"{s} ({S_AXIS[s]['name']})" for s in s_data.index], fontsize=12)
        ax.set_xlabel('Mean AIC')
        ax.set_title(f'Exp {exp_num}: S-Axis (Switch)', fontweight='bold')
        ax.invert_yaxis()
        despine(ax)

        # B-axis (Bias Mechanism)
        ax = axes[row, 2]
        b_data = comparison['by_b'].sort_values('AIC_mean')
        ax.barh(range(len(b_data)), b_data['AIC_mean'], color='#9467bd', alpha=0.7)
        ax.set_yticks(range(len(b_data)))
        ax.set_yticklabels([f"{b} ({B_AXIS[b]['name']})" for b in b_data.index], fontsize=12)
        ax.set_xlabel('Mean AIC')
        ax.set_title(f'Exp {exp_num}: B-Axis (Bias)', fontweight='bold')
        ax.invert_yaxis()
        despine(ax)

    plt.tight_layout()
    plt.savefig(FIG_DIR / 'figC1_model_comparison.png', dpi=300, bbox_inches='tight')
    plt.savefig(FIG_DIR / 'figC1_model_comparison.pdf', bbox_inches='tight')
    plt.close()
    print(f"Saved figC1_model_comparison")


# ============================================================
# FIGURE 7: Best Model Parameters (grouped by parameter)
# ============================================================

# Map long model names to short appendix notation
_MODEL_SHORT_NAMES = {
    'C_Q1__S_baseline__B_B2': 'C1_S0_B2',
    'C_Q1__S_x_reset__B_B2': 'C1_S1_B2',
    'C_Q1__S_K_reset__B_B2': 'C1_S2_B2',
    'C_Q1__S_baseline__B_B1': 'C1_S0_B1',
    'C_Q1__S_baseline__B_B3': 'C1_S0_B3',
    'C_Q1__S_x_reset__B_B1': 'C1_S1_B1',
    'C_Q1__S_x_reset__B_B3': 'C1_S1_B3',
}

def _short_model_name(long_name):
    """Convert long model name to short appendix notation."""
    if long_name in _MODEL_SHORT_NAMES:
        return _MODEL_SHORT_NAMES[long_name]
    # Fallback: extract components from C_{name}__S_{name}__B_{name}
    parts = long_name.split('__')
    c_part = parts[0].replace('C_', '')
    s_part = parts[1].replace('S_', '')
    b_part = parts[2].replace('B_', '')
    s_map = {'baseline': 'S0', 'x_reset': 'S1', 'K_reset': 'S2'}
    c_map = {'Q1': 'C1', 'Q2': 'C2', 'Q3': 'C3', 'R': 'C4',
             'Q1_R': 'C5', 'Q2_R': 'C6', 'Q3_R': 'C7',
             'Q1_Q2': 'C8', 'Q1_Q3': 'C9', 'Q2_Q3': 'C10'}
    c_id = c_map.get(c_part, c_part)
    s_id = s_map.get(s_part, s_part)
    return f"{c_id}_{s_id}_{b_part}"


def plot_fig7_parameters(results_df, get_best_model):
    """Figure 7: Key parameters grouped by parameter, both experiments side by side."""

    # Key parameters to plot (user-specified selection)
    key_params = [
        ('q1',       '$q_1$\n(Fast\nstate)'),
        ('q2',       '$q_2$\n(Slow\nstate)'),
        ('q3',       '$q_3$\n(Bias\nstate)'),
        ('alpha_q1', r'$\alpha_{q_1}$' + '\n(Coherence\nmodulation)'),
        ('lambda',   '$\\lambda$\n(Bias\ndecay)'),
    ]

    # Get best model data for both experiments
    best_name_1, best_data_1 = get_best_model(results_df, exp_num=1)
    best_name_2, best_data_2 = get_best_model(results_df, exp_num=2)
    short_1 = _short_model_name(best_name_1)
    short_2 = _short_model_name(best_name_2)

    n_params = len(key_params)
    x = np.arange(n_params)
    bar_width = 0.35

    fig, ax = plt.subplots(figsize=(10, 5))

    # Collect means and SEMs for both experiments
    means_1, sems_1 = [], []
    means_2, sems_2 = [], []

    # Determine which s_id each best model uses (for filtering switch params)
    s_id_1 = best_data_1['s_id'].iloc[0] if 's_id' in best_data_1.columns else 'S0'
    s_id_2 = best_data_2['s_id'].iloc[0] if 's_id' in best_data_2.columns else 'S0'

    def _is_active(param, s_id):
        """Check if a parameter is a free parameter in the model."""
        if param == 'x_reset':
            return s_id == 'S1'
        if param == 'k_reset':
            return s_id == 'S2'
        return True

    for param, _ in key_params:
        # Experiment 1
        if _is_active(param, s_id_1) and param in best_data_1.columns:
            vals = best_data_1[param].dropna()
            means_1.append(vals.mean() if len(vals) > 0 else np.nan)
            sems_1.append(vals.std() / np.sqrt(len(vals)) if len(vals) > 1 else 0)
        else:
            means_1.append(np.nan)
            sems_1.append(0)
        # Experiment 2
        if _is_active(param, s_id_2) and param in best_data_2.columns:
            vals = best_data_2[param].dropna()
            means_2.append(vals.mean() if len(vals) > 0 else np.nan)
            sems_2.append(vals.std() / np.sqrt(len(vals)) if len(vals) > 1 else 0)
        else:
            means_2.append(np.nan)
            sems_2.append(0)

    means_1 = np.array(means_1, dtype=float)
    means_2 = np.array(means_2, dtype=float)
    sems_1 = np.array(sems_1, dtype=float)
    sems_2 = np.array(sems_2, dtype=float)

    # Plot bars side by side
    mask_1 = ~np.isnan(means_1)
    mask_2 = ~np.isnan(means_2)

    bars1 = ax.bar(x[mask_1] - bar_width/2, means_1[mask_1], bar_width,
                   yerr=sems_1[mask_1], capsize=3,
                   color=COLORS['exp1'], alpha=0.85, edgecolor='black', lw=0.6,
                   label=f'Exp 1 ({short_1})')
    bars2 = ax.bar(x[mask_2] + bar_width/2, means_2[mask_2], bar_width,
                   yerr=sems_2[mask_2], capsize=3,
                   color=COLORS['exp2'], alpha=0.85, edgecolor='black', lw=0.6,
                   label=f'Exp 2 ({short_2})')

    # Add value annotations
    for i in range(n_params):
        if mask_1[i]:
            m, s = means_1[i], sems_1[i]
            y_pos = m + s + 0.08 if m >= 0 else m - s - 0.08
            va = 'bottom' if m >= 0 else 'top'
            ax.text(i - bar_width/2, y_pos, f'{m:.2f}',
                    ha='center', va=va, fontsize=12, color=COLORS['exp1'])
        if mask_2[i]:
            m, s = means_2[i], sems_2[i]
            y_pos = m + s + 0.08 if m >= 0 else m - s - 0.08
            va = 'bottom' if m >= 0 else 'top'
            ax.text(i + bar_width/2, y_pos, f'{m:.2f}',
                    ha='center', va=va, fontsize=12, color=COLORS['exp2'])

    # Mark slots where a parameter is not estimated for one experiment
    for i, (param, _) in enumerate(key_params):
        if not mask_1[i] and mask_2[i]:
            ax.text(i - bar_width/2, 0, 'not\nestimated', ha='center', va='center',
                    fontsize=10, color='gray', style='italic')
        if not mask_2[i] and mask_1[i]:
            ax.text(i + bar_width/2, 0, 'not\nestimated', ha='center', va='center',
                    fontsize=10, color='gray', style='italic')

    ax.set_xticks(x)
    ax.set_xticklabels([label for _, label in key_params], fontsize=12, rotation=0)
    ax.set_ylabel('Parameter Value', fontsize=12)
    ax.axhline(0, color='gray', linestyle='--', alpha=0.3, lw=0.8)
    ax.legend(fontsize=12, loc='upper right', framealpha=0.9)
    despine(ax)

    plt.tight_layout()
    save_figure(fig, 'fig7_parameters', aliases=('fig7_parameter_estimates',))
    plt.close()

    # Print summary
    print("\n" + "=" * 60)
    print("Best Models Summary:")
    print("=" * 60)
    for exp_num, (best_name, best_data) in enumerate(
            [(best_name_1, best_data_1), (best_name_2, best_data_2)], start=1):
        short = _short_model_name(best_name)
        print(f"\nExperiment {exp_num}: {short} (internal: {best_name})")
        print(f"  N subjects: {len(best_data)}")
        print(f"  Key parameters:")
        for param, label in key_params:
            if param in best_data.columns:
                values = best_data[param].dropna()
                if len(values) > 0 and not np.isnan(values.mean()):
                    print(f"    {param}: {values.mean():.4f} ± {values.std()/np.sqrt(len(values)):.4f}")

    print(f"\nSaved fig7_parameters")


# ============================================================
# FIGURE C2 (was fig9): CTI & SDI Recovery
# ============================================================

def _compute_cti(duration, reproduction):
    """Central Tendency Index: slope of Reproduction ~ Duration.
    Returns ABSOLUTE VALUE.
    """
    valid = ~(np.isnan(duration) | np.isnan(reproduction))
    if valid.sum() < 20:
        return np.nan
    slope = np.polyfit(duration[valid], reproduction[valid], 1)[0]
    return np.abs(slope - 1)


def _compute_sdi(prev_stim, bias):
    """Serial Dependence Index: slope of Bias ~ prev_stim."""
    valid = ~(np.isnan(prev_stim) | np.isnan(bias))
    if valid.sum() < 20:
        return np.nan
    return np.polyfit(prev_stim[valid], bias[valid], 1)[0]


def _run_ppc_cti_sdi(data, results_df, exp_num, model_name,
                     generate_ppc_single_subject, verbose=False):
    """Run PPC and compute CTI/SDI recovery for all subjects."""
    exp_data = data[data['exp'] == exp_num]
    model_results = results_df[(results_df['exp'] == exp_num) &
                                (results_df['model_name'] == model_name)]

    obs_metrics, sim_metrics = [], []

    for sub in exp_data['Sub'].unique():
        sub_data = exp_data[exp_data['Sub'] == sub].copy()
        params_row = model_results[model_results['Sub'] == sub]

        if len(params_row) == 0 or not params_row.iloc[0]['success']:
            continue

        try:
            ppc = generate_ppc_single_subject(sub_data, results_df, model_name=model_name)
            sim_resp = ppc['pred']
        except Exception as e:
            if verbose:
                print(f'  Warning: PPC failed for Sub {sub}: {e}')
            continue

        prev_dur = sub_data['Duration'].shift(1).values

        obs_cti = _compute_cti(sub_data['Duration'].values, sub_data['Reproduction'].values)
        obs_sdi = _compute_sdi(prev_dur, sub_data['Bias'].values)
        obs_metrics.append({'Sub': sub, 'CTI': obs_cti, 'SDI': obs_sdi})

        sim_cti = _compute_cti(sub_data['Duration'].values, sim_resp)
        sim_bias = sim_resp - sub_data['Duration'].values
        sim_sdi = _compute_sdi(prev_dur, sim_bias)
        sim_metrics.append({'Sub': sub, 'CTI': sim_cti, 'SDI': sim_sdi})

    obs_df = pd.DataFrame(obs_metrics)
    sim_df = pd.DataFrame(sim_metrics)
    merged = obs_df.merge(sim_df, on='Sub', suffixes=('_obs', '_sim'))

    return merged


def plot_figC2_cti_sdi_recovery(df_model, results_df, best_exp1, best_exp2,
                                generate_ppc_single_subject):
    """Figure C2: CTI & SDI Recovery scatter plots.

    Returns ppc_results dict (needed by later figures).
    """
    print('Running PPC for CTI & SDI Recovery...')

    ppc_results = {}

    for exp_num, model_name in [(1, best_exp1), (2, best_exp2)]:
        print(f'\n  Exp {exp_num}: {model_name}')
        merged = _run_ppc_cti_sdi(df_model, results_df, exp_num, model_name,
                                  generate_ppc_single_subject)

        if len(merged) > 0:
            cti_r, _ = pearsonr(merged['CTI_obs'].dropna(), merged['CTI_sim'].dropna())
            sdi_r, _ = pearsonr(merged['SDI_obs'].dropna(), merged['SDI_sim'].dropna())

            cti_rec = merged['CTI_sim'].mean() / merged['CTI_obs'].mean() * 100
            sdi_rec = merged['SDI_sim'].mean() / merged['SDI_obs'].mean() * 100

            ppc_results[exp_num] = {
                'merged': merged,
                'CTI_r': cti_r, 'SDI_r': sdi_r,
                'CTI_rec': cti_rec, 'SDI_rec': sdi_rec,
                'model': model_name
            }

            print(f'    CTI: r = {cti_r:.3f}, Recovery = {cti_rec:.0f}%')
            print(f'    SDI: r = {sdi_r:.3f}, Recovery = {sdi_rec:.0f}%')
        else:
            print(f'    Warning: No valid PPC data')

    # Figure C2: CTI & SDI Recovery Scatter Plots
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))

    for exp_idx, exp in enumerate([1, 2]):
        if exp not in ppc_results:
            continue
        res = ppc_results[exp]
        merged = res['merged']

        for metric_idx, (metric, color) in enumerate([('CTI', '#E65100'), ('SDI', '#1565C0')]):
            ax = axes[exp_idx, metric_idx]
            valid = merged.dropna(subset=[f'{metric}_obs', f'{metric}_sim'])

            x = valid[f'{metric}_obs']
            y = valid[f'{metric}_sim']

            ax.scatter(x, y, s=80, alpha=0.7, color=color, edgecolors='k', lw=0.8)

            lims = [min(x.min(), y.min()) - 0.02, max(x.max(), y.max()) + 0.02]
            ax.plot(lims, lims, 'k--', alpha=0.5, lw=2, label='Identity')

            if len(valid) > 2:
                slope, intercept = np.polyfit(x, y, 1)
                ax.plot(lims, [slope * l + intercept for l in lims],
                        color=color, lw=2.5, alpha=0.7, label=f'Fit (slope={slope:.2f})')

            ax.set_xlabel(f'Observed {metric}', fontsize=14, fontweight='bold')
            ax.set_ylabel(f'Simulated {metric}', fontsize=14, fontweight='bold')
            ax.set_title(f'Exp {exp}: {metric}\nr={res[f"{metric}_r"]:.3f}, Recovery={res[f"{metric}_rec"]:.0f}%')
            ax.legend(frameon=True, fontsize=12)
            ax.grid(alpha=0.3)
            ax.set_aspect('equal', adjustable='box')
            despine(ax)

    plt.tight_layout()
    plt.savefig(FIG_DIR / 'figC2_cti_sdi_recovery.png', dpi=300, bbox_inches='tight')
    plt.savefig(FIG_DIR / 'figC2_cti_sdi_recovery.pdf', bbox_inches='tight')
    plt.close()
    print(f"\nSaved figC2_cti_sdi_recovery")

    return ppc_results


# ============================================================
# FIGURE 8: Trial-Level Serial Dependence
# ============================================================

def plot_fig8_trial_level_sd(df_model, results_df, best_exp1, best_exp2,
                             generate_ppc_single_subject):
    """Figure 8: Trial-Level Serial Dependence by Condition.

    EXPERIMENT SWAP applied:
        exp_num == 1 (Dynamic Coherence) -> coherence grouping
        exp_num == 2 (Fixed Coherence)   -> Same/Switch grouping
    """

    def plot_sd_by_condition(data, results_df, exp_num, model_name, ax):
        exp_data = data[data['exp'] == exp_num].copy()
        model_results = results_df[(results_df['exp'] == exp_num) &
                                    (results_df['model_name'] == model_name)]

        # Generate simulated data
        sim_data_list = []
        for sub in exp_data['Sub'].unique():
            sub_data = exp_data[exp_data['Sub'] == sub].copy()
            params = model_results[model_results['Sub'] == sub]

            if len(params) == 0 or not params.iloc[0]['success']:
                continue

            try:
                ppc = generate_ppc_single_subject(sub_data, results_df, model_name=model_name)
                sub_data['sim_resp'] = ppc['pred']
                sub_data['sim_bias'] = ppc['pred'] - sub_data['Duration'].values
                sub_data['preDur'] = sub_data['Duration'].shift(1)
                sim_data_list.append(sub_data)
            except Exception:
                continue

        if not sim_data_list:
            return

        sim_data = pd.concat(sim_data_list, ignore_index=True)
        exp_data['preDur'] = exp_data.groupby('Sub')['Duration'].shift(1)

        # SWAPPED conditions:
        # exp_num == 1 -> Dynamic Coherence -> coherence grouping
        # exp_num == 2 -> Fixed Coherence   -> Same/Switch grouping
        if exp_num == 1:
            exp_data['coh_cat'] = np.where(exp_data['coherence'] < 0.5, 'Low coherence', 'High coherence')
            sim_data['coh_cat'] = np.where(sim_data['coherence'] < 0.5, 'Low coherence', 'High coherence')
            cond_col = 'coh_cat'
            conds = ['Low coherence', 'High coherence']
            colors = [COLORS['low'], COLORS['high']]
            title_suffix = 'Current coherence'
        else:
            cond_col = 'Structure'
            conds = ['Same', 'Switch']
            colors = [COLORS['same'], COLORS['switch']]
            title_suffix = 'Task Structure'

        for i, cond in enumerate(conds):
            # Observed
            obs_cond = exp_data[exp_data[cond_col] == cond].dropna(subset=['preDur', 'Bias'])
            if len(obs_cond) > 0:
                g_obs = obs_cond.groupby('preDur')['Bias'].agg(['mean', 'sem']).reset_index()
                ax.errorbar(g_obs['preDur'], g_obs['mean'], yerr=g_obs['sem'],
                            marker='o', ls='none', color=colors[i],
                            label=f'{cond} (Observed)', capsize=3, lw=1.5, ms=6)

            # Simulated
            sim_cond = sim_data[sim_data[cond_col] == cond].dropna(subset=['preDur', 'sim_bias'])
            if len(sim_cond) > 0:
                g_sim = sim_cond.groupby('preDur')['sim_bias'].agg(['mean', 'sem']).reset_index()
                g_sim = g_sim.sort_values('preDur')

                ci_lower = g_sim['mean'] - 1.96 * g_sim['sem']
                ci_upper = g_sim['mean'] + 1.96 * g_sim['sem']

                ax.fill_between(g_sim['preDur'], ci_lower, ci_upper, color=colors[i], alpha=0.2)
                ax.plot(g_sim['preDur'], g_sim['mean'], ls='--', color=colors[i], lw=2,
                        label=f'{cond} (Model)')

        ax.axhline(0, color='gray', ls=':', alpha=0.5)
        ax.set_xlabel('Previous Duration (s)')
        ax.set_ylabel('Bias (s)')
        ax.legend(frameon=True, fontsize=12, loc='upper left',
                  bbox_to_anchor=(0.01, 0.98), framealpha=0.9)
        despine(ax)

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    short_1 = _short_model_name(best_exp1)
    short_2 = _short_model_name(best_exp2)

    plot_sd_by_condition(df_model, results_df, exp_num=1, model_name=best_exp1, ax=axes[0])
    axes[0].set_title('A: Experiment 1 — Current coherence',
                      loc='left', fontweight='bold', fontsize=14, pad=8)
    axes[0].text(0.98, 0.04, f'Model: {short_1}', transform=axes[0].transAxes,
                 ha='right', va='bottom', fontsize=12, style='italic', color='dimgray')

    plot_sd_by_condition(df_model, results_df, exp_num=2, model_name=best_exp2, ax=axes[1])
    axes[1].set_title('B: Experiment 2 — Same vs. Switch',
                      loc='left', fontweight='bold', fontsize=14, pad=8)
    axes[1].text(0.98, 0.04, f'Model: {short_2}', transform=axes[1].transAxes,
                 ha='right', va='bottom', fontsize=12, style='italic', color='dimgray')

    # Shared y-limits across both panels
    for a in axes:
        a.set_ylim(-0.10, 0.20)

    plt.tight_layout()
    save_figure(fig, 'fig8_trial_level_sd', aliases=('fig8_trial_level_serial_dependence',))
    plt.close()
    print(f"Saved fig8_trial_level_sd")
