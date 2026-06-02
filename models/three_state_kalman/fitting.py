"""
Model fitting for 135-model framework - NO LOG TRANSFORM VERSION.
Optimizes in original duration space.
"""
import numpy as np
import pandas as pd
from scipy.optimize import least_squares
from pathlib import Path
import gc

try:
    from statsmodels.stats.stattools import durbin_watson
except ImportError:
    def durbin_watson(x):
        x = np.asarray(x)
        diff = np.diff(x)
        return np.sum(diff**2) / np.sum(x**2) if np.sum(x**2) > 0 else np.nan

try:
    from tqdm import tqdm
except ImportError:
    def tqdm(x, **kwargs):
        return x

from .config import get_model_by_id, get_all_models, C_AXIS, S_AXIS, B_AXIS
from .parameters import get_parameter_config, parse_parameters
from .engine import run_3state_kf, generate_predictions


def fit_single_model(stimrep, coherence, structure, c_id, s_id, b_id,
                     n_starts=5, max_nfev=2000, seed=None):
    """
    Fit a single (C, S, B) model to one subject's data.
    Optimization in ORIGINAL SPACE (no log transform).
    """
    if seed is not None:
        np.random.seed(seed)

    config = get_parameter_config(c_id, s_id, b_id)
    p0 = config['p0']
    lower = config['lower']
    upper = config['upper']
    n_params = config['n_params']
    names = config['names']

    def objective(par):
        # Always use original space residuals
        return run_3state_kf(par, stimrep, coherence, structure,
                            c_id, s_id, b_id, residual_space='orig')

    best_res = None
    best_cost = np.inf

    for start in range(n_starts):
        if start == 0:
            x0 = p0
        else:
            x0 = [np.random.uniform(l, min(h, l + 2.0))
                  for l, h in zip(lower, upper)]

        try:
            res = least_squares(
                objective, x0,
                bounds=(lower, upper),
                method='trf',
                max_nfev=max_nfev,
                ftol=1e-4, xtol=1e-4, gtol=1e-4
            )
            if res.cost < best_cost:
                best_cost = res.cost
                best_res = res
        except Exception:
            continue

    if best_res is None:
        model_info = get_model_by_id(c_id, s_id, b_id)
        return {
            'model_id': model_info['model_id'],
            'model_name': model_info['model_name'],
            'c_id': c_id, 's_id': s_id, 'b_id': b_id,
            'n_params': n_params,
            'AIC': np.nan, 'BIC': np.nan, 'RMSE': np.nan, 'DW': np.nan,
            'success': False
        }

    # Compute metrics in original space
    resid = run_3state_kf(best_res.x, stimrep, coherence, structure,
                          c_id, s_id, b_id, residual_space='orig')
    resid = resid[np.isfinite(resid)]

    rss = np.sum(resid ** 2)
    N = len(resid)
    dw = durbin_watson(resid) if len(resid) > 2 else np.nan

    # Log-likelihood and information criteria
    ll = -N / 2 * (np.log(2 * np.pi) + np.log(rss / N) + 1.0)
    aic = 2 * n_params - 2 * ll
    bic = n_params * np.log(N) - 2 * ll
    rmse = np.sqrt(rss / N)

    model_info = get_model_by_id(c_id, s_id, b_id)
    result = dict(zip(names, best_res.x))
    result.update({
        'model_id': model_info['model_id'],
        'model_name': model_info['model_name'],
        'c_id': c_id, 's_id': s_id, 'b_id': b_id,
        'n_params': n_params,
        'AIC': aic, 'BIC': bic, 'RMSE': rmse, 'DW': dw,
        'RSS': rss, 'N': N,
        'success': bool(best_res.success),
    })

    return result


def fit_all_models_subject(stimrep, coherence, structure, subject_id, exp_num,
                           models=None, verbose=False):
    """Fit all 135 models (or subset) to one subject."""
    if models is None:
        models = get_all_models(include_switching=True)

    results = []
    iterator = tqdm(models, desc=f"Sub {subject_id}", leave=False) if verbose else models

    for model in iterator:
        seed = hash((subject_id, exp_num, model['model_id'])) % (2**31)
        result = fit_single_model(
            stimrep, coherence, structure,
            model['c_id'], model['s_id'], model['b_id'],
            seed=seed
        )
        result['Sub'] = subject_id
        result['exp'] = exp_num
        results.append(result)

    return results


def fit_all_subjects(data, exp_num, models=None, verbose=True):
    """Fit models to all subjects in an experiment."""
    exp_data = data[data['exp'] == exp_num] if 'exp' in data.columns else data
    subjects = exp_data['Sub'].unique()

    all_results = []

    for sub in tqdm(subjects, desc=f"Exp{exp_num}"):
        sub_data = exp_data[exp_data['Sub'] == sub]
        stimrep = sub_data[['Duration', 'Reproduction']].values
        coherence = sub_data['coherence'].values
        structure = sub_data['Structure'].values if 'Structure' in sub_data.columns else None

        results = fit_all_models_subject(
            stimrep, coherence, structure,
            subject_id=sub, exp_num=exp_num,
            models=models, verbose=verbose
        )
        all_results.extend(results)

    return pd.DataFrame(all_results)


def fit_all_subjects_incremental(data, exp_num, output_file, models=None,
                                  verbose=True, save_every=1):
    """Fit models with incremental saving and checkpoint/resume."""
    output_file = Path(output_file)
    output_file.parent.mkdir(parents=True, exist_ok=True)

    # Check for existing results
    completed = set()
    if output_file.exists():
        try:
            df = pd.read_csv(output_file)
            completed = set(zip(df['Sub'], df['exp'], df['model_id']))
        except:
            pass

    if models is None:
        models = get_all_models(include_switching=True)

    exp_data = data[data['exp'] == exp_num] if 'exp' in data.columns else data
    subjects = sorted(exp_data['Sub'].unique())

    model_ids = {m['model_id'] for m in models}
    subjects_to_fit = []
    for sub in subjects:
        sub_completed = {mid for (s, e, mid) in completed if s == sub and e == exp_num}
        if not model_ids.issubset(sub_completed):
            subjects_to_fit.append(sub)

    if verbose:
        print(f"Exp{exp_num}: {len(subjects)} subjects, {len(subjects_to_fit)} need fitting")

    if not subjects_to_fit:
        return pd.read_csv(output_file)

    pending_results = []
    write_header = not output_file.exists()

    pbar = tqdm(subjects_to_fit, desc=f"Exp{exp_num}") if verbose else subjects_to_fit

    for i, sub in enumerate(pbar):
        sub_data = exp_data[exp_data['Sub'] == sub]
        stimrep = sub_data[['Duration', 'Reproduction']].values
        coherence = sub_data['coherence'].values
        structure = sub_data['Structure'].values if 'Structure' in sub_data.columns else None

        sub_completed = {mid for (s, e, mid) in completed if s == sub and e == exp_num}
        models_to_fit = [m for m in models if m['model_id'] not in sub_completed]

        for model in models_to_fit:
            seed = hash((sub, exp_num, model['model_id'])) % (2**31)
            result = fit_single_model(
                stimrep, coherence, structure,
                model['c_id'], model['s_id'], model['b_id'],
                seed=seed
            )
            result['Sub'] = sub
            result['exp'] = exp_num
            pending_results.append(result)

        if (i + 1) % save_every == 0 or (i + 1) == len(subjects_to_fit):
            if pending_results:
                df = pd.DataFrame(pending_results)
                df.to_csv(output_file, mode='w' if write_header else 'a',
                         header=write_header, index=False)
                write_header = False
                pending_results = []
                gc.collect()

    return pd.read_csv(output_file)


# =============================================================================
# MODEL COMPARISON
# =============================================================================

def get_best_model(results_df, exp_num=None, criterion='AIC'):
    """Get best model by AIC or BIC."""
    df = results_df.copy()
    if exp_num is not None:
        df = df[df['exp'] == exp_num]

    model_scores = df.groupby('model_name')[criterion].mean()
    best_model = model_scores.idxmin()
    best_data = df[df['model_name'] == best_model]

    return best_model, best_data


def rank_models(results_df, exp_num=None, criterion='AIC'):
    """Rank all models by AIC or BIC."""
    df = results_df.copy()
    if exp_num is not None:
        df = df[df['exp'] == exp_num]

    ranking = df.groupby(['model_name', 'c_id', 's_id', 'b_id']).agg({
        'AIC': 'mean',
        'BIC': 'mean',
        'RMSE': 'mean',
        'DW': 'mean',
        'n_params': 'first'
    }).reset_index()

    ranking = ranking.sort_values(criterion)
    ranking['Rank'] = range(1, len(ranking) + 1)
    ranking[f'delta_{criterion}'] = ranking[criterion] - ranking[criterion].min()

    return ranking


# =============================================================================
# PPC: POSTERIOR PREDICTIVE CHECKS
# =============================================================================

def generate_ppc_single_subject(sub_data, results_df, model_name=None,
                                 c_id=None, s_id=None, b_id=None):
    """Generate PPC predictions for a single subject."""
    sub = sub_data['Sub'].iloc[0]
    exp = sub_data['exp'].iloc[0] if 'exp' in sub_data.columns else 1

    if model_name is not None:
        model_results = results_df[
            (results_df['Sub'] == sub) &
            (results_df['model_name'] == model_name)
        ]
        if len(model_results) == 0:
            raise ValueError(f"No results for subject {sub}, model {model_name}")
        c_id = model_results['c_id'].iloc[0]
        s_id = model_results['s_id'].iloc[0]
        b_id = model_results['b_id'].iloc[0]

    model_results = results_df[
        (results_df['Sub'] == sub) &
        (results_df['c_id'] == c_id) &
        (results_df['s_id'] == s_id) &
        (results_df['b_id'] == b_id)
    ]

    config = get_parameter_config(c_id, s_id, b_id)
    par = [model_results[name].iloc[0] for name in config['names']]

    stimrep = sub_data[['Duration', 'Reproduction']].values
    coherence = sub_data['coherence'].values
    structure = sub_data['Structure'].values if 'Structure' in sub_data.columns else None

    tracking = generate_predictions(par, stimrep, coherence, structure,
                                   c_id, s_id, b_id)

    return {
        'Sub': sub,
        'exp': exp,
        'model_name': get_model_by_id(c_id, s_id, b_id)['model_name'],
        'pred': tracking['pred_orig'],
        'obs': tracking['obs_orig'],
        'stimulus': stimrep[:, 0],
        'coherence': coherence,
        'structure': structure,
        'tracking': tracking
    }


def generate_ppc_all_subjects(data, results_df, model_name=None,
                               c_id=None, s_id=None, b_id=None,
                               exp_num=None, verbose=True):
    """Generate PPC predictions for all subjects."""
    df = data.copy()
    if exp_num is not None:
        df = df[df['exp'] == exp_num]

    subjects = df['Sub'].unique()
    all_ppc = []

    iterator = tqdm(subjects, desc="Generating PPC") if verbose else subjects

    for sub in iterator:
        sub_data = df[df['Sub'] == sub]
        try:
            ppc = generate_ppc_single_subject(
                sub_data, results_df,
                model_name=model_name,
                c_id=c_id, s_id=s_id, b_id=b_id
            )

            n_trials = len(ppc['pred'])
            ppc_rows = pd.DataFrame({
                'Sub': [ppc['Sub']] * n_trials,
                'exp': [ppc['exp']] * n_trials,
                'model_name': [ppc['model_name']] * n_trials,
                'stimulus': ppc['stimulus'],
                'coherence': ppc['coherence'],
                'structure': ppc['structure'] if ppc['structure'] is not None else ['NA'] * n_trials,
                'pred': ppc['pred'],
                'obs': ppc['obs'],
                'mu_post': ppc['tracking']['mu_post'],
                'b_post': ppc['tracking']['b_post'],
                'K_mu': ppc['tracking']['K_mu'],
                'resid': ppc['obs'] - ppc['pred'],
            })
            all_ppc.append(ppc_rows)

        except Exception as e:
            if verbose:
                print(f"Warning: Failed for subject {sub}: {e}")
            continue

    return pd.concat(all_ppc, ignore_index=True) if all_ppc else pd.DataFrame()


def compare_axes(results_df, exp_num=None, criterion='AIC'):
    """
    Compare performance by each axis.

    Returns
    -------
    comparison : dict
        'by_c': C-axis comparison
        'by_s': S-axis comparison
        'by_b': B-axis comparison
    """
    df = results_df.copy()
    if exp_num is not None:
        df = df[df['exp'] == exp_num]

    # By C-axis
    c_summary = df.groupby('c_id').agg({
        criterion: ['mean', 'min', 'std'],
        'n_params': 'mean'
    }).round(2)
    c_summary.columns = [f'{criterion}_mean', f'{criterion}_min', f'{criterion}_std', 'n_params_mean']
    c_summary = c_summary.sort_values(f'{criterion}_mean')

    # By S-axis
    s_summary = df.groupby('s_id').agg({
        criterion: ['mean', 'min', 'std']
    }).round(2)
    s_summary.columns = [f'{criterion}_mean', f'{criterion}_min', f'{criterion}_std']
    s_summary = s_summary.sort_values(f'{criterion}_mean')

    # By B-axis
    b_summary = df.groupby('b_id').agg({
        criterion: ['mean', 'min', 'std']
    }).round(2)
    b_summary.columns = [f'{criterion}_mean', f'{criterion}_min', f'{criterion}_std']
    b_summary = b_summary.sort_values(f'{criterion}_mean')

    return {
        'by_c': c_summary,
        'by_s': s_summary,
        'by_b': b_summary
    }


# =============================================================================
# PPC EVALUATION METRICS
# =============================================================================

def compute_ppc_metrics(ppc_df):
    """
    Compute PPC evaluation metrics.

    Returns
    -------
    metrics : dict
        'corr': correlation between pred and obs
        'rmse': root mean squared error
        'by_coherence': metrics split by coherence level
    """
    from scipy.stats import pearsonr

    pred = ppc_df['pred'].values
    obs = ppc_df['obs'].values
    resid = ppc_df['resid'].values

    # Overall metrics
    corr, _ = pearsonr(pred, obs)
    rmse = np.sqrt(np.mean(resid ** 2))
    mae = np.mean(np.abs(resid))

    # By coherence
    by_coh = {}
    for coh in ppc_df['coherence'].unique():
        mask = ppc_df['coherence'] == coh
        pred_c = pred[mask]
        obs_c = obs[mask]
        resid_c = resid[mask]
        r, _ = pearsonr(pred_c, obs_c)
        by_coh[coh] = {
            'corr': r,
            'rmse': np.sqrt(np.mean(resid_c ** 2)),
            'n': mask.sum()
        }

    return {
        'corr': corr,
        'rmse': rmse,
        'mae': mae,
        'by_coherence': by_coh
    }
