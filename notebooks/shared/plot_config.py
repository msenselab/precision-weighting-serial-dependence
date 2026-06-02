# Nature Style Plot Configuration
# Unified plotting style for all analysis notebooks

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Nature-style color palette
# High/low now refer to objective motion coherence (not subjective uncertainty).
COLORS = {
    'high': '#4DBBD5',      # Blue - high coherence / clearer stimulus
    'low': '#E64B35',       # Red - low coherence / degraded stimulus
    # TransitionType values are raw data codes from the old uncertainty notation:
    # raw HH->display LL, raw HL->display LH, raw LH->display HL, raw LL->display HH.
    'HH': '#E64B35',        # Display: low-low coherence transition
    'HL': '#F39B7F',        # Display: low-high coherence transition
    'LH': '#7E6148',        # Display: high-low coherence transition
    'LL': '#4DBBD5',        # Display: high-high coherence transition
    'same': '#3C5488',      # Same condition
    'switch': '#F39B7F',    # Switch condition
    'exp1': '#00A087',      # Experiment 1
    'exp2': '#8491B4',      # Experiment 2
    'neutral': '#666666',   # Neutral/control
}

# Transition type order and colors. TRANSITION_ORDER is raw-data order; labels
# are remapped for display so H/L denote objective coherence in figures.
TRANSITION_ORDER = ['HH', 'HL', 'LH', 'LL']
COHERENCE_TRANSITION_LABELS = {
    'HH': 'LL',  # old high uncertainty -> low coherence on both trials
    'HL': 'LH',  # old high->low uncertainty -> low->high coherence
    'LH': 'HL',  # old low->high uncertainty -> high->low coherence
    'LL': 'HH',  # old low uncertainty -> high coherence on both trials
}
TRANSITION_COLORS = [COLORS['HH'], COLORS['HL'], COLORS['LH'], COLORS['LL']]
TRANSITION_PALETTE = dict(zip(TRANSITION_ORDER, TRANSITION_COLORS))

# Duration levels
DURATION_ORDER = [0.6, 0.9, 1.2, 1.5, 1.8]

def set_nature_style():
    """Set matplotlib parameters for Nature-style figures."""
    plt.rcParams.update({
        # Figure
        'figure.figsize': (3.5, 3),
        'figure.dpi': 300,
        'figure.facecolor': 'white',
        'savefig.dpi': 300,
        'savefig.bbox': 'tight',
        'savefig.transparent': False,

        # Font - Nature uses Helvetica/Arial. Sizes bumped for publication legibility.
        'font.family': 'sans-serif',
        'font.sans-serif': ['Arial', 'Helvetica', 'DejaVu Sans'],
        'font.size': 12,
        'axes.labelsize': 13,
        'axes.titlesize': 14,
        'xtick.labelsize': 12,
        'ytick.labelsize': 12,
        'legend.fontsize': 12,

        # Axes
        'axes.linewidth': 1.0,
        'axes.spines.top': False,
        'axes.spines.right': False,
        'axes.labelpad': 5,
        'axes.titlepad': 10,

        # Ticks
        'xtick.major.width': 1.0,
        'ytick.major.width': 1.0,
        'xtick.major.size': 4,
        'ytick.major.size': 4,
        'xtick.direction': 'out',
        'ytick.direction': 'out',

        # Lines
        'lines.linewidth': 1.4,
        'lines.markersize': 5,

        # Legend
        'legend.frameon': False,
        'legend.borderpad': 0.3,
        'legend.labelspacing': 0.35,

        # Grid (off by default for Nature)
        'axes.grid': False,
    })

    # Set seaborn style
    sns.set_style("ticks")
    sns.set_context("paper")

def get_transition_palette():
    """Return palette for transition types."""
    return TRANSITION_PALETTE

def get_exp_palette():
    """Return palette for experiments."""
    return {'Exp1': COLORS['exp1'], 'Exp2': COLORS['exp2']}

def add_significance_stars(ax, x1, x2, y, p_value, height=0.02):
    """Add significance annotation between two bars."""
    if p_value < 0.001:
        stars = '***'
    elif p_value < 0.01:
        stars = '**'
    elif p_value < 0.05:
        stars = '*'
    else:
        stars = 'n.s.'

    ax.plot([x1, x1, x2, x2], [y, y+height, y+height, y],
            lw=0.8, color='black')
    ax.text((x1+x2)/2, y+height, stars, ha='center', va='bottom', fontsize=7)

def despine(ax=None):
    """Remove top and right spines."""
    if ax is None:
        ax = plt.gca()
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

# Initialize style when imported
set_nature_style()
