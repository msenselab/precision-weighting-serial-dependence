# Nature Style Plot Configuration
# Unified plotting style for all analysis notebooks

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Nature-style color palette
COLORS = {
    'high': '#E64B35',      # Red - High uncertainty/coherence
    'low': '#4DBBD5',       # Blue - Low uncertainty/coherence
    'HH': '#E64B35',        # High-High transition
    'HL': '#F39B7F',        # High-Low transition
    'LH': '#7E6148',        # Low-High transition
    'LL': '#4DBBD5',        # Low-Low transition
    'same': '#3C5488',      # Same condition
    'switch': '#F39B7F',    # Switch condition
    'exp1': '#00A087',      # Experiment 1
    'exp2': '#8491B4',      # Experiment 2
    'neutral': '#666666',   # Neutral/control
}

# Transition type order and colors
TRANSITION_ORDER = ['HH', 'HL', 'LH', 'LL']
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

        # Font - Nature uses Helvetica/Arial
        'font.family': 'sans-serif',
        'font.sans-serif': ['Arial', 'Helvetica', 'DejaVu Sans'],
        'font.size': 10,
        'axes.labelsize': 10,
        'axes.titlesize': 10,
        'xtick.labelsize': 10,
        'ytick.labelsize': 10,
        'legend.fontsize': 9,

        # Axes
        'axes.linewidth': 0.8,
        'axes.spines.top': False,
        'axes.spines.right': False,
        'axes.labelpad': 4,
        'axes.titlepad': 8,

        # Ticks
        'xtick.major.width': 0.8,
        'ytick.major.width': 0.8,
        'xtick.major.size': 3,
        'ytick.major.size': 3,
        'xtick.direction': 'out',
        'ytick.direction': 'out',

        # Lines
        'lines.linewidth': 1.2,
        'lines.markersize': 4,

        # Legend
        'legend.frameon': False,
        'legend.borderpad': 0.2,
        'legend.labelspacing': 0.3,

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
