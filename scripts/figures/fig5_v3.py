"""
Figure 5 for revision 1: the submitted three-state Kalman filter figure (135-model space), changed as little as
possible to show 3SKF v3 (2026-11-13): prior = learned mean; same gain at both coherence levels; the B2 bias drifts
outside the filter; the trace f = x - m of the last trial enters only the readout, with share phi set by the current
coherence (0 after a switch, S3); the response is a separate column.
Usage (repo root): python scripts/figures/fig5_v3.py <output path without extension>

Original header:
Three-State Kalman Filter — Revised Figure
B2/B3 pathways explicitly shown.
Good elements from kalman_schematic: Prediction/Error labels, text-based K description,
= Innovation, straight Response arrow.
"""

import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Rectangle
import matplotlib.patches as mpatches
import numpy as np
from scipy import stats

fig, ax = plt.subplots(1, 1, figsize=(16.8, 8))
ax.set_xlim(0, 16.8)
ax.set_ylim(0.3, 7.95)
ax.set_aspect('equal')
ax.axis('off')

# ── Color palette ──────────────────────────────────────────────────────────────
FAST_COL  = '#2980B9'
SLOW_COL  = '#27AE60'
BIAS_COL  = '#8E44AD'
PROC_DARK = '#2C5F6E'
BG_LIGHT  = '#FFFFFF'
BG_ALT    = '#F0F0F0'
TEXT_DARK = '#1A3A44'
TEXT_MED  = '#555555'
ARROW_COL = '#7F8C8D'
COH_COL   = '#7A9AA5'
WHITE     = '#FFFFFF'
DIST_HIGH = '#27AE60'
DIST_LOW  = '#C0392B'

# ── Column geometry ────────────────────────────────────────────────────────────
cols = [(0.0, 2.6), (2.6, 5.2), (5.2, 8.2), (8.2, 11.2), (11.2, 14.0), (14.0, 16.8)]
col_labels = ['PRIOR STATES', 'COMPARE', 'UPDATE', 'POSTERIOR STATES', 'READOUT', 'RESPONSE']
col_bg     = [BG_LIGHT, BG_ALT, BG_LIGHT, BG_ALT, BG_LIGHT, BG_ALT]
GOLD = '#C9A227'

Y_HEADER = 7.65
Y_TOP    = 6.20   # fast state
Y_MID    = 4.10   # slow state / pipeline
Y_BOT    = 2.00   # bias state
Y_FLOOR  = 0.30
H_STATE  = 1.50
XC = [(x0 + x1) / 2 for x0, x1 in cols]

# ── Column backgrounds & headers ───────────────────────────────────────────────
for (x0, x1), label, bg in zip(cols, col_labels, col_bg):
    ax.add_patch(FancyBboxPatch(
        (x0, Y_FLOOR), x1 - x0, Y_HEADER - Y_FLOOR + 0.2,
        boxstyle="round,pad=0,rounding_size=0.08",
        facecolor=bg, edgecolor='#CCCCCC', linewidth=0.8))
    ax.text((x0 + x1) / 2, Y_HEADER, label,
            ha='center', va='center', fontsize=9, fontweight='bold', color=TEXT_DARK)


# ── Helpers ────────────────────────────────────────────────────────────────────
def state_box(x, y, lines, w=2.2, h=H_STATE, color=FAST_COL, border_lw=2.0):
    ax.add_patch(FancyBboxPatch(
        (x - w/2, y - h/2), w, h,
        boxstyle="round,pad=0.06,rounding_size=0.15",
        facecolor=WHITE, edgecolor=color, linewidth=border_lw))
    top = y + h/2 - 0.10
    for j, (txt, fs, fw, tc) in enumerate(lines):
        ax.text(x, top - 0.22 - j * 0.22, txt,
                ha='center', va='center', fontsize=fs, fontweight=fw, color=tc)


def gauss(x, yc, w, h, sigma=1.0, color=FAST_COL,
          ls='solid', fill=True, alpha=0.22, frac=0.36, ref_sigma=None, mu=0.0):
    xr = np.linspace(-3.5, 3.5, 300)
    xv = (xr / 7.0) * (w * 0.78) + x
    yb = yc - h / 2 + 0.12
    yv = stats.norm.pdf(xr, mu, sigma)
    if ref_sigma is not None:
        ref_peak = stats.norm.pdf(0, 0, ref_sigma)
        yv = (yv / ref_peak) * (h * frac) + yb
    else:
        yv = (yv / np.max(yv)) * (h * frac) + yb
    ax.plot(xv, yv, color=color, lw=1.8, linestyle=ls, zorder=5)
    ax.plot([xv[0], xv[-1]], [yb, yb], color=color, lw=0.8, alpha=0.5, zorder=4)
    if fill:
        ax.fill_between(xv, yb, yv, color=color, alpha=alpha, zorder=4)


def arr(start, end, color=ARROW_COL, lw=1.5, rad=0,
        style='-|>', mutation=12, ls='solid'):
    ax.add_patch(FancyArrowPatch(
        start, end, connectionstyle=f"arc3,rad={rad}",
        arrowstyle=style, mutation_scale=mutation,
        lw=lw, color=color, linestyle=ls, zorder=3))


def badge(x, y, text, color):
    ax.text(x, y, text, ha='center', va='center',
            fontsize=7.5, fontweight='bold', color=color, zorder=8,
            clip_on=False,
            bbox=dict(boxstyle='round,pad=0.25', facecolor='white',
                      edgecolor=color, linewidth=1.2))


def col_label(x, y, text):
    ax.text(x, y, text, ha='center', va='center',
            fontsize=7.5, color=TEXT_MED, fontstyle='italic', zorder=5)


# ══════════════════════════════════════════════════════════════════════════════
# COLUMN 1 — PRIOR STATES
# ══════════════════════════════════════════════════════════════════════════════
state_box(XC[0], Y_TOP, [
    ('Fast state ($\\hat{x}_{i-1}$)', 9, 'bold', FAST_COL),
    ('last estimate', 7, 'normal', TEXT_MED),
], color=FAST_COL)
gauss(XC[0], Y_TOP, 2.2, H_STATE, sigma=0.7, color=FAST_COL, ls='--', fill=False, mu=1.3, ref_sigma=0.7, frac=0.3)
gauss(XC[0], Y_TOP, 2.2, H_STATE, sigma=0.35, color=SLOW_COL, ls=':', fill=False, mu=0.0, ref_sigma=0.35, frac=0.22)
_yt = Y_TOP - H_STATE/2 + 0.12 + 0.5
ax.annotate('', ((1.3 / 7.0) * (2.2 * 0.78) + XC[0], _yt), (XC[0], _yt),
            arrowprops=dict(arrowstyle='-|>', color=GOLD, lw=2.0), zorder=7)
ax.text(XC[0] - 0.55, _yt, r'$\hat{f}_{i-1}$', ha='center', va='center', fontsize=8.5, color=GOLD, fontweight='bold', zorder=7)

state_box(XC[0], Y_MID, [
    ('Slow state ($\\hat{m}_{i-1}$)', 9, 'bold', SLOW_COL),
    ('learned mean', 7, 'normal', TEXT_MED),
], color=SLOW_COL)
gauss(XC[0], Y_MID, 2.2, H_STATE, sigma=0.35, color=SLOW_COL, ls='--', fill=False)

state_box(XC[0], Y_BOT, [
    ('Bias state ($b$)', 9, 'bold', BIAS_COL),
    ('response tendency', 7, 'normal', TEXT_MED),
], color=BIAS_COL)
gauss(XC[0], Y_BOT, 2.2, H_STATE, sigma=0.8, color=BIAS_COL, ls='--', fill=False)

ax.text(XC[0], Y_BOT - H_STATE/2 - 0.22,
        '$\\lambda$ decay (between trials)',
        ha='center', va='center', fontsize=6.5, color=TEXT_MED, fontstyle='italic')


# ══════════════════════════════════════════════════════════════════════════════
# COLUMN 3 — COMPARE
# ══════════════════════════════════════════════════════════════════════════════
state_box(XC[1], Y_TOP, [
    ('Stimulus ($z_i$)', 9, 'bold', TEXT_DARK),
    ('presented duration', 7, 'normal', TEXT_MED),
], color='#7F8C8D')
gauss(XC[1], Y_TOP, 2.2, H_STATE, sigma=0.50, color='#7F8C8D',
      fill=True, frac=0.36, ref_sigma=0.50)

# "−" sign on vertical arrow
ax.text(XC[1] + 0.18, (Y_TOP - H_STATE/2 + Y_MID + H_STATE/2) / 2,
        '$-$', ha='center', va='center', fontsize=14,
        fontweight='bold', color=PROC_DARK, zorder=6)

# Pred. error box
state_box(XC[1], Y_MID, [
    ('Pred. error', 8.5, 'bold', PROC_DARK),
    (r'$z_i - \hat{x}_i^-$,  $\hat{x}_i^- = \hat{m}_{i-1}$', 7.5, 'normal', TEXT_MED),
], color=PROC_DARK, border_lw=1.8, w=2.3)
gauss(XC[1], Y_MID, 2.3, H_STATE, sigma=0.80, color=PROC_DARK,
      fill=True, alpha=0.15, frac=0.26, ref_sigma=0.50)
# "= Innovation" label between formula and Gaussian
ax.text(XC[1], Y_MID + 0.04, '= Innovation',
        ha='center', va='center', fontsize=6.5, color=TEXT_MED,
        fontstyle='italic', zorder=6)

arr((XC[0] + 1.22, Y_MID),       (XC[1] - 1.15, Y_MID), color=SLOW_COL)
arr((XC[1], Y_TOP - H_STATE/2 - 0.10),  (XC[1], Y_MID + H_STATE/2))


# Coherence box — bottom of Compare column
state_box(XC[1], Y_BOT, [
    ('Coherence', 8.5, 'bold', PROC_DARK),
    (r'sets $\phi$ and bias noise', 7.5, 'normal', TEXT_MED),
], color=PROC_DARK, border_lw=1.5, w=2.3)
gauss(XC[1], Y_BOT, 2.3, H_STATE, sigma=0.50, color=DIST_HIGH,
      fill=True, frac=0.34, ref_sigma=0.50)
gauss(XC[1], Y_BOT, 2.3, H_STATE, sigma=1.17, color=DIST_LOW,
      fill=True, alpha=0.15, frac=0.34, ref_sigma=0.50)


# ══════════════════════════════════════════════════════════════════════════════
# COLUMN 4 — UPDATE  (Kalman gain)
# ══════════════════════════════════════════════════════════════════════════════
K_H = 2.5
state_box(XC[2], Y_MID, [
    ('Kalman gain ($\\mathbf{K}$)', 9, 'bold', PROC_DARK),
    ('$\\mathbf{K} = \\mathbf{P}^-\\mathbf{H}^T'
     '(\\mathbf{H}\\mathbf{P}^-\\mathbf{H}^T\\!+\\!R)^{-1}$',
     6.5, 'normal', TEXT_MED),
], color=PROC_DARK, border_lw=2.0, w=2.6, h=K_H)

arr((XC[1] + 1.28, Y_MID), (XC[2] - 1.30, Y_MID))

# ── Vertical stacked bars (sensory on top, prior on bottom) ──────────────────
bar_base  = Y_MID - K_H/2 + 0.57   # raised to leave room for legend at bottom
bar_total = 1.05                     # total bar height
bar_w     = 0.65
gap       = 0.38
x_left_b  = XC[2] - bar_w - gap/2   # 9.3 - 0.65 - 0.19 = 8.46
x_right_b = XC[2] + gap/2           # 9.3 + 0.19 = 9.49

xb, K_val = XC[2] - bar_w / 2, 0.6
prior_h, sensory_h = bar_total * (1 - K_val), bar_total * K_val
ax.add_patch(Rectangle((xb, bar_base), bar_w, prior_h, facecolor=FAST_COL, alpha=0.55, lw=0, zorder=3))
ax.add_patch(Rectangle((xb, bar_base + prior_h), bar_w, sensory_h, facecolor=DIST_HIGH, alpha=0.82, lw=0, zorder=3))
ax.add_patch(Rectangle((xb, bar_base), bar_w, bar_total, edgecolor='#7f8c8d', facecolor='none', lw=0.9, zorder=4))
ax.text(xb + bar_w + 0.12, bar_base + prior_h + sensory_h / 2, '$K$', ha='left', va='center', fontsize=8.5, color=TEXT_DARK, zorder=5)
ax.text(xb + bar_w + 0.12, bar_base + prior_h / 2, '$1-K$\nweight left\nfor the past', ha='left', va='center',
        fontsize=6.8, color=TEXT_DARK, linespacing=1.05, zorder=5)
ax.text(XC[2], bar_base - 0.16, 'High = Low', ha='center', va='top', fontsize=7.5, fontweight='bold', color=TEXT_MED, zorder=5)



# Coherence → bottom center edge of Kalman gain box




# ══════════════════════════════════════════════════════════════════════════════
# COLUMN 5 — POSTERIOR STATES
# ══════════════════════════════════════════════════════════════════════════════
state_box(XC[3], Y_TOP, [
    ('Fast state ($\\hat{x}$)', 9, 'bold', FAST_COL),
    ('updated estimate', 7, 'normal', TEXT_MED),
], color=FAST_COL)
# Prior (dashed blue) — mirrors col 1 fast state, mu=0: reference for the shift
gauss(XC[3], Y_TOP, 2.2, H_STATE, sigma=1.3, color=FAST_COL,
      ls='--', fill=False, frac=0.36, ref_sigma=0.50, mu=0.0)
# posterior: the same at both coherence levels (same gain)
gauss(XC[3], Y_TOP, 2.2, H_STATE, sigma=0.60, color=FAST_COL,
      fill=True, frac=0.36, ref_sigma=0.50, mu=0.6)

state_box(XC[3], Y_MID, [
    ('Slow state ($\\hat{m}$)', 9, 'bold', SLOW_COL),
    ('updated mean', 7, 'normal', TEXT_MED),
], color=SLOW_COL)
gauss(XC[3], Y_MID, 2.2, H_STATE, sigma=0.32, color=SLOW_COL, fill=True, frac=0.36)

state_box(XC[3], Y_BOT, [
    ('Bias state ($b$)', 9, 'bold', BIAS_COL),
    (r'$\lambda\,b_{i-1} + \eta_i$', 7.5, 'normal', TEXT_MED),
], color=BIAS_COL)
gauss(XC[3], Y_BOT, 2.2, H_STATE, sigma=0.75, color=BIAS_COL, fill=True, frac=0.36)
# the bias drifts on its own, outside the filter (B2)
ax.plot([XC[0], XC[0], XC[3]], [Y_BOT - H_STATE/2 - 0.06, 0.62, 0.62], color=BIAS_COL, lw=1.3, ls=(0, (4, 2)), zorder=3)
arr((XC[3], 0.62), (XC[3], Y_BOT - H_STATE/2 - 0.08), color=BIAS_COL, lw=1.3, ls=(0, (4, 2)))

# K → all three posterior states
arr((XC[2] + 1.42, Y_MID + 0.20), (XC[3] - 1.10, Y_TOP - 0.20), rad=-0.12)
arr((XC[2] + 1.42, Y_MID),        (XC[3] - 1.10, Y_MID))


# ══════════════════════════════════════════════════════════════════════════════
# COLUMN 6 — READOUT & RESPONSE
# ══════════════════════════════════════════════════════════════════════════════
state_box(XC[4], Y_TOP, [
    ('Readout ($\\hat{u}_i$)', 9, 'bold', PROC_DARK),
    (r'$\hat{x}_i + (1-K^{(x)})\,\phi_i\,\hat{f}_{i-1}$', 7, 'normal', TEXT_MED),
], color=PROC_DARK)
# the readout: the updated estimate (dashed) shifted by the retained share of the trace (schematic size)
gauss(XC[4], Y_TOP, 2.2, H_STATE, sigma=0.60, color=FAST_COL, ls='--', fill=False, frac=0.32, ref_sigma=0.5, mu=0.6)
gauss(XC[4], Y_TOP, 2.2, H_STATE, sigma=0.60, color=PROC_DARK, fill=True, alpha=0.18, frac=0.32, ref_sigma=0.5, mu=1.4)
yr = Y_TOP - H_STATE/2 + 0.12 + 0.62
ax.annotate('', ((1.4 / 7.0) * (2.2 * 0.78) + XC[4], yr), ((0.6 / 7.0) * (2.2 * 0.78) + XC[4], yr),
            arrowprops=dict(arrowstyle='-|>', color=GOLD, lw=1.8), zorder=7)
arr((XC[3] + 1.22, Y_TOP), (XC[4] - 1.15, Y_TOP))

# continuity phi: the share of the previous trace used at readout, set by the current coherence
# continuity: the bar style of before, with the same schematic fill for both levels (no ordering implied)
state_box(XC[4], Y_MID, [('Continuity ($\\phi_i$)', 8.5, 'bold', PROC_DARK)], color=PROC_DARK, border_lw=1.5, h=1.3)
bx0, bw, bh = XC[4] - 0.45, 1.25, 0.18
for c, y, lab, f in (('High', Y_MID - 0.02, r'$\phi^{H}$', 0.25), ('Low', Y_MID - 0.27, r'$\phi^{L}$', 0.62)):
    ax.add_patch(Rectangle((bx0, y - bh/2), bw * f, bh, facecolor=GOLD, alpha=0.85, lw=0, zorder=5))
    ax.add_patch(Rectangle((bx0, y - bh/2), bw, bh, facecolor='none', edgecolor='#7f8c8d', lw=0.8, zorder=6))
    ax.text(bx0 - 0.07, y, c, ha='right', va='center', fontsize=7, fontweight='bold',
            color=DIST_HIGH if c == 'High' else DIST_LOW, zorder=6)
    ax.text(bx0 + bw * f + 0.05, y, lab, ha='left', va='center', fontsize=7, color=GOLD, fontweight='bold', zorder=7) if f < 0.3 else ax.text(bx0 + bw * f / 2, y, lab, ha='center', va='center', fontsize=7, color='white', fontweight='bold', zorder=7)
ax.text(XC[4], Y_MID - 0.5, '0 after a switch (S3)', ha='center', va='center', fontsize=6.5,
        color=TEXT_MED)
arr((XC[4], Y_MID + 0.72), (XC[4], Y_TOP - H_STATE/2 - 0.08), color=GOLD, lw=1.6)

# trace of the previous trial: around the filter, into the readout only
ax.plot([XC[0], XC[0], XC[4] + 0.35], [Y_TOP + H_STATE/2 + 0.06, 7.15, 7.15], color=GOLD, lw=1.8, zorder=3)
arr((XC[4] + 0.35, 7.15), (XC[4] + 0.35, Y_TOP + H_STATE/2 + 0.08), color=GOLD, lw=1.8)
ax.text((XC[0] + XC[4]) / 2, 7.3, r'trace $\hat{f}_{i-1} = \hat{x}_{i-1} - \hat{m}_{i-1}$: enters the readout only',
        ha='center', va='center', fontsize=7.5, color=GOLD, fontweight='bold',
        bbox=dict(facecolor='white', edgecolor='none', pad=0.6))

# coherence: to phi and to the bias noise (C3), not to the gain
ax.plot([XC[1] + 1.15, 8.32, 8.32, XC[4] - 0.35], [Y_BOT + 0.45, Y_BOT + 0.45, 3.05, 3.05], color=COH_COL, lw=1.4, zorder=3)
arr((XC[4] - 0.35, 3.05), (XC[4] - 0.35, Y_MID - 0.72), color=COH_COL, lw=1.4)
arr((8.32, Y_BOT + 0.45), (XC[3] - 1.12, Y_BOT + 0.45), color=COH_COL, lw=1.4)

# response
state_box(XC[5], Y_TOP, [
    ('Response ($y_i$)', 9, 'bold', TEXT_DARK),
    (r'$\hat{u}_i + b_i + d_0^{\mathrm{eff}} + e_i$', 7, 'normal', TEXT_MED),
], color=TEXT_DARK)
gauss(XC[5], Y_TOP, 2.2, H_STATE, sigma=0.9, color=DIST_HIGH, fill=True, frac=0.3, ref_sigma=0.6, mu=0.7)
gauss(XC[5], Y_TOP, 2.2, H_STATE, sigma=0.9, color=DIST_LOW, fill=True, alpha=0.15, frac=0.3, ref_sigma=0.6, mu=1.5)
arr((XC[4] + 1.22, Y_TOP), (XC[5] - 1.15, Y_TOP))
ax.plot([XC[3] + 1.22, XC[5]], [Y_BOT, Y_BOT], color=BIAS_COL, lw=1.5, zorder=3)
arr((XC[5], Y_BOT), (XC[5], Y_TOP - H_STATE/2 - 0.08), color=BIAS_COL, lw=1.5)
ax.text(XC[5], Y_MID, 'bias added\nto the response', ha='center', va='center', fontsize=7, color=BIAS_COL,
        fontstyle='italic', bbox=dict(facecolor=BG_LIGHT, edgecolor='none', pad=1))


# ── Legend ─────────────────────────────────────────────────────────────────────
ax.plot([], [], color=DIST_HIGH, lw=2.0, label='High coherence (low uncertainty)')
ax.plot([], [], color=DIST_LOW,  lw=2.0, label='Low coherence (high uncertainty)')
ax.plot([], [], color=FAST_COL, lw=1.8, linestyle='--', label='Prior (before update)')
ax.plot([], [], color=GOLD, lw=2.5, label='trace and its share $\\phi$ at readout')
ax.legend(loc='upper center', fontsize=8, frameon=True, framealpha=0.85,
          bbox_to_anchor=(0.5, -0.04), ncol=4)


# ── Save ───────────────────────────────────────────────────────────────────────
import sys
BASE = sys.argv[1]
plt.tight_layout(pad=0.3)
plt.savefig(BASE + '.png', dpi=300, bbox_inches='tight', facecolor='white')
plt.savefig(BASE + '.pdf', bbox_inches='tight', facecolor='white')
print("Saved:", BASE)
plt.close()
