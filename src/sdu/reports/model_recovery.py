"""Figures of the model recovery and the high-continuity profiles (notebooks/recovery.ipynb, Appendix B.6)."""
import numpy as np
import matplotlib.pyplot as plt

INK = "#263940"
COLORS = {1: "#287CA6", 2: "#617D71"}


def switch_confusion(matrix):
    fig, ax = plt.subplots(figsize=(4.8, 4.1), layout="constrained")
    proportions = matrix.to_numpy() / matrix.sum(axis=1).to_numpy()[:, None]
    ax.imshow(proportions, cmap="Blues", vmin=0, vmax=1)
    ax.set_xticks([0, 1], ["S0", "S3"])
    ax.set_yticks([0, 1], ["S0", "S3"])
    ax.set(xlabel="Selected model", ylabel="Generating model")
    for row in range(2):
        for col in range(2):
            value = proportions[row, col]
            ax.text(col, row, f"{value:.0%}", ha="center", va="center", fontsize=19,
                    color="white" if value > .5 else INK)
    return fig


def phi_profiles(curves):
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.8), sharex=True, sharey=True, layout="constrained")
    for ax, exp in zip(axes, (1, 2)):
        d = curves[curves.exp == exp]
        for _, g in d.groupby("Sub"):
            g = g.sort_values("phi_high")
            ax.plot(g.phi_high, g.loss, color=COLORS[exp], alpha=.2, lw=.9)
        regular = d[np.isclose(d.phi_high*10, np.round(d.phi_high*10))].copy()
        regular["grid"] = regular.phi_high.round(1)
        med = regular.drop_duplicates(["Sub", "grid"]).groupby("grid").loss.median()
        ax.plot(med.index, med.values, color=COLORS[exp], lw=2.5, label="Median profile")
        ax.axhline(3.8414588207, color=INK, ls="--", lw=1, label="Loss reference: 3.84")
        ax.set(xlim=(0, 1), ylim=(0, 20), xlabel=r"Fixed $\phi^H$",
               title=f"Experiment {exp}")
        ax.grid(axis="y", color="#E8ECED", lw=.6)
    axes[0].set_ylabel("Twice the log-likelihood loss")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="outside lower center", frameon=False, ncol=2)
    return fig
