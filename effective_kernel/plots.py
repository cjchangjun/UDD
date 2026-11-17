"""Figures of the T3 analysis, shared by run_t3.py and the notebook.

Every figure shows a conditional summary of the effective kernels (power |rfft|^2 of the kernel rows,
see spectra.py), not "the frequency response of Mamba". Lines are the mean over seeds, bands +/- 1 std
over seeds. Frequencies are non-negative, in Hz; the dataset was low-pass filtered at 20 Hz.
"""
import numpy as np
import matplotlib.pyplot as plt

from spectra import channel_mean

CLASSES = ["WALKING", "UPSTAIRS", "DOWNSTAIRS", "SITTING", "STANDING", "LAYING"]
COLORS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300"]   # fixed class order
STYLES = ["-", "-", "-", "--", "--", "--"]                                     # moving: solid, static: dashed
LPF_HZ = 20.0
INK, MUTED, GRID = "#2b3140", "#6b7280", "#e5e7eb"


def _axes(ax, xlabel=None, ylabel=None, title=None):
    ax.grid(color=GRID, lw=0.7)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.set_xlabel(xlabel or "", color=INK)
    ax.set_ylabel(ylabel or "", color=INK)
    if title:
        ax.set_title(title, fontsize=10, color=INK, loc="left")


def seed_stats(specs, layer):
    """Channel-averaged class spectra of one layer, mean and std over seeds: two arrays [classes, F]."""
    S = np.stack([channel_mean(s)[:, layer] for s in specs])                  # [seeds, classes, F]
    return S.mean(0), (S.std(0, ddof=1) if len(S) > 1 else np.zeros_like(S[0]))


def draw_class_spectra(ax, specs, layer, title=None, ylabel="mean |FFT of kernel row|$^2$"):
    f = specs[0]["freq"]
    mean, std = seed_stats(specs, layer)
    for c in range(len(mean)):
        ax.plot(f, mean[c], color=COLORS[c], ls=STYLES[c], lw=1.6, label=CLASSES[c])
        ax.fill_between(f, mean[c] - std[c], mean[c] + std[c], color=COLORS[c], alpha=0.15, lw=0)
    ax.axvline(LPF_HZ, color=MUTED, ls=":", lw=1)
    ax.set_xlim(0, f[-1])
    _axes(ax, "frequency [Hz]", ylabel, title)


def _legend(fig, ax):
    handles, labels = ax.get_legend_handles_labels()
    fig.legend(handles, labels, loc="outside lower center", ncol=7, frameon=False, fontsize=9)


def draw_lti_curve(ax, lti, layer):
    """The LTI model's K does not depend on the input: one curve for all classes (mean over seeds)."""
    S = np.mean([channel_mean(s)[0, layer] for s in lti], axis=0)
    ax.plot(lti[0]["freq"], S, color=INK, ls=":", lw=2.0, label="LTI (same for every input)")


def fig_class_spectra(selective, lti=None):
    """1 x 2: columns = layer. `lti` (the LTI model's input-independent kernel) is drawn as one curve."""
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.9), constrained_layout=True)
    for layer in range(2):
        draw_class_spectra(axes[layer], selective, layer, f"layer {layer}")
        if lti:
            draw_lti_curve(axes[layer], lti, layer)
    _legend(fig, axes[0])
    fig.suptitle("Class-conditional mean row spectrum of the effective kernel (selective Mamba, validation windows)",
                 fontsize=10.5, color=INK, x=0.01, ha="left")
    return fig
