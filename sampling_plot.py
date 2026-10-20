"""
sampling_plot.py - distribution of sampled normal coordinates/momenta, one panel per mode.

Gaussian N(0, sigma_i^2) for that mode.
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from vibfreq import omega_to_wavenumber

UNITS = {"q": "amu$^{1/2}$·Å", "p": "amu$^{1/2}$·Å / ASE time"}

def plot_sampling(omega, x, sigma, label, out="sampling_check.png", bins=60):
    """
    omega  : (n_modes,)            rad/s
    x      : (n_samples, n_modes)  sampled q or p
    sigma  : (n_modes,)            theoretical width, sigma_q(omega) or sigma_p(omega)
    """
    cm = omega_to_wavenumber(omega)
    std = x.std(axis=0)

    n = len(omega)
    ncol = int(np.ceil(np.sqrt(n)))
    nrow = int(np.ceil(n / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(max(3.2 * ncol, 5), 2.6 * nrow + 0.8),
                             squeeze=False, facecolor="#fcfcfb")

    legend_ax = None
    for i, ax in enumerate(axes.flat):
        if i >= n:
            ax.axis("off")
            continue
        ax.set_facecolor("#fcfcfb")
        for side in ("top", "right", "left"):
            ax.spines[side].set_visible(False)
        ax.set_yticks([])
        ax.tick_params(colors="#52514e", labelsize=7)

        if not np.isfinite(sigma[i]):            # imaginary mode: nothing to sample
            ax.set_title(f"{cm[i]:.0f} cm$^{{-1}}$  imaginary", fontsize=9, color="#e34948")
            ax.set_xticks([])
            ax.spines["bottom"].set_visible(False)
            continue

        s = sigma[i]
        t = np.linspace(-4 * s, 4 * s, 200)      # each mode on its own scale
        pdf = np.exp(-t**2 / (2 * s**2)) / np.sqrt(2 * np.pi * s**2)

        ax.hist(x[:, i], bins=bins, range=(-4 * s, 4 * s), density=True,
                color="#2a78d6", edgecolor="#fcfcfb", linewidth=0.4, label="samples")
        ax.plot(t, pdf, color="#0b0b0b", linewidth=1.3, label="theory")
        if legend_ax is None:
            legend_ax = ax
        ax.set_title(f"{cm[i]:.0f} cm$^{{-1}}$   σ = {std[i]:.3g} (theory {s:.3g})",
                     fontsize=8, color="#52514e")
        ax.ticklabel_format(axis="x", style="sci", scilimits=(-2, 3))
        ax.xaxis.get_offset_text().set_fontsize(7)

    if legend_ax is not None:
        legend_ax.legend(loc="upper left", frameon=False, fontsize=7)
    fig.suptitle(f"Wigner sampling: {label} per mode ({len(x)} samples)",
                 x=0.02, ha="left", fontsize=11)
    fig.supxlabel(f"{label}  [{UNITS.get(label, '')}]", fontsize=10, color="#52514e")
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(out, dpi=150)
    plt.close(fig)