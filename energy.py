import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def potential_energies(frames, calc):
    for a in frames:
        a.calc = calc
    return np.array([a.get_potential_energy() for a in frames])

def kinetic_energies(frames):
    return np.array([a.get_kinetic_energy() for a in frames])

def total_energies(frames, calc):
    E_pot = potential_energies(frames, calc)
    E_kin = kinetic_energies(frames)
    return E_pot, E_kin, E_pot + E_kin

def plot_energy(E_tot, out="energy.png", bins=80):
    fig, ax = plt.subplots(figsize=(6, 4), facecolor="#fcfcfb")
    ax.set_facecolor("#fcfcfb")
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.tick_params(colors="#52514e", labelsize=8)

    ax.hist(E_tot, bins=bins, color="#2a78d6", edgecolor="#fcfcfb", linewidth=0.4)
    ax.set_xlabel("E_tot  [eV]", fontsize=10, color="#52514e")
    ax.set_ylabel("samples", fontsize=10, color="#52514e")
    ax.set_title(f"Total energy ({len(E_tot)} samples)   mean = {E_tot.mean():.4f} eV   σ = {E_tot.std():.3g} eV",
                 x=0.0, ha="left", fontsize=10)

    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)
