import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from ase import units


def harmonic_energies(omega, q, p):
    w = omega / units.second
    E_pot = 0.5 * (w**2 * q**2).sum(axis=1)
    E_kin = 0.5 * (p**2).sum(axis=1)
    return E_pot, E_kin

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

def plot_energy(E_tot, E0, out="energy.png", bins=80):
    fig, ax = plt.subplots(figsize=(6, 4), facecolor="#fcfcfb")
    ax.set_facecolor("#fcfcfb")
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.tick_params(colors="#52514e", labelsize=8)

    E = E_tot - E0
    ax.hist(E, bins=bins, color="#2a78d6", edgecolor="#fcfcfb", linewidth=0.4)
    ax.set_xlabel("E_tot − E0  [eV]", fontsize=10, color="#52514e")
    ax.set_ylabel("samples", fontsize=10, color="#52514e")
    ax.set_title(f"Total energy ({len(E)} samples)   mean = {E.mean():.4f} eV   σ = {E.std():.3g} eV",
                 x=0.0, ha="left", fontsize=10)

    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)
