"""
sampling.py - Wigner sampling of positions from normal modes.

Everything is in ASE units (eV, Å, amu, ASE time)。
"""

import numpy as np
from ase import units

HBAR = units._hbar * units.J * units.second      # ħ in eV·(ASE time)


def sigma_q(omega):
    w = omega / units.second                     # rad/s -> 1/(ASE time)
    return np.sqrt(HBAR / (2 * w))

def sample_q(omega, n_samples, seed=None):
    rng = np.random.default_rng(seed)
    return rng.standard_normal((n_samples, omega.size)) * sigma_q(omega) #(n_samples, n_modes)

def q_to_R(atoms, modes, q):
    sqrt_m = np.sqrt(atoms.get_masses())[None, :, None]
    return atoms.get_positions()[None] + np.einsum("si,iaj->saj", q, modes) / sqrt_m #(n_samples, n, 3), Å

def to_frames(atoms, R):
    frames = []
    for r in R:
        a = atoms.copy()
        a.set_positions(r)
        frames.append(a)
    return frames