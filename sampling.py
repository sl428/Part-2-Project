"""
sampling.py - Wigner sampling of positions from normal modes.

Everything is in ASE units (eV, Å, amu, ASE time)。
"""

import numpy as np
from ase import units

HBAR = units._hbar * units.J * units.second      # ħ in eV·(ASE time)


def wigner_sample(atoms, omega, modes, n_samples, seed=None):
    """
    omega : (n_modes,)       rad/s, from vib_freq
    modes : (n_modes, n, 3)  mass-weighted orthonormal eigenvectors, from vib_freq

    Returns R : (n_samples, n, 3) sampled positions, Å
    """

    w = omega / units.second                     # rad/s -> 1/(ASE time)
    sigma_q = np.sqrt(HBAR / (2 * w))           

    rng = np.random.default_rng(seed)
    q = rng.standard_normal((n_samples, w.size)) * sigma_q     # (n_samples, n_modes)

    sqrt_m = np.sqrt(atoms.get_masses())[None, :, None]
    return atoms.get_positions()[None] + np.einsum("si,iaj->saj", q, modes) / sqrt_m #(n_samples, n_atoms, 3)


def to_frames(atoms, R):
    frames = []
    for r in R:
        a = atoms.copy()                        
        a.set_positions(r)
        frames.append(a)
    return frames