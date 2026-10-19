"""
sampling.py - Wigner sampling of positions and momenta from normal modes.

Everything is in ASE units (eV, Å, amu, ASE time)。
"""

import numpy as np
from ase import units

HBAR = units._hbar * units.J * units.second      # ħ in eV·(ASE time)


def sigma_q(omega):
    w = omega / units.second                     # rad/s -> 1/(ASE time)
    return np.sqrt(HBAR / (2 * w))

def sigma_p(omega):
    w = omega / units.second
    return np.sqrt(HBAR * w / 2)

def sample_gaussian(sigma, n_samples, rng):
    """(n_samples, n_modes) Gaussian samples with per-mode width sigma.
    Pass the SAME rng for q and p so they are independent."""
    return rng.standard_normal((n_samples, sigma.size)) * sigma

def q_to_R(atoms, modes, q):
    """Positions, (n_samples, n_atoms, 3), Å."""
    sqrt_m = np.sqrt(atoms.get_masses())[None, :, None]
    return atoms.get_positions()[None] + np.einsum("si,iaj->saj", q, modes) / sqrt_m

def p_to_P(atoms, modes, p):
    """Momenta, (n_samples, n_atoms, 3), amu·Å/(ASE time)."""
    sqrt_m = np.sqrt(atoms.get_masses())[None, :, None]
    return np.einsum("si,iaj->saj", p, modes) * sqrt_m

def to_frames(atoms, R, P=None):
    frames = []
    for i, r in enumerate(R):
        a = atoms.copy()                      
        a.set_positions(r)
        if P is not None:
            a.set_momenta(P[i])
        frames.append(a)
    return frames