"""
vibfreq.py - vibrational frequencies from the fairchem Hessian.

Workflow
  1. Cartesian Hessian H from fairchem (exact, via autograd), mass-weighted
  2. Remove the 3 translations + 3 rotations (2 for linear molecules)
  3. Diagonalise H_mw; eigenvalues -> omega (rad/s) and wavenumbers (cm^-1)

Usage
    freqs, omega, modes = vib_freq(atoms)
"""

import numpy as np
from ase import units


def mass_weighted_hessian(atoms):
    """atoms.calc must be loaded with InferenceSettings(predict_untrained_hessian={"omol"})."""
    H = atoms.calc.get_property("hessian", atoms).reshape(3 * len(atoms), -1)   # (3n, 3n) eV/Å²
    m = np.repeat(atoms.get_masses(), 3)      # (3n,) [m₀, m₀, m₀, m₁, m₁, m₁, ...]
    return H / np.sqrt(np.outer(m, m))        # eV/(Å²·amu)

def n_trans_rot(atoms):
    I = atoms.get_moments_of_inertia() # shape (3,)
    return 5 if I.min() < 1e-3 * I.max() else 6

def eigval_to_omega(eigvals):
    """Eigenvalues of H_mw, eV/(Å²·amu)  ->  angular frequency omega, rad/s."""
    lam = eigvals * units._e / (units._amu * 1e-20)        # -> s⁻²
    return np.sign(lam) * np.sqrt(np.abs(lam))             # -> rad/s

def omega_to_wavenumber(omega):
    """omega, rad/s  ->  wavenumbers, cm⁻¹."""
    return omega / (2 * np.pi * units._c * 100)

def vib_freq(atoms):
    eigvals, eigvecs = np.linalg.eigh(mass_weighted_hessian(atoms)) 

    keep = np.argsort(np.abs(eigvals))[n_trans_rot(atoms):]   # remove rot & trs mode
    keep = keep[np.argsort(eigvals[keep])]                     # sort by signed value

    omega = eigval_to_omega(eigvals[keep])           # (n_modes,) rad/s
    #freqs = omega_to_wavenumber(omega)               # (n_modes,) cm⁻¹
    modes = eigvecs[:, keep].T.reshape(-1, len(atoms), 3) #(n_modes, n, 3)
    return omega, modes
