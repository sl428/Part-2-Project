"""
vibfreq.py - Hessian, mass-weighted Hessian and vibrational frequencies.

Workflow
  1. Solve Cartesian Hessian H
  2. Mass-weight
  3. Project out the 3 translations + 3 rotations (2 for linear molecules)
  4. Diagonalise H_mw; eigenvalues -> frequencies in cm^-1
"""

import numpy as np
from ase import units


def gradient(atoms, coords):
    atoms.set_positions(coords)
    return -atoms.get_forces()

def hessian(atoms, step=0.005):
    x0 = atoms.get_positions().copy() #(n, 3)
    n_coords = x0.size
    moves = np.eye(n_coords).reshape(n_coords, *x0.shape) * step #	(3n, n, 3)
    H = []

    for d in moves:
        diff = gradient(atoms, x0 + d) - gradient(atoms, x0 - d)
        H.append(diff.ravel() / (2 * step))  #(3n,)

    atoms.set_positions(x0) # restore the coords
    H = np.array(H) #(3n, 3n)
    return (H + H.T) / 2

def mass_weight(H, masses):
    m = np.repeat(masses, 3)      # (3n,) [m₀, m₀, m₀, m₁, m₁, m₁, ...]
    H_mw = H / np.sqrt(np.outer(m, m))
    return H_mw

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

def vib_freq(H_mw, masses, n_remove):
    eigvals, eigvecs = np.linalg.eigh(H_mw)          # (3n,) and (3n, 3n); column k = mode k

    keep = np.argsort(np.abs(eigvals))[n_remove:]    # remove rot & trs mode
    keep = keep[np.argsort(eigvals[keep])]           # sort by signed value
    freqs = eigval_to_wavenumber(eigvals[keep])

    modes = eigvecs[:, keep].T.reshape(-1, len(masses), 3) #(n_modes, n, 3)
    return freqs, modes

def run_vibfreq(atoms):
    masses = atoms.get_masses()
    H_mw = mass_weight(hessian(atoms), masses)
    return vib_freq(H_mw, masses, n_trans_rot(atoms))