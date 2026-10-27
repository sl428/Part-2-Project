"""
sampling.py - Wigner sampling of positions and momenta from normal modes.

Everything is in ASE units (eV, Å, amu, ASE time)
"""

import numpy as np
from ase import units
from scipy.stats import truncnorm

HBAR = units._hbar * units.J * units.second      # ħ in eV·(ASE time)

def thermal_factor(omega, temperature):
    if temperature == 0:
        return np.ones_like(omega)
    w = omega / units.second
    return 1 / np.tanh(HBAR * w / (2 * units.kB * temperature))   # ħω [eV] / k_B·T [eV] -> unitless

def width_factor(omega, temperature, extra_energy):
    w = omega / units.second
    eps = extra_energy / np.count_nonzero(omega > 0)              # eV per mode
    return thermal_factor(omega, temperature) + 2 * eps / (HBAR * w)   # eps [eV] / ħω [eV] -> unitless

def sigma_q(omega, temperature, extra_energy):
    w = omega / units.second                     # rad/s -> 1/(ASE time)
    return np.sqrt(HBAR / (2 * w) * width_factor(omega, temperature, extra_energy))

def sigma_p(omega, temperature, extra_energy):
    w = omega / units.second
    return np.sqrt(HBAR * w / 2 * width_factor(omega, temperature, extra_energy))

def sample_qp(omega, n_samples, rng, cutoff, temperature, extra_energy):
    xi = truncnorm.rvs(-cutoff, cutoff, size=(2, n_samples, omega.size), random_state=rng)
    return (xi[0] * sigma_q(omega, temperature, extra_energy),
            xi[1] * sigma_p(omega, temperature, extra_energy))

def to_cartesian(atoms, modes, q, p):
    sqrt_m = np.sqrt(atoms.get_masses())[None, :, None]
    R = atoms.get_positions()[None] + np.einsum("si,iaj->saj", q, modes) / sqrt_m
    P = np.einsum("si,iaj->saj", p, modes) * sqrt_m
    return R, P

def to_frames(atoms, R, P):
    frames = []
    for i, r in enumerate(R):
        a = atoms.copy()                      
        a.set_positions(r)
        a.set_momenta(P[i])
        frames.append(a)
    return frames