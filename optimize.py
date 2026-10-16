from ase.io import write
import os
import numpy as np


def gradient(atoms):
    return -atoms.get_forces()

#parameters
#epsilon: the step-size parameter
#fmax: maximum force tolerance for convergence (eV/Å)
def minimize_gd(atoms, epsilon=0.001, fmax=0.005, max_steps=50000):
    for i in range(max_steps):
        g = gradient(atoms)
        if np.abs(g).max() < fmax:
            break
        atoms.set_positions(atoms.get_positions() - epsilon * g)

    return atoms.get_potential_energy()