# main.py
import os

from fairchem.core import FAIRChemCalculator
from fairchem.core.units.mlip_unit import load_predict_unit
from fairchem.core.units.mlip_unit.api.inference import InferenceSettings
from ase.io import read, write
import numpy as np

from optimize import minimize_gd
from vibfreq import vib_freq
from sampling import sigma_q, sigma_p, sample_gaussian, q_to_R, p_to_P, to_frames
from sampling_plot import plot_sampling


filename = 'nitrogen.xyz'
charge = 0
spin = 1
model = "/u/ajmk/applications/omol25/esen_sm_conserving_all.pt"
n_samples = 1000
seed = 0
momenta = False     
debug = True        

def main():
    predictor = load_predict_unit(model, device='cpu')
    atoms = read(filename)
    atoms.info.update({"charge": charge, "spin": spin})
    atoms.calc = FAIRChemCalculator(predictor, task_name="omol")

    energy = minimize_gd(atoms)
    name, ext = os.path.splitext(filename)
    write(f"{name}_opt{ext}", atoms)

    hess_settings = InferenceSettings(predict_untrained_hessian={"omol"})
    hess_predictor = load_predict_unit(model, inference_settings=hess_settings, device='cpu')
    atoms.calc = FAIRChemCalculator(hess_predictor, task_name="omol")

    omega, modes = vib_freq(atoms)

    rng = np.random.default_rng(seed)       
    q = sample_gaussian(sigma_q(omega), n_samples, rng)      # (n_samples, n_modes)
    R = q_to_R(atoms, modes, q)                              # (n_samples, n_atoms, 3)
    if momenta:
        p = sample_gaussian(sigma_p(omega), n_samples, rng)
        P = p_to_P(atoms, modes, p)
    else:
        P = None

    write(f"{name}_sampling_geo.extxyz", to_frames(atoms, R, P))

    if debug:
        plot_sampling(omega, q, sigma_q(omega), "q", f"{name}_check_q.png")
        if momenta:
            plot_sampling(omega, p, sigma_p(omega), "p", f"{name}_check_p.png")


if __name__ == "__main__":
    main()