# main.py
import os

from fairchem.core import FAIRChemCalculator
from fairchem.core.units.mlip_unit import load_predict_unit
from fairchem.core.units.mlip_unit.api.inference import InferenceSettings
from ase.io import read, write
import numpy as np

from optimize import minimize_gd
from vibfreq import vib_freq
from sampling import wigner_sample, to_frames


filename = 'nitrogen.xyz'
charge = 0
spin = 1
model = "/u/ajmk/applications/omol25/esen_sm_conserving_all.pt"
n_samples = 1000                                 
seed = 0

def main():
    predictor = load_predict_unit(model, device='cpu')
    atoms = read(filename)
    atoms.info.update({"charge": charge, "spin": spin})
    atoms.calc = FAIRChemCalculator(predictor, task_name="omol")

    energy = minimize_gd(atoms)
    #print(energy)
    name, ext = os.path.splitext(filename)
    write(f"{name}_opt{ext}", atoms)
    R_opt = atoms.get_positions().copy()

    hess_settings = InferenceSettings(predict_untrained_hessian={"omol"})
    hess_predictor = load_predict_unit(model, inference_settings=hess_settings, device='cpu')
    atoms.calc = FAIRChemCalculator(hess_predictor, task_name="omol")

    omega, modes = vib_freq(atoms)
    #np.set_printoptions(suppress=True, precision=2)

    R = wigner_sample(atoms, omega, modes, n_samples=n_samples, seed=seed) 
    write(f"{name}_sampling_geo.extxyz", to_frames(atoms, R))               



if __name__ == "__main__":
    main()