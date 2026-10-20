# main.py
import os

from fairchem.core import FAIRChemCalculator
from fairchem.core.units.mlip_unit import load_predict_unit
from fairchem.core.units.mlip_unit.api.inference import InferenceSettings
from ase.io import read, write
import numpy as np

from optimize import minimize_gd
from vibfreq import vib_freq
from sampling import sigma_q, sigma_p, sample_qp, to_cartesian, to_frames
from sampling_plot import plot_sampling
from energy import total_energies, plot_energy


filename = 'furan.xyz'
charge = 0
spin = 1
model = "/u/ajmk/applications/omol25/esen_sm_conserving_all.pt"
n_samples = 1000
seed = 0
cutoff = 2.0
temperature = 300.0
debug = True

def main():
    predictor = load_predict_unit(model, device='cpu')
    atoms = read(filename)
    atoms.info.clear()
    atoms.info.update({"charge": charge, "spin": spin})
    atoms.calc = FAIRChemCalculator(predictor, task_name="omol")

    energy = minimize_gd(atoms)
    print(energy)
    name, ext = os.path.splitext(filename)
    write(f"{name}_opt{ext}", atoms)

    hess_settings = InferenceSettings(predict_untrained_hessian={"omol"})
    hess_predictor = load_predict_unit(model, inference_settings=hess_settings, device='cpu')
    atoms.calc = FAIRChemCalculator(hess_predictor, task_name="omol")

    omega, modes = vib_freq(atoms)

    rng = np.random.default_rng(seed)
    q, p = sample_qp(omega, n_samples, rng, cutoff, temperature)
    R, P = to_cartesian(atoms, modes, q, p)
    frames = to_frames(atoms, R, P)
    write(f"{name}_sampling_geo.extxyz", frames)

    E_pot, E_kin, E_tot = total_energies(frames, FAIRChemCalculator(predictor, task_name="omol"))
    np.savez(f"{name}_energy.npz", E_pot=E_pot, E_kin=E_kin, E_tot=E_tot)
    plot_energy(E_tot, f"{name}_energy.png")

    if debug:
        plot_sampling(omega, q, sigma_q(omega, temperature), "q", f"{name}_check_q.png")
        plot_sampling(omega, p, sigma_p(omega, temperature), "p", f"{name}_check_p.png")

if __name__ == "__main__":
    main()
