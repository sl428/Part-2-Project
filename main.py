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
cutoff = np.inf
temperature = 0
extra_energy = 5.0                # eV, total added energy, split equally over all modes
debug = False

def sample_energies(atoms, omega, modes, calc, E0, tag, extra_energy):
    rng = np.random.default_rng(seed)
    q, p = sample_qp(omega, n_samples, rng, cutoff, temperature, extra_energy)
    R, P = to_cartesian(atoms, modes, q, p)
    frames = to_frames(atoms, R, P)
    write(f"{tag}_sampling_geo.extxyz", frames)

    E_pot, E_kin, E_tot = total_energies(frames, calc)
    np.savez(f"{tag}_energy.npz", E0=E0, E_pot=E_pot, E_kin=E_kin, E_tot=E_tot,
             omega=omega, q=q, p=p)
    return q, p, E_tot

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

    calc = FAIRChemCalculator(predictor, task_name="omol")
    tag = f"{name}_T{temperature:g}K_E{extra_energy:g}eV"
    q, p, E_tot = sample_energies(atoms, omega, modes, calc, energy, tag, extra_energy)
    plot_energy(E_tot, energy, f"{tag}_energy.png")

    base_tag = f"{name}_T{temperature:g}K_E0eV"
    _, _, E_base = sample_energies(atoms, omega, modes, calc, energy, base_tag, 0.0)

    before = (E_base - energy).mean()
    after = (E_tot - energy).mean()
    print(f"<E_tot - E0>  before (E = 0 eV):   {before:.4f} eV")
    print(f"<E_tot - E0>  after  (E = {extra_energy:g} eV):   {after:.4f} eV")
    print(f"difference:   {after - before:.4f} eV")

    if debug:
        plot_sampling(omega, q, sigma_q(omega, temperature, extra_energy), "q", f"{tag}_check_q.png")
        plot_sampling(omega, p, sigma_p(omega, temperature, extra_energy), "p", f"{tag}_check_p.png")

if __name__ == "__main__":
    main()
