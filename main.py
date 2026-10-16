# main.py
from fairchem.core import FAIRChemCalculator
from fairchem.core.units.mlip_unit import load_predict_unit
from ase.io import read
import numpy as np

from optimize import minimize_gd
from vibfreq import run_vibfreq


filename = 'furan.xyz'
charge = 0
spin = 1
model = "/u/ajmk/applications/omol25/esen_sm_conserving_all.pt"


def main():
    predictor = load_predict_unit(model, device='cpu')
    atoms = read(filename)
    atoms.info.update({"charge": charge, "spin": spin})
    atoms.calc = FAIRChemCalculator(predictor, task_name="omol")

    energy = minimize_gd(atoms)
    print(energy)
    name, ext = os.path.splitext(filename)
    write(f"{name}_opt{ext}", atoms)



    freqs, modes = run_vibfreq(atoms)
    np.set_printoptions(suppress=True, precision=2)
    print(freqs)
    print(modes)


if __name__ == "__main__":
    main()