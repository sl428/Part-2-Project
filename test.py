import numpy as np
from ase import units
from energy import harmonic_energies
from sampling import HBAR

d = np.load("furan_T0K_E5eV_energy.npz")
H_pot, H_kin = harmonic_energies(d["omega"], d["q"], d["p"])
zpe = (HBAR * d["omega"] / units.second / 2).sum()

print("ZPE:", zpe)
print("<E_pot>  harmonic:", (H_pot).mean())
print("<E_pot>  OMol:    ", (d["E_pot"] - d["E0"]).mean())
print("<E_tot - ZPE>  harmonic:", (H_pot + H_kin).mean() - zpe)
print("<E_tot - ZPE>  OMol:    ", (d["E_tot"] - d["E0"]).mean() - zpe)
