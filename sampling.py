import numpy as np

def wigner_geometry(omega, modes, masses, R_eq, n_samples=1000, seed=None):
    omega = np.asarray(omega)
    masses = np.asarray(masses)

    rng = np.random.default_rng(seed)

    # 1. Sample normal-mode amplitudes
    q = rng.standard_normal((n_samples, len(omega)))
    q /= np.sqrt(2 * omega)

    # 2. Transform into Cartesian displacements
    displacement = np.einsum("si,iaj->saj", q, modes)
    displacement /= np.sqrt(masses)[None, :, None]

    # 3. Add equilibrium geometry
    geometries = R_eq[None, :, :] + displacement

    return geometries