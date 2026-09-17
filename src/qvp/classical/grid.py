import numpy as np


def create_grid(L, v_min, v_max, Nx, Nv):
    """
    Create the spatial and velocity grids for a 1D1V
    Vlasov-Poisson simulation.
    """

    x = np.linspace(0, L, Nx, endpoint=False)
    v = np.linspace(v_min, v_max, Nv)

    dx = L / Nx
    dv = (v_max - v_min) / (Nv - 1)

    X, V = np.meshgrid(x, v, indexing="ij")

    kx = 2 * np.pi * np.fft.fftfreq(Nx, d=dx)

    return x, v, X, V, dx, dv, kx