import numpy as np


def maxwellian(V):
    """
    Normalized 1D Maxwellian equilibrium distribution.
    """

    return (
        1 / np.sqrt(2 * np.pi)
        * np.exp(-V**2 / 2)
    )


def perturbed_maxwellian(V, X, alpha, k):
    """
    Maxwellian equilibrium with a small spatial perturbation.

    f(x,v,0) = f0(v) * [1 + alpha*cos(k*x)]
    """

    f0 = maxwellian(V)

    return f0 * (
        1 + alpha * np.cos(k * X)
    )