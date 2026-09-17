import numpy as np


def total_mass(f, dx, dv):
    """
    Total particle number:
        N = ∫∫ f(x,v) dx dv
    """
    return np.sum(f) * dx * dv


def total_momentum(f, V, dx, dv):
    """
    Total momentum:
        P = ∫∫ v f(x,v) dx dv
    """
    return np.sum(V * f) * dx * dv


def kinetic_energy(f, V, dx, dv):
    """
    Particle kinetic energy:
        K = ∫∫ (1/2) v^2 f(x,v) dx dv
    """
    return 0.5 * np.sum(V**2 * f) * dx * dv


def field_energy(E, dx):
    """
    Electrostatic field energy:
        W_E = ∫ (1/2) E^2 dx
    """
    return 0.5 * np.sum(E**2) * dx


def total_energy(f, V, E, dx, dv):
    """
    Total energy:
        H = K + W_E
    """
    return kinetic_energy(f, V, dx, dv) + field_energy(E, dx)


def conservation_drift(values):
    """
    Relative drift from the initial value.

    Returns:
        absolute drift
        relative drift
    """
    values = np.asarray(values)

    initial = values[0]
    final = values[-1]

    absolute_drift = final - initial

    if np.abs(initial) > 1e-14:
        relative_drift = absolute_drift / np.abs(initial)
    else:
        relative_drift = np.nan

    return absolute_drift, relative_drift