import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

import numpy as np
from scipy.signal import find_peaks
from scipy.stats import linregress

from qvp.classical.grid import create_grid
from qvp.physics.distributions import perturbed_maxwellian
from qvp.classical.solver import rk4_step
from qvp.classical.poisson import calculate_electric_field

from qvp.analysis.observables import (
    total_mass,
    total_momentum,
    total_energy,
)


def run_landau_damping(
    Nx,
    Nv,
    alpha=0.01,
    k=0.5,
    v_min=-6.0,
    v_max=6.0,
    dt=0.01,
    t_final=20.0,
    fit_damping=True,
):
    """
    Run a Landau-damping simulation and estimate
    the damping rate from electric-field peaks.
    """

    L = 2 * np.pi / k

    x, v, X, V, dx, dv, kx = create_grid(
        L,
        v_min,
        v_max,
        Nx,
        Nv,
    )

    f = perturbed_maxwellian(
        V,
        X,
        alpha,
        k,
    )

    E, _, _ = calculate_electric_field(f, dv, kx)

    mass_history = []
    momentum_history = []
    energy_history = []

    mass_history.append(total_mass(f, dx, dv))
    momentum_history.append(total_momentum(f, V, dx, dv))
    energy_history.append(total_energy(f, V, E, dx, dv))

    num_steps = int(t_final / dt)

    times = []
    field_amplitudes = []

    for step in range(num_steps):

        t = step * dt

        E, _, _ = calculate_electric_field(
            f,
            dv,
            kx,
        )

        times.append(t)
        field_amplitudes.append(
            np.max(np.abs(E))
        )
        if step % 100 == 0:
            print(
                f"t={t:.4f}, "
                f"max|f|={np.max(np.abs(f)):.6e}, "
                f"max|E|={np.max(np.abs(E)):.6e}"
            )

        f = rk4_step(
            f,
            dt,
            V,
            kx,
            dv,
        )
        E, _, _ = calculate_electric_field(f, dv, kx)

        mass_history.append(total_mass(f, dx, dv))
        momentum_history.append(total_momentum(f, V, dx, dv))
        energy_history.append(total_energy(f, V, E, dx, dv))

    if not np.all(np.isfinite(f)):
        raise RuntimeError(
            f"Non-finite distribution detected at t={t:.6f}"
        )

    times = np.array(times)
    field_amplitudes = np.array(field_amplitudes)

    # Detect electric-field peaks
    peaks, _ = find_peaks(
        field_amplitudes,
        distance=20,
    )

    peak_times = times[peaks]
    peak_values = field_amplitudes[peaks]
    if len(peak_values) < 3 or not fit_damping:
        gamma = np.nan
        r_squared = np.nan

    else:
        log_peaks = np.log(peak_values)

        fit = linregress(
            peak_times,
            log_peaks,
        )

        gamma = -fit.slope
        r_squared = fit.rvalue ** 2

    return {
        "Nx": Nx,
        "Nv": Nv,
        "gamma": gamma,
        "r_squared": r_squared,
        "times": times,
        "field_amplitudes": field_amplitudes,
        "peak_times": peak_times,
        "peak_values": peak_values,
        "f_final": f,
        "x": x,
        "v": v,
        "mass": np.array(mass_history),
        "momentum": np.array(momentum_history),
        "energy": np.array(energy_history),
    }