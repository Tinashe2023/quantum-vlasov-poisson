import numpy as np


def calculate_electric_field(f, dv, kx, ion_density=1.0):
    """
    Calculate the self-consistent electric field
    from the distribution function.

    Uses a Fourier-space solution of Poisson's equation
    for a periodic spatial domain.
    """

    # Integrate over velocity
    density = np.sum(f, axis=1) * dv

    # Charge density
    rho = ion_density - density

    # Fourier transform
    rho_hat = np.fft.fft(rho)

    # Potential in Fourier space
    phi_hat = np.zeros_like(
        rho_hat,
        dtype=complex
    )

    nonzero = kx != 0

    phi_hat[nonzero] = (
        rho_hat[nonzero]
        / kx[nonzero]**2
    )

    # E = -d(phi)/dx
    E_hat = -1j * kx * phi_hat

    E = np.real(
        np.fft.ifft(E_hat)
    )

    return E, rho, phi_hat