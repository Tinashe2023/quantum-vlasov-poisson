import numpy as np


def spectral_derivative_x(f, kx):
    """
    Spectral derivative with respect to x.
    """

    f_hat = np.fft.fft(
        f,
        axis=0
    )

    df_dx_hat = (
        1j
        * kx[:, None]
        * f_hat
    )

    df_dx = np.real(
        np.fft.ifft(
            df_dx_hat,
            axis=0
        )
    )

    return df_dx


def finite_difference_v(f, dv):
    """
    Centered finite-difference derivative
    with respect to velocity.
    """

    df_dv = np.zeros_like(f)

    df_dv[:, 1:-1] = (
        f[:, 2:]
        - f[:, :-2]
    ) / (2 * dv)

    # Forward difference
    df_dv[:, 0] = (
        f[:, 1]
        - f[:, 0]
    ) / dv

    # Backward difference
    df_dv[:, -1] = (
        f[:, -1]
        - f[:, -2]
    ) / dv

    return df_dv


def vlasov_rhs(f, V, E, kx, dv):
    """
    Right-hand side of the normalized
    1D1V Vlasov equation.
    """

    df_dx = spectral_derivative_x(
        f,
        kx
    )

    df_dv = finite_difference_v(
        f,
        dv
    )

    rhs = (
        -V * df_dx
        + E[:, None] * df_dv
    )

    return rhs