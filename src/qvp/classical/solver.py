from .poisson import calculate_electric_field
from .vlasov import vlasov_rhs


def rk4_step(f, dt, V, kx, dv):
    """
    Perform one fourth-order Runge-Kutta
    time step of the Vlasov-Poisson system.
    """

    E1, _, _ = calculate_electric_field(
        f,
        dv,
        kx
    )

    k1 = vlasov_rhs(
        f,
        V,
        E1,
        kx,
        dv
    )

    f2 = f + 0.5 * dt * k1

    E2, _, _ = calculate_electric_field(
        f2,
        dv,
        kx
    )

    k2 = vlasov_rhs(
        f2,
        V,
        E2,
        kx,
        dv
    )

    f3 = f + 0.5 * dt * k2

    E3, _, _ = calculate_electric_field(
        f3,
        dv,
        kx
    )

    k3 = vlasov_rhs(
        f3,
        V,
        E3,
        kx,
        dv
    )

    f4 = f + dt * k3

    E4, _, _ = calculate_electric_field(
        f4,
        dv,
        kx
    )

    k4 = vlasov_rhs(
        f4,
        V,
        E4,
        kx,
        dv
    )

    f_new = f + (
        dt / 6.0
    ) * (
        k1
        + 2 * k2
        + 2 * k3
        + k4
    )

    return f_new