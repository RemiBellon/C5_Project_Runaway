"""Time integrators for the motion of charged particles: one step of a pusher."""

from collections.abc import Callable
from numbers import Real

import numpy as np
from numpy.typing import NDArray

from ppf_c5_runaways.constants import c
from ppf_c5_runaways.fields import Field

# The interface shared by all pushers: push(x, u, field, t, dt, q, m) -> (x_new, u_new).
type Pusher = Callable[
    [NDArray[np.float64], NDArray[np.float64], Field, float, float, float, float],
    tuple[NDArray[np.float64], NDArray[np.float64]],
]


# --- helpers shared by all the pushers ---------------------------------------------


def _check_push_arguments(
    x: NDArray[np.float64],
    u: NDArray[np.float64],
    t: float,
    dt: float,
    q: float,
    m: float,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Check the arguments of a pusher, before anything is computed.

    Every pusher starts with this call, so they all refuse the same wrong inputs
    with the same messages. The field is checked separately, when it is evaluated
    (see ``_evaluate_field``).

    Parameters
    ----------
    x, u, t, dt, q, m
        The arguments of a pusher, see ``boris``.

    Returns
    -------
    x, u : ndarray, shape (n_part, 3)
        The positions [m] and proper velocities [m/s] as ``float64`` arrays.

    Raises
    ------
    ValueError
        If ``x`` or ``u`` does not have shape ``(n_part, 3)``, if ``u`` and ``x``
        differ in shape, or if ``t``, ``dt``, ``q`` or ``m`` is not a finite real
        number in its range. The message says what was expected and what was
        received.
    """
    x = np.asarray(x, dtype=np.float64)
    u = np.asarray(u, dtype=np.float64)
    if x.ndim != 2 or x.shape[1] != 3:
        raise ValueError(
            "x must have shape (n_part, 3), one row per particle; "
            f"got shape {x.shape}. For a single particle use shape (1, 3), "
            "for example x[np.newaxis, :]."
        )
    if u.shape != x.shape:
        raise ValueError(
            f"u must have the same shape as x, {x.shape}, one row per particle; "
            f"got shape {u.shape}."
        )
    if not isinstance(t, Real) or not np.isfinite(t):
        raise ValueError(f"t must be a finite number [s], got {t!r}")
    if not isinstance(dt, Real) or not np.isfinite(dt) or dt == 0.0:
        raise ValueError(
            "dt must be a finite, non-zero number [s] (a negative dt steps "
            f"backward in time), got {dt!r}"
        )
    if not isinstance(q, Real) or not np.isfinite(q):
        raise ValueError(f"q must be a finite number [C], got {q!r}")
    if not isinstance(m, Real) or not np.isfinite(m) or m <= 0.0:
        raise ValueError(
            f"m must be a finite, strictly positive number [kg], got {m!r}"
        )
    return x, u


def _evaluate_field(
    field: Field, x: NDArray[np.float64], t: float
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Evaluate the field once at ``(x, t)`` and check the shape of what it returns.

    Parameters
    ----------
    field : Field
        Function ``field(x, t) -> (E, B)`` (see ``ppf_c5_runaways.fields``).
    x : ndarray, shape (n_part, 3)
        Positions [m] at which the field is wanted.
    t : float
        Time [s] at which the field is wanted.

    Returns
    -------
    e_field, b_field : ndarray, shape (n_part, 3)
        The electric [V/m] and magnetic [T] fields at ``(x, t)``.

    Raises
    ------
    ValueError
        If the field returns an array whose shape is not that of ``x``.
    """
    e_field, b_field = field(x, t)
    for name, array in (("E", e_field), ("B", b_field)):
        if np.shape(array) != x.shape:
            raise ValueError(
                f"field(x, t) must return {name} of shape {x.shape}, the shape of x; "
                f"got shape {np.shape(array)}."
            )
    return e_field, b_field


def _lorentz_factor(u: NDArray[np.float64]) -> NDArray[np.float64]:
    """Return ``gamma = sqrt(1 + |u|^2 / c^2)`` of each particle.

    Parameters
    ----------
    u : ndarray, shape (n_part, 3)
        Proper velocities [m/s].

    Returns
    -------
    ndarray, shape (n_part, 1)
        The Lorentz factors. The last axis has length 1, so that the result
        broadcasts against the 3 components of a vector of shape ``(n_part, 3)``.
    """
    return np.sqrt(1.0 + np.sum(u**2, axis=1, keepdims=True) / c**2)


def _gamma_from_quartic(
    a: NDArray[np.float64], b: NDArray[np.float64]
) -> NDArray[np.float64]:
    """Return the positive ``gamma`` that solves ``gamma^4 - a gamma^2 - b = 0``.

    Higuera & Cary (2017, Eq. 20) and Vay (2008, Eq. 11) both find the Lorentz
    factor at the end of the step as the positive root of this equation; only the
    coefficients ``a`` and ``b`` differ. The two roots in ``gamma^2`` are
    ``(a +/- sqrt(a^2 + 4 b)) / 2``, and the physical one is the ``+``.

    The formula is written so that no two nearly equal numbers are subtracted. For
    ``a >= 0`` the ``+`` root is computed as it stands. For ``a < 0`` (a very large
    ``dt * B``) ``a + sqrt(a^2 + 4 b)`` would cancel, so the same root is computed
    as ``b`` divided by the larger root in absolute value, which is exact
    (the product of the two roots is ``-b``).

    Parameters
    ----------
    a : ndarray, shape (n_part, 1)
        Coefficient of ``gamma^2``, real.
    b : ndarray, shape (n_part, 1)
        Constant term, non-negative, and strictly positive whenever ``a < 0``.

    Returns
    -------
    ndarray, shape (n_part, 1)
        ``gamma``, dimensionless, at least 1 for physical inputs.
    """
    larger = 0.5 * (np.abs(a) + np.sqrt(a**2 + 4.0 * b))  # > 0: no division by zero
    gamma_squared = np.where(a >= 0.0, larger, b / larger)
    return np.sqrt(gamma_squared)


# --- the pushers -----------------------------------------------------------------


def boris(
    x: NDArray[np.float64],
    u: NDArray[np.float64],
    field: Field,
    t: float,
    dt: float,
    q: float,
    m: float,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Advance the particles by one time step with the classic Boris scheme.

    Positions and velocities are known at the same time: the step takes
    ``(x^n, u^n)`` at ``t`` and returns ``(x^{n+1}, u^{n+1})`` at ``t + dt``
    (synchronized leapfrog of Ripperda et al. 2018, Eqs. 5-8). No half step is needed
    to start a run, and ``u`` can be used as it is in any diagnostic at ``t``.

    Parameters
    ----------
    x : ndarray, shape (n_part, 3)
        Positions ``x^n`` [m] at time ``t``.
    u : ndarray, shape (n_part, 3)
        Proper velocities ``u^n = gamma * v`` [m/s] at time ``t``.
    field : Field
        Function ``field(x, t) -> (E, B)``, each of shape ``(n_part, 3)``, in V/m
        and T (see ``ppf_c5_runaways.fields``). It is called once, at the middle of
        the step: ``(x^{n+1/2}, t + dt / 2)``.
    t : float
        Time [s] of ``x`` and ``u``.
    dt : float
        Time step [s], finite and non-zero. A negative ``dt`` steps backward.
    q : float
        Signed charge [C], for example ``-e`` for an electron. The sign sets the
        sense of gyration, so it is never replaced by an absolute value.
    m : float
        Rest mass [kg], strictly positive.

    Returns
    -------
    x_new : ndarray, shape (n_part, 3)
        Positions ``x^{n+1}`` [m] at time ``t + dt``.
    u_new : ndarray, shape (n_part, 3)
        Proper velocities ``u^{n+1}`` [m/s] at time ``t + dt``.

        Both are new arrays: ``x`` and ``u`` are not modified.

    Raises
    ------
    ValueError
        If ``x`` or ``u`` does not have shape ``(n_part, 3)``, if ``u`` and ``x``
        differ in shape, if the field returns arrays of another shape, or if
        ``t``, ``dt``, ``q`` or ``m`` is not a finite real number in its range. The
        message says what was expected and what was received.

    Notes
    -----
    One step (Ripperda et al. 2018, Eqs. 6, 10-12 and 8), with
    ``a = q dt E / (2 m)`` and ``E``, ``B`` taken at ``(x_half, t + dt / 2)``::

        x_half = x + (dt / 2) u / gamma(u)           first half of the move
        u- = u + a                                   half acceleration by E
        gamma- = sqrt(1 + |u-|^2 / c^2)
        tvec = q dt B / (2 m gamma-)                 (called ``t`` by Ripperda)
        svec = 2 tvec / (1 + |tvec|^2)               (called ``s`` by Ripperda)
        u+ = u- + (u- + u- x tvec) x svec            rotation by B, |u+| = |u-|
        u_new = u+ + a                               half acceleration by E
        x_new = x_half + (dt / 2) u_new / gamma(u_new)    second half of the move

    The step is symmetric in time: applying it with ``-dt`` to ``(x_new, u_new)`` at
    ``t + dt`` gives back ``(x, u)``, up to rounding.
    """
    x, u = _check_push_arguments(x, u, t, dt, q, m)

    # First half of the move, with the velocity at the start of the step.
    x_half = x + 0.5 * dt * u / _lorentz_factor(u)
    e_field, b_field = _evaluate_field(field, x_half, t + 0.5 * dt)

    # Half acceleration by E, shape (n_part, 3) [m/s].
    half_kick = q * dt / (2.0 * m) * e_field
    u_minus = u + half_kick

    # Rotation by B. gamma_minus has shape (n_part, 1) so that it broadcasts on the
    # 3 components. tvec and svec are t and s of Ripperda et al. (2018); ``t`` is
    # already the name of the time here.
    gamma_minus = _lorentz_factor(u_minus)
    tvec = q * dt / (2.0 * m) * b_field / gamma_minus
    svec = 2.0 * tvec / (1.0 + np.sum(tvec**2, axis=1, keepdims=True))
    u_prime = u_minus + np.cross(u_minus, tvec)
    u_plus = u_minus + np.cross(u_prime, svec)

    u_new = u_plus + half_kick

    # Second half of the move, with the velocity at the end of the step.
    x_new = x_half + 0.5 * dt * u_new / _lorentz_factor(u_new)
    return x_new, u_new


def higuera_cary(
    x: NDArray[np.float64],
    u: NDArray[np.float64],
    field: Field,
    t: float,
    dt: float,
    q: float,
    m: float,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Advance the particles by one time step with the Higuera-Cary scheme.

    It is the Boris scheme with one change: the magnetic rotation uses the Lorentz
    factor ``gamma_new`` of the *average* momentum of the step, from a closed
    formula, instead of ``gamma-``. Like Boris it preserves the phase-space volume
    and, in a pure magnetic field, ``|u|``; unlike Boris it also gives the exact
    E x B drift velocity at relativistic speed (Higuera & Cary 2017).

    As for ``boris``, positions and velocities are known at the same time: the step
    takes ``(x^n, u^n)`` at ``t`` and returns ``(x^{n+1}, u^{n+1})`` at ``t + dt``
    (synchronized leapfrog of Ripperda et al. 2018, Eqs. 5-8).

    Parameters
    ----------
    x : ndarray, shape (n_part, 3)
        Positions ``x^n`` [m] at time ``t``.
    u : ndarray, shape (n_part, 3)
        Proper velocities ``u^n = gamma * v`` [m/s] at time ``t``.
    field : Field
        Function ``field(x, t) -> (E, B)``, each of shape ``(n_part, 3)``, in V/m
        and T. It is called once, at the middle of the step:
        ``(x^{n+1/2}, t + dt / 2)``.
    t : float
        Time [s] of ``x`` and ``u``.
    dt : float
        Time step [s], finite and non-zero. A negative ``dt`` steps backward.
    q : float
        Signed charge [C], for example ``-e`` for an electron.
    m : float
        Rest mass [kg], strictly positive.

    Returns
    -------
    x_new : ndarray, shape (n_part, 3)
        Positions ``x^{n+1}`` [m] at time ``t + dt``.
    u_new : ndarray, shape (n_part, 3)
        Proper velocities ``u^{n+1}`` [m/s] at time ``t + dt``.

        Both are new arrays: ``x`` and ``u`` are not modified.

    Raises
    ------
    ValueError
        Same as ``boris``: wrong shapes, or ``t``, ``dt``, ``q``, ``m`` not finite
        real numbers in their range.

    Notes
    -----
    One step (Higuera & Cary 2017, Eqs. 10, 16, 18, 20, 21, 24, between the two
    half moves of Ripperda et al. 2018, Eqs. 6 and 8), with ``eps = q dt E / (2 m)``,
    ``beta = q dt B / (2 m)`` and ``E``, ``B`` taken at ``(x_half, t + dt / 2)``::

        x_half = x + (dt / 2) u / gamma(u)           first half of the move
        u- = u + eps                                 half acceleration by E
        gamma-^2 = 1 + |u-|^2 / c^2
        gamma_new^4 - (gamma-^2 - |beta|^2) gamma_new^2
                    - (|beta|^2 + (beta . u-)^2 / c^2) = 0        (Eq. 20)
        tvec = beta / gamma_new                      rotation by B, as in Boris
        svec = 2 tvec / (1 + |tvec|^2)
        u+ = u- + (u- + u- x tvec) x svec
        u_new = u+ + eps                             half acceleration by E
        x_new = x_half + (dt / 2) u_new / gamma(u_new)    second half of the move

    The paper writes Eq. 20 in units where ``c = 1``; the ``c`` above comes from
    writing every momentum as ``u / c``. Only the rotation differs from ``boris``:
    Boris would use ``gamma-`` where this uses ``gamma_new`` (``gamma- >= gamma_new``,
    Eq. 25). ``gamma_new`` is the Lorentz factor of the mean momentum
    ``(u + u_new) / 2`` of the step, not that of ``u_new``. Like ``boris``, the step
    is symmetric in time.
    """
    x, u = _check_push_arguments(x, u, t, dt, q, m)

    # First half of the move, with the velocity at the start of the step.
    x_half = x + 0.5 * dt * u / _lorentz_factor(u)
    e_field, b_field = _evaluate_field(field, x_half, t + 0.5 * dt)

    half_kick = q * dt / (2.0 * m) * e_field  # eps, Eq. 10
    u_minus = u + half_kick  # Eq. 18
    beta = q * dt / (2.0 * m) * b_field  # Eq. 16, dimensionless

    # Lorentz factor of the rotation: positive root of Eq. 20 (see _gamma_from_quartic).
    gamma_minus_squared = 1.0 + np.sum(u_minus**2, axis=1, keepdims=True) / c**2
    beta_squared = np.sum(beta**2, axis=1, keepdims=True)
    beta_dot_u = np.sum(beta * u_minus, axis=1, keepdims=True)
    gamma_new = _gamma_from_quartic(
        a=gamma_minus_squared - beta_squared,
        b=beta_squared + (beta_dot_u / c) ** 2,
    )

    # Rotation by B: the Boris rotation, with gamma_new in place of gamma-.
    tvec = beta / gamma_new
    svec = 2.0 * tvec / (1.0 + np.sum(tvec**2, axis=1, keepdims=True))
    u_prime = u_minus + np.cross(u_minus, tvec)
    u_plus = u_minus + np.cross(u_prime, svec)

    u_new = u_plus + half_kick

    # Second half of the move, with the velocity at the end of the step.
    x_new = x_half + 0.5 * dt * u_new / _lorentz_factor(u_new)
    return x_new, u_new


def vay(
    x: NDArray[np.float64],
    u: NDArray[np.float64],
    field: Field,
    t: float,
    dt: float,
    q: float,
    m: float,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Advance the particles by one time step with the relativistic Vay scheme.

    The magnetic force is computed with the *average of the velocities* at the
    start and at the end of the step, ``(v^i + v^{i+1}) / 2``, so that the electric
    and magnetic contributions to the Lorentz force cancel correctly when
    ``E + v x B = 0``. This gives the exact E x B drift velocity at relativistic
    speed, which Boris does not (Vay 2008). It does **not** preserve the
    phase-space volume, unlike Boris and Higuera-Cary (Higuera & Cary 2017, Sec. V).

    As for ``boris``, positions and velocities are known at the same time: the step
    takes ``(x^n, u^n)`` at ``t`` and returns ``(x^{n+1}, u^{n+1})`` at ``t + dt``
    (synchronized leapfrog of Ripperda et al. 2018, Eqs. 5-8).

    Parameters
    ----------
    x : ndarray, shape (n_part, 3)
        Positions ``x^n`` [m] at time ``t``.
    u : ndarray, shape (n_part, 3)
        Proper velocities ``u^n = gamma * v`` [m/s] at time ``t``.
    field : Field
        Function ``field(x, t) -> (E, B)``, each of shape ``(n_part, 3)``, in V/m
        and T. It is called once, at the middle of the step:
        ``(x^{n+1/2}, t + dt / 2)``.
    t : float
        Time [s] of ``x`` and ``u``.
    dt : float
        Time step [s], finite and non-zero. A negative ``dt`` steps backward.
    q : float
        Signed charge [C], for example ``-e`` for an electron.
    m : float
        Rest mass [kg], strictly positive.

    Returns
    -------
    x_new : ndarray, shape (n_part, 3)
        Positions ``x^{n+1}`` [m] at time ``t + dt``.
    u_new : ndarray, shape (n_part, 3)
        Proper velocities ``u^{n+1}`` [m/s] at time ``t + dt``.

        Both are new arrays: ``x`` and ``u`` are not modified.

    Raises
    ------
    ValueError
        Same as ``boris``: wrong shapes, or ``t``, ``dt``, ``q``, ``m`` not finite
        real numbers in their range.

    Notes
    -----
    Vay's paper puts the velocities at whole steps ``u^i`` and the positions at half
    steps ``x^{i+1/2}``. His ``u^i`` is our ``u^n``, his ``x^{i+1/2}`` our ``x_half``,
    and his ``E^{i+1/2}``, ``B^{i+1/2}`` are the fields at ``(x_half, t + dt / 2)``.
    The two half moves around his momentum update are those of Ripperda et al.
    (2018, Eqs. 6 and 8).

    One step (Vay 2008, Eqs. 9-12), with ``eps = q dt E / (2 m)``,
    ``tau = q dt B / (2 m)`` and ``E``, ``B`` taken at ``(x_half, t + dt / 2)``::

        x_half = x + (dt / 2) u / gamma(u)                  first half of the move
        u' = u + 2 eps + (u / gamma(u)) x tau               (Eq. 9)
        gamma_new^4 - (gamma'^2 - |tau|^2) gamma_new^2
                    - (|tau|^2 + u*^2) = 0                  (Eq. 11)
            with gamma'^2 = 1 + |u'|^2 / c^2 and u* = (u' . tau) / c
        tvec = tau / gamma_new,   s = 1 / (1 + |tvec|^2)
        u_new = s (u' + (u' . tvec) tvec + u' x tvec)       (Eq. 12)
        x_new = x_half + (dt / 2) u_new / gamma(u_new)      second half of the move

    ``u_new`` solves the implicit equation ``u_new = u' + (u_new / gamma_new) x tau``
    (Eq. 10); the explicit formulas above are its exact solution. Here ``gamma_new``
    is the Lorentz factor of ``u_new`` itself. Like ``boris``, the step is symmetric
    in time.
    """
    x, u = _check_push_arguments(x, u, t, dt, q, m)

    # First half of the move, with the velocity at the start of the step.
    x_half = x + 0.5 * dt * u / _lorentz_factor(u)
    e_field, b_field = _evaluate_field(field, x_half, t + 0.5 * dt)

    half_kick = q * dt / (2.0 * m) * e_field  # eps
    tau = q * dt / (2.0 * m) * b_field  # dimensionless

    # First half: the electric kick, plus the magnetic rotation with the *old*
    # velocity v = u / gamma. Eq. 9.
    u_prime = u + 2.0 * half_kick + np.cross(u / _lorentz_factor(u), tau)

    # Lorentz factor at the end of the step: positive root of Eq. 11.
    gamma_prime_squared = 1.0 + np.sum(u_prime**2, axis=1, keepdims=True) / c**2
    tau_squared = np.sum(tau**2, axis=1, keepdims=True)
    u_star = np.sum(u_prime * tau, axis=1, keepdims=True) / c
    gamma_new = _gamma_from_quartic(
        a=gamma_prime_squared - tau_squared, b=tau_squared + u_star**2
    )

    # Second half: solve the implicit magnetic rotation. Eq. 12.
    tvec = tau / gamma_new
    scale = 1.0 / (1.0 + np.sum(tvec**2, axis=1, keepdims=True))
    u_dot_t = np.sum(u_prime * tvec, axis=1, keepdims=True)
    u_new = scale * (u_prime + u_dot_t * tvec + np.cross(u_prime, tvec))

    # Second half of the move, with the velocity at the end of the step.
    x_new = x_half + 0.5 * dt * u_new / _lorentz_factor(u_new)
    return x_new, u_new


def rk4(
    x: NDArray[np.float64],
    u: NDArray[np.float64],
    field: Field,
    t: float,
    dt: float,
    q: float,
    m: float,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Advance the particles by one time step with the classical Runge-Kutta scheme.

    This is the witness of the project, not a production pusher: it is fourth order
    in ``dt``, far more accurate than the three others over a short time, but it
    preserves no structure of the motion. In a pure magnetic field ``|u|`` is not
    kept: the energy drifts steadily, step after step, where Boris and
    Higuera-Cary keep it to rounding. The step is not symmetric in time and does
    not preserve the phase-space volume.

    Like the other pushers, it takes ``(x^n, u^n)`` at ``t`` and returns
    ``(x^{n+1}, u^{n+1})`` at ``t + dt``. Unlike them, it does not evaluate the
    field once at the middle of the step, but at its four stages.

    Parameters
    ----------
    x : ndarray, shape (n_part, 3)
        Positions ``x^n`` [m] at time ``t``.
    u : ndarray, shape (n_part, 3)
        Proper velocities ``u^n = gamma * v`` [m/s] at time ``t``.
    field : Field
        Function ``field(x, t) -> (E, B)``, each of shape ``(n_part, 3)``, in V/m
        and T. It is called four times: at ``t``, twice at ``t + dt / 2`` and at
        ``t + dt``, each time at the position of the stage.
    t : float
        Time [s] of ``x`` and ``u``.
    dt : float
        Time step [s], finite and non-zero. A negative ``dt`` steps backward.
    q : float
        Signed charge [C], for example ``-e`` for an electron.
    m : float
        Rest mass [kg], strictly positive.

    Returns
    -------
    x_new : ndarray, shape (n_part, 3)
        Positions ``x^{n+1}`` [m] at time ``t + dt``.
    u_new : ndarray, shape (n_part, 3)
        Proper velocities ``u^{n+1}`` [m/s] at time ``t + dt``.

        Both are new arrays: ``x`` and ``u`` are not modified.

    Raises
    ------
    ValueError
        Same as ``boris``: wrong shapes, or ``t``, ``dt``, ``q``, ``m`` not finite
        real numbers in their range.

    Notes
    -----
    The equations of motion (Ripperda et al. 2018, Eqs. 1-2) are written as
    ``d(x, u)/dt = f(x, u, t)`` with::

        dx/dt = u / gamma,                   gamma = sqrt(1 + |u|^2 / c^2)
        du/dt = (q / m) (E + (u / gamma) x B)

    and one step is the classical four-stage formula::

        k1 = f(x, u, t)
        k2 = f(x + (dt / 2) k1_x, u + (dt / 2) k1_u, t + dt / 2)
        k3 = f(x + (dt / 2) k2_x, u + (dt / 2) k2_u, t + dt / 2)
        k4 = f(x + dt k3_x, u + dt k3_u, t + dt)
        (x_new, u_new) = (x, u) + (dt / 6) (k1 + 2 k2 + 2 k3 + k4)
    """
    x, u = _check_push_arguments(x, u, t, dt, q, m)

    def rate_of_change(
        x_stage: NDArray[np.float64], u_stage: NDArray[np.float64], t_stage: float
    ) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        """Return ``(dx/dt, du/dt)`` at one stage: velocity and Lorentz force."""
        e_field, b_field = _evaluate_field(field, x_stage, t_stage)
        velocity = u_stage / _lorentz_factor(u_stage)
        return velocity, q / m * (e_field + np.cross(velocity, b_field))

    half = 0.5 * dt
    k1_x, k1_u = rate_of_change(x, u, t)
    k2_x, k2_u = rate_of_change(x + half * k1_x, u + half * k1_u, t + half)
    k3_x, k3_u = rate_of_change(x + half * k2_x, u + half * k2_u, t + half)
    k4_x, k4_u = rate_of_change(x + dt * k3_x, u + dt * k3_u, t + dt)

    x_new = x + dt / 6.0 * (k1_x + 2.0 * k2_x + 2.0 * k3_x + k4_x)
    u_new = u + dt / 6.0 * (k1_u + 2.0 * k2_u + 2.0 * k3_u + k4_u)
    return x_new, u_new


# --- choosing a pusher by name -----------------------------------------------------


def get_pusher(name: str) -> Pusher:
    """Return the pusher function called ``name``.

    This is how ``run`` turns the ``pusher:`` line of a ``.yaml`` file into a
    function. To add a pusher to the project, write it above and add one line to
    the dictionary inside this function.

    Parameters
    ----------
    name : str
        ``"boris"``, ``"higuera_cary"``, ``"vay"`` or ``"rk4"``.

    Returns
    -------
    Pusher
        The function ``push(x, u, field, t, dt, q, m) -> (x_new, u_new)``.

    Raises
    ------
    ValueError
        If there is no pusher of that name. The message lists the available ones.

    Examples
    --------
    >>> push = get_pusher("higuera_cary")
    >>> push is higuera_cary
    True
    """
    # Built at each call, not stored at module level: the project keeps no global
    # mutable state.
    available: dict[str, Pusher] = {
        "boris": boris,
        "higuera_cary": higuera_cary,
        "rk4": rk4,
        "vay": vay,
    }
    if name not in available:
        raise ValueError(
            f"The pusher {name!r} is not implemented yet. "
            f"Available: {sorted(available)}."
        )
    return available[name]
