"""
Time integrators for the motion of charged particles: one step of a pusher.
"""

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
    field: Field,
    t: float,
    dt: float,
    q: float,
    m: float,
) -> tuple[
    NDArray[np.float64],
    NDArray[np.float64],
    NDArray[np.float64],
    NDArray[np.float64],
]:
    """Check the arguments of a pusher, then evaluate the field once.

    Every pusher starts with this call, so they all refuse the same wrong inputs
    with the same messages.

    Parameters
    ----------
    x, u, field, t, dt, q, m
        The arguments of a pusher, see ``boris``.

    Returns
    -------
    x, u : ndarray, shape (n_part, 3)
        The positions [m] and proper velocities [m/s] as ``float64`` arrays.
    e_field, b_field : ndarray, shape (n_part, 3)
        The electric [V/m] and magnetic [T] fields at ``(x, t)``.

    Raises
    ------
    ValueError
        If ``x`` or ``u`` does not have shape ``(n_part, 3)``, if ``u`` and ``x``
        differ in shape, if the field returns arrays of another shape, or if
        ``t``, ``dt``, ``q`` or ``m`` is not a finite real number in its range. The
        message says what was expected and what was received.
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

    e_field, b_field = field(x, t)
    for name, array in (("E", e_field), ("B", b_field)):
        if np.shape(array) != x.shape:
            raise ValueError(
                f"field(x, t) must return {name} of shape {x.shape}, the shape of x; "
                f"got shape {np.shape(array)}."
            )
    return x, u, e_field, b_field


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

    Parameters
    ----------
    x : ndarray, shape (n_part, 3)
        Positions ``x^n`` [m] at time ``t``.
    u : ndarray, shape (n_part, 3)
        Proper velocities ``u^{n-1/2} = gamma * v`` [m/s], half a step *before*
        ``t``. This offset is the leapfrog convention of the project: to start a
        run, the initial ``u`` must be moved back by half a step.
    field : Field
        Function ``field(x, t) -> (E, B)``, each of shape ``(n_part, 3)``, in V/m
        and T (see ``ppf_c5_runaways.fields``). It is called once, at ``(x, t)``.
    t : float
        Time [s] of the positions ``x``.
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
        Proper velocities ``u^{n+1/2}`` [m/s]. To get the velocity at a whole step,
        as needed for an energy diagnostic, average two half steps:
        ``u^n = (u^{n-1/2} + u^{n+1/2}) / 2``. Using ``u^{n-1/2}`` directly there
        silently destroys the second order of the scheme.

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
    One step (Ripperda et al. 2018, Eqs. 10-12), with ``a = q dt E / (2 m)``::

        u- = u + a                                   half acceleration by E
        gamma- = sqrt(1 + |u-|^2 / c^2)
        tvec = q dt B / (2 m gamma-)                 (called ``t`` by Ripperda)
        svec = 2 tvec / (1 + |tvec|^2)               (called ``s`` by Ripperda)
        u+ = u- + (u- + u- x tvec) x svec            rotation by B, |u+| = |u-|
        u_new = u+ + a                               half acceleration by E
        x_new = x + dt u_new / gamma_new             move with the new velocity

    """
    x, u, e_field, b_field = _check_push_arguments(x, u, field, t, dt, q, m)

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

    # Move with the velocity at the middle of the step, v = u / gamma.
    x_new = x + dt * u_new / _lorentz_factor(u_new)
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

    Parameters
    ----------
    x : ndarray, shape (n_part, 3)
        Positions ``x^n`` [m] at time ``t``.
    u : ndarray, shape (n_part, 3)
        Proper velocities ``u^{n-1/2} = gamma * v`` [m/s], half a step *before*
        ``t`` (leapfrog convention of the project, as for ``boris``).
    field : Field
        Function ``field(x, t) -> (E, B)``, each of shape ``(n_part, 3)``, in V/m
        and T. It is called once, at ``(x, t)``.
    t : float
        Time [s] of the positions ``x``.
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
        Proper velocities ``u^{n+1/2}`` [m/s]. For a velocity at a whole step,
        average two half steps: ``u^n = (u^{n-1/2} + u^{n+1/2}) / 2``.

        Both are new arrays: ``x`` and ``u`` are not modified.

    Raises
    ------
    ValueError
        Same as ``boris``: wrong shapes, or ``t``, ``dt``, ``q``, ``m`` not finite
        real numbers in their range.

    Notes
    -----
    One step (Higuera & Cary 2017, Eqs. 10, 16, 18, 20, 21, 24), with
    ``eps = q dt E / (2 m)`` and ``beta = q dt B / (2 m)``::

        u- = u + eps                                 half acceleration by E
        gamma-^2 = 1 + |u-|^2 / c^2
        gamma_new^4 - (gamma-^2 - |beta|^2) gamma_new^2
                    - (|beta|^2 + (beta . u-)^2 / c^2) = 0        (Eq. 20)
        tvec = beta / gamma_new                      rotation by B, as in Boris
        svec = 2 tvec / (1 + |tvec|^2)
        u+ = u- + (u- + u- x tvec) x svec
        u_new = u+ + eps                             half acceleration by E
        x_new = x + dt u_new / gamma(u_new)          move with the new velocity

    The paper writes Eq. 20 in units where ``c = 1``; the ``c`` above comes from
    writing every momentum as ``u / c``. Only the rotation differs from ``boris``:
    Boris would use ``gamma-`` where this uses ``gamma_new`` (``gamma- >= gamma_new``,
    Eq. 25).
    """
    x, u, e_field, b_field = _check_push_arguments(x, u, field, t, dt, q, m)

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

    x_new = x + dt * u_new / _lorentz_factor(u_new)
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

    Parameters
    ----------
    x : ndarray, shape (n_part, 3)
        Positions ``x^n`` [m] at time ``t``.
    u : ndarray, shape (n_part, 3)
        Proper velocities ``u^{n-1/2} = gamma * v`` [m/s], half a step *before*
        ``t`` (leapfrog convention of the project, as for ``boris``).
    field : Field
        Function ``field(x, t) -> (E, B)``, each of shape ``(n_part, 3)``, in V/m
        and T. It is called once, at ``(x, t)``.
    t : float
        Time [s] of the positions ``x``.
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
        Proper velocities ``u^{n+1/2}`` [m/s]. For a velocity at a whole step,
        average two half steps: ``u^n = (u^{n-1/2} + u^{n+1/2}) / 2``.

        Both are new arrays: ``x`` and ``u`` are not modified.

    Raises
    ------
    ValueError
        Same as ``boris``: wrong shapes, or ``t``, ``dt``, ``q``, ``m`` not finite
        real numbers in their range.

    Notes
    -----
    Vay's paper puts the velocities at whole steps ``u^i`` and the positions at half
    steps ``x^{i+1/2}``. This is the project's leapfrog with the labels shifted by
    half a step: Vay's ``u^i`` is our ``u^{n-1/2}``, his ``x^{i+1/2}`` our ``x^n``,
    and his ``E^{i+1/2}``, ``B^{i+1/2}`` are the fields at our ``(x^n, t)``.

    One step (Vay 2008, Eqs. 9-12), with ``eps = q dt E / (2 m)`` and
    ``tau = q dt B / (2 m)``::

        u' = u + 2 eps + (u / gamma(u)) x tau               (Eq. 9)
        gamma_new^4 - (gamma'^2 - |tau|^2) gamma_new^2
                    - (|tau|^2 + u*^2) = 0                  (Eq. 11)
            with gamma'^2 = 1 + |u'|^2 / c^2 and u* = (u' . tau) / c
        tvec = tau / gamma_new,   s = 1 / (1 + |tvec|^2)
        u_new = s (u' + (u' . tvec) tvec + u' x tvec)       (Eq. 12)
        x_new = x + dt u_new / gamma(u_new)                 move with the new velocity

    ``u_new`` solves the implicit equation ``u_new = u' + (u_new / gamma_new) x tau``
    (Eq. 10); the explicit formulas above are its exact solution.
    """
    x, u, e_field, b_field = _check_push_arguments(x, u, field, t, dt, q, m)

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

    x_new = x + dt * u_new / _lorentz_factor(u_new)
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
        ``"boris"``, ``"higuera_cary"`` or ``"vay"``.

    Returns
    -------
    Pusher
        The function ``push(x, u, field, t, dt, q, m) -> (x_new, u_new)``.

    Raises
    ------
    ValueError
        If there is no pusher of that name (or it is not written yet, like
        ``"rk4"``). The message lists the available ones.

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
        "vay": vay,
    }
    if name not in available:
        raise ValueError(
            f"The pusher {name!r} is not implemented yet. "
            f"Available: {sorted(available)}."
        )
    return available[name]
