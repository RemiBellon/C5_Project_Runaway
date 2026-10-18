"""Diagnostics computed from trajectories: pure functions of arrays, in SI units.

This module reads no configuration and writes nothing. It receives arrays, as
returned by ``integrate`` or read back by ``io.read_result``, and returns arrays.
"""

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.integrate import quad
from scipy.special import kv

from ppf_c5_runaways.constants import c, eps_0
from ppf_c5_runaways.fields import Field
from ppf_c5_runaways.pushers import Pusher


def energy_error(u: NDArray[np.float64]) -> NDArray[np.float64]:
    """Relative change of the kinetic energy of each particle since the first state.

    ``(K - K_0) / K_0``, with ``K = (gamma - 1) m c^2``. The mass cancels, so it
    is not needed. In a static magnetic field without electric field ``K`` is a
    constant of motion, and this is the energy error of the scheme. With an electric
    field it is only the change of ``K``, which may be physical.

    Parameters
    ----------
    u : ndarray, shape (n_saved, n_part, 3)
        Proper velocities ``u = gamma * v`` [m/s], one set per saved state: ``u``
        of ``integrate`` or of the results files, which is the velocity at the
        times of the positions. In a pure magnetic field Boris and Higuera-Cary
        keep ``|u|`` exactly, so the error measured is rounding only, as in
        Ripperda et al. (2018, Figs. 3 and 11).

    Returns
    -------
    ndarray, shape (n_saved, n_part)
        Relative energy change, dimensionless. The first row is 0.

    Raises
    ------
    ValueError
        If ``u`` does not have shape ``(n_saved, n_part, 3)``, or if a particle is
        at rest in the first state (``K_0 = 0``: a relative change has no meaning).
        The message names the particles concerned.

    Notes
    -----
    ``gamma - 1`` is computed as ``(|u|/c)^2 / (gamma + 1)``, the same number
    written without subtraction. The direct ``gamma - 1`` loses all its digits in
    the non-relativistic regime: at ``|u|/c = 1e-8``, ``sqrt(1 + 1e-16)`` rounds to
    exactly 1.
    """
    u = np.asarray(u, dtype=np.float64)
    if u.ndim != 3 or u.shape[2] != 3:
        raise ValueError(
            "u must have shape (n_saved, n_part, 3), one vector per saved "
            f"time and per particle; got shape {u.shape}"
        )

    u_over_c_squared = np.sum(u**2, axis=2) / c**2
    gamma = np.sqrt(1.0 + u_over_c_squared)
    gamma_minus_one = u_over_c_squared / (gamma + 1.0)  # K / (m c^2)

    at_rest = np.flatnonzero(gamma_minus_one[0] == 0.0)
    if at_rest.size > 0:
        raise ValueError(
            f"Particle(s) {at_rest.tolist()} are at rest at t = 0: their kinetic "
            "energy is 0, so a relative energy change has no meaning"
        )
    return (gamma_minus_one - gamma_minus_one[0]) / gamma_minus_one[0]


def kinetic_energy(u: NDArray[np.float64], m: float) -> NDArray[np.float64]:
    """Kinetic energy ``(gamma - 1) m c^2`` of each particle, without cancellation.

    Computed as ``m c^2 (|u| / c)^2 / (gamma + 1)``, which is the same number
    written without the subtraction ``gamma - 1``: from
    ``gamma^2 - 1 = (|u| / c)^2`` one gets
    ``gamma - 1 = (|u| / c)^2 / (gamma + 1)``. The direct form loses its digits in
    the non-relativistic regime, and all of them below ``|u| / c = 1e-8``, where
    ``sqrt(1 + 1e-16)`` rounds to exactly 1. This form keeps full precision from a
    particle at rest to the ultra-relativistic regime.

    Parameters
    ----------
    u : ndarray, shape (..., 3)
        Proper velocities ``u = gamma * v`` [m/s]. Any leading shape: ``(3,)`` for
        one particle, ``(n_part, 3)``, or ``(n_saved, n_part, 3)``.
    m : float
        Rest mass [kg], finite and strictly positive.

    Returns
    -------
    ndarray, shape (...)
        Kinetic energy [J], the shape of ``u`` without its last axis. Divide by
        ``constants.e`` for electron-volts.

    Raises
    ------
    ValueError
        If the last axis of ``u`` does not have length 3, or ``m`` is not a finite,
        strictly positive number.
    """
    u = np.asarray(u, dtype=np.float64)
    if u.ndim == 0 or u.shape[-1] != 3:
        raise ValueError(
            "u must have shape (..., 3), the last axis holding the 3 components; "
            f"got shape {u.shape}"
        )
    if (
        isinstance(m, bool)
        or not isinstance(m, (int, float))
        or not np.isfinite(m)
        or m <= 0.0
    ):
        raise ValueError(
            f"m must be a finite, strictly positive number [kg], got {m!r}"
        )

    u_over_c_squared = np.sum(u**2, axis=-1) / c**2
    gamma = np.sqrt(1.0 + u_over_c_squared)
    return m * c**2 * u_over_c_squared / (gamma + 1.0)


def drift_velocity(
    t: NDArray[np.float64], x: NDArray[np.float64]
) -> NDArray[np.float64]:
    """Mean velocity of each particle over the whole run, by a least-squares fit.

    Each coordinate is fitted by a straight line ``x(t) = x_0 + v_d t``; the slope
    ``v_d`` is the drift velocity of the guiding centre. The fit uses every saved
    point, not only the first and the last one.

    Parameters
    ----------
    t : ndarray, shape (n_saved,)
        Times of the saved states [s], at least two different ones.
    x : ndarray, shape (n_saved, n_part, 3)
        Positions [m] at the times ``t``.

    Returns
    -------
    ndarray, shape (n_part, 3)
        Drift velocity of each particle [m/s].

    Raises
    ------
    ValueError
        If the shapes do not match, or if fewer than two different times are given.

    Notes
    -----
    A gyration of radius ``r`` and angular frequency ``omega`` on top of the drift
    biases the slope. Over a whole number of turns and a duration ``L``, the bias
    is ``12 r sin(phi) / (omega L^2)`` for a gyration ``r cos(omega t + phi)``
    (continuous limit): it falls as ``1 / L^2``, so a long run is needed when the
    particle gyrates. A particle launched exactly at the drift velocity does not
    gyrate, and the slope is then the drift velocity of the scheme itself.
    """
    t = np.asarray(t, dtype=np.float64)
    x = np.asarray(x, dtype=np.float64)
    if t.ndim != 1 or x.ndim != 3 or x.shape[0] != t.size or x.shape[2] != 3:
        raise ValueError(
            "t must have shape (n_saved,) and x shape (n_saved, n_part, 3) with the "
            f"same n_saved; got t of shape {t.shape} and x of shape {x.shape}"
        )
    t_centred = t - np.mean(t)
    spread = np.sum(t_centred**2)
    if spread == 0.0:
        raise ValueError(
            f"at least two different times are needed to fit a velocity, got t = {t!r}"
        )

    # Least-squares slope, sum((t - mean t) (x - mean x)) / sum((t - mean t)^2),
    # for every particle and every coordinate at once (sum over the time axis).
    x_centred = x - np.mean(x, axis=0)
    t_column = t_centred[:, np.newaxis, np.newaxis]  # broadcasts on (n_part, 3)
    return np.sum(t_column * x_centred, axis=0) / spread


def gyration_phase(
    u: NDArray[np.float64], b_direction: NDArray[np.float64]
) -> NDArray[np.float64]:
    """Gyration phase of each particle: angle of ``u`` about a fixed field direction.

    The part of ``u`` perpendicular to ``b_direction`` is written in a fixed
    right-handed basis ``(e1, e2, b_hat)``, and its angle ``atan2(u.e2, u.e1)`` is
    unwrapped in time, so that it grows without jumps of ``2 pi``. In a uniform
    field it turns at ``d phase / dt = -q |B| / (gamma m)``: counter-clockwise
    about ``B`` for an electron. This is the phase ``theta_c`` of Ripperda et al.
    (2018, Eq. 46).

    Parameters
    ----------
    u : ndarray, shape (n_saved, n_part, 3)
        Proper velocities [m/s], one set per saved state: ``u`` of ``integrate`` or
        of the results files. The phase is that of the particle at the times of
        these states.
    b_direction : array_like, shape (3,)
        Direction of the (uniform) magnetic field, any non-zero length.

    Returns
    -------
    ndarray, shape (n_saved, n_part)
        Unwrapped phase [rad].

    Raises
    ------
    ValueError
        If the shapes are wrong, ``b_direction`` is zero, or a velocity has no
        part perpendicular to the field (its phase is not defined).

    Notes
    -----
    Unwrapping assumes that the phase changes by less than ``pi`` between two
    saved states: keep ``save_every * (angle per step) < pi``, otherwise turns are
    silently lost.
    """
    u = np.asarray(u, dtype=np.float64)
    b_direction = np.asarray(b_direction, dtype=np.float64)
    if u.ndim != 3 or u.shape[2] != 3 or b_direction.shape != (3,):
        raise ValueError(
            "u must have shape (n_saved, n_part, 3) and b_direction shape (3,); "
            f"got {u.shape} and {b_direction.shape}"
        )
    b_norm = np.linalg.norm(b_direction)
    if b_norm == 0.0:
        raise ValueError("b_direction must not be the zero vector")
    b_hat = b_direction / b_norm

    # e1: perpendicular to b_hat, built from the coordinate axis least aligned
    # with it (never nearly parallel, so the cross product is well conditioned).
    axis = np.zeros(3)
    axis[np.argmin(np.abs(b_hat))] = 1.0
    e1 = np.cross(b_hat, axis)
    e1 /= np.linalg.norm(e1)
    e2 = np.cross(b_hat, e1)

    u_1 = u @ e1
    u_2 = u @ e2
    if np.any(np.hypot(u_1, u_2) == 0.0):
        raise ValueError(
            "a velocity is parallel to the field: its gyration phase is not defined"
        )
    return np.unwrap(np.arctan2(u_2, u_1), axis=0)


def gyration_phase_error(
    t: NDArray[np.float64],
    u: NDArray[np.float64],
    u0: NDArray[np.float64],
    b_field: NDArray[np.float64],
    q: float,
    m: float,
) -> NDArray[np.float64]:
    """Error on the gyration phase in a uniform magnetic field, without electric field.

    The exact velocity turns about ``B`` at the constant angular velocity
    ``Omega = -q |B| / (gamma m)``, with ``gamma`` that of the initial velocity
    ``u0``, from the phase of ``u0`` (Ripperda et al. 2018, Eq. 46). The error is
    ``phase(u(t)) - (phase(u0) + Omega t)``, signed: for an electron (``Omega >
    0``) a negative error is a lag, for a positive charge a positive one.

    Parameters
    ----------
    t : ndarray, shape (n_saved,)
        Times [s] of the velocities ``u``.
    u : ndarray, shape (n_saved, n_part, 3)
        Proper velocities [m/s] at the times ``t``: ``u`` of ``integrate`` or of
        the results files.
    u0 : ndarray, shape (n_part, 3)
        Exact initial proper velocities [m/s] at ``t = 0``, those of the
        configuration. They are also the first saved state, so the error is zero
        at ``t = 0``.
    b_field : array_like, shape (3,)
        The uniform magnetic field [T].
    q, m : float
        Signed charge [C] and mass [kg].

    Returns
    -------
    ndarray, shape (n_saved, n_part)
        Phase error [rad].

    Raises
    ------
    ValueError
        If the shapes are wrong, if a velocity has no part perpendicular to ``B``,
        or if the saved states are too far apart for the phase to be followed:
        the exact phase must advance by less than ``pi / 2`` between two saved
        states (see Notes).

    Notes
    -----
    The phase is known only modulo ``2 pi`` at each saved state; it is followed by
    unwrapping, which is right only if it moves by less than ``pi`` between two
    states. The exact advance ``|Omega| (t_{k+1} - t_k)`` is required below
    ``pi / 2``: the angle of a scheme per step, ``alpha``, differs from the exact
    one by less than ``pi / 2`` for the schemes of the project, so the true jump
    stays below ``pi``. Above, a turn could be lost silently: this raises instead.

    Both phases come from ``atan2``, in ``(-pi, pi]``; a velocity on the cut (angle
    ``pi``) can be read ``pi`` in ``u0`` and ``-pi`` in ``u[0]``. The initial
    error is therefore brought back into ``(-pi, pi]``.
    """
    t = np.asarray(t, dtype=np.float64)
    u0 = np.asarray(u0, dtype=np.float64)
    b_field = np.asarray(b_field, dtype=np.float64)
    if u0.ndim != 2 or u0.shape[1] != 3 or np.shape(u)[1:] != u0.shape:
        raise ValueError(
            "u must have shape (n_saved, n_part, 3) and u0 shape (n_part, 3); got "
            f"{np.shape(u)} and {u0.shape}"
        )
    if t.shape != np.shape(u)[:1]:
        raise ValueError(f"t must have shape (n_saved,), got {t.shape}")

    gamma0 = np.sqrt(1.0 + np.sum(u0**2, axis=1) / c**2)
    omega = -q * np.linalg.norm(b_field) / (gamma0 * m)  # (n_part,) [rad/s]
    if t.size > 1:
        largest_advance = np.max(np.abs(omega)) * np.max(np.abs(np.diff(t)))
        if largest_advance >= np.pi / 2.0:
            raise ValueError(
                f"the exact phase advances by up to {largest_advance:.3g} rad between "
                "two saved states, the phase cannot be followed above pi / 2: save "
                "more often (smaller save_every)"
            )

    phase = gyration_phase(u, b_field)
    phase0 = gyration_phase(u0[np.newaxis], b_field)[0]
    initial_error = phase[0] - phase0
    initial_error -= 2.0 * np.pi * np.round(initial_error / (2.0 * np.pi))
    exact_start = phase[0] - initial_error
    return phase - (exact_start + omega * t[:, np.newaxis])


def rotation_angle_per_step(
    scheme: str,
    u: NDArray[np.float64],
    b_field: NDArray[np.float64],
    dt: float,
    q: float,
    m: float,
) -> NDArray[np.float64]:
    """Angle by which a scheme turns ``u`` in one step of a uniform magnetic field.

    Without electric field the three schemes turn ``u`` about ``B`` by the Boris
    rotation with ``t = beta / gamma_r``, ``beta = q B dt / (2 m)``, whose angle is
    ``2 arctan(|beta| / gamma_r)`` (Boris 1970; Zenitani & Umeda 2018, Eq. 12). Only
    the Lorentz factor ``gamma_r`` of the rotation differs:

    * Boris: ``gamma_r = gamma``, since ``gamma^- = gamma`` when ``E = 0``
      (Ripperda et al. 2018, Eqs. 10-12).
    * Vay: ``gamma_r = gamma`` too. With ``E = 0``, ``gamma = gamma(u)`` solves the
      quartic of Vay (2008, Eq. 11), because
      ``gamma'^2 = gamma^2 + |u x tau|^2 / (gamma^2 c^2)``; Eqs. 10-12 are then the
      Crank-Nicolson form ``u_new - u = (u_new + u) x tau / gamma`` of the Boris
      rotation.
    * Higuera-Cary: ``gamma_r`` is the positive root of
      ``gamma_r^4 - (gamma^2 - |beta|^2) gamma_r^2 - (|beta|^2 + (beta.u / c)^2)
      = 0`` (Higuera & Cary 2017, Eq. 20). It depends on the pitch angle through
      ``beta . u``.

    Parameters
    ----------
    scheme : {"boris", "vay", "higuera_cary"}
        Name of the scheme.
    u : ndarray, shape (n_part, 3)
        Proper velocities [m/s]. In a uniform ``B`` without ``E`` the norm and the
        parallel part are constant, so the angle is the same at every step.
    b_field : array_like, shape (3,)
        The uniform magnetic field [T].
    dt : float
        Time step [s]; a negative ``dt`` turns the other way.
    q, m : float
        Signed charge [C] and mass [kg].

    Returns
    -------
    ndarray, shape (n_part,)
        Signed angle per step [rad], with the sign of the exact rotation
        ``-q dt``: positive for an electron and ``dt > 0``.

    Raises
    ------
    ValueError
        If ``scheme`` is not one of the three names, or the shapes are wrong.

    Notes
    -----
    The exact angle is ``theta = |q| |B| |dt| / (gamma m) = 2 |beta| / gamma``. Boris
    and Vay turn by ``2 arctan(theta / 2) = theta - theta^3 / 12 + ...``. For
    ``gamma >> 1`` Higuera-Cary turns by ``2 arctan((theta / 2) / sqrt(x))`` with
    ``x = [(1 - theta^2 / 4) + sqrt((1 - theta^2 / 4)^2 + theta^2 cos^2 psi)] / 2``
    (``psi`` the pitch angle; limit of Eq. 20 with ``|beta| / gamma = theta / 2``
    and ``beta . u / (c gamma^2) -> (theta / 2) cos psi``), that is
    ``theta + theta^3 (3 sin^2 psi - 2) / 24 + ...``: a lead of half the Boris lag
    at ``psi = 90`` degrees, no error of order ``theta^3`` at
    ``sin^2 psi = 2 / 3`` (``psi = 54.7`` degrees), and the Boris lag as ``psi``
    goes to 0.
    """
    if scheme not in ("boris", "vay", "higuera_cary"):
        raise ValueError(
            f"scheme must be 'boris', 'vay' or 'higuera_cary', got {scheme!r}"
        )
    u = np.asarray(u, dtype=np.float64)
    b_field = np.asarray(b_field, dtype=np.float64)
    if u.ndim != 2 or u.shape[1] != 3 or b_field.shape != (3,):
        raise ValueError(
            f"u must have shape (n_part, 3) and b_field shape (3,); got {u.shape} "
            f"and {b_field.shape}"
        )

    beta = abs(q) * abs(dt) / (2.0 * m) * b_field  # (3,), dimensionless
    beta_squared = beta @ beta
    gamma_squared = 1.0 + np.sum(u**2, axis=1) / c**2
    if scheme == "higuera_cary":
        # Positive root in gamma_r^2 of x^2 - a x - b = 0, written without
        # cancellation (see pushers._gamma_from_quartic for the same care).
        a = gamma_squared - beta_squared
        b = beta_squared + (u @ beta / c) ** 2
        larger = 0.5 * (np.abs(a) + np.sqrt(a**2 + 4.0 * b))
        gamma_rotation = np.sqrt(np.where(a >= 0.0, larger, b / larger))
    else:
        gamma_rotation = np.sqrt(gamma_squared)
    angle = 2.0 * np.arctan(np.sqrt(beta_squared) / gamma_rotation)
    return np.sign(-q * dt) * angle


def predicted_gyration_phase_error(
    scheme: str,
    t: NDArray[np.float64],
    u0: NDArray[np.float64],
    b_field: NDArray[np.float64],
    dt: float,
    q: float,
    m: float,
) -> NDArray[np.float64]:
    """Phase error that ``gyration_phase_error`` must find for a scheme, exactly.

    In a uniform ``B`` without ``E``, for the velocities ``u^n`` kept by
    ``integrate`` (at the times ``t^n = n dt``, starting from ``u^0 = u0``), the
    phase error at step ``n`` is, with ``alpha`` the angle of the scheme for one step
    ``dt`` (``rotation_angle_per_step``) and ``theta`` the exact angle per step:

        e_n = n (alpha - theta)

    Derivation: each step turns ``u`` by ``alpha``, so the phase of ``u^n`` is
    ``phase(u0) + n alpha``; the exact phase is ``phase(u0) + n theta``. The error
    is zero at ``t = 0`` and grows linearly: a pure secular drift, with no offset.

    Parameters
    ----------
    scheme : {"boris", "vay", "higuera_cary"}
        Name of the scheme.
    t : ndarray, shape (n_saved,)
        Saved times [s], multiples of ``dt``.
    u0 : ndarray, shape (n_part, 3)
        Initial proper velocities [m/s].
    b_field : array_like, shape (3,)
        The uniform magnetic field [T].
    dt : float
        Time step [s], strictly positive.
    q, m : float
        Signed charge [C] and mass [kg].

    Returns
    -------
    ndarray, shape (n_saved, n_part)
        Predicted phase error [rad].

    Raises
    ------
    ValueError
        As ``rotation_angle_per_step``, or if ``dt`` is not strictly positive.
    """
    if not dt > 0.0:
        raise ValueError(f"dt must be strictly positive, got {dt!r}")
    u0 = np.asarray(u0, dtype=np.float64)
    step_angle = rotation_angle_per_step(scheme, u0, b_field, dt, q, m)
    gamma0 = np.sqrt(1.0 + np.sum(u0**2, axis=1) / c**2)
    exact_angle = -q * np.linalg.norm(b_field) * dt / (gamma0 * m)

    steps = np.round(np.asarray(t, dtype=np.float64) / dt)[:, np.newaxis]
    return steps * (step_angle - exact_angle)


def pitch_angle(
    u: NDArray[np.float64], b_field: NDArray[np.float64]
) -> NDArray[np.float64]:
    """Pitch angle of each velocity: angle between ``u`` and the field, in ``[0, pi]``.

    ``psi = arctan(u_perp / u_par)``, taken in ``[0, pi]``: 0 along ``B``,
    ``pi / 2`` perpendicular to it, ``pi`` against it (Carbajal et al. 2017, Sec. II).
    It is computed as ``atan2(|u x b_hat|, u . b_hat)``, which is accurate at every
    angle. The equivalent ``arccos(u . b_hat / |u|)`` is not: near 0 and ``pi`` the
    cosine is flat, so an angle of ``1e-8`` rad is returned as exactly 0.

    Parameters
    ----------
    u : ndarray, shape (n_part, 3)
        Proper velocities [m/s], non-zero.
    b_field : array_like, shape (3,)
        Magnetic field [T], non-zero; only its direction is used.

    Returns
    -------
    ndarray, shape (n_part,)
        Pitch angles [rad].

    Raises
    ------
    ValueError
        If the shapes are wrong, or ``u`` or ``b_field`` is zero.
    """
    u = np.asarray(u, dtype=np.float64)
    b_field = np.asarray(b_field, dtype=np.float64)
    if u.ndim != 2 or u.shape[1] != 3 or b_field.shape != (3,):
        raise ValueError(
            f"u must have shape (n_part, 3) and b_field shape (3,); got {u.shape} "
            f"and {b_field.shape}"
        )
    speed = np.linalg.norm(u, axis=1)
    b_norm = np.linalg.norm(b_field)
    if b_norm == 0.0 or np.any(speed == 0.0):
        raise ValueError("u and b_field must be non-zero to define a pitch angle")
    b_hat = b_field / b_norm
    u_parallel = u @ b_hat
    u_perpendicular = np.linalg.norm(np.cross(u, b_hat), axis=1)
    return np.arctan2(u_perpendicular, u_parallel)


def phase_error_per_turn(
    t: NDArray[np.float64],
    phase_error: NDArray[np.float64],
    period: NDArray[np.float64],
) -> NDArray[np.float64]:
    """Secular gyration phase error per exact turn, by a least-squares fit.

    The phase error of a scheme in a uniform field is a drift linear in time, zero
    at ``t = 0`` (``predicted_gyration_phase_error``). The slope of the straight
    line fitted to all the saved points, times the exact period, is the drift per
    turn. The line is fitted with a free constant, so a constant offset in
    ``phase_error``, whatever its origin, does not enter the result.

    Parameters
    ----------
    t : ndarray, shape (n_saved,)
        Times [s], at least two different ones.
    phase_error : ndarray, shape (n_saved, n_part)
        Phase error [rad] (``gyration_phase_error``).
    period : ndarray, shape (n_part,)
        Exact gyration period of each particle [s], ``2 pi gamma m / (|q| |B|)``.

    Returns
    -------
    ndarray, shape (n_part,)
        Phase error per exact turn [rad], signed as ``phase_error``.

    Raises
    ------
    ValueError
        If the shapes do not match, or fewer than two different times are given.
    """
    t = np.asarray(t, dtype=np.float64)
    phase_error = np.asarray(phase_error, dtype=np.float64)
    period = np.asarray(period, dtype=np.float64)
    if (
        t.ndim != 1
        or phase_error.shape[:1] != t.shape
        or phase_error.ndim != 2
        or period.shape != phase_error.shape[1:]
    ):
        raise ValueError(
            "t must have shape (n_saved,), phase_error (n_saved, n_part) and period "
            f"(n_part,); got {t.shape}, {phase_error.shape} and {period.shape}"
        )
    t_centred = t - np.mean(t)
    spread = np.sum(t_centred**2)
    if spread == 0.0:
        raise ValueError("at least two different times are needed to fit a slope")
    error_centred = phase_error - np.mean(phase_error, axis=0)
    slope = np.sum(t_centred[:, np.newaxis] * error_centred, axis=0) / spread
    return slope * period


def magnetic_moment(
    u: NDArray[np.float64], b_field: NDArray[np.float64], m: float
) -> NDArray[np.float64]:
    """Relativistic magnetic moment ``mu = p_perp^2 / (2 m |B|)`` of each particle.

    ``p_perp = m u_perp`` is the momentum perpendicular to the local field. ``mu``
    is the adiabatic invariant of relativistic gyration (Ripperda et al. 2018,
    Sec. 3.2.1; Northrop 1963): it is conserved to order ``eps = r_L / L`` when the
    field varies over a length ``L`` much larger than the gyroradius ``r_L``.

    Parameters
    ----------
    u : ndarray, shape (n_saved, n_part, 3)
        Proper velocities [m/s], one set per saved state: ``u`` of ``integrate`` or
        of the results files.
    b_field : ndarray, shape (n_saved, n_part, 3)
        Magnetic field [T] at the positions of the particles, at the same times as
        ``u`` (``b_field`` of the results files). ``u`` and the positions are known
        at the same times, so ``mu`` is that of the particle at these times.
    m : float
        Rest mass [kg], strictly positive.

    Returns
    -------
    ndarray, shape (n_saved, n_part)
        Magnetic moment [J/T].

    Raises
    ------
    ValueError
        If the shapes differ, ``m`` is not strictly positive, or ``|B| = 0``
        somewhere (no perpendicular direction then).

    Notes
    -----
    The field is taken at the particle, not at its guiding centre, as the stored
    field is. In a non-uniform field this adds an oscillation of relative size
    ``eps`` at the gyration frequency: a property of the definition, present in
    the exact motion as well.
    """
    u = np.asarray(u, dtype=np.float64)
    b_field = np.asarray(b_field, dtype=np.float64)
    if u.ndim != 3 or u.shape[2] != 3 or b_field.shape != u.shape:
        raise ValueError(
            "u and b_field must have the same shape (n_saved, n_part, 3); "
            f"got {u.shape} and {b_field.shape}"
        )
    if not np.isfinite(m) or m <= 0.0:
        raise ValueError(f"m must be a finite, strictly positive mass [kg], got {m!r}")
    b_norm = np.linalg.norm(b_field, axis=2)
    if np.any(b_norm == 0.0):
        raise ValueError("|B| = 0 at some point: the magnetic moment is not defined")

    b_hat = b_field / b_norm[:, :, np.newaxis]
    u_parallel = np.sum(u * b_hat, axis=2)
    u_perp_squared = np.sum(u**2, axis=2) - u_parallel**2
    return m * u_perp_squared / (2.0 * b_norm)


def step_jacobian_determinant(
    push: Pusher,
    field: Field,
    t: NDArray[np.float64],
    x: NDArray[np.float64],
    u: NDArray[np.float64],
    dt: float,
    q: float,
    m: float,
    *,
    relative_shift: float = 1.0e-5,
) -> NDArray[np.float64]:
    """Measure the Jacobian determinant of one step of ``push`` at each given state.

    The step maps ``(x^n, u^n)`` to ``(x^{n+1}, u^{n+1})``. Its Jacobian determinant
    is the factor by which it changes phase-space volumes: exactly 1 for Boris and
    Higuera-Cary, ``J(u^n) / J(u^{n+1})`` for Vay with
    ``J(u) = 1 + (|beta|^2 + (beta . u / c)^2) / gamma(u)^4`` and
    ``beta = q B(x^{n+1/2}) dt / (2 m)``, the field being taken at the middle of the
    step (Higuera & Cary 2017, Sec. V, Eqs. 38-41). The two half moves of the step
    are shears, of determinant 1, so they do not change these values. The
    determinant is measured here by central differences on the whole 6 x 6
    Jacobian, without using these formulas, so that they can be checked. The
    product of the determinants of all the steps of a run is the volume change of
    the whole run.

    Parameters
    ----------
    push : Pusher
        The scheme, for example ``pushers.vay``.
    field : Field
        The field of the run.
    t : ndarray, shape (n_saved,)
        Times of the states [s].
    x : ndarray, shape (n_saved, n_part, 3)
        Positions ``x^n`` [m].
    u : ndarray, shape (n_saved, n_part, 3)
        Proper velocities ``u^n`` [m/s] at the same times (``u`` of the results
        files).
    dt, q, m : float
        Time step [s], signed charge [C] and mass [kg] of the run.
    relative_shift : float, optional
        Size of the differences, relative to ``max(|x|, |v| dt)`` for a position
        and to ``|u|`` for a velocity. Default ``1e-5``, close to the optimum
        ``eps^(1/3) = 6e-6`` of central differences, where the truncation error
        (about ``1e-10``) and the rounding error (about ``eps / 1e-5 = 2e-11`` on
        each entry) balance: the determinant is then known to about ``2e-10``. With
        ``1e-6`` the rounding dominates and the error reaches ``3e-9``.

    Returns
    -------
    ndarray, shape (n_saved, n_part)
        Determinant of one step at each state, dimensionless.

    Raises
    ------
    ValueError
        If the shapes do not match, or ``relative_shift`` is not in ``(0, 1)``.

    Notes
    -----
    The field is evaluated by ``push`` itself, at the middle of each step, so any
    field works, uniform or not. The cost is one call to ``push`` with
    ``12 n_part`` particles per state.
    """
    t = np.asarray(t, dtype=np.float64)
    x = np.asarray(x, dtype=np.float64)
    u = np.asarray(u, dtype=np.float64)
    if x.ndim != 3 or x.shape[2] != 3 or u.shape != x.shape or t.shape != x.shape[:1]:
        raise ValueError(
            "t must have shape (n_saved,), x and u shape (n_saved, n_part, 3); got "
            f"{t.shape}, {x.shape} and {u.shape}"
        )
    if not 0.0 < relative_shift < 1.0:
        raise ValueError(f"relative_shift must be in (0, 1), got {relative_shift!r}")

    n_part = x.shape[1]
    determinants = np.empty(x.shape[:2])
    for k in range(t.size):
        state = np.concatenate([x[k], u[k]], axis=1)  # (n_part, 6)
        speed = np.linalg.norm(u[k], axis=1)
        speed = np.where(speed > 0.0, speed, c)  # a particle at rest: scale c
        gamma = np.sqrt(1.0 + (speed / c) ** 2)
        length = np.maximum(np.linalg.norm(x[k], axis=1), speed / gamma * abs(dt))
        # shift[p, j]: step of the differences for particle p along coordinate j.
        shift = relative_shift * np.concatenate(
            [np.tile(length[:, np.newaxis], 3), np.tile(speed[:, np.newaxis], 3)],
            axis=1,
        )

        # 12 shifted copies of all the particles, pushed in a single call:
        # rows [2j n_part, (2j+1) n_part) are shifted by +shift along j, the next
        # n_part rows by -shift.
        shifted = np.tile(state, (12, 1))
        for j in range(6):
            rows_plus = slice(2 * j * n_part, (2 * j + 1) * n_part)
            rows_minus = slice((2 * j + 1) * n_part, (2 * j + 2) * n_part)
            shifted[rows_plus, j] += shift[:, j]
            shifted[rows_minus, j] -= shift[:, j]
        x_new, u_new = push(shifted[:, :3], shifted[:, 3:], field, t[k], dt, q, m)
        pushed = np.concatenate([x_new, u_new], axis=1)  # (12 n_part, 6)

        # jacobian[p, i, j] = d state_i / d state_j, each coordinate counted in
        # units of its own shift: this rescaling (D J D^-1) keeps the determinant
        # and makes all the entries dimensionless and of order 1.
        jacobian = np.empty((n_part, 6, 6))
        for j in range(6):
            plus = pushed[2 * j * n_part : (2 * j + 1) * n_part]
            minus = pushed[(2 * j + 1) * n_part : (2 * j + 2) * n_part]
            jacobian[:, :, j] = (plus - minus) / (2.0 * shift)
        determinants[k] = np.linalg.det(jacobian)
    return determinants


def boost_proper_velocity(
    u: NDArray[np.float64], frame_velocity: ArrayLike
) -> NDArray[np.float64]:
    """Proper velocities seen from a frame moving at a constant velocity.

    Lorentz transformation of ``u = gamma * v`` from the frame of the simulation
    to a frame that moves at ``frame_velocity`` with respect to it. With ``V`` the
    frame velocity, ``n`` its direction and ``kappa = 1 / sqrt(1 - |V|^2 / c^2)``
    its Lorentz factor, the part of ``u`` along ``n`` and the Lorentz factor of the
    particle become

        u_par' = kappa (u_par - gamma |V|),    gamma' = kappa (gamma - V . u / c^2),

    and the part of ``u`` perpendicular to ``n`` does not change. This is the
    transformation of the four-vector ``(gamma c, u)``; written for the velocity
    ``v`` it gives Eqs. 53-56 of Ripperda et al. (2018).

    Parameters
    ----------
    u : ndarray, shape (..., 3)
        Proper velocities [m/s] in the frame of the simulation. Any leading shape:
        ``(3,)``, ``(n_part, 3)`` or ``(n_saved, n_part, 3)``.
    frame_velocity : array_like, shape (3,)
        Velocity ``V`` [m/s] of the new frame, slower than light. It is a
        velocity, not a proper velocity. For the frame of the E x B drift it is
        ``E x B / |B|^2``.

    Returns
    -------
    ndarray, shape (..., 3)
        Proper velocities ``u'`` [m/s] in the moving frame, along the same axes.
        The Lorentz factor there is ``sqrt(1 + |u'|^2 / c^2)``.

    Raises
    ------
    ValueError
        If the shapes are wrong, or ``frame_velocity`` is not finite or not slower
        than light.

    Notes
    -----
    For a particle that moves almost with the frame, ``u_par`` and ``gamma |V|``
    are two numbers of size ``kappa c`` that nearly cancel, and the result is
    multiplied by ``kappa``: the rounding error on ``u'`` is of order
    ``kappa^2 eps c``. It is negligible at ``kappa = 10`` (``2e-14 c``) and grows
    as ``kappa^2`` beyond.
    """
    u = np.asarray(u, dtype=np.float64)
    frame_velocity = np.asarray(frame_velocity, dtype=np.float64)
    if u.ndim == 0 or u.shape[-1] != 3 or frame_velocity.shape != (3,):
        raise ValueError(
            "u must have shape (..., 3) and frame_velocity shape (3,); got "
            f"{u.shape} and {frame_velocity.shape}"
        )
    frame_speed = np.linalg.norm(frame_velocity)
    if not np.isfinite(frame_speed) or frame_speed >= c:
        raise ValueError(
            "frame_velocity must be finite and slower than light, got a speed of "
            f"{frame_speed!r} m/s"
        )
    if frame_speed == 0.0:
        return u.copy()

    direction = frame_velocity / frame_speed
    kappa = 1.0 / np.sqrt(1.0 - (frame_speed / c) ** 2)
    gamma = np.sqrt(1.0 + np.sum(u**2, axis=-1) / c**2)
    u_parallel = u @ direction
    # u_par' - u_par, added along the direction of the frame velocity.
    change = (kappa - 1.0) * u_parallel - kappa * gamma * frame_speed
    return u + change[..., np.newaxis] * direction


def curvature(
    u: NDArray[np.float64],
    e_field: ArrayLike,
    b_field: ArrayLike,
    q: float,
    m: float,
) -> NDArray[np.float64]:
    """Instantaneous curvature of the orbit of each particle, from the Lorentz force.

    The curvature of a path of velocity ``v`` and acceleration ``a`` is
    ``kappa = |v x a| / |v|^3`` (Carbajal et al. 2017, Eq. 13); its inverse is the
    radius of the circle that follows the path locally. With the acceleration of
    the Lorentz force, ``a = (q / (gamma m)) (E + v x B - v (v . E) / c^2)``, whose
    last term is along ``v`` and drops out of the cross product:

        kappa = |q| |v x (E + v x B)| / (gamma m |v|^3)

    Without electric field this is ``|q| |B| sin(theta) / (gamma m |v|)``, with
    ``theta`` the pitch angle (Eq. 14 of the paper). It is computed here from the
    proper velocity, as ``|q| |u x (gamma E + u x B)| / (m |u|^3)``, which is the
    same number and never forms ``v``.

    Parameters
    ----------
    u : ndarray, shape (..., 3)
        Proper velocities ``u = gamma * v`` [m/s], non-zero. Any leading shape:
        ``(3,)``, ``(n_part, 3)`` or ``(n_saved, n_part, 3)``.
    e_field, b_field : array_like, shape (..., 3)
        Electric field [V/m] and magnetic field [T] at the position of each
        particle, at the time of ``u``. Any shape that broadcasts with ``u``, for
        example ``(3,)`` for uniform fields.
    q : float
        Signed charge [C]; only its absolute value matters here.
    m : float
        Rest mass [kg], finite and strictly positive.

    Returns
    -------
    ndarray, shape (...)
        Curvature [1/m], positive or zero, the shape of ``u`` without its last
        axis. It is zero when the force is along the velocity or vanishes.

    Raises
    ------
    ValueError
        If a last axis does not have length 3, a velocity is zero (a particle at
        rest has no path, hence no curvature), or ``m`` is not a finite, strictly
        positive number.

    Notes
    -----
    ``u`` and the fields must be taken at the same time and position: the states
    ``(x^n, u^n)`` kept by ``integrate`` and the fields evaluated at ``x^n``.
    """
    u = np.asarray(u, dtype=np.float64)
    e_field = np.asarray(e_field, dtype=np.float64)
    b_field = np.asarray(b_field, dtype=np.float64)
    for name, array in (("u", u), ("e_field", e_field), ("b_field", b_field)):
        if array.ndim == 0 or array.shape[-1] != 3:
            raise ValueError(
                f"{name} must have shape (..., 3), the last axis holding the 3 "
                f"components; got shape {array.shape}"
            )
    if (
        isinstance(m, bool)
        or not isinstance(m, (int, float))
        or not np.isfinite(m)
        or m <= 0.0
    ):
        raise ValueError(
            f"m must be a finite, strictly positive number [kg], got {m!r}"
        )
    speed_squared = np.sum(u**2, axis=-1)
    if np.any(speed_squared == 0.0):
        raise ValueError(
            "a velocity is zero: a particle at rest has no path, its curvature is "
            "not defined"
        )

    gamma = np.sqrt(1.0 + speed_squared / c**2)
    force_over_q = gamma[..., np.newaxis] * e_field + np.cross(u, b_field)
    turning = np.linalg.norm(np.cross(u, force_over_q), axis=-1)
    return abs(q) * turning / (m * speed_squared**1.5)


def synchrotron_power(
    u: NDArray[np.float64], kappa: ArrayLike, q: float
) -> NDArray[np.float64]:
    """Power radiated by each particle because its orbit is curved.

    ``P = q^2 gamma^4 |v|^4 kappa^2 / (6 pi eps_0 c^3)`` (Carbajal et al. 2017,
    Eq. 15), with ``kappa`` the curvature of the orbit (``curvature``). Since
    ``gamma |v| = |u|`` it is computed as ``q^2 |u|^4 kappa^2 / (6 pi eps_0 c^3)``.

    Parameters
    ----------
    u : ndarray, shape (..., 3)
        Proper velocities ``u = gamma * v`` [m/s]. Any leading shape.
    kappa : array_like, shape (...)
        Curvature of the orbit [1/m] at the same states, of the shape of ``u``
        without its last axis, or any shape that broadcasts with it.
    q : float
        Charge [C]; only its square matters.

    Returns
    -------
    ndarray, shape (...)
        Radiated power [W], positive or zero.

    Raises
    ------
    ValueError
        If the last axis of ``u`` does not have length 3, or ``kappa`` does not
        broadcast with the leading shape of ``u``.

    Notes
    -----
    This is the part of the radiated power due to the acceleration perpendicular
    to the velocity, the one that curves the orbit. The Lienard formula has a
    second part, due to the acceleration along the velocity,
    ``q^4 E_par^2 / (6 pi eps_0 c^3 m^2)`` with ``E_par`` the electric field along
    ``v``. It is not included: it vanishes without electric field, which is the
    case of Eq. 15, and it is smaller than the first part by the factor
    ``(E_par / (gamma |v| |B| sin(theta)))^2`` otherwise.
    """
    u = np.asarray(u, dtype=np.float64)
    kappa = np.asarray(kappa, dtype=np.float64)
    if u.ndim == 0 or u.shape[-1] != 3:
        raise ValueError(
            "u must have shape (..., 3), the last axis holding the 3 components; "
            f"got shape {u.shape}"
        )
    try:
        np.broadcast_shapes(u.shape[:-1], kappa.shape)
    except ValueError:
        raise ValueError(
            "kappa must have the shape of u without its last axis, or broadcast "
            f"with it; got {kappa.shape} for u of shape {u.shape}"
        ) from None

    u_squared = np.sum(u**2, axis=-1)
    return q**2 * u_squared**2 * kappa**2 / (6.0 * np.pi * eps_0 * c**3)


def synchrotron_kernel_integral(x: ArrayLike) -> NDArray[np.float64]:
    """Integral of the modified Bessel function ``K_{5/3}`` from ``x`` to infinity.

    ``F(x) = int_x^inf K_{5/3}(eta) d eta``, dimensionless. It gives the shape of
    the synchrotron spectrum (Carbajal et al. 2017, Eq. 16, with
    ``x = lambda_c / lambda``).

    Near zero the integrand grows as ``eta^(-5/3)``, which a quadrature handles
    badly. The recurrence ``K_{nu-1} + K_{nu+1} = -2 K'_nu`` (DLMF 10.29.1) at
    ``nu = 2/3``, with ``K_{-1/3} = K_{1/3}``, removes this growth:

        F(x) = 2 K_{2/3}(x) - int_x^inf K_{1/3}(eta) d eta

    The remaining integrand grows only as ``eta^(-1/3)`` near zero and decays as
    ``exp(-eta)``; it is integrated by ``scipy.integrate.quad``, one call per
    value of ``x``.

    Parameters
    ----------
    x : array_like
        Lower bound of the integral, dimensionless, finite and strictly positive.
        Any shape.

    Returns
    -------
    ndarray
        ``F(x)``, of the shape of ``x``, strictly positive up to ``x`` of about 700.

    Raises
    ------
    ValueError
        If a value of ``x`` is not finite or not strictly positive.

    Notes
    -----
    Limits: ``F(x) ~ 2^(2/3) Gamma(2/3) x^(-2/3)`` as ``x -> 0`` and
    ``F(x) ~ sqrt(pi / (2 x)) exp(-x)`` as ``x -> inf``. Beyond ``x`` of about 700,
    ``scipy.special.kv`` underflows and the function returns 0, where the exact
    value is below ``1e-305``. The accuracy is that of ``scipy.special.kv``: about
    ``1e-13`` relative for ``x`` between 1 and 2, ``1e-15`` elsewhere. Each value
    costs about 0.1 ms.
    """
    x = np.asarray(x, dtype=np.float64)
    if not np.all(np.isfinite(x)) or np.any(x <= 0.0):
        raise ValueError(
            "x must hold finite, strictly positive values: the integral diverges "
            f"at x = 0; got min {np.min(x, initial=np.inf)!r}"
        )

    integral = np.empty_like(x)
    for index, x_value in np.ndenumerate(x):
        # Relative tolerance only: F(x) falls below the default absolute
        # tolerance of quad (1.5e-8) as soon as x > 17.
        tail_of_k_one_third, _ = quad(
            lambda eta: kv(1.0 / 3.0, eta),
            x_value,
            np.inf,
            epsabs=0.0,
            epsrel=1e-13,
            limit=200,
        )
        integral[index] = 2.0 * kv(2.0 / 3.0, x_value) - tail_of_k_one_third
    return integral


def synchrotron_spectrum(
    wavelength: ArrayLike,
    u: NDArray[np.float64],
    kappa: ArrayLike,
    q: float,
) -> NDArray[np.float64]:
    """Synchrotron power radiated per unit wavelength by each particle.

    Carbajal et al. (2017, Eq. 16):

        P(lambda) = (1 / sqrt 3) q^2 c / (eps_0 gamma^2 lambda^3) F(lambda_c / lambda)

    with ``F(x) = int_x^inf K_{5/3}`` (``synchrotron_kernel_integral``) and the
    critical wavelength ``lambda_c = (2 lambda_0 / 3) (m c^2 / E)^3
    = 4 pi / (3 kappa gamma^3)``, where ``lambda_0 = 2 pi / kappa`` and
    ``E = gamma m c^2``.

    Parameters
    ----------
    wavelength : array_like, shape (n_lambda,)
        Wavelengths [m], finite and strictly positive.
    u : ndarray, shape (..., 3)
        Proper velocities ``u = gamma * v`` [m/s]. Any leading shape.
    kappa : array_like, shape (...)
        Curvature of the orbit [1/m] at the same states (``curvature``), finite
        and positive or zero, of the shape of ``u`` without its last axis, or any
        shape that broadcasts with it.
    q : float
        Charge [C]; only its square matters.

    Returns
    -------
    ndarray, shape (..., n_lambda)
        Spectral power density [W/m], positive or zero. It is zero for every
        wavelength where ``kappa = 0``: a straight path does not radiate.

    Raises
    ------
    ValueError
        If ``wavelength`` is not one-dimensional or holds a value that is not
        finite and strictly positive, the last axis of ``u`` does not have length
        3, or ``kappa`` does not broadcast with ``u`` or holds a value that is not
        finite and positive or zero.

    Notes
    -----
    This is the spectrum of an ultra-relativistic particle. Its integral over all
    wavelengths is ``q^2 c gamma^4 kappa^2 / (6 pi eps_0)``, the power of
    ``synchrotron_power`` with ``|v|`` replaced by ``c``; the two differ by the
    factor ``(|v| / c)^4``, that is by about ``2 / gamma^2``. One call of
    ``synchrotron_kernel_integral`` is made per state and wavelength, about
    0.1 ms each.
    """
    wavelength = np.asarray(wavelength, dtype=np.float64)
    u = np.asarray(u, dtype=np.float64)
    kappa = np.asarray(kappa, dtype=np.float64)
    if (
        wavelength.ndim != 1
        or not np.all(np.isfinite(wavelength))
        or np.any(wavelength <= 0.0)
    ):
        raise ValueError(
            "wavelength must be a one-dimensional array of finite, strictly "
            f"positive values [m]; got shape {wavelength.shape}"
        )
    if u.ndim == 0 or u.shape[-1] != 3:
        raise ValueError(
            "u must have shape (..., 3), the last axis holding the 3 components; "
            f"got shape {u.shape}"
        )
    try:
        state_shape = np.broadcast_shapes(u.shape[:-1], kappa.shape)
    except ValueError:
        raise ValueError(
            "kappa must have the shape of u without its last axis, or broadcast "
            f"with it; got {kappa.shape} for u of shape {u.shape}"
        ) from None
    if not np.all(np.isfinite(kappa)) or np.any(kappa < 0.0):
        raise ValueError("kappa must hold finite values, positive or zero [1/m]")

    gamma = np.broadcast_to(np.sqrt(1.0 + np.sum(u**2, axis=-1) / c**2), state_shape)
    kappa = np.broadcast_to(kappa, state_shape)
    spectrum = np.zeros(state_shape + wavelength.shape)

    # A straight path (kappa = 0) has an infinite critical wavelength and no
    # spectrum: those states keep their zeros.
    curved = kappa > 0.0
    gamma_curved = gamma[curved][:, np.newaxis]
    critical_wavelength = 4.0 * np.pi / (3.0 * kappa[curved] * gamma[curved] ** 3)
    x = critical_wavelength[:, np.newaxis] / wavelength
    prefactor = q**2 * c / (np.sqrt(3.0) * eps_0 * gamma_curved**2 * wavelength**3)
    spectrum[curved] = prefactor * synchrotron_kernel_integral(x)
    return spectrum
