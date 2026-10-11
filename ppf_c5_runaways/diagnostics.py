"""Diagnostics computed from trajectories: pure functions of arrays, in SI units.

This module reads no configuration and writes nothing. It receives arrays, as
returned by ``integrate`` or read back by ``io.read_result``, and returns arrays.
"""

import numpy as np
from numpy.typing import NDArray

from ppf_c5_runaways.constants import c
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

    ``psi = arccos(u . b_hat / |u|)``: 0 along ``B``, ``pi / 2`` perpendicular to it.

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
    cosine = (u @ b_field) / (speed * b_norm)
    return np.arccos(np.clip(cosine, -1.0, 1.0))  # clip: rounding may give 1 + eps


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
    relative_shift: float = 1.0e-6,
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
        and to ``|u|`` for a velocity. Default ``1e-6``: truncation error about
        ``1e-12`` and rounding error about ``eps / 1e-6 = 2e-10`` on each entry.

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
            plus = slice(2 * j * n_part, (2 * j + 1) * n_part)
            minus = slice((2 * j + 1) * n_part, (2 * j + 2) * n_part)
            shifted[plus, j] += shift[:, j]
            shifted[minus, j] -= shift[:, j]
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
