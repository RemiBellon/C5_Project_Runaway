"""Radiation reaction: the Landau-Lifshitz force and the power it takes away.

Pure numerics in SI units, like ``diagnostics``: this module reads no
configuration and writes nothing. It receives arrays and returns arrays.
"""

import math
from numbers import Real

import numpy as np
from numpy.typing import ArrayLike, NDArray

from ppf_c5_runaways.constants import c, eps_0, hbar


def _check_charge_and_mass(q: float, m: float) -> None:
    """Refuse a charge that is not a finite number, or a mass that is not > 0.

    Raises
    ------
    ValueError
        With a message that says what was expected and what was received.
    """
    if isinstance(q, bool) or not isinstance(q, (int, float)) or not np.isfinite(q):
        raise ValueError(f"q must be a finite number [C], got {q!r}")
    if (
        isinstance(m, bool)
        or not isinstance(m, (int, float))
        or not np.isfinite(m)
        or m <= 0.0
    ):
        raise ValueError(
            f"m must be a finite, strictly positive number [kg], got {m!r}"
        )


def _check_radiation_arguments(
    u: ArrayLike, e_field: ArrayLike, b_field: ArrayLike, q: float, m: float
) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
    """Check the arguments shared by the functions of this module.

    Parameters
    ----------
    u, e_field, b_field, q, m
        See ``landau_lifshitz_force``.

    Returns
    -------
    u, e_field, b_field : ndarray, shape (..., 3)
        The inputs as ``float64`` arrays.

    Raises
    ------
    ValueError
        If a last axis does not have length 3, the three arrays do not broadcast
        together, ``q`` is not a finite number or ``m`` is not a finite, strictly
        positive number. The message says what was expected and what was received.
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
    try:
        np.broadcast_shapes(u.shape, e_field.shape, b_field.shape)
    except ValueError:
        raise ValueError(
            "u, e_field and b_field must broadcast together; got shapes "
            f"{u.shape}, {e_field.shape} and {b_field.shape}"
        ) from None
    _check_charge_and_mass(q, m)
    return u, e_field, b_field


def landau_lifshitz_force(
    u: ArrayLike, e_field: ArrayLike, b_field: ArrayLike, q: float, m: float
) -> NDArray[np.float64]:
    """Radiation reaction force on each particle, in the Landau-Lifshitz form.

    Carbajal et al. (2017, Eq. 3) without its first term, the one in ``D/Dt`` of
    the fields (roadmap 2.5)::

        F_R = K { (E . v) E / c^2 + (E + v x B) x B
                  - (gamma^2 / c^2) [ (E + v x B)^2 - (E . v)^2 / c^2 ] v },

        K = q^4 / (6 pi eps_0 m^2 c^3).

    The two terms kept go as ``q^4``: the force does not depend on the sign of the
    charge. Both are kept, as Tamburini et al. (2010) recommend.

    Parameters
    ----------
    u : array_like, shape (..., 3)
        Proper velocities ``u = gamma * v`` [m/s]. Any leading shape: ``(3,)``,
        ``(n_part, 3)`` or ``(n_saved, n_part, 3)``.
    e_field, b_field : array_like, shape (..., 3)
        Electric field [V/m] and magnetic field [T] at the position of each
        particle, at the time of ``u``. Any shape that broadcasts with ``u``, for
        example ``(3,)`` for uniform fields.
    q : float
        Signed charge [C]; only ``q^4`` enters.
    m : float
        Rest mass [kg], finite and strictly positive.

    Returns
    -------
    ndarray, shape (..., 3)
        The force [N], of the broadcast shape of the three arrays.

    Raises
    ------
    ValueError
        If a last axis does not have length 3, the arrays do not broadcast
        together, ``q`` is not finite, or ``m`` is not finite and strictly positive.

    Notes
    -----
    The omitted term, ``q^3 gamma (DE/Dt + v x DB/Dt) / (6 pi eps_0 m c^3)``, is zero
    in static uniform fields. In the static but non-uniform tokamak field it is not
    zero; Carbajal et al. report it three to four orders of magnitude below the
    last term, and drop it as done here.
    """
    u, e_field, b_field = _check_radiation_arguments(u, e_field, b_field, q, m)

    gamma = np.sqrt(1.0 + np.sum(u**2, axis=-1, keepdims=True) / c**2)
    v = u / gamma
    lorentz = e_field + np.cross(v, b_field)  # E + v x B [V/m]
    e_dot_v = np.sum(e_field * v, axis=-1, keepdims=True)
    lorentz_squared = np.sum(lorentz**2, axis=-1, keepdims=True) - (e_dot_v / c) ** 2

    prefactor = q**4 / (6.0 * np.pi * eps_0 * m**2 * c**3)
    return prefactor * (
        e_dot_v / c**2 * e_field
        + np.cross(lorentz, b_field)
        - gamma**2 / c**2 * lorentz_squared * v
    )


def radiated_power(
    u: ArrayLike, e_field: ArrayLike, b_field: ArrayLike, q: float, m: float
) -> NDArray[np.float64]:
    """Power taken from each particle by the Landau-Lifshitz force, ``-F_R . v``.

    Carbajal et al. (2017, Eq. 8)::

        P_R = -K { E^2 + (v x B) . E
                   - gamma^2 [ (E + v x B)^2 - (E . v)^2 / c^2 ] },

    with ``K = q^4 / (6 pi eps_0 m^2 c^3)`` as in ``landau_lifshitz_force``. It
    equals ``-F_R . v`` exactly (the identity ``gamma^2 v^2 / c^2 = gamma^2 - 1``
    turns one into the other). Without electric field it is
    ``K gamma^2 |v x B|^2``, the synchrotron power of Eq. 12 of the paper.

    Parameters
    ----------
    u, e_field, b_field, q, m
        As ``landau_lifshitz_force``.

    Returns
    -------
    ndarray, shape (...)
        The power [W], of the broadcast shape of the three arrays without the
        last axis.

    Raises
    ------
    ValueError
        As ``landau_lifshitz_force``.

    Notes
    -----
    With an electric field this is not the power of the Lienard formula, which is
    ``K gamma^2 [ (E + v x B)^2 - (E . v)^2 / c^2 ]``: the two differ by
    ``K E . (E + v x B)``, the rate of change of the Schott energy (relation
    established for this project, OPEN_POINTS N42). This difference is smaller than
    the power by about ``1 / gamma^2`` for a runaway electron. For a slow particle
    in an electric field it is not small: the power can then be negative, and it is
    the difference of two nearly equal terms, so its relative rounding error grows
    as ``E^2`` over the result.
    """
    u, e_field, b_field = _check_radiation_arguments(u, e_field, b_field, q, m)

    gamma_squared = 1.0 + np.sum(u**2, axis=-1) / c**2
    v = u / np.sqrt(gamma_squared)[..., np.newaxis]
    v_cross_b = np.cross(v, b_field)
    e_dot_v = np.sum(e_field * v, axis=-1)
    lorentz_squared = np.sum((e_field + v_cross_b) ** 2, axis=-1) - (e_dot_v / c) ** 2

    prefactor = q**4 / (6.0 * np.pi * eps_0 * m**2 * c**3)
    return -prefactor * (
        np.sum(e_field**2, axis=-1)
        + np.sum(v_cross_b * e_field, axis=-1)
        - gamma_squared * lorentz_squared
    )


def _rest_frame_electric_field(
    u: NDArray[np.float64], e_field: NDArray[np.float64], b_field: NDArray[np.float64]
) -> NDArray[np.float64]:
    """Electric field in the rest frame of each particle [V/m], shape (...).

    ``gamma sqrt((E + v x B)^2 - (E . v)^2 / c^2)``, the Lorentz invariant that both
    validity parameters below are built on (Landau & Lifshitz, The classical theory
    of fields, Sec. 76). Rounding can make the bracket slightly negative when ``v``
    is nearly along ``E`` with no magnetic force; it is then set to zero.
    """
    gamma = np.sqrt(1.0 + np.sum(u**2, axis=-1) / c**2)
    v = u / gamma[..., np.newaxis]
    lorentz = e_field + np.cross(v, b_field)
    e_dot_v = np.sum(e_field * v, axis=-1)
    bracket = np.sum(lorentz**2, axis=-1) - (e_dot_v / c) ** 2
    return gamma * np.sqrt(np.maximum(bracket, 0.0))


def quantum_parameter(
    u: ArrayLike, e_field: ArrayLike, b_field: ArrayLike, q: float, m: float
) -> NDArray[np.float64]:
    """Quantum parameter ``chi`` of each particle: classical radiation needs chi << 1.

    ``chi = E_rest / E_S``, the electric field in the rest frame of the particle
    over the critical field of quantum electrodynamics ``E_S = m^2 c^3 / (|q| hbar)``
    (1.3e18 V/m for the electron)::

        chi = (|q| hbar / (m^2 c^3)) gamma sqrt((E + v x B)^2 - (E . v)^2 / c^2).

    Roadmap D5. When ``chi`` is no longer small, the emission of a photon takes a
    sizeable part of the energy of the particle and the classical force of
    Landau and Lifshitz overestimates the losses.

    Parameters
    ----------
    u, e_field, b_field, q, m
        As ``landau_lifshitz_force``.

    Returns
    -------
    ndarray, shape (...)
        ``chi``, dimensionless, of the broadcast shape of the three arrays without
        the last axis.

    Raises
    ------
    ValueError
        As ``landau_lifshitz_force``.

    Notes
    -----
    ``chi`` does not depend on ``rr_scale``: scaling the radiation force does not
    change the photon energies. For a runaway electron of a few tens of MeV in a
    tokamak field of a few teslas it is of order 1e-8.
    """
    u, e_field, b_field = _check_radiation_arguments(u, e_field, b_field, q, m)
    critical_field = m**2 * c**3 / (abs(q) * hbar)  # E_S [V/m]
    return _rest_frame_electric_field(u, e_field, b_field) / critical_field


def landau_lifshitz_parameter(
    u: ArrayLike,
    e_field: ArrayLike,
    b_field: ArrayLike,
    q: float,
    m: float,
    rr_scale: float = 1.0,
) -> NDArray[np.float64]:
    """Small parameter ``eta`` of the Landau-Lifshitz force: it must be << 1.

    The force of Landau and Lifshitz is the first term of an expansion of the
    Lorentz-Abraham-Dirac force in ``tau_0`` times the rest-frame frequency of the
    motion. Its small parameter is::

        eta = rr_scale tau_0 |q| E_rest / (m c),   tau_0 = q^2 / (6 pi eps_0 m c^3),

    with ``E_rest`` the rest-frame electric field of ``quantum_parameter``. In a
    magnetic field alone, for ``v`` perpendicular to ``B`` and ``v -> c``, it is
    the ``tau_0 Omega gamma`` of the roadmap (D5), ``Omega = |q| B / m``. The
    factor ``rr_scale`` enters because the pushers multiply the force by it, as
    if ``tau_0`` were ``rr_scale tau_0``.

    Parameters
    ----------
    u, e_field, b_field, q, m
        As ``landau_lifshitz_force``.
    rr_scale : float, optional
        Factor on the radiation force, as in the pushers. Default 1, the physical
        force.

    Returns
    -------
    ndarray, shape (...)
        ``eta``, dimensionless, of the broadcast shape of the three arrays without
        the last axis.

    Raises
    ------
    ValueError
        As ``landau_lifshitz_force``, or if ``rr_scale`` is not a finite number
        ``>= 0``.

    Notes
    -----
    ``eta = (2 alpha / 3) rr_scale chi`` (relation established for this project),
    with ``alpha`` the fine-structure constant: at ``rr_scale = 1`` the classical
    condition is met whenever the quantum one is.
    """
    u, e_field, b_field = _check_radiation_arguments(u, e_field, b_field, q, m)
    if (
        isinstance(rr_scale, bool)
        or not isinstance(rr_scale, Real)
        or not math.isfinite(rr_scale)
        or rr_scale < 0.0
    ):
        raise ValueError(f"rr_scale must be a finite number >= 0, got {rr_scale!r}")
    tau_0 = q**2 / (6.0 * np.pi * eps_0 * m * c**3)  # [s]
    rest_field = _rest_frame_electric_field(u, e_field, b_field)
    return rr_scale * tau_0 * abs(q) * rest_field / (m * c)


def maximum_lorentz_factor(
    e_parallel: ArrayLike,
    b_field: ArrayLike,
    pitch_angle: ArrayLike,
    q: float,
    m: float,
) -> NDArray[np.float64]:
    """Lorentz factor at which synchrotron losses balance the electric acceleration.

    For an ultra-relativistic particle at pitch angle ``theta`` in a magnetic field
    ``B``, with an electric field ``E`` along ``B`` that accelerates it, the power
    gained ``|q| E c cos(theta)`` equals the power radiated
    ``K gamma^2 c^2 B^2 sin^2(theta)`` (``K`` of ``landau_lifshitz_force``) at::

        gamma_max^2 = 6 pi eps_0 m^2 c^2 E cos(theta) / (|q|^3 B^2 sin^2(theta)).

    Roadmap 2.5 (the formula behind the 146, 47 and 25 MeV that Carbajal et al.
    2017, Sec. III.D, quote for 4 V/m and 10, 30 and 50 degrees). It takes the
    pitch angle as fixed, which is the "0-D" limit; in a tokamak, ``theta`` and
    ``B`` change along the orbit.

    Parameters
    ----------
    e_parallel : array_like
        Magnitude of the accelerating electric field along ``B`` [V/m], > 0.
    b_field : array_like
        Magnitude of the magnetic field [T], > 0.
    pitch_angle : array_like
        Pitch angle ``theta`` [rad], strictly between 0 and pi/2.
    q, m
        As ``landau_lifshitz_force``.

    Returns
    -------
    ndarray
        ``gamma_max``, of the broadcast shape of the three arrays.

    Raises
    ------
    ValueError
        If a field is not strictly positive, a pitch angle is outside (0, pi/2), the
        arrays do not broadcast, ``q`` is not finite, or ``m`` is not finite and
        strictly positive.

    Notes
    -----
    The balance neglects terms of relative order ``1 / gamma^2`` (``v = c``) and
    ``(E / (c B))^2`` in the radiated power of ``radiated_power``.
    """
    _check_charge_and_mass(q, m)
    e_parallel = np.asarray(e_parallel, dtype=np.float64)
    b_field = np.asarray(b_field, dtype=np.float64)
    pitch_angle = np.asarray(pitch_angle, dtype=np.float64)
    if not np.all(e_parallel > 0.0) or not np.all(b_field > 0.0):
        raise ValueError(
            "e_parallel [V/m] and b_field [T] must be strictly positive magnitudes"
        )
    if not np.all((pitch_angle > 0.0) & (pitch_angle < 0.5 * np.pi)):
        raise ValueError(
            "pitch_angle must lie strictly between 0 and pi/2 [rad]: there is no "
            "balance for theta = 0 (no loss) or theta >= pi/2 (no gain)"
        )
    try:
        np.broadcast_shapes(e_parallel.shape, b_field.shape, pitch_angle.shape)
    except ValueError:
        raise ValueError(
            "e_parallel, b_field and pitch_angle must broadcast together; got "
            f"shapes {e_parallel.shape}, {b_field.shape} and {pitch_angle.shape}"
        ) from None
    gamma_squared = (
        6.0
        * np.pi
        * eps_0
        * m**2
        * c**2
        * e_parallel
        * np.cos(pitch_angle)
        / (abs(q) ** 3 * b_field**2 * np.sin(pitch_angle) ** 2)
    )
    return np.sqrt(gamma_squared)


def radiating_helix(
    u0: ArrayLike,
    b_z: float,
    time: ArrayLike,
    q: float,
    m: float,
    rr_scale: float = 1.0,
) -> NDArray[np.float64]:
    """Exact proper velocity of particles that radiate in a uniform magnetic field.

    The field is ``B = b_z z`` with ``b_z > 0`` and no electric field; the force is
    ``q v x B + rr_scale F_R`` with ``F_R`` of ``landau_lifshitz_force``. Its part
    along ``B`` reduces ``u_par`` in the same ratio as ``gamma``, so
    ``beta_par = u_par / (gamma c)`` is constant, and ``gamma`` obeys::

        d gamma / dt = -(gamma^2 / gamma_par^2 - 1) / tau_R,
        tau_R = 6 pi eps_0 m^3 c^3 / (rr_scale q^4 b_z^2),

    with ``gamma_par = 1 / sqrt(1 - beta_par^2)``, the value ``gamma`` decays to
    (roadmap 2.5, derived again in RAPPORT §5.44). Its solution is::

        gamma = gamma_par (1 + y) / (1 - y),   y = A exp(-k t),
        k = 2 / (gamma_par tau_R),   A = (gamma_0 - gamma_par) / (gamma_0 + gamma_par).

    In the plane perpendicular to ``B``, ``F_R`` is along ``-u_perp``: ``u_perp``
    shrinks and turns as without radiation, by the angle ``-(q b_z / m) tau``,
    where ``tau = integral of dt / gamma`` is the proper time (the form of Elkina
    et al. 2014, Eq. 29)::

        tau = [ t + (2 / k) ln((1 + y) / (1 + A)) ] / gamma_par.

    Parameters
    ----------
    u0 : array_like, shape (n_part, 3)
        Proper velocities ``u = gamma v`` [m/s] at ``t = 0``.
    b_z : float
        Magnetic field along ``z`` [T], finite and strictly positive.
    time : array_like, shape (n_t,)
        Times [s] at which the velocity is returned, finite and ``>= 0``.
    q : float
        Signed charge [C], for example ``-e`` for an electron.
    m : float
        Rest mass [kg], finite and strictly positive.
    rr_scale : float, optional
        Factor on the radiation force, as in the pushers; finite and strictly
        positive (without radiation the motion is the helix of the pushers' tests).
        Default 1, the physical force.

    Returns
    -------
    ndarray, shape (n_t, n_part, 3)
        Proper velocities [m/s] at the times ``time``.

    Raises
    ------
    ValueError
        If ``u0`` is not of shape (n_part, 3) with finite values, ``time`` is not
        one-dimensional with finite values ``>= 0``, ``b_z`` or ``rr_scale`` is not
        a finite number ``> 0``, ``q`` is not finite, or ``m`` is not finite and
        strictly positive.

    Notes
    -----
    The formulas are written so that nothing cancels when ``gamma -> 1``:
    ``A = gamma_par^2 w / (gamma_0 + gamma_par)^2`` with ``w = |u0_perp|^2 / c^2``,
    ``|u_perp| = 2 c sqrt(y) / (1 - y)``, and the logarithm of ``tau`` through
    ``log1p`` and ``expm1``. The field and the radiation force are imposed and the
    particle is a test particle; the term in ``D/Dt`` of the Landau-Lifshitz force
    is left out, as in ``landau_lifshitz_force``.
    """
    _check_charge_and_mass(q, m)
    for name, value in (("b_z", b_z), ("rr_scale", rr_scale)):
        if (
            isinstance(value, bool)
            or not isinstance(value, Real)
            or not math.isfinite(value)
            or value <= 0.0
        ):
            raise ValueError(f"{name} must be a finite number > 0, got {value!r}")
    u0 = np.asarray(u0, dtype=np.float64)
    time = np.asarray(time, dtype=np.float64)
    if u0.ndim != 2 or u0.shape[1] != 3 or not np.all(np.isfinite(u0)):
        raise ValueError(
            f"u0 must have shape (n_part, 3) with finite values; got shape {u0.shape}"
        )
    if time.ndim != 1 or not np.all(np.isfinite(time)) or np.any(time < 0.0):
        raise ValueError(
            f"time must be one-dimensional with finite values >= 0; got shape "
            f"{time.shape}"
        )

    radiation_time = (
        6.0 * np.pi * eps_0 * m**3 * c**3 / (rr_scale * q**4 * b_z**2)
    )  # tau_R [s]
    w = (u0[:, 0] ** 2 + u0[:, 1] ** 2) / c**2
    gamma0 = np.sqrt(1.0 + w + u0[:, 2] ** 2 / c**2)
    gamma_par = gamma0 / np.sqrt(1.0 + w)
    a = gamma_par**2 * w / (gamma0 + gamma_par) ** 2
    k = 2.0 / (gamma_par * radiation_time)  # [1/s]

    t = time[:, np.newaxis]  # (n_t, 1), against (n_part,) for each particle
    y = a * np.exp(-k * t)
    gamma = gamma_par * (1.0 + y) / (1.0 - y)
    proper_time = (t + (2.0 / k) * np.log1p(a * np.expm1(-k * t) / (1.0 + a))) / (
        gamma_par
    )
    angle = np.arctan2(u0[:, 1], u0[:, 0]) - q * b_z / m * proper_time
    u_perp = 2.0 * c * np.sqrt(y) / (1.0 - y)

    u = np.empty((time.size, u0.shape[0], 3))
    u[..., 0] = u_perp * np.cos(angle)
    u[..., 1] = u_perp * np.sin(angle)
    u[..., 2] = u0[:, 2] * gamma / gamma0
    return u
