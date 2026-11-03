"""Initial conditions of the particles, built from physical quantities.

A run starts from positions and proper velocities in cartesian coordinates. A
physicist rather thinks of a particle by its kinetic energy, its pitch angle and its
gyration phase: this module does the conversion. It is pure numerics: it receives
numbers and arrays and returns arrays, reads no configuration and writes nothing.
"""

import math
from numbers import Integral, Real

import numpy as np
from numpy.typing import ArrayLike, NDArray

from ppf_c5_runaways.constants import c
from ppf_c5_runaways.fields import Field, toroidal_to_cartesian


def proper_velocity(
    kinetic_energy: ArrayLike,
    pitch_angle: ArrayLike,
    gyrophase: ArrayLike,
    b_direction: ArrayLike,
    m: float,
) -> NDArray[np.float64]:
    """Build the proper velocity ``u = gamma * v`` of particles of given energy.

    The size of ``u`` comes from the kinetic energy ``K = (gamma - 1) m c^2``, and
    its direction from two angles measured from the magnetic field::

        |u| = c sqrt(k (k + 2)),  k = K / (m c^2)
        u = |u| [cos(psi) b_hat + sin(psi) (cos(phi) e1 + sin(phi) e2)]

    with ``psi`` the pitch angle and ``phi`` the gyration phase. ``|u|`` is
    ``c sqrt(gamma^2 - 1)`` with ``gamma = 1 + k``, written so that nothing cancels
    when the particle is slow (``k << 1``).

    ``(e1, e2, b_hat)`` is the right-handed basis of ``diagnostics.gyration_phase``:
    ``e1`` is ``b_hat`` crossed with the coordinate axis least aligned with it, and
    ``e2 = b_hat x e1``. So this function is the inverse of
    ``diagnostics.kinetic_energy``, ``diagnostics.pitch_angle`` and
    ``diagnostics.gyration_phase`` taken together.

    Parameters
    ----------
    kinetic_energy : float or array_like, shape (n_part,)
        Kinetic energy ``K`` [J], finite and not negative. Multiply an energy in
        electron-volts by ``constants.e`` to get joules.
    pitch_angle : float or array_like, shape (n_part,)
        Angle ``psi`` [rad] between the velocity and the magnetic field, in
        ``[0, pi]``: 0 along the field, ``pi / 2`` perpendicular to it.
    gyrophase : float or array_like, shape (n_part,)
        Gyration phase ``phi`` [rad], any finite value. It has no effect when the
        velocity is along the field.
    b_direction : array_like, shape (3,) or (n_part, 3)
        Direction of the magnetic field at each particle, any non-zero length:
        only the direction is used. One vector for all the particles, or one each.
    m : float
        Rest mass [kg], finite and strictly positive.

    Returns
    -------
    ndarray, shape (n_part, 3)
        Proper velocities [m/s], one row per particle (``n_part = 1`` when every
        argument is a single number or vector).

    Raises
    ------
    ValueError
        If an energy is negative or not finite, a pitch angle is outside
        ``[0, pi]``, a phase is not finite, a field direction is zero or not
        finite, ``m`` is not a finite, strictly positive number, or the shapes do
        not agree.
    """
    if (
        isinstance(m, bool)
        or not isinstance(m, (int, float))
        or not np.isfinite(m)
        or m <= 0.0
    ):
        raise ValueError(
            f"m must be a finite, strictly positive number [kg], got {m!r}"
        )
    b_direction = np.atleast_2d(np.asarray(b_direction, dtype=np.float64))
    if b_direction.ndim != 2 or b_direction.shape[1] != 3:
        raise ValueError(
            "b_direction must have shape (3,) or (n_part, 3); got shape "
            f"{np.shape(b_direction)}"
        )
    try:
        energy, psi, phi, _ = np.broadcast_arrays(
            np.atleast_1d(np.asarray(kinetic_energy, dtype=np.float64)),
            np.atleast_1d(np.asarray(pitch_angle, dtype=np.float64)),
            np.atleast_1d(np.asarray(gyrophase, dtype=np.float64)),
            b_direction[:, 0],
        )
    except ValueError as error:
        raise ValueError(
            "kinetic_energy, pitch_angle and gyrophase must be single numbers or "
            "have one value per particle, as many as the rows of b_direction"
        ) from error
    if energy.ndim != 1:
        raise ValueError(
            "kinetic_energy, pitch_angle and gyrophase must be single numbers or "
            f"have shape (n_part,); got an array of {energy.ndim} dimensions"
        )
    if not np.all(np.isfinite(energy)) or np.any(energy < 0.0):
        raise ValueError("kinetic_energy must be finite and not negative [J]")
    if not np.all((psi >= 0.0) & (psi <= np.pi)):
        raise ValueError("pitch_angle must be in [0, pi] [rad]")
    if not np.all(np.isfinite(phi)):
        raise ValueError("gyrophase must be finite [rad]")
    b_norm = np.linalg.norm(b_direction, axis=1, keepdims=True)
    if not np.all(np.isfinite(b_direction)) or np.any(b_norm == 0.0):
        raise ValueError("b_direction must be finite and not the zero vector")

    b_hat = np.broadcast_to(b_direction / b_norm, (energy.size, 3))

    # e1: perpendicular to b_hat, built from the coordinate axis least aligned with
    # it, so that the cross product is well conditioned. Same rule as in
    # ``diagnostics.gyration_phase``.
    axis = np.zeros((energy.size, 3))
    axis[np.arange(energy.size), np.argmin(np.abs(b_hat), axis=1)] = 1.0
    e1 = np.cross(b_hat, axis)
    e1 /= np.linalg.norm(e1, axis=1, keepdims=True)
    e2 = np.cross(b_hat, e1)

    k = energy / (m * c**2)
    speed = (c * np.sqrt(k * (k + 2.0)))[:, np.newaxis]  # |u| [m/s], (n_part, 1)
    psi, phi = psi[:, np.newaxis], phi[:, np.newaxis]
    return speed * (
        np.cos(psi) * b_hat + np.sin(psi) * (np.cos(phi) * e1 + np.sin(phi) * e2)
    )


def position_from_guiding_centre(
    guiding_centre: ArrayLike,
    u: ArrayLike,
    b_field: ArrayLike,
    q: float,
    m: float,
) -> NDArray[np.float64]:
    """Place each particle on its Larmor circle around a given guiding centre.

    A particle gyrates around its guiding centre ``X`` at the distance of the
    Larmor radius. Its position is ``x = X + rho`` with the Larmor vector::

        rho = m (B x u) / (q |B|^2),    |rho| = m |u_perp| / (|q| |B|)

    ``rho`` is perpendicular to ``B`` and to ``u``. It follows from the equation of
    motion ``du/dt = (q / (m gamma)) u x B``: with ``gamma`` constant,
    ``d rho/dt = u_perp / gamma``, the perpendicular velocity, so ``x - rho`` only
    moves along the field. The charge is signed: an electron and a positron of the
    same velocity are on opposite sides of their guiding centre.

    Parameters
    ----------
    guiding_centre : array_like, shape (3,) or (n_part, 3)
        Positions ``X`` of the guiding centres [m].
    u : array_like, shape (3,) or (n_part, 3)
        Proper velocities ``u = gamma * v`` [m/s] of the particles.
    b_field : array_like, shape (3,) or (n_part, 3)
        Magnetic field [T] seen by each particle, not zero. In a field that is not
        uniform, give the field at the guiding centre: the result is then exact to
        first order in the Larmor radius over the length on which the field varies.
    q : float
        Signed charge [C], finite and not zero, for example ``-e`` for an electron.
    m : float
        Rest mass [kg], finite and strictly positive.

    Returns
    -------
    ndarray, shape (n_part, 3)
        Positions ``x`` of the particles [m] (``n_part = 1`` when every argument is
        a single vector).

    Raises
    ------
    ValueError
        If an array does not have shape ``(3,)`` or ``(n_part, 3)``, the numbers of
        particles do not agree, a value is not finite, a magnetic field is zero,
        ``q`` is zero or not finite, or ``m`` is not finite and strictly positive.
    """
    for name, value, positive in (("q", q, False), ("m", m, True)):
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not np.isfinite(value)
            or value == 0.0
            or (positive and value < 0.0)
        ):
            wanted = "strictly positive" if positive else "non-zero"
            raise ValueError(f"{name} must be a finite, {wanted} number, got {value!r}")
    arrays = []
    for name, vectors in (
        ("guiding_centre", guiding_centre),
        ("u", u),
        ("b_field", b_field),
    ):
        array = np.atleast_2d(np.asarray(vectors, dtype=np.float64))
        if array.ndim != 2 or array.shape[1] != 3:
            raise ValueError(
                f"{name} must have shape (3,) or (n_part, 3); got shape "
                f"{np.shape(vectors)}"
            )
        if not np.all(np.isfinite(array)):
            raise ValueError(f"{name} must be finite (no nan, no inf)")
        arrays.append(array)
    try:
        centre, u, b_field = np.broadcast_arrays(*arrays)
    except ValueError as error:
        raise ValueError(
            "guiding_centre, u and b_field must have the same number of particles "
            "(or be single vectors); got shapes "
            f"{[array.shape for array in arrays]}"
        ) from error
    b_squared = np.sum(b_field**2, axis=1, keepdims=True)
    if np.any(b_squared == 0.0):
        raise ValueError("b_field must not be the zero vector")

    larmor_vector = m * np.cross(b_field, u) / (q * b_squared)
    return centre + larmor_vector


def uniform_poloidal_disc(
    n_part: int, radius: float, r0: float, rng: np.random.Generator
) -> NDArray[np.float64]:
    """Draw positions uniformly in area in a disc of a poloidal plane of the torus.

    The disc is centred on the magnetic axis, in the poloidal plane ``zeta = 0`` of
    the toroidal coordinates of Carbajal (``fields.toroidal_to_cartesian``): the
    half-plane ``x = 0``, ``y > 0``. Carbajal et al. (2017, Sec. III) distribute
    the electrons of their confinement study "uniformly on a poloidal plane"; for
    their Fig. 12 they only say that the electrons start inside a circle of that
    plane (Sec. III.D), and uniform in area there is the choice of this project
    (OPEN_POINTS N48). Uniform in area means that the fraction of points within
    the minor radius ``r`` is ``(r / radius)^2``, hence, with ``U`` and ``V``
    independent and uniform in ``[0, 1)``::

        r = radius sqrt(U),   theta = 2 pi V,   zeta = 0.

    Parameters
    ----------
    n_part : int
        Number of positions, at least 1.
    radius : float
        Radius of the disc [m], finite, strictly positive and smaller than ``r0``.
    r0 : float
        Major radius ``R0`` [m], finite and strictly positive.
    rng : numpy.random.Generator
        Source of the random numbers, built by the caller with an explicit seed
        (``numpy.random.default_rng(seed)``).

    Returns
    -------
    ndarray, shape (n_part, 3)
        Cartesian positions ``(x, y, z)`` [m]; ``x`` is exactly 0.

    Raises
    ------
    ValueError
        If ``n_part`` is not an integer ``>= 1``, ``r0`` is not a finite number
        ``> 0``, ``radius`` is not a finite number in ``(0, r0)``, or ``rng`` is
        not a ``numpy.random.Generator``.

    Notes
    -----
    The two numbers ``(U, V)`` of each particle are drawn one after the other, as
    one row of ``rng.random((n_part, 2))``. So the first ``k`` positions drawn
    from a seed do not depend on ``n_part``: a run on the first ``k`` particles of
    an ensemble starts from the same positions as the full run.

    These are positions of the particles, not of their guiding centres, which lie
    one Larmor radius away (``position_from_guiding_centre``).
    """
    if isinstance(n_part, bool) or not isinstance(n_part, Integral) or n_part < 1:
        raise ValueError(f"n_part must be an integer >= 1, got {n_part!r}")
    if (
        isinstance(r0, bool)
        or not isinstance(r0, Real)
        or not math.isfinite(r0)
        or r0 <= 0.0
    ):
        raise ValueError(f"r0 must be a finite number > 0 [m], got {r0!r}")
    if (
        isinstance(radius, bool)
        or not isinstance(radius, Real)
        or not math.isfinite(radius)
        or not 0.0 < radius < r0
    ):
        raise ValueError(
            f"radius must be a finite number in (0, r0) = (0, {r0}) m, got {radius!r}"
        )
    if not isinstance(rng, np.random.Generator):
        raise ValueError(
            "rng must be a numpy.random.Generator, built with "
            f"numpy.random.default_rng(seed); got {type(rng).__name__}"
        )

    draws = rng.random((int(n_part), 2))  # row i: (U, V) of particle i
    minor_radius = radius * np.sqrt(draws[:, 0])
    poloidal_angle = 2.0 * np.pi * draws[:, 1]
    return toroidal_to_cartesian(minor_radius, poloidal_angle, 0.0, r0)


def poloidal_disc_ensemble(
    n_part: int,
    radius: float,
    kinetic_energy: float,
    pitch_angle: float,
    field: Field,
    r0: float,
    m: float,
    seed: int,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Build an ensemble of particles of one energy and one pitch angle in a disc.

    This is the kind of ensemble of Carbajal et al. (2017, Sec. III.D, Fig. 12):
    every particle has the same kinetic energy and the same pitch angle, measured
    from the magnetic field where it starts, and they differ by their position and
    their gyration phase. The positions are those of ``uniform_poloidal_disc``; the
    magnetic field is evaluated there at ``t = 0``, the start of a run; the
    gyration phases are drawn uniformly in ``[0, 2 pi)``; ``proper_velocity`` then
    builds the velocities.

    Parameters
    ----------
    n_part : int
        Number of particles, at least 1.
    radius : float
        Radius of the disc of starting positions [m], in ``(0, r0)``.
    kinetic_energy : float
        Kinetic energy ``(gamma - 1) m c^2`` [J] of every particle, finite and not
        negative. Multiply an energy in electron-volts by ``constants.e``.
    pitch_angle : float
        Pitch angle [rad] of every particle from the local magnetic field, in
        ``[0, pi]``; below ``pi / 2`` the particles move along ``B``.
    field : Field
        Function ``field(x, t) -> (E, B)`` (see ``ppf_c5_runaways.fields``). ``B``
        must not be zero at the starting positions.
    r0 : float
        Major radius ``R0`` [m] of the torus, the centre of the disc being on the
        magnetic axis at ``R = r0``. Finite and strictly positive.
    m : float
        Rest mass [kg], finite and strictly positive.
    seed : int
        Seed of the random numbers, an integer ``>= 0``, to be stored with the run.

    Returns
    -------
    x0 : ndarray, shape (n_part, 3)
        Starting positions [m], in the poloidal plane ``zeta = 0``.
    u0 : ndarray, shape (n_part, 3)
        Starting proper velocities ``u = gamma v`` [m/s].

    Raises
    ------
    ValueError
        If ``seed`` is not an integer ``>= 0``, or if ``uniform_poloidal_disc`` or
        ``proper_velocity`` refuses the other arguments (they give the reason).

    Notes
    -----
    The positions and the phases come from two independent streams of random
    numbers, built from ``seed`` by ``numpy.random.SeedSequence(seed).spawn(2)``.
    Within each stream the numbers of a particle are drawn in the order of the
    particles, so the first ``k`` particles built from a seed do not depend on
    ``n_part``: a run on the first ``k`` particles of an ensemble starts from the
    same states as the full run. The pitch angle is exact at the starting
    position only; along the orbit it changes with the field.
    """
    if isinstance(seed, bool) or not isinstance(seed, Integral) or seed < 0:
        raise ValueError(f"seed must be an integer >= 0, got {seed!r}")
    position_seed, phase_seed = np.random.SeedSequence(int(seed)).spawn(2)

    x0 = uniform_poloidal_disc(n_part, radius, r0, np.random.default_rng(position_seed))
    b_field = field(x0, 0.0)[1]
    gyrophase = 2.0 * np.pi * np.random.default_rng(phase_seed).random(int(n_part))
    u0 = proper_velocity(kinetic_energy, pitch_angle, gyrophase, b_field, m)
    return x0, u0
