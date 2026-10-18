"""Initial conditions of the particles, built from physical quantities.

A run starts from positions and proper velocities in cartesian coordinates. A
physicist rather thinks of a particle by its kinetic energy, its pitch angle and its
gyration phase: this module does the conversion. It is pure numerics: it receives
numbers and arrays and returns arrays, reads no configuration and writes nothing.
"""

import numpy as np
from numpy.typing import ArrayLike, NDArray

from ppf_c5_runaways.constants import c


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
