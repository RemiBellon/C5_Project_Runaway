"""Guiding-centre estimates from particle states."""

import numpy as np
from numpy.typing import ArrayLike, NDArray

from ppf_c5_runaways.constants import e, m_e


def guiding_centre_from_position(
    position: ArrayLike,
    u: ArrayLike,
    b_field: ArrayLike,
    q: float = -e,
    m: float=m_e,
) -> NDArray[np.float64]:
    """Estimate guiding-centre positions from particle positions and velocities.

    The particle position is ``x = X + rho``, where ``X`` is the guiding centre and
    ``rho = m (B x u) / (q |B|^2)`` is the Larmor vector. Therefore::

        X = x - m (B x u) / (q |B|^2)

    This is exact in a uniform magnetic field. In a varying field, ``b_field`` is
    evaluated at the particle position and the estimate is accurate to first order
    when the Larmor radius is small compared with the field-variation length.

    Parameters
    ----------
    position : array_like, shape (..., 3)
        Particle positions ``x`` [m]. Leading dimensions can represent time and
        particles.
    u : array_like, shape (..., 3)
        Proper velocities ``u = gamma * v`` [m/s], broadcastable with ``position``.
    b_field : array_like, shape (..., 3)
        Magnetic fields at the particle positions [T], not zero, broadcastable with
        ``position`` and ``u``.
    q : float
        Signed particle charge [C], finite and not zero.
    m : float
        Particle rest mass [kg], finite and strictly positive.

    Returns
    -------
    ndarray, shape (..., 3)
        Estimated guiding-centre positions [m], with leading dimensions broadcast
        across the inputs.

    Raises
    ------
    ValueError
        If an input does not have a final dimension of 3 or contains a non-finite
        value, leading dimensions cannot be broadcast, the magnetic field is zero,
        ``q`` is zero or invalid, or ``m`` is not finite and strictly positive.
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
        ("position", position),
        ("u", u),
        ("b_field", b_field),
    ):
        array = np.asarray(vectors, dtype=np.float64)
        if array.ndim < 1 or array.shape[-1] != 3:
            raise ValueError(
                f"{name} must have shape (..., 3); got shape "
                f"{np.shape(vectors)}"
            )
        if not np.all(np.isfinite(array)):
            raise ValueError(f"{name} must be finite (no nan, no inf)")
        arrays.append(array)

    try:
        position, u, b_field = np.broadcast_arrays(*arrays)
    except ValueError as error:
        raise ValueError(
            "position, u and b_field must have broadcastable leading dimensions; "
            "got shapes "
            f"{[array.shape for array in arrays]}"
        ) from error

    b_squared = np.sum(b_field**2, axis=-1, keepdims=True)
    if np.any(b_squared == 0.0):
        raise ValueError("b_field must not be the zero vector")

    larmor_vector = m * np.cross(b_field, u) / (q * b_squared)
    return position - larmor_vector
