"""
Analytic electromagnetic fields, in SI units.

A *field* is any function ``field(x, t) -> (E, B)`` where ``x`` holds the positions of
all particles, shape ``(n_part, 3)`` [m], ``t`` is the time [s], and ``E`` [V/m] and
``B`` [T] are the fields felt by each particle, both of shape ``(n_part, 3)``. This
is the interface the pushers rely on: any function that follows it can be used.
"""

from collections.abc import Callable

import numpy as np
from numpy.typing import ArrayLike, NDArray

type Field = Callable[
    [NDArray[np.float64], float], tuple[NDArray[np.float64], NDArray[np.float64]]
]


def uniform(*, b_field: ArrayLike, e_field: ArrayLike) -> Field:
    """Build a field that is the same at every point and at every time.

    Both arguments are required, and given by name: "no electric field" is written
    ``e_field=[0, 0, 0]``, it is never assumed.

    b_field : array_like of 3 numbers
        Magnetic field ``(Bx, By, Bz)`` [T]. A list, a tuple or an array.
    e_field : array_like of 3 numbers
        Electric field ``(Ex, Ey, Ez)`` [V/m]. A list, a tuple or an array.
    """
    vectors = {}
    for name, value in (("b_field", b_field), ("e_field", e_field)):
        message = f"{name} must be exactly 3 numbers [x, y, z], got {value!r}"
        try:
            vector = np.array(value)  # np.array copies its input, np.asarray does not
        except ValueError as error:  # e.g. a ragged nested list
            raise ValueError(message) from error
        # dtype kinds i, u, f: integers and floats. Strings, booleans, None and
        # complex numbers are refused rather than silently converted.
        if vector.shape != (3,) or vector.dtype.kind not in "iuf":
            raise ValueError(message)
        vector = vector.astype(np.float64)
        if not np.all(np.isfinite(vector)):
            raise ValueError(f"{name} must be finite (no nan, no inf), got {value!r}")
        vectors[name] = vector
    e_vector = vectors["e_field"]
    b_vector = vectors["b_field"]

    def field(
        x: NDArray[np.float64], t: float
    ) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        """Return ``(E, B)``, each of shape ``(n_part, 3)``, for positions ``x``."""
        x = np.asarray(x)
        if x.ndim != 2 or x.shape[1] != 3:
            raise ValueError(
                "x must have shape (n_part, 3), one row per particle; "
                f"got shape {x.shape}. For a single particle use shape (1, 3), "
                "for example x[np.newaxis, :]."
            )
        # np.tile builds new arrays: the caller may modify them freely.
        return np.tile(e_vector, (x.shape[0], 1)), np.tile(b_vector, (x.shape[0], 1))

    return field
