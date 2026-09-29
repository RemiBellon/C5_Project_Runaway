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


def uniform(*, b_field: ArrayLike, e_field: ArrayLike,) -> Field:
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

"""
def custom_toroidal(*, b0: float, r0: float, b_r: str, b_theta: str, b_phi: str) -> Field:
    context = {"b0": b0, "r0": r0, "cos": np.cos, "sin": np.sin, "sqrt": np.sqrt}

    def field(x, t):
        r, theta, phi = _cartesian_to_toroidal(x, r0)
        local_vars = {**context, "r": r, "theta": theta, "phi": phi}
        b_r_val = eval(b_r, {}, local_vars)
        b_theta_val = eval(b_theta, {}, local_vars)
        b_phi_val = eval(b_phi, {}, local_vars)
        b = _toroidal_to_cartesian_vector(b_r_val, b_theta_val, b_phi_val, theta, phi)
        e = np.zeros_like(b)
        return e, b

    return field
"""

def custom_toroidal(*, b0, r0, q0,lamb, b_r, b_theta, b_phi):
    context = {"b0": b0, "r0": r0, "q0": q0, "lamb":lamb, "cos": np.cos, "sin": np.sin, "sqrt": np.sqrt}

    # Compile once, outside the loop
    code_r = compile(b_r, "<b_r>", "eval")
    code_theta = compile(b_theta, "<b_theta>", "eval")
    code_phi = compile(b_phi, "<b_phi>", "eval")

    def field(x, t):
        r, theta, phi = _cartesian_to_toroidal(x, r0)
        local_vars = {**context, "r": r, "theta": theta, "phi": phi}
        b_r_val = eval(code_r, {}, local_vars)
        b_theta_val = eval(code_theta, {}, local_vars)
        b_phi_val = eval(code_phi, {}, local_vars)
        b = _toroidal_to_cartesian_vector(b_r_val, b_theta_val, b_phi_val, theta, phi)
        e = np.zeros_like(b)
        return e, b

    return field

def _cartesian_to_toroidal(
    x: NDArray, r0: float
) -> tuple[NDArray, NDArray, NDArray]:
    """Convert cartesian x (n_part, 3) to toroidal (r, theta, phi)."""
    x_, y_, z_ = x[:, 0], x[:, 1], x[:, 2]
    big_r = np.hypot(x_, y_)          # distance to the main (z) axis
    phi = np.arctan2(y_, x_)          # toroidal angle
    r = np.hypot(big_r - r0, z_)      # distance to the small circle
    theta = np.arctan2(z_, big_r - r0)  # poloidal angle
    return r, theta, phi

def _toroidal_to_cartesian_vector(b_r_val: NDArray[np.float64],b_theta_val: NDArray[np.float64],b_phi_val: NDArray[np.float64],
    theta: NDArray[np.float64],phi: NDArray[np.float64],
) -> NDArray[np.float64]:
    """Convert vector components (v_r, v_theta, v_phi) into cartesian.
    
    b_r_val, b_theta_val, b_phi_val : array_like, shape (n_part,)
        Vector components in the toroidal basis, at each particle's position.
    theta, phi : array_like, shape (n_part,)
        Poloidal and toroidal angles of each particle.

    Returns
    -------
    NDArray of shape (n_part, 3)
        The same vector, expressed in cartesian (x, y, z).
    """
    cos_t, sin_t = np.cos(theta), np.sin(theta)
    cos_p, sin_p = np.cos(phi), np.sin(phi)

    vx = b_r_val * cos_t * cos_p - b_theta_val * sin_t * cos_p - b_phi_val * sin_p
    vy = b_r_val * cos_t * sin_p - b_theta_val * sin_t * sin_p + b_phi_val * cos_p
    vz = b_r_val * sin_t + b_theta_val * cos_t

    return np.stack([vx, vy, vz], axis=-1)