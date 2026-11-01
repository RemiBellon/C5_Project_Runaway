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


def uniform(
    *,
    b_field: ArrayLike,
    e_field: ArrayLike,
) -> Field:
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


def magnetic_mirror(*, b0: float, length: float) -> Field:
    """Build the static magnetic bottle of Ripperda et al. (2018, Eq. 59).

    ``B(x, y, z) = B0 (-x z / L^2, -y z / L^2, 1 + z^2 / L^2)`` and ``E = 0``.

    On the axis the field grows from ``B0`` at ``z = 0`` to ``2 B0`` at
    ``z = +/- L``. The radial part is the paraxial one, ``B_r = -r B0 z / L^2``,
    chosen so that ``div B = 0`` exactly. The field is not curl-free: it is the
    paraxial model of a bottle, valid near the axis (``r << L``), as in the paper.
    A particle of pitch angle ``alpha`` at ``z = 0`` on the axis is reflected at
    ``z_max = L sqrt(1 / sin^2(alpha) - 1)`` while its magnetic moment is
    conserved (Ripperda et al. 2018, Eq. 61).

    Parameters
    ----------
    b0 : float
        Field at the centre [T], finite and strictly positive.
    length : float
        Length ``L`` [m] over which the field doubles along the axis, finite and
        strictly positive.

    Returns
    -------
    Field
        ``field(x, t) -> (E, B)``, both of shape ``(n_part, 3)``; ``t`` is unused.

    Raises
    ------
    ValueError
        If ``b0`` or ``length`` is not a finite, strictly positive number.
    """
    for name, value in (("b0", b0), ("length", length)):
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not np.isfinite(value)
            or value <= 0.0
        ):
            raise ValueError(
                f"{name} must be a finite, strictly positive number, got {value!r}"
            )

    def field(
        x: NDArray[np.float64], t: float
    ) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        """Return ``(E, B)``, each of shape ``(n_part, 3)``, for positions ``x``."""
        x = np.asarray(x, dtype=np.float64)
        if x.ndim != 2 or x.shape[1] != 3:
            raise ValueError(
                "x must have shape (n_part, 3), one row per particle; "
                f"got shape {x.shape}. For a single particle use shape (1, 3), "
                "for example x[np.newaxis, :]."
            )
        z_over_l = x[:, 2] / length
        b_field = np.empty_like(x)
        b_field[:, 0] = -b0 * z_over_l * x[:, 0] / length
        b_field[:, 1] = -b0 * z_over_l * x[:, 1] / length
        b_field[:, 2] = b0 * (1.0 + z_over_l**2)
        return np.zeros_like(x), b_field

    return field


def magnetic_gradient(*, b0: float, length: float) -> Field:
    """Build the static field with a perpendicular gradient of Ripperda et al. (2018).

    ``B(x, y, z) = B0 (1 + x / L) z_hat`` and ``E = 0`` (Eq. 63 of the paper).

    The field keeps the direction ``z`` everywhere and its strength grows linearly
    with ``x``: ``grad |B| = (B0 / L) x_hat`` for ``x > -L``, perpendicular to the
    field. ``div B = 0`` exactly, since ``B_z`` does not depend on ``z``. The field
    is not curl-free (``curl B = -(B0 / L) y_hat``): it is a test field, not a
    vacuum field. It is the simplest field in which what a particle feels depends
    on where the field is evaluated, which a uniform field cannot show.

    Parameters
    ----------
    b0 : float
        Field on the plane ``x = 0`` [T], finite and strictly positive.
    length : float
        Gradient length ``L`` [m], finite and strictly positive: the field doubles
        between ``x = 0`` and ``x = L``.

    Returns
    -------
    Field
        ``field(x, t) -> (E, B)``, both of shape ``(n_part, 3)``; ``t`` is unused.

    Raises
    ------
    ValueError
        If ``b0`` or ``length`` is not a finite, strictly positive number.

    Notes
    -----
    The field vanishes on the plane ``x = -L`` and points along ``-z`` beyond it.
    The gyration and the drift formulas of the paper assume ``x > -L``; keeping
    the particles on that side is left to the caller.
    """
    for name, value in (("b0", b0), ("length", length)):
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not np.isfinite(value)
            or value <= 0.0
        ):
            raise ValueError(
                f"{name} must be a finite, strictly positive number, got {value!r}"
            )

    def field(
        x: NDArray[np.float64], t: float
    ) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        """Return ``(E, B)``, each of shape ``(n_part, 3)``, for positions ``x``."""
        x = np.asarray(x, dtype=np.float64)
        if x.ndim != 2 or x.shape[1] != 3:
            raise ValueError(
                "x must have shape (n_part, 3), one row per particle; "
                f"got shape {x.shape}. For a single particle use shape (1, 3), "
                "for example x[np.newaxis, :]."
            )
        b_field = np.zeros_like(x)
        b_field[:, 2] = b0 * (1.0 + x[:, 0] / length)
        return np.zeros_like(x), b_field

    return field


def higuera_cary_shear(*, a: float, b: float) -> Field:
    """Build the static sheared field of Higuera and Cary (2017, Sec. VI, Eqs. 48-51).

    ``E = -a x x_hat`` and ``B = b y x_hat``, derived from the potentials
    ``phi = a x^2 / 2`` and ``A = (b y^2 / 2) z_hat`` (``E = -grad phi``,
    ``B = curl A``). Both fields point along ``x``; ``B`` changes sign on the plane
    ``y = 0``. The motion conserves the energy, the canonical momentum ``p_z`` and
    ``I_y = p_y^2 + (p_z - q A_z)^2`` (Eq. 47): it is integrable, which is what
    makes its Poincare section a test of the pushers (roadmap C7).

    Parameters
    ----------
    a : float
        Gradient of the electric field [V/m^2], finite and signed; zero is allowed.
    b : float
        Gradient of the magnetic field [T/m], finite and signed; zero is allowed.

    Returns
    -------
    Field
        ``field(x, t) -> (E, B)``, both of shape ``(n_part, 3)``; ``t`` is unused.

    Raises
    ------
    ValueError
        If ``a`` or ``b`` is not a finite number; when the field is called, if
        ``x`` does not have shape ``(n_part, 3)``.

    Notes
    -----
    The paper works in units where ``q = m = c = 1``, with ``a = 1`` and ``b = 2``.
    The signs are left free because the motion does not change when ``q``, ``E``
    and ``B`` all change sign: an electron (``q = -e``) follows the orbit of the
    paper with ``a`` and ``b`` negative. With the unit of length ``L0`` and the unit
    of time ``L0 / c``, ``a = a' m c^2 / (q L0^2)`` and ``b = b' m c / (q L0^2)``,
    primes denoting the values of the paper.
    """
    for name, value in (("a", a), ("b", b)):
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not np.isfinite(value)
        ):
            raise ValueError(f"{name} must be a finite number, got {value!r}")

    def field(
        x: NDArray[np.float64], t: float
    ) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        """Return ``(E, B)``, each of shape ``(n_part, 3)``, for positions ``x``."""
        x = np.asarray(x, dtype=np.float64)
        if x.ndim != 2 or x.shape[1] != 3:
            raise ValueError(
                "x must have shape (n_part, 3), one row per particle; "
                f"got shape {x.shape}. For a single particle use shape (1, 3), "
                "for example x[np.newaxis, :]."
            )
        e_field = np.zeros_like(x)
        e_field[:, 0] = -a * x[:, 0]
        b_field = np.zeros_like(x)
        b_field[:, 0] = b * x[:, 1]
        return e_field, b_field

    return field


def toroidal_to_cartesian(
    r: ArrayLike, theta: ArrayLike, zeta: ArrayLike, r0: float
) -> NDArray[np.float64]:
    """Cartesian position of points given in the toroidal coordinates of Carbajal.

    ``x = (R0 + r cos(theta)) sin(zeta)``, ``y = (R0 + r cos(theta)) cos(zeta)`` and
    ``z = r sin(theta)`` (Carbajal et al. 2017, Sec. II, below Eq. 3).

    ``r`` is the minor radius, the distance to the magnetic axis, which is the circle
    of radius ``R0`` in the plane ``z = 0``. ``theta`` is the poloidal angle, zero on
    the side of the torus away from the vertical axis and growing towards ``+z``.
    ``zeta`` is the toroidal angle, zero on the ``+y`` axis and turning clockwise
    seen from ``+z``: ``zeta = pi/2 - phi``, with ``phi`` the azimuth of the usual
    cylindrical coordinates. This is the convention of the paper, chosen for the
    whole project (OPEN_POINTS N2); with it, ``(r, theta, zeta)`` is right-handed.

    Parameters
    ----------
    r : array_like
        Minor radius [m], finite, at least 0 and smaller than ``r0``.
    theta, zeta : array_like
        Poloidal and toroidal angles [rad], finite. ``r``, ``theta`` and ``zeta``
        broadcast together: one of them can be a single number.
    r0 : float
        Major radius ``R0`` [m], finite and strictly positive.

    Returns
    -------
    ndarray, shape (..., 3)
        Positions ``(x, y, z)`` [m], one per broadcast point.

    Raises
    ------
    ValueError
        If ``r0`` is not a finite, strictly positive number, the coordinates do not
        broadcast together or are not finite, or ``r`` is not in ``[0, r0)``.

    Notes
    -----
    ``r < R0`` keeps the distance to the vertical axis, ``R0 + r cos(theta)``,
    positive: beyond, two sets of coordinates give the same point, and the
    coordinates could not be recovered from the position.
    """
    if (
        isinstance(r0, bool)
        or not isinstance(r0, (int, float))
        or not np.isfinite(r0)
        or r0 <= 0.0
    ):
        raise ValueError(f"r0 must be a finite, strictly positive number, got {r0!r}")
    try:
        r, theta, zeta = np.broadcast_arrays(
            np.asarray(r, dtype=np.float64),
            np.asarray(theta, dtype=np.float64),
            np.asarray(zeta, dtype=np.float64),
        )
    except ValueError:
        raise ValueError(
            "r, theta and zeta must broadcast together; got shapes "
            f"{np.shape(r)}, {np.shape(theta)} and {np.shape(zeta)}"
        ) from None
    for name, values in (("r", r), ("theta", theta), ("zeta", zeta)):
        if not np.all(np.isfinite(values)):
            raise ValueError(f"{name} must be finite (no nan, no inf)")
    if np.any(r < 0.0) or np.any(r >= r0):
        raise ValueError(
            f"r must be in [0, r0) = [0, {r0}) m, the inside of the torus; got values "
            f"from {np.min(r)} to {np.max(r)} m"
        )

    major_radius = r0 + r * np.cos(theta)  # distance to the vertical axis
    return np.stack(
        [major_radius * np.sin(zeta), major_radius * np.cos(zeta), r * np.sin(theta)],
        axis=-1,
    )


def cartesian_to_toroidal(
    x: ArrayLike, r0: float
) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
    """Toroidal coordinates of Carbajal of cartesian positions.

    This is the inverse of ``toroidal_to_cartesian``. With ``R = sqrt(x^2 + y^2)``
    the distance to the vertical axis:

        zeta = atan2(x, y),   r = sqrt((R - R0)^2 + z^2),   theta = atan2(z, R - R0)

    which follow from ``x = R sin(zeta)``, ``y = R cos(zeta)``,
    ``R - R0 = r cos(theta)`` and ``z = r sin(theta)`` (Carbajal et al. 2017, Sec.
    II; same conventions as ``toroidal_to_cartesian``, OPEN_POINTS N2).

    Parameters
    ----------
    x : array_like, shape (..., 3)
        Positions ``(x, y, z)`` [m], finite, none on the vertical axis
        ``x = y = 0``. Any leading shape, for example ``(n_part, 3)``.
    r0 : float
        Major radius ``R0`` [m], finite and strictly positive.

    Returns
    -------
    r, theta, zeta : ndarray, shape (...)
        Minor radius [m], at least 0; poloidal and toroidal angles [rad], in
        ``(-pi, pi]``.

    Raises
    ------
    ValueError
        If ``r0`` is not a finite, strictly positive number, the last axis of ``x``
        does not have length 3, a position is not finite, or a position is on the
        vertical axis, where ``zeta`` is not defined.

    Notes
    -----
    On the magnetic axis (``r = 0``) ``theta`` is not defined either; ``theta = 0``
    is returned there, the value of ``atan2(0, 0)``. Positions with ``r >= R0`` are
    accepted, since their coordinates are well defined, but they are outside the
    domain of ``toroidal_to_cartesian``: the round trip holds for ``r < R0``.
    """
    if (
        isinstance(r0, bool)
        or not isinstance(r0, (int, float))
        or not np.isfinite(r0)
        or r0 <= 0.0
    ):
        raise ValueError(f"r0 must be a finite, strictly positive number, got {r0!r}")
    x = np.asarray(x, dtype=np.float64)
    if x.ndim == 0 or x.shape[-1] != 3:
        raise ValueError(
            "x must have shape (..., 3), the last axis holding (x, y, z); "
            f"got shape {x.shape}"
        )
    if not np.all(np.isfinite(x)):
        raise ValueError("x must be finite (no nan, no inf)")

    major_radius = np.hypot(x[..., 0], x[..., 1])  # distance to the vertical axis
    if np.any(major_radius == 0.0):
        raise ValueError(
            "a position is on the vertical axis x = y = 0, where the toroidal angle "
            "zeta is not defined"
        )
    zeta = np.arctan2(x[..., 0], x[..., 1])
    r = np.hypot(major_radius - r0, x[..., 2])
    theta = np.arctan2(x[..., 2], major_radius - r0)
    return r, theta, zeta


def toroidal_basis(
    x: ArrayLike, r0: float
) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
    """Return the unit vectors of the toroidal coordinates of Carbajal at positions.

    They are the derivatives of ``toroidal_to_cartesian`` along each coordinate,
    divided by their length (1, ``r`` and ``R = R0 + r cos(theta)``)::

        e_r     = ( cos(theta) sin(zeta),  cos(theta) cos(zeta), sin(theta))
        e_theta = (-sin(theta) sin(zeta), -sin(theta) cos(zeta), cos(theta))
        e_zeta  = ( cos(zeta),            -sin(zeta),            0)

    ``(e_r, e_theta, e_zeta)`` is orthonormal and right-handed:
    ``e_r x e_theta = e_zeta``. ``e_zeta = -e_phi``, with ``e_phi`` the azimuthal
    vector of the usual cylindrical coordinates (``zeta`` turns clockwise,
    OPEN_POINTS N2).

    Parameters
    ----------
    x : array_like, shape (..., 3)
        Positions ``(x, y, z)`` [m], finite, none on the vertical axis.
    r0 : float
        Major radius ``R0`` [m], finite and strictly positive.

    Returns
    -------
    e_r, e_theta, e_zeta : ndarray, shape (..., 3)
        The three unit vectors at each position (dimensionless).

    Raises
    ------
    ValueError
        As ``cartesian_to_toroidal``, whose angles are used.

    Notes
    -----
    On the magnetic axis (``r = 0``) ``e_r`` and ``e_theta`` are not defined; they
    are those of ``theta = 0`` there, the convention of ``cartesian_to_toroidal``.
    """
    _, theta, zeta = cartesian_to_toroidal(x, r0)
    cos_theta, sin_theta = np.cos(theta), np.sin(theta)
    cos_zeta, sin_zeta = np.cos(zeta), np.sin(zeta)
    e_r = np.stack([cos_theta * sin_zeta, cos_theta * cos_zeta, sin_theta], axis=-1)
    e_theta = np.stack(
        [-sin_theta * sin_zeta, -sin_theta * cos_zeta, cos_theta], axis=-1
    )
    e_zeta = np.stack([cos_zeta, -sin_zeta, np.zeros_like(zeta)], axis=-1)
    return e_r, e_theta, e_zeta


def toroidal(*, b0: float, r0: float) -> Field:
    """Build the static, purely toroidal magnetic field ``B = B0 R0 / R e_zeta``.

    ``R`` is the distance to the vertical axis and ``E = 0``. This is the toroidal
    part of the field of Carbajal et al. (2017, Eq. 4), without its poloidal part:
    ``|B| = B0`` on the magnetic axis ``R = R0`` and decreases as ``1 / R``.
    ``div B = 0`` and ``curl B = 0`` away from the vertical axis: it is the vacuum
    field of a current along that axis. The field lines are horizontal circles
    around the vertical axis, so a particle drifts vertically, by the gradient and
    curvature drifts (roadmap test C3).

    Parameters
    ----------
    b0 : float
        Field on the magnetic axis [T], finite and strictly positive: the field
        points along ``+e_zeta``, as in the paper.
    r0 : float
        Major radius ``R0`` [m], finite and strictly positive.

    Returns
    -------
    Field
        ``field(x, t) -> (E, B)``, both of shape ``(n_part, 3)``; ``t`` is unused.

    Raises
    ------
    ValueError
        If ``b0`` or ``r0`` is not a finite, strictly positive number; when the
        field is called, if ``x`` does not have shape ``(n_part, 3)``, is not
        finite, or a position is on the vertical axis, where the field is infinite.
    """
    for name, value in (("b0", b0), ("r0", r0)):
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not np.isfinite(value)
            or value <= 0.0
        ):
            raise ValueError(
                f"{name} must be a finite, strictly positive number, got {value!r}"
            )

    def field(
        x: NDArray[np.float64], t: float
    ) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        """Return ``(E, B)``, each of shape ``(n_part, 3)``, for positions ``x``."""
        x = np.asarray(x, dtype=np.float64)
        if x.ndim != 2 or x.shape[1] != 3:
            raise ValueError(
                "x must have shape (n_part, 3), one row per particle; "
                f"got shape {x.shape}. For a single particle use shape (1, 3), "
                "for example x[np.newaxis, :]."
            )
        # toroidal_basis refuses the vertical axis before R = 0 can divide below.
        _, _, e_zeta = toroidal_basis(x, r0)
        major_radius = np.hypot(x[:, 0], x[:, 1])
        b_field = (b0 * r0 / major_radius)[:, np.newaxis] * e_zeta
        return np.zeros_like(x), b_field

    return field


def carbajal(*, b0: float, r0: float, q0: float, lam: float, e0: float) -> Field:
    """Build the analytic tokamak field of Carbajal et al. (2017, Eqs. 4, 5 and 7).

    With the toroidal coordinates ``(r, theta, zeta)`` of ``cartesian_to_toroidal``
    and the aspect ratio ``eta = r / R0``::

        B = (B0 e_zeta + B_theta(r) e_theta) / (1 + eta cos(theta))     (Eq. 4)
        B_theta(r) = eta B0 / q(r),   q(r) = q0 (1 + r^2 / lam^2)       (Eq. 5)
        E = E0 / (1 + eta cos(theta)) e_zeta                            (Eq. 7)

    ``1 + eta cos(theta) = R / R0``, with ``R`` the distance to the vertical axis:
    the toroidal part is the field of ``toroidal``. ``q`` is the safety factor,
    ``q0`` on the magnetic axis. The fields are static and axisymmetric.

    Parameters
    ----------
    b0 : float
        Toroidal field on the magnetic axis [T], finite and strictly positive.
    r0 : float
        Major radius ``R0`` [m], finite and strictly positive.
    q0 : float
        Safety factor on the magnetic axis (dimensionless), finite and strictly
        positive.
    lam : float
        Length ``lambda`` of the safety-factor profile [m], finite and strictly
        positive. For ``q(r_edge) = q_edge``: ``lam = r_edge / sqrt(q_edge / q0 - 1)``.
    e0 : float
        Toroidal electric field on the magnetic axis [V/m], finite, signed and
        applied as it is (OPEN_POINTS D3): ``e0 < 0`` is antiparallel to the
        toroidal field and accelerates electrons moving along it. Zero is allowed.

    Returns
    -------
    Field
        ``field(x, t) -> (E, B)``, both of shape ``(n_part, 3)``; ``t`` is unused.

    Raises
    ------
    ValueError
        If ``b0``, ``r0``, ``q0`` or ``lam`` is not a finite, strictly positive
        number, or ``e0`` is not a finite number; when the field is called, if
        ``x`` does not have shape ``(n_part, 3)``, is not finite, or a position is on
        the vertical axis.

    Notes
    -----
    On the magnetic axis ``theta`` is not defined, but ``B_theta = 0`` there: the
    convention ``theta = 0`` of ``cartesian_to_toroidal`` does not change the field.
    No plasma edge is imposed: the field is defined beyond ``r_edge`` (OPEN_POINTS
    N7).
    """
    for name, value in (("b0", b0), ("r0", r0), ("q0", q0), ("lam", lam)):
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not np.isfinite(value)
            or value <= 0.0
        ):
            raise ValueError(
                f"{name} must be a finite, strictly positive number, got {value!r}"
            )
    if isinstance(e0, bool) or not isinstance(e0, (int, float)) or not np.isfinite(e0):
        raise ValueError(f"e0 must be a finite number, got {e0!r}")

    def field(
        x: NDArray[np.float64], t: float
    ) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        """Return ``(E, B)``, each of shape ``(n_part, 3)``, for positions ``x``."""
        x = np.asarray(x, dtype=np.float64)
        if x.ndim != 2 or x.shape[1] != 3:
            raise ValueError(
                "x must have shape (n_part, 3), one row per particle; "
                f"got shape {x.shape}. For a single particle use shape (1, 3), "
                "for example x[np.newaxis, :]."
            )
        r, theta, _ = cartesian_to_toroidal(x, r0)
        _, e_theta, e_zeta = toroidal_basis(x, r0)
        eta = r / r0
        toroidicity = (1.0 + eta * np.cos(theta))[:, np.newaxis]  # R / R0
        safety_factor = q0 * (1.0 + (r / lam) ** 2)
        b_theta = (eta * b0 / safety_factor)[:, np.newaxis]
        b_field = (b0 * e_zeta + b_theta * e_theta) / toroidicity
        e_field = e0 * e_zeta / toroidicity
        return e_field, b_field

    return field


def carbajal_poloidal_flux(
    r: ArrayLike, *, b0: float, q0: float, lam: float
) -> NDArray[np.float64]:
    """Poloidal flux function of the field of ``carbajal``, zero on the magnetic axis.

        Psi(r) = B0 lam^2 / (2 q0) ln(1 + r^2 / lam^2)

    It is the term in ``log`` of the canonical momentum of Carbajal et al. (2017,
    Eq. 6): ``gamma m lam^2 Omega_e / (2 q0) log(1 + r^2 / lam^2) = e Psi``, with
    ``Omega_e = e B0 / (gamma m)``. It satisfies ``dPsi/dr = R B_theta = r B0 / q(r)``,
    which does not depend on ``theta``: ``Psi`` is constant on each magnetic surface
    ``r = const``. It enters the invariant ``P_zeta = R u_zeta - (q / m) Psi``.

    Parameters
    ----------
    r : array_like
        Minor radius [m], finite and at least 0, any shape.
    b0 : float
        Toroidal field on the magnetic axis [T], finite and strictly positive.
    q0 : float
        Safety factor on the magnetic axis, finite and strictly positive.
    lam : float
        Length ``lambda`` of the safety-factor profile [m], finite and strictly
        positive.

    Returns
    -------
    ndarray, shape of ``r``
        Poloidal flux function ``Psi`` [T m^2].

    Raises
    ------
    ValueError
        If ``b0``, ``q0`` or ``lam`` is not a finite, strictly positive number, or
        ``r`` is not finite or is negative.

    Notes
    -----
    The additive constant (``Psi = 0`` on the axis) and the sign are conventions,
    both those of Carbajal et al. (2017, Eq. 6), with ``zeta`` turning clockwise
    (OPEN_POINTS N1, N2).
    """
    for name, value in (("b0", b0), ("q0", q0), ("lam", lam)):
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not np.isfinite(value)
            or value <= 0.0
        ):
            raise ValueError(
                f"{name} must be a finite, strictly positive number, got {value!r}"
            )
    r = np.asarray(r, dtype=np.float64)
    if not np.all(np.isfinite(r)):
        raise ValueError("r must be finite (no nan, no inf)")
    if np.any(r < 0.0):
        raise ValueError(f"r must be at least 0, got a minimum of {np.min(r)} m")

    # log1p keeps the digits of Psi near the axis, where r^2 / lam^2 << 1.
    return b0 * lam**2 / (2.0 * q0) * np.log1p((r / lam) ** 2)
