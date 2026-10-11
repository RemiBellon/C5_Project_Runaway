"""Time loop: advance particles for many steps and keep some of the states."""

import numpy as np
from numpy.typing import NDArray

from ppf_c5_runaways.fields import Field
from ppf_c5_runaways.pushers import Pusher


def integrate(
    x0: NDArray[np.float64],
    u0: NDArray[np.float64],
    field: Field,
    push: Pusher,
    dt: float,
    n_steps: int,
    save_every: int,
    q: float,
    m: float,
) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
    """
    Advance the particles for ``n_steps`` fixed time steps, keeping some states.

    Parameters
    ----------
    x0 : ndarray, shape (n_part, 3)
        Initial positions [m] at ``t = 0``.
    u0 : ndarray, shape (n_part, 3)
        Initial proper velocities ``u = gamma * v`` [m/s] at ``t = 0``, the same
        time as ``x0``. They are used as they are: no half step is taken.
    field : Field
        Function ``field(x, t) -> (E, B)`` (see ``ppf_c5_runaways.fields``).
    push : Pusher
        Function that does one step, for example ``pushers.boris``.
    dt : float
        Time step [s], finite and strictly positive.
    n_steps : int
        Number of time steps, strictly positive.
    save_every : int
        A state is kept every ``save_every`` steps, strictly positive. It must
        divide ``n_steps``, so that the last step is kept.
    q : float
        Signed charge [C], for example ``-e`` for an electron.
    m : float
        Rest mass [kg].

    Returns
    -------
    t : ndarray, shape (n_saved,)
        Times ``t^n = n dt`` [s] of the kept states, ``n = 0, save_every, ...,
        n_steps``, so ``n_saved = n_steps // save_every + 1``.
    x : ndarray, shape (n_saved, n_part, 3)
        Positions ``x^n`` [m] at the times ``t``.
    u : ndarray, shape (n_saved, n_part, 3)
        Proper velocities ``u^n`` [m/s] at the same times ``t``. ``x`` and ``u`` are
        the state of the scheme: any diagnostic at ``t`` (energy, drift, invariants)
        uses them directly, and a run can be continued from the last one.

    Raises
    ------
    ValueError
        If ``n_steps`` or ``save_every`` is not a strictly positive integer, if
        ``save_every`` does not divide ``n_steps``, if ``dt`` is not a finite,
        strictly positive number, or if ``push`` refuses ``x0``, ``u0``, ``q`` or
        ``m`` (see ``pushers.boris``).

    Notes
    -----
    Synchronized convention of the project (Ripperda et al. 2018, Eqs. 5-8):
    positions and velocities are known at the same times, and step ``n`` is one call
    ``push(x^n, u^n, field, n dt, dt, q, m)``. ``push`` is called exactly
    ``n_steps`` times, so the field is evaluated at the times ``(n + 1/2) dt``.
    """
    for name, value in (("n_steps", n_steps), ("save_every", save_every)):
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            raise ValueError(
                f"{name} must be a strictly positive integer, got {value!r}"
            )
    if n_steps % save_every:
        raise ValueError(
            f"save_every={save_every} must divide n_steps={n_steps}, "
            "so that the last step is kept"
        )
    if (
        isinstance(dt, bool)
        or not isinstance(dt, (int, float))
        or not np.isfinite(dt)
        or dt <= 0.0
    ):
        raise ValueError(
            f"dt must be a finite, strictly positive number [s], got {dt!r}"
        )

    x = np.array(x0, dtype=np.float64)
    u = np.array(u0, dtype=np.float64)

    n_saved = n_steps // save_every + 1
    t_saved = np.empty(n_saved)
    x_saved = np.empty((n_saved, *x.shape))
    u_saved = np.empty((n_saved, *u.shape))
    t_saved[0], x_saved[0], u_saved[0] = 0.0, x, u

    # ``push`` checks x, u, q and m at the first step, before anything is computed.
    for step in range(1, n_steps + 1):
        x, u = push(x, u, field, (step - 1) * dt, dt, q, m)
        if step % save_every == 0:
            i = step // save_every
            t_saved[i], x_saved[i], u_saved[i] = step * dt, x, u

    return t_saved, x_saved, u_saved
