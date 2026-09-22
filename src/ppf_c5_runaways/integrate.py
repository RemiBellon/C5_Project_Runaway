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

    x0 : ndarray, shape (n_part, 3)
        Initial positions [m] at ``t = 0``.
    u0 : ndarray, shape (n_part, 3)
        Initial proper velocities ``u = gamma * v`` [m/s] at ``t = 0``, that is *at
        a whole step*, as one naturally writes an initial condition.
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

    # Half a step backward: (x^0, u^0) -> u^{-1/2}. The returned position is not used.
    # ``push`` also checks x0, u0, q and m here, before anything is computed.
    _, u = push(x0, u0, field, 0.0, -dt / 2.0, q, m)
    x = np.array(x0, dtype=np.float64)

    n_saved = n_steps // save_every + 1
    t_saved = np.empty(n_saved)
    x_saved = np.empty((n_saved, *x.shape))
    u_saved = np.empty((n_saved, *x.shape))
    t_saved[0], x_saved[0], u_saved[0] = 0.0, x, u

    for step in range(1, n_steps + 1):
        # The state is (x^{step-1}, u^{step-3/2}) at time (step-1)*dt.
        x, u = push(x, u, field, (step - 1) * dt, dt, q, m)
        if step % save_every == 0:
            i = step // save_every
            t_saved[i], x_saved[i], u_saved[i] = step * dt, x, u

    return t_saved, x_saved, u_saved
