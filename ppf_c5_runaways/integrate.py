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
) -> tuple[
    NDArray[np.float64],
    NDArray[np.float64],
    NDArray[np.float64],
    NDArray[np.float64],
]:
    """
    Advance the particles for ``n_steps`` fixed time steps, keeping some states.

    Parameters
    ----------
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

    Returns
    -------
    t : ndarray, shape (n_saved,)
        Times ``t^n = n dt`` [s] of the kept states, ``n = 0, save_every, ...,
        n_steps``, so ``n_saved = n_steps // save_every + 1``.
    x : ndarray, shape (n_saved, n_part, 3)
        Positions ``x^n`` [m] at the times ``t``.
    u_half : ndarray, shape (n_saved, n_part, 3)
        Proper velocities ``u^{n-1/2}`` [m/s], as the pushers hold them: half a
        step *before* ``t``. They are the exact state of the leapfrog, from which a
        run can be continued, but not the velocity at ``t``.
    u_centered : ndarray, shape (n_saved, n_part, 3)
        Proper velocities at the times ``t``, ``u^n = (u^{n-1/2} + u^{n+1/2}) / 2``
        [m/s]. This is the velocity to pair with ``x^n`` in any diagnostic (energy,
        drift, invariants): using ``u_half`` instead makes an error of order ``dt``
        and silently destroys the second order of the scheme. Getting ``u^n`` at the
        last kept step needs ``u^{n_steps+1/2}``: one extra step is computed for it,
        and not kept.

    Notes
    -----
    Leapfrog convention of the project: positions at whole steps, velocities at half
    steps. The initial ``u0``, given at ``t = 0``, is first moved back by half a
    step to ``u^{-1/2}`` with the same pusher.
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
    u_half_saved = np.empty((n_saved, *x.shape))
    u_centered_saved = np.empty((n_saved, *x.shape))
    t_saved[0], x_saved[0], u_half_saved[0] = 0.0, x, u

    for step in range(1, n_steps + 1):
        # Before the push the state is (x^{step-1}, u^{step-3/2}), after it
        # (x^{step}, u^{step-1/2}): the two velocities around the time step - 1.
        u_before = u
        x, u = push(x, u, field, (step - 1) * dt, dt, q, m)
        if (step - 1) % save_every == 0:
            u_centered_saved[(step - 1) // save_every] = 0.5 * (u_before + u)
        if step % save_every == 0:
            i = step // save_every
            t_saved[i], x_saved[i], u_half_saved[i] = step * dt, x, u

    # One extra step, only for u^{n_steps+1/2}, to center the last kept velocity.
    _, u_after = push(x, u, field, n_steps * dt, dt, q, m)
    u_centered_saved[-1] = 0.5 * (u + u_after)

    return t_saved, x_saved, u_half_saved, u_centered_saved
