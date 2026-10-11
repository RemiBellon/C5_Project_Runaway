"""Tests of ``ppf_c5_runaways.integrate`` (``integrate``)."""

import math

import numpy as np
import pytest

from ppf_c5_runaways import fields
from ppf_c5_runaways.constants import c, e, m_e
from ppf_c5_runaways.integrate import integrate
from ppf_c5_runaways.pushers import boris, higuera_cary, vay

Q = -e
M = m_e
BZ = 2.5  # [T]


@pytest.mark.physics
def test_one_gyro_period_brings_the_particle_back_above_its_start():
    # Independent truth: the exact orbit. After exactly one gyro-period
    # T = 2 pi gamma m / (|q| B) the particle is back at x = y = 0, and it has
    # moved by u_z * T / gamma along z, whatever the scheme.
    # Tolerance in the plane: theta^2 * r_L, with theta = omega * dt = 2 pi / 200.
    # In a uniform B the synchronized Boris step keeps the particle on the exact
    # Larmor circle; only its phase is late, by 2 pi theta^2 / 12 = 0.52 theta^2
    # after one turn, which is below the tolerance.
    # Along z the motion is uniform: only rounding errors, 200 steps of 1e-16.
    n_steps = 200
    speed = math.sqrt(3.0) * c  # gamma = 2
    u0 = np.array([[0.8 * speed, 0.0, 0.6 * speed]])
    gamma = 2.0
    omega = abs(Q) * BZ / (M * gamma)
    period = 2.0 * math.pi / omega
    dt = period / n_steps
    r_larmor = M * 0.8 * speed / (abs(Q) * BZ)
    field = fields.uniform(b_field=[0.0, 0.0, BZ], e_field=[0.0, 0.0, 0.0])

    t, x, u = integrate(np.zeros((1, 3)), u0, field, boris, dt, n_steps, 10, Q, M)

    theta = omega * dt
    assert np.hypot(x[-1, 0, 0], x[-1, 0, 1]) < theta**2 * r_larmor
    assert x[-1, 0, 2] == pytest.approx(0.6 * speed * period / gamma, rel=1e-12)
    # Shapes and times: 200 steps kept every 10, plus the initial state.
    assert t.shape == (21,)
    assert x.shape == u.shape == (21, 1, 3)
    assert t[0] == 0.0
    assert t[-1] == pytest.approx(n_steps * dt, rel=1e-15)
    np.testing.assert_array_equal(x[0], np.zeros((1, 3)))  # the start is kept as is
    np.testing.assert_array_equal(u[0], u0)


def test_initial_state_is_used_as_it_is_and_push_is_called_once_per_step():
    # Independent truth: the definition of the loop. The state given at t = 0 is the
    # first one kept, untouched (no half step backward), and n_steps steps are
    # n_steps calls of the pusher, so n_steps evaluations of the field, at the
    # middles of the steps (n + 1/2) dt (no extra step at the end). A field that
    # depends on the time is used so that a wrong time would change the result.
    times = []

    def recording_field(x, t):
        times.append(t)
        return np.zeros_like(x), np.full_like(x, 1.0 + 1.0e9 * t)

    x0 = np.array([[1.0e-3, -2.0e-3, 3.0e-3], [0.0, 0.0, 0.0]])
    u0 = np.array([[0.3 * c, -0.5 * c, 0.8 * c], [2.0 * c, 0.0, c]])
    dt, n_steps = 1.0e-12, 12

    t, x, u = integrate(x0, u0, recording_field, boris, dt, n_steps, 4, Q, M)

    np.testing.assert_array_equal(x[0], x0)
    np.testing.assert_array_equal(u[0], u0)
    np.testing.assert_allclose(t, [0.0, 4.0e-12, 8.0e-12, 12.0e-12], rtol=1e-15)
    assert len(times) == n_steps
    np.testing.assert_allclose(times, (np.arange(n_steps) + 0.5) * dt, rtol=1e-15)


# |u| / c for the three regimes: gamma = sqrt(1 + (u/c)^2) is about 1, 2 and 100.
REGIMES = [
    pytest.param(1.0e-3, id="non_relativistic"),
    pytest.param(math.sqrt(3.0), id="gamma_2"),
    pytest.param(math.sqrt(9999.0), id="gamma_100"),
]
all_pushers = pytest.mark.parametrize(
    "push", [boris, higuera_cary, vay], ids=["boris", "higuera_cary", "vay"]
)


@pytest.mark.physics
@all_pushers
@pytest.mark.parametrize("save_every", [1, 7])
@pytest.mark.parametrize("u_over_c", REGIMES)
def test_velocity_is_exact_in_a_pure_electric_field(push, u_over_c, save_every):
    # Independent truth: in a uniform E alone, du/dt = q E / m exactly, so
    # u(t) = u0 + q E t / m, whatever gamma. The three pushers add q E dt / m per
    # step with no other change (no rotation when B = 0), so the velocity kept at
    # each time must be the exact one at that same time, up to rounding.
    # Tolerance 1e-13 of the largest |u|: 70 additions of rounding 1.1e-16 each give
    # at most 8e-15. A velocity half a step off would differ by q E dt / (2 m),
    # which is 1e4 times larger (2e-9 in the ultra-relativistic case, the smallest
    # one): checked below, so that the test can tell them apart.
    e_vector = np.array([1.0e7, 0.0, 0.0])  # [V/m]
    u0 = np.array([[0.0, u_over_c * c, 0.0]])
    dt, n_steps = 1.0e-13, 70
    field = fields.uniform(b_field=[0.0, 0.0, 0.0], e_field=e_vector)

    t, _, u = integrate(
        np.zeros((1, 3)), u0, field, push, dt, n_steps, save_every, Q, M
    )

    u_exact = u0 + Q * e_vector / M * t[:, np.newaxis, np.newaxis]
    scale = np.max(np.linalg.norm(u_exact, axis=-1))
    np.testing.assert_allclose(u, u_exact, rtol=0.0, atol=1e-13 * scale)
    half_step_offset = abs(Q) * e_vector[0] / M * dt / 2.0
    assert half_step_offset > 1e3 * 1e-13 * scale


def _exact_state_in_uniform_bz(x0, u0, t, gamma):
    """Exact ``x(t)`` and ``u(t)``, shape ``(n_times, 3)``, in ``B = BZ z``.

    ``du/dt = (Q / (gamma M)) u x B`` gives ``du_x/dt = Omega u_y`` and
    ``du_y/dt = -Omega u_x`` with ``Omega = Q BZ / (gamma M)``: a rotation of the
    perpendicular part of ``u0``, with ``u_z`` and ``gamma`` constant. ``x(t)`` is
    ``x0`` plus the integral of ``u / gamma``.
    """
    rate = Q * BZ / (gamma * M)
    cos, sin = np.cos(rate * t), np.sin(rate * t)
    u_exact = np.empty((t.size, 3))
    u_exact[:, 0] = u0[0] * cos + u0[1] * sin
    u_exact[:, 1] = -u0[0] * sin + u0[1] * cos
    u_exact[:, 2] = u0[2]
    x_exact = np.empty((t.size, 3))
    x_exact[:, 0] = x0[0] + (u0[0] * sin - u0[1] * (cos - 1.0)) / (rate * gamma)
    x_exact[:, 1] = x0[1] + (u0[0] * (cos - 1.0) + u0[1] * sin) / (rate * gamma)
    x_exact[:, 2] = x0[2] + u0[2] * t / gamma
    return x_exact, u_exact


@pytest.mark.convergence
@pytest.mark.physics
@all_pushers
@pytest.mark.parametrize("u_over_c", REGIMES)
def test_position_and_velocity_converge_as_dt_squared(push, u_over_c):
    # Independent truth: the exact gyration in a uniform B, over one gyro-period.
    # The pushers are second order and x and u are kept at the same times, so the
    # errors of x^n and of u^n against the exact state at t^n must both fall as
    # dt^2, with no averaging of any kind. The error is the largest one over all the
    # kept times of the period, so a state half a step off (an error of order dt)
    # cannot hide at a particular time. Slopes fitted on 4 steps from
    # theta = 2 pi / 50 to 2 pi / 400, tolerance +/- 0.1 as in the other convergence
    # tests of the project.
    x0 = np.array([1.0e-3, -2.0e-3, 0.5e-3])
    u0 = u_over_c * c * np.array([0.8, 0.0, 0.6])
    gamma = math.sqrt(1.0 + u_over_c**2)
    period = 2.0 * math.pi * gamma * M / (abs(Q) * BZ)
    r_larmor = M * 0.8 * u_over_c * c / (abs(Q) * BZ)
    field = fields.uniform(b_field=[0.0, 0.0, BZ], e_field=[0.0, 0.0, 0.0])

    speed = np.linalg.norm(u0)
    steps_per_period = np.array([50, 100, 200, 400])
    x_errors, u_errors = [], []
    for n_steps in steps_per_period:
        dt = period / n_steps
        t, x, u = integrate(
            x0[np.newaxis, :], u0[np.newaxis, :], field, push, dt, int(n_steps), 1, Q, M
        )
        x_exact, u_exact = _exact_state_in_uniform_bz(x0, u0, t, gamma)
        # Index 0: the only particle.
        x_errors.append(np.max(np.linalg.norm(x[:, 0] - x_exact, axis=-1)) / r_larmor)
        u_errors.append(np.max(np.linalg.norm(u[:, 0] - u_exact, axis=-1)) / speed)

    log_dt = np.log(1.0 / steps_per_period)
    assert np.polyfit(log_dt, np.log(x_errors), 1)[0] == pytest.approx(2.0, abs=0.1)
    assert np.polyfit(log_dt, np.log(u_errors), 1)[0] == pytest.approx(2.0, abs=0.1)


@pytest.mark.parametrize(
    ("change", "message"),
    [
        ({"n_steps": 0}, "n_steps must be a strictly positive integer"),
        ({"n_steps": 10.0}, "n_steps must be a strictly positive integer"),
        ({"save_every": 0}, "save_every must be a strictly positive integer"),
        ({"n_steps": 15, "save_every": 10}, "must divide n_steps"),
        ({"dt": 0.0}, "dt must be a finite, strictly positive"),
        ({"dt": -1.0e-12}, "dt must be a finite, strictly positive"),
        ({"dt": float("inf")}, "dt must be a finite, strictly positive"),
        ({"u0": np.zeros((2, 3))}, "u must have the same shape as x"),  # from push
    ],
)
def test_wrong_arguments_are_refused_with_a_clear_message(change, message):
    arguments = {
        "x0": np.zeros((1, 3)),
        "u0": np.ones((1, 3)),
        "field": fields.uniform(b_field=[0.0, 0.0, BZ], e_field=[0.0, 0.0, 0.0]),
        "push": boris,
        "dt": 1.0e-12,
        "n_steps": 20,
        "save_every": 10,
        "q": Q,
        "m": M,
        **change,
    }

    with pytest.raises(ValueError, match=message):
        integrate(**arguments)
