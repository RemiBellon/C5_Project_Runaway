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
    # Boris has three errors of this order in a uniform B (phase delay of
    # 2 pi theta^2 / 12 = 0.52 theta^2, radius and centre of about theta^2 / 8
    # each): their sum, 0.77 theta^2, is below the tolerance.
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

    t, x, u_half, u_centered = integrate(
        np.zeros((1, 3)), u0, field, boris, dt, n_steps, 10, Q, M
    )

    theta = omega * dt
    assert np.hypot(x[-1, 0, 0], x[-1, 0, 1]) < theta**2 * r_larmor
    assert x[-1, 0, 2] == pytest.approx(0.6 * speed * period / gamma, rel=1e-12)
    # Shapes and times: 200 steps kept every 10, plus the initial state.
    assert t.shape == (21,)
    assert x.shape == u_half.shape == u_centered.shape == (21, 1, 3)
    assert t[0] == 0.0
    assert t[-1] == pytest.approx(n_steps * dt, rel=1e-15)
    np.testing.assert_array_equal(x[0], np.zeros((1, 3)))  # the start is kept as is


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
def test_centered_velocity_is_exact_in_a_pure_electric_field(
    push, u_over_c, save_every
):
    # Independent truth: in a uniform E alone, du/dt = q E / m exactly, so
    # u(t) = u0 + q E t / m, whatever gamma. The three pushers add q E dt / m per
    # step with no other change (no rotation when B = 0), so the velocity at the
    # whole steps must be exact up to rounding, and the half-step velocity must be
    # the exact one half a step earlier, off by q E dt / (2 m) from u(t).
    # Tolerance 1e-13 of the largest |u|: 70 additions of rounding 1.1e-16 each give
    # at most 8e-15; the half-step offset tested below is 1e4 times larger (2e-9 in
    # the ultra-relativistic case, the smallest one).
    e_vector = np.array([1.0e7, 0.0, 0.0])  # [V/m]
    u0 = np.array([[0.0, u_over_c * c, 0.0]])
    dt, n_steps = 1.0e-13, 70
    field = fields.uniform(b_field=[0.0, 0.0, 0.0], e_field=e_vector)

    t, _, u_half, u_centered = integrate(
        np.zeros((1, 3)), u0, field, push, dt, n_steps, save_every, Q, M
    )

    time = t[:, np.newaxis, np.newaxis]
    u_exact = u0 + Q * e_vector / M * time
    u_exact_half_step_before = u0 + Q * e_vector / M * (time - dt / 2.0)
    scale = np.max(np.linalg.norm(u_exact, axis=-1))
    np.testing.assert_allclose(u_centered, u_exact, rtol=0.0, atol=1e-13 * scale)
    np.testing.assert_allclose(
        u_half, u_exact_half_step_before, rtol=0.0, atol=1e-13 * scale
    )
    half_step_offset = abs(Q) * e_vector[0] / M * dt / 2.0
    assert half_step_offset > 1e3 * 1e-13 * scale  # the test can tell them apart


def _exact_velocity_in_uniform_bz(u0, t, gamma):
    """Exact ``u(t)``, shape ``(n_times, 3)``, of a particle in ``B = BZ z``.

    ``du/dt = (Q / (gamma M)) u x B`` gives ``du_x/dt = Omega u_y`` and
    ``du_y/dt = -Omega u_x`` with ``Omega = Q BZ / (gamma M)``: a rotation of the
    perpendicular part of ``u0``, with ``u_z`` and ``gamma`` constant.
    """
    angle = Q * BZ / (gamma * M) * t
    u_exact = np.empty((t.size, 3))
    u_exact[:, 0] = u0[0] * np.cos(angle) + u0[1] * np.sin(angle)
    u_exact[:, 1] = -u0[0] * np.sin(angle) + u0[1] * np.cos(angle)
    u_exact[:, 2] = u0[2]
    return u_exact


@pytest.mark.convergence
@pytest.mark.physics
@all_pushers
@pytest.mark.parametrize("u_over_c", REGIMES)
def test_centered_velocity_converges_as_dt_squared_and_half_step_as_dt(push, u_over_c):
    # Independent truth: the exact gyration in a uniform B, over one gyro-period.
    # The pushers are second order, so the error of u^n at the times t^n must fall
    # as dt^2. The half-step velocity u^{n-1/2} is the exact velocity at t^n - dt/2:
    # compared with u(t^n) it is off by about |du/dt| dt / 2, an error of order dt
    # only. This is the error that a forgotten recentering would bring into every
    # diagnostic. Slopes fitted on 4 steps from theta = 2 pi / 50 to 2 pi / 400,
    # tolerance +/- 0.1 as in the other convergence tests of the project. Observed:
    # 2.000 to 2.002 for u^n; 1.002 to 1.051 for u^{n-1/2}, a little above 1 because
    # its dt^2 part (the u^n error, 12 % of it at theta = 2 pi / 50) fades with dt.
    u0 = u_over_c * c * np.array([0.8, 0.0, 0.6])
    gamma = math.sqrt(1.0 + u_over_c**2)
    period = 2.0 * math.pi * gamma * M / (abs(Q) * BZ)
    field = fields.uniform(b_field=[0.0, 0.0, BZ], e_field=[0.0, 0.0, 0.0])

    speed = np.linalg.norm(u0)
    steps_per_period = np.array([50, 100, 200, 400])
    errors_centered, errors_half = [], []
    for n_steps in steps_per_period:
        dt = period / n_steps
        t, _, u_half, u_centered = integrate(
            np.zeros((1, 3)), u0[np.newaxis, :], field, push, dt, int(n_steps), 1, Q, M
        )
        u_exact = _exact_velocity_in_uniform_bz(u0, t, gamma)
        # Largest error over the period, relative to |u|. Index 0: the only particle.
        error_centered = np.linalg.norm(u_centered[:, 0] - u_exact, axis=-1)
        error_half = np.linalg.norm(u_half[:, 0] - u_exact, axis=-1)
        errors_centered.append(np.max(error_centered) / speed)
        errors_half.append(np.max(error_half) / speed)

    log_dt = np.log(1.0 / steps_per_period)
    slope_centered = np.polyfit(log_dt, np.log(errors_centered), 1)[0]
    slope_half = np.polyfit(log_dt, np.log(errors_half), 1)[0]
    assert slope_centered == pytest.approx(2.0, abs=0.1)
    assert slope_half == pytest.approx(1.0, abs=0.1)


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
