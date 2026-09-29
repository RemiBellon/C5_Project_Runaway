"""Tests of ``ppf_c5_runaways.integrate`` (``integrate``)."""

import math

import numpy as np
import pytest

from ppf_c5_runaways import fields
from ppf_c5_runaways.constants import c, e, m_e
from ppf_c5_runaways.integrate import integrate
from ppf_c5_runaways.pushers import boris

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
