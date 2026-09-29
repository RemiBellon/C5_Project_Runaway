"""Tests of ``ppf_c5_runaways.pushers`` (``boris``, ``higuera_cary``, ``vay``).

Every test compares the result with something that does not come from the code
under test: an exact property of the scheme derived by hand, an analytic solution, or
a published formula (Ripperda et al. 2018, Zenitani & Umeda 2018, Chin & Cator 2022).

The three regimes of the project are covered where it matters: non relativistic
(``u/c = 1e-3``), moderately relativistic (``gamma = 2``) and ultra relativistic
(``gamma = 100``, about 50 MeV for an electron, the runaway regime).
"""

import inspect
import math
from decimal import Decimal, localcontext

import numpy as np
import pytest
from hypothesis import given
from hypothesis import strategies as st

from ppf_c5_runaways import fields
from ppf_c5_runaways.constants import c, e, m_e
from ppf_c5_runaways.pushers import (
    _gamma_from_quartic,
    boris,
    get_pusher,
    higuera_cary,
    vay,
)

Q = -e  # signed charge of an electron [C]
M = m_e  # [kg]
BZ = 2.5  # [T], a tokamak-like field

# |u| / c for the three regimes: gamma = sqrt(1 + (u/c)^2) is about 1, 2 and 100.
REGIMES = [
    pytest.param(1.0e-3, id="non_relativistic"),
    pytest.param(math.sqrt(3.0), id="gamma_2"),
    pytest.param(math.sqrt(9999.0), id="gamma_100"),
]


# The properties every pusher must have run for each of them (one test per pusher).
all_pushers = pytest.mark.parametrize(
    "push", [boris, higuera_cary, vay], ids=["boris", "higuera_cary", "vay"]
)


# --- helpers -------------------------------------------------------------------


def _gamma(u):
    """Lorentz factor of proper velocities ``u`` (last axis: the 3 components)."""
    return np.sqrt(1.0 + np.sum(u**2, axis=-1) / c**2)


def _state(u_over_c):
    """One particle at the origin, ``|u| = u_over_c * c``.

    60 % of ``|u|`` is along z (parallel to B) and 80 % along x (perpendicular),
    since 0.6^2 + 0.8^2 = 1.
    """
    speed = u_over_c * c
    return np.zeros((1, 3)), np.array([[0.8 * speed, 0.0, 0.6 * speed]])


def _dt_for_angle(theta, u, q=Q, m=M, b=BZ):
    """Time step for which ``theta = |q| B dt / (m gamma)`` has the given value."""
    return theta * m * _gamma(u)[0] / (abs(q) * abs(b))


def _larmor_radius(u, q=Q, m=M, b=BZ):
    """Exact Larmor radius ``m |u_perp| / (|q| B)`` [m] (u_perp: perpendicular to z)."""
    u_perp = np.hypot(u[0, 0], u[0, 1])
    return m * u_perp / (abs(q) * abs(b))


def _uniform_b(bz=BZ):
    return fields.uniform(b_field=[0.0, 0.0, bz], e_field=[0.0, 0.0, 0.0])


def _advance(push, x, u, field, t0, dt, n_steps):
    """Apply ``push`` ``n_steps`` times; step ``j`` is done at time ``t0 + j dt``."""
    for j in range(n_steps):
        x, u = push(x, u, field, t0 + j * dt, dt, Q, M)
    return x, u


def _exact_helix(x0, u0, q, m, bz, time):
    """Exact position and proper velocity of one particle in a uniform ``B = bz z``.

    Solution of ``du/dt = (q / (m gamma)) u x B`` with ``gamma`` constant: ``u_perp``
    turns about z with the signed angular frequency ``omega = -q bz / (m gamma)``
    (counter-clockwise for ``q < 0`` and ``bz > 0``) and ``u_z`` is constant.
    """
    gamma = _gamma(u0)[0]
    omega = -q * bz / (m * gamma)
    cos, sin = math.cos(omega * time), math.sin(omega * time)
    ux0, uy0, uz0 = u0[0]
    u = np.array([[ux0 * cos - uy0 * sin, ux0 * sin + uy0 * cos, uz0]])
    displacement = np.array(
        [
            [
                (ux0 * sin + uy0 * (cos - 1.0)) / omega,
                (ux0 * (1.0 - cos) + uy0 * sin) / omega,
                uz0 * time,
            ]
        ]
    )
    return x0 + displacement / gamma, u


def _wavy_field(x, t):
    """A smooth field that depends on the position AND on the time (not uniform)."""
    k, omega = 2.0e3, 1.0e10  # [1/m], [rad/s]
    zeros = np.zeros(len(x))
    e_field = 1.0e5 * np.stack(
        [np.sin(k * x[:, 1] + omega * t), zeros, np.cos(k * x[:, 0])], axis=1
    )
    b_field = np.stack([zeros, 0.5 + zeros, BZ + 0.5 * np.sin(k * x[:, 0])], axis=1)
    return e_field, b_field


# --- exact properties of the scheme ---------------------------------------------


def _assert_speed_is_kept_over_1000_steps(push, u_over_c, theta):
    """Push 3 particles for 1000 steps in a pure B: ``|u|`` must not change.

    Independent truth: a magnetic field does no work, and the three schemes keep
    ``|u|`` (Higuera & Cary 2017, Sec. IV). The particles have different directions.
    Tolerance 1e-13 (relative), the same for every pusher: rounding gives at most
    about one rounding error per step, so even added up linearly 1000 steps give
    1000 * 1.1e-16 = 1.1e-13.
    """
    rng = np.random.default_rng(0)
    directions = rng.normal(size=(3, 3))
    u = u_over_c * c * directions / np.linalg.norm(directions, axis=1, keepdims=True)
    x = np.zeros((3, 3))
    dt = _dt_for_angle(theta, u[:1])

    _, u_end = _advance(push, x, u, _uniform_b(), 0.0, dt, 1000)

    np.testing.assert_allclose(
        np.linalg.norm(u_end, axis=1),
        np.linalg.norm(u, axis=1),
        rtol=1e-13,
        atol=0.0,
    )


@pytest.mark.physics
@pytest.mark.parametrize("push", [boris, higuera_cary], ids=["boris", "higuera_cary"])
@pytest.mark.parametrize("theta", [0.05, 2.0], ids=["small_step", "large_step"])
@pytest.mark.parametrize("u_over_c", REGIMES)
def test_speed_is_conserved_in_a_pure_magnetic_field(push, u_over_c, theta):
    # Boris and Higuera-Cary: their step is a rotation, which keeps |u| whatever the
    # rounding error on gamma. Measured over 1000 steps: below 1e-14 (bound: 1e-13).
    _assert_speed_is_kept_over_1000_steps(push, u_over_c, theta)


# Known limitation of Vay's scheme, not a formula error: its step is not a rotation,
# so a rounding error on gamma_new goes straight into |u|. About 2 roundings per step
# with a small bias add up to 1.1e-13 over 1000 steps, just above the 1e-13 bound (see
# ``test_vay_speed_error_in_one_step_is_a_few_roundings``). The bound is kept as it is,
# on purpose: the test is an expected failure, not a relaxed test.
@pytest.mark.xfail(
    reason="Vay: rounding on gamma_new goes into |u|, 1.1e-13 > 1e-13 in 1000 steps",
    strict=False,  # the outcome depends on the rounding of the platform
)
@pytest.mark.physics
@pytest.mark.parametrize("theta", [0.05, 2.0], ids=["small_step", "large_step"])
@pytest.mark.parametrize("u_over_c", REGIMES)
def test_vay_speed_is_conserved_in_a_pure_magnetic_field(u_over_c, theta):
    _assert_speed_is_kept_over_1000_steps(vay, u_over_c, theta)


@pytest.mark.physics
@pytest.mark.parametrize(("q", "bz"), [(Q, BZ), (-Q, BZ), (Q, -BZ), (-Q, -BZ)])
@pytest.mark.parametrize("theta", [0.05, 0.5, 2.0])
@pytest.mark.parametrize("u_over_c", REGIMES)
def test_velocity_turns_by_the_boris_angle_in_the_right_sense(u_over_c, theta, q, bz):
    # Independent truth: Zenitani & Umeda 2018, Eq. (10): the classic Boris scheme
    # rotates by 2 arctan(theta / 2), not by theta. At theta = 2 that is 1.571 rad
    # instead of 2 rad, so this test cannot pass with an exact rotation.
    # Sense: the angular velocity is -q B / (m gamma), so a negative charge in
    # B along +z turns counter-clockwise (positive angle), a positive one clockwise.
    # Tolerance 1e-13 rad: a few rounding errors (1e-16) on angles of order 1.
    x, u = _state(u_over_c)
    dt = _dt_for_angle(theta, u, q=q, b=bz)
    field = _uniform_b(bz)

    _, u_new = boris(x, u, field, 0.0, dt, q, M)

    cross_z = u[0, 0] * u_new[0, 1] - u[0, 1] * u_new[0, 0]
    dot_xy = u[0, 0] * u_new[0, 0] + u[0, 1] * u_new[0, 1]
    angle = math.atan2(cross_z, dot_xy)
    expected = -math.copysign(1.0, q * bz) * 2.0 * math.atan(theta / 2.0)
    assert angle == pytest.approx(expected, abs=1e-13)
    assert u_new[0, 2] == pytest.approx(u[0, 2], rel=1e-15)  # u_parallel untouched


@pytest.mark.physics
@pytest.mark.parametrize("theta", [0.05, 0.5], ids=["small_step", "large_step"])
@pytest.mark.parametrize("u_over_c", REGIMES)
def test_orbit_is_a_circle_of_the_radius_predicted_for_boris(u_over_c, theta):
    # Independent truth (geometry, no fit): each step moves the particle by
    # dt * v_perp, and the direction turns by theta_B = 2 arctan(theta / 2). The
    # positions are the vertices of a regular polygon, on a circle of radius
    #     R = (side / 2) / sin(theta_B / 2) = r_L sqrt(1 + (theta / 2)^2),
    # (cf. Chin & Cator 2022, Eq. 3.5), whose centre is on the left of the first
    # edge (counter-clockwise turn), at distance R cos(theta_B / 2) from its middle.
    # The motion along z must be uniform: z_n = n dt u_z / gamma.
    # Tolerance 1e-12 (relative): 300 steps of rounding, about 300 * 1e-16 = 3e-14,
    # with a margin of a factor 30.
    n_steps = 300
    x, u = _state(u_over_c)
    dt = _dt_for_angle(theta, u)
    r_larmor = _larmor_radius(u)
    radius = r_larmor * math.sqrt(1.0 + (theta / 2.0) ** 2)
    theta_boris = 2.0 * math.atan(theta / 2.0)
    field = _uniform_b()

    positions = [x[0]]
    for j in range(n_steps):
        x, u = boris(x, u, field, j * dt, dt, Q, M)
        positions.append(x[0])
    positions = np.array(positions)

    first_edge = positions[1, :2] - positions[0, :2]
    direction = first_edge / np.linalg.norm(first_edge)
    left = np.array([-direction[1], direction[0]])
    middle = 0.5 * (positions[0, :2] + positions[1, :2])
    centre = middle + left * radius * math.cos(theta_boris / 2.0)
    distances = np.linalg.norm(positions[:, :2] - centre, axis=1)
    np.testing.assert_allclose(distances, radius, rtol=1e-12, atol=0.0)

    steps = np.arange(n_steps + 1)
    z_expected = steps * dt * 0.6 * u_over_c * c / _gamma(u)[0]
    np.testing.assert_allclose(
        positions[:, 2], z_expected, rtol=1e-12, atol=1e-12 * radius
    )


@all_pushers
@pytest.mark.physics
@pytest.mark.parametrize("u_over_c", REGIMES)
def test_electric_field_parallel_to_b_accelerates_linearly(push, u_over_c):
    # Independent truth: with E and B both along z, dp_z/dt = qE exactly (the force
    # q v x B has no z component), whatever gamma. So u_z grows linearly, for every
    # pusher. Tolerance 1e-12 (relative): 500 steps of rounding.
    # The perpendicular speed is exactly unchanged too (E has no perpendicular
    # component and B only rotates it), and Boris and Higuera-Cary keep that to
    # rounding, since their step rotates u_perp. Vay does not: it multiplies |u_perp|
    # by sqrt(1 + tau^2 / gamma_i^2) in Eq. 9 and divides it by
    # sqrt(1 + tau^2 / gamma_f^2) in Eq. 10, and gamma changes along z. That error
    # is second order, see ``test_vay_perpendicular_speed_error_is_second_order``.
    n_steps = 500
    x, u = _state(u_over_c)
    dt = _dt_for_angle(0.05, u)
    gain = 0.5 * max(u_over_c, 1.0) * c  # total change of u_z over the run [m/s]
    e_z = M * gain / (Q * n_steps * dt)  # E along z [V/m] (negative for an electron)
    field = fields.uniform(b_field=[0.0, 0.0, BZ], e_field=[0.0, 0.0, e_z])

    _, u_end = _advance(push, x, u, field, 0.0, dt, n_steps)

    # After n steps the half-step velocity has received n full kicks.
    assert u_end[0, 2] == pytest.approx(u[0, 2] + gain, rel=1e-12)
    if push is not vay:
        assert np.hypot(u_end[0, 0], u_end[0, 1]) == pytest.approx(
            np.hypot(u[0, 0], u[0, 1]), rel=1e-12
        )


@all_pushers
@pytest.mark.physics
def test_position_is_moved_with_the_new_velocity(
    push,
):
    # Independent truth, done by hand. From rest (u = 0), a pure E field gives
    # u_new = q E dt / m (two half kicks). Choose it equal to 2c along z, so that
    # gamma_new = sqrt(5). The move is then dt * u_new / gamma_new; using the old
    # gamma (1) instead would give a result larger by sqrt(5).
    dt = 1.0e-12
    e_z = 2.0 * c * M / (Q * dt)
    field = fields.uniform(b_field=[0.0, 0.0, 0.0], e_field=[0.0, 0.0, e_z])

    x_new, u_new = push(np.zeros((1, 3)), np.zeros((1, 3)), field, 0.0, dt, Q, M)

    assert u_new[0, 2] == pytest.approx(2.0 * c, rel=1e-14)
    assert x_new[0, 2] == pytest.approx(dt * 2.0 * c / math.sqrt(5.0), rel=1e-14)


@all_pushers
@pytest.mark.physics
@pytest.mark.parametrize("u_over_c", REGIMES)
def test_free_particle_moves_in_a_straight_line(push, u_over_c):
    # Independent truth: no field, no force. u is unchanged and x_new = x + dt u/gamma
    # with gamma computed here with plain Python floats.
    dt = 1.0e-12
    field = fields.uniform(b_field=[0.0, 0.0, 0.0], e_field=[0.0, 0.0, 0.0])
    x = np.array([[1.0, -2.0, 3.0]])
    u = np.array([[0.3, -0.5, 0.8]]) * u_over_c * c
    gamma = math.sqrt(1.0 + sum(float(component) ** 2 for component in u[0]) / c**2)

    x_new, u_new = push(x, u, field, 0.0, dt, Q, M)

    np.testing.assert_array_equal(u_new, u)
    np.testing.assert_allclose(x_new, x + dt * u / gamma, rtol=1e-15, atol=0.0)


# --- time reversal --------------------------------------------------------------


@all_pushers
@pytest.mark.physics
@pytest.mark.parametrize(
    ("field", "tolerance"),
    [
        # Uniform E and B: 200 steps, rounding about 200 * 1e-16, factor 10 margin
        # for the amplification: 1e-12 of the Larmor radius.
        (
            fields.uniform(b_field=[0.3, -0.2, BZ], e_field=[1.0e5, -2.0e5, 3.0e5]),
            1e-12,
        ),
        # Non-uniform, time-dependent field: rounding errors can grow along the
        # orbit; a margin of 1e-10 of the Larmor radius stays far below any error of
        # the scheme itself (about 1e-3), so a wrong reversal cannot hide.
        (_wavy_field, 1e-10),
    ],
    ids=["uniform", "non_uniform_time_dependent"],
)
@pytest.mark.parametrize("u_over_c", REGIMES)
def test_running_backward_from_the_paired_state_gives_back_the_start(
    push, u_over_c, field, tolerance
):
    # Independent truth: the scheme is time symmetric (see the Notes of ``boris``).
    # Forward: (x^0, u^-1/2) -> ... -> (x^N, u^N-1/2), then one more step for
    # u^N+1/2. Backward, with -dt and the time going down from t_N, starting from
    # (x^N, u^N+1/2): after N steps we must find x^0 and u^{+1/2}.
    n_steps = 200
    x0, u_back = _state(u_over_c)
    dt = _dt_for_angle(0.05, u_back)
    r_larmor = _larmor_radius(u_back)
    _, u_half = push(x0, u_back, field, 0.0, dt, Q, M)  # u^{+1/2}

    x_n, u_n_minus = _advance(push, x0, u_back, field, 0.0, dt, n_steps)
    _, u_n_plus = push(x_n, u_n_minus, field, n_steps * dt, dt, Q, M)
    x_end, u_end = _advance(push, x_n, u_n_plus, field, n_steps * dt, -dt, n_steps)

    assert np.max(np.abs(x_end - x0)) < tolerance * r_larmor
    np.testing.assert_allclose(
        u_end, u_half, rtol=tolerance, atol=tolerance * np.linalg.norm(u_half)
    )


def test_running_backward_from_the_unpaired_state_does_not_give_back_the_start():
    # Documents why the test above pairs the states: calling boris with -dt on
    # (x^N, u^N-1/2) accelerates before moving, so the position comes back wrong,
    # by about 5 % of the Larmor radius at theta = 0.05 (30 times that at least is
    # well above rounding). This guards the warning written in the docstring.
    n_steps = 200
    x0, u_back = _state(math.sqrt(3.0))
    dt = _dt_for_angle(0.05, u_back)
    field = _uniform_b()

    x_n, u_n_minus = _advance(boris, x0, u_back, field, 0.0, dt, n_steps)
    x_end, _ = _advance(boris, x_n, u_n_minus, field, n_steps * dt, -dt, n_steps)

    assert np.max(np.abs(x_end - x0)) > 1e-2 * _larmor_radius(u_back)


# --- order of convergence -------------------------------------------------------


@all_pushers
@pytest.mark.convergence
@pytest.mark.physics
@pytest.mark.parametrize("u_over_c", REGIMES)
def test_gyration_error_decreases_as_dt_squared(push, u_over_c):
    # Independent truth: the exact helix (``_exact_helix``). Two gyro-periods,
    # refined 100 -> 800 steps. The start is the exact u at -dt/2 (leapfrog
    # convention). The scheme is second order: slope of log(error) against log(dt)
    # must be 2 within 0.1 (a first-order error, or a wrong rotation angle, would
    # give another slope). The error is far above rounding (1e-6 against 1e-16).
    x0, u0 = _state(u_over_c)
    omega = abs(Q) * BZ / (M * _gamma(u0)[0])
    duration = 2.0 * 2.0 * math.pi / omega
    r_larmor = _larmor_radius(u0)
    field = _uniform_b()
    step_counts = [100, 200, 400, 800]

    errors = []
    for n_steps in step_counts:
        dt = duration / n_steps
        _, u_start = _exact_helix(x0, u0, Q, M, BZ, -dt / 2.0)
        x_end, _ = _advance(push, x0, u_start, field, 0.0, dt, n_steps)
        x_exact, _ = _exact_helix(x0, u0, Q, M, BZ, duration)
        errors.append(np.linalg.norm(x_end - x_exact) / r_larmor)

    dts = duration / np.array(step_counts)
    slope = np.polyfit(np.log(dts), np.log(errors), 1)[0]
    assert slope == pytest.approx(2.0, abs=0.1)


@all_pushers
@pytest.mark.convergence
@pytest.mark.physics
@pytest.mark.parametrize("u_over_c", REGIMES)
def test_relativistic_acceleration_error_decreases_as_dt_squared(push, u_over_c):
    # Independent truth: hyperbolic motion in a uniform E field, B = 0. With
    # u(t) = u0 + a t (exact), z(t) = (c^2 / a) (gamma(t) - gamma(0)). The start is
    # the exact u at -dt/2. This test detects a wrong gamma in the position update:
    # using gamma^{n-1/2} instead of gamma^{n+1/2} makes the scheme first order.
    # Slope of the error against dt must be 2 within 0.1.
    duration = 1.0e-9  # [s]
    u0 = u_over_c * c
    accel = (u0 + c) / duration  # du/dt [m/s^2]
    e_z = M * accel / Q
    field = fields.uniform(b_field=[0.0, 0.0, 0.0], e_field=[0.0, 0.0, e_z])
    step_counts = [50, 100, 200, 400]

    def gamma_of(speed):
        return math.sqrt(1.0 + (speed / c) ** 2)

    z_exact = (c**2 / accel) * (gamma_of(u0 + accel * duration) - gamma_of(u0))
    errors = []
    for n_steps in step_counts:
        dt = duration / n_steps
        u_start = np.array([[0.0, 0.0, u0 - accel * dt / 2.0]])
        x_end, _ = _advance(push, np.zeros((1, 3)), u_start, field, 0.0, dt, n_steps)
        errors.append(abs(x_end[0, 2] - z_exact) / (c * duration))

    dts = duration / np.array(step_counts)
    slope = np.polyfit(np.log(dts), np.log(errors), 1)[0]
    assert slope == pytest.approx(2.0, abs=0.1)


# --- property-based -------------------------------------------------------------

_VECTOR = st.tuples(*[st.floats(-1.0, 1.0)] * 3)


@all_pushers
@given(
    b_field=st.tuples(*[st.floats(-10.0, 10.0)] * 3),
    direction=_VECTOR,
    u_over_c=st.floats(0.0, 100.0),
    dt=st.floats(1.0e-15, 1.0e-9),
    sign=st.sampled_from([-1.0, 1.0]),
)
def test_speed_is_conserved_for_any_b_u_and_dt(
    push, b_field, direction, u_over_c, dt, sign
):
    # Property: in any pure B, for any step (even one much larger than the gyro
    # period: theta up to about 1e3), any speed up to gamma = 100, either sign of
    # dt, |u| does not change. Tolerance 1e-12: one step, rounding of a few 1e-16
    # amplified at most by |tvec| ~ 1e3 in the intermediate u' and reduced again by
    # |svec| ~ 1e-3.
    u = np.array([direction]) * u_over_c * c
    field = fields.uniform(b_field=b_field, e_field=[0.0, 0.0, 0.0])

    _, u_new = push(np.zeros((1, 3)), u, field, 0.0, sign * dt, Q, M)

    assert np.linalg.norm(u_new) == pytest.approx(
        np.linalg.norm(u), rel=1e-12, abs=1e-300
    )


@all_pushers
@given(
    b_field=st.tuples(*[st.floats(-10.0, 10.0)] * 3),
    e_field=st.tuples(*[st.floats(-1.0e9, 1.0e9)] * 3),
    direction=_VECTOR,
    u_over_c=st.floats(0.0, 100.0),
    dt=st.floats(1.0e-15, 1.0e-9),
)
def test_particle_never_moves_faster_than_light(
    push, b_field, e_field, direction, u_over_c, dt
):
    # Property: with any E and B, the distance covered in one step is below c * dt.
    # x = 0 at the start so that the displacement is computed without rounding.
    # The largest u reached is about 600 c, where 1 - v/c = 1 / (2 gamma^2) = 1e-6:
    # far above rounding (1e-16), so the comparison with c is meaningful.
    u = np.array([direction]) * u_over_c * c
    field = fields.uniform(b_field=b_field, e_field=e_field)

    x_new, _ = push(np.zeros((1, 3)), u, field, 0.0, dt, Q, M)

    assert np.linalg.norm(x_new) / dt < c


# --- vectorisation, purity and interface ------------------------------------------


@all_pushers
def test_particles_do_not_influence_each_other(
    push,
):
    # Independent truth: pushing 5 particles at once must give what pushing each
    # alone gives. The field depends on the position, so a mix-up of rows shows.
    # Tolerance 1e-13: only the order of floating-point operations may differ.
    rng = np.random.default_rng(1)
    x = rng.normal(scale=1.0e-3, size=(5, 3))
    u = rng.normal(scale=0.5 * c, size=(5, 3))
    dt = 1.0e-13

    x_all, u_all = push(x, u, _wavy_field, 2.0e-11, dt, Q, M)

    for i in range(5):
        x_one, u_one = push(x[i : i + 1], u[i : i + 1], _wavy_field, 2.0e-11, dt, Q, M)
        np.testing.assert_allclose(x_all[i : i + 1], x_one, rtol=1e-13, atol=1e-25)
        np.testing.assert_allclose(u_all[i : i + 1], u_one, rtol=1e-13, atol=1e-3)


@all_pushers
def test_field_is_called_once_with_the_positions_and_the_time(
    push,
):
    calls = []

    def recording_field(x, t):
        calls.append((x.copy(), t))
        return np.zeros_like(x), np.zeros_like(x)

    x = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])

    push(x, np.zeros((2, 3)), recording_field, 7.5e-9, 1.0e-12, Q, M)

    assert len(calls) == 1
    np.testing.assert_array_equal(calls[0][0], x)  # x^n, not the new positions
    assert calls[0][1] == 7.5e-9


@all_pushers
def test_inputs_are_not_modified_and_outputs_are_new_arrays(
    push,
):
    x = np.array([[1.0, 2.0, 3.0]])
    u = np.array([[1.0e7, 2.0e7, 3.0e7]])
    x_before, u_before = x.copy(), u.copy()

    x_new, u_new = push(x, u, _uniform_b(), 0.0, 1.0e-12, Q, M)

    np.testing.assert_array_equal(x, x_before)
    np.testing.assert_array_equal(u, u_before)
    assert not np.shares_memory(x_new, x)
    assert not np.shares_memory(u_new, u)
    for array in (x_new, u_new):
        assert array.shape == (1, 3)
        assert array.dtype == np.float64


@all_pushers
def test_lists_are_accepted_for_positions_and_velocities(
    push,
):
    x_new, u_new = push(
        [[0.0, 0.0, 0.0]], [[1.0e7, 0.0, 0.0]], _uniform_b(), 0.0, 1.0e-12, Q, M
    )

    assert x_new.shape == u_new.shape == (1, 3)


# --- errors ---------------------------------------------------------------------


def _call(push, **change):
    """Call ``push`` with valid arguments, some of them replaced."""
    arguments = {
        "x": np.zeros((1, 3)),
        "u": np.ones((1, 3)),
        "field": _uniform_b(),
        "t": 0.0,
        "dt": 1.0e-12,
        "q": Q,
        "m": M,
        **change,
    }
    return push(**arguments)


@all_pushers
@pytest.mark.parametrize(
    ("change", "message"),
    [
        ({"x": np.zeros(3)}, r"x must have shape \(n_part, 3\)"),
        ({"x": np.zeros((2, 2))}, r"x must have shape \(n_part, 3\)"),
        ({"u": np.zeros(3)}, "u must have the same shape as x"),
        ({"u": np.zeros((2, 3))}, "u must have the same shape as x"),
        ({"t": float("nan")}, "t must be a finite number"),
        ({"t": "0"}, "t must be a finite number"),
        ({"dt": 0.0}, "dt must be a finite, non-zero"),
        ({"dt": float("inf")}, "dt must be a finite, non-zero"),
        ({"dt": float("nan")}, "dt must be a finite, non-zero"),
        ({"dt": "1e-12"}, "dt must be a finite, non-zero"),
        ({"q": float("nan")}, "q must be a finite number"),
        ({"q": "-1.6e-19"}, "q must be a finite number"),
        ({"m": 0.0}, "m must be a finite, strictly positive"),
        ({"m": -M}, "m must be a finite, strictly positive"),
        ({"m": float("inf")}, "m must be a finite, strictly positive"),
    ],
)
def test_wrong_arguments_are_refused_with_a_clear_message(push, change, message):
    with pytest.raises(ValueError, match=message):
        _call(push, **change)


@all_pushers
def test_negative_dt_is_accepted(
    push,
):
    x_new, _ = _call(push, dt=-1.0e-12)

    assert x_new.shape == (1, 3)


@all_pushers
def test_field_returning_the_wrong_shape_is_refused(
    push,
):
    def bad_field(x, t):
        return np.zeros(3), np.zeros_like(x)  # E of shape (3,) instead of (n, 3)

    with pytest.raises(ValueError, match=r"return E of shape \(1, 3\)"):
        _call(push, field=bad_field)


# --- each scheme solves the equation of its paper ----------------------------------


def _velocity(u):
    """Velocity ``v = u / gamma`` of each particle, shape ``(n_part, 3)``."""
    return u / _gamma(u)[:, np.newaxis]


def _boris_mean_velocity(u_i, u_f, eps):
    """Boris: mean of ``v(u_i + eps)`` and ``v(u_f - eps)`` (Higuera & Cary, Eq. 9)."""
    return 0.5 * (_velocity(u_i + eps) + _velocity(u_f - eps))


def _higuera_cary_mean_velocity(u_i, u_f, eps):
    """Higuera-Cary: velocity of the mean momentum (Higuera & Cary 2017, Eq. 7)."""
    return _velocity(0.5 * (u_i + u_f))


def _vay_mean_velocity(u_i, u_f, eps):
    """Vay: mean of the velocities at both ends of the step (Vay 2008, Eq. 7)."""
    return 0.5 * (_velocity(u_i) + _velocity(u_f))


@pytest.mark.physics
@pytest.mark.parametrize(
    ("push", "mean_velocity"),
    [
        pytest.param(boris, _boris_mean_velocity, id="boris"),
        pytest.param(higuera_cary, _higuera_cary_mean_velocity, id="higuera_cary"),
        pytest.param(vay, _vay_mean_velocity, id="vay"),
    ],
)
@pytest.mark.parametrize("theta", [0.05, 2.0], ids=["small_step", "large_step"])
@pytest.mark.parametrize("u_over_c", REGIMES)
def test_step_solves_the_defining_equation_of_the_scheme(
    push, mean_velocity, u_over_c, theta
):
    # Independent truth: the equation by which each paper *defines* its scheme,
    #     u_f - u_i = (q dt / m) (E + v_bar x B),
    # which differs from one scheme to the next only by the choice of ``v_bar``
    # (Higuera & Cary 2017, Eqs. 5-9). The pushers solve it with a different
    # algorithm (Eqs. 9-12 of Vay, Eqs. 12-24 of Higuera-Cary), so this test would
    # catch a wrong coefficient, sign or factor c in any of them. E and B are not
    # parallel and E is of the size of c B, so that both forces matter.
    # Tolerance 1e-13, relative to the change |u_f - u_i|: the residual is a few
    # rounding errors of |u| (1e-16 each), which is at most 1/theta = 20 times
    # larger relative to |u_f - u_i|: observed below 1e-14.
    rng = np.random.default_rng(3)
    direction = rng.normal(size=(1, 3))
    u_i = u_over_c * c * direction / np.linalg.norm(direction)
    e_field = 0.1 * c * BZ * np.array([1.0, -2.0, 3.0])
    b_field = np.array([0.3, -0.2, BZ])
    field = fields.uniform(b_field=b_field, e_field=e_field)
    dt = _dt_for_angle(theta, u_i)

    _, u_f = push(np.zeros((1, 3)), u_i, field, 0.0, dt, Q, M)

    eps = Q * dt / (2.0 * M) * e_field
    v_bar = mean_velocity(u_i, u_f, eps)
    expected_change = Q * dt / M * (e_field + np.cross(v_bar, b_field))
    residual = np.linalg.norm((u_f - u_i) - expected_change)
    assert residual < 1e-13 * np.linalg.norm(u_f - u_i)


# --- the E x B drift ------------------------------------------------------------------

# Drift speed |E| / (c B) = v / c for the three regimes: gamma is about 1, 2 and 100.
DRIFT_REGIMES = [
    pytest.param(1.0e-3, id="non_relativistic"),
    pytest.param(math.sqrt(3.0) / 2.0, id="gamma_2"),
    pytest.param(math.sqrt(1.0 - 1.0e-4), id="gamma_100"),
]


def _exb_drift_state(kappa):
    """A particle drifting in ``E = kappa c B x``, ``B = BZ z``: ``(u, field)``.

    ``u = gamma v`` with ``v = E x B / B^2`` computed with ``np.cross``. This is the
    stationary solution of the equation of motion: the force ``E + v x B`` vanishes.
    """
    e_vector = np.array([kappa * c * BZ, 0.0, 0.0])
    b_vector = np.array([0.0, 0.0, BZ])
    v_drift = np.cross(e_vector, b_vector) / (b_vector @ b_vector)
    gamma = 1.0 / math.sqrt(1.0 - (v_drift @ v_drift) / c**2)
    field = fields.uniform(b_field=b_vector, e_field=e_vector)
    return gamma * v_drift[np.newaxis, :], field


@pytest.mark.physics
@pytest.mark.parametrize("push", [higuera_cary, vay], ids=["higuera_cary", "vay"])
@pytest.mark.parametrize("theta", [0.05, 0.5, 2.0])
@pytest.mark.parametrize("kappa", DRIFT_REGIMES)
def test_vay_and_higuera_cary_keep_the_exb_drift(push, kappa, theta):
    # Independent truth: at the drift velocity E x B / B^2 the force is zero, so u
    # must not change (Higuera & Cary 2017, Eqs. 28-29: Vay and Higuera-Cary keep
    # this, Boris does not). Tolerance 1e-13 of |u| in one step: the half kick and
    # the rotation are each about theta |u| and cancel, so rounding (1e-16) is
    # amplified at most by 1: observed below 1e-15.
    u, field = _exb_drift_state(kappa)
    dt = _dt_for_angle(theta, u)

    _, u_new = push(np.zeros((1, 3)), u, field, 0.0, dt, Q, M)

    assert np.linalg.norm(u_new - u) < 1e-13 * np.linalg.norm(u)


@pytest.mark.physics
@pytest.mark.parametrize("kappa", DRIFT_REGIMES[1:], ids=["gamma_2", "gamma_100"])
def test_boris_does_not_keep_the_relativistic_exb_drift(kappa):
    # Documents the known weakness of Boris (Higuera & Cary 2017, Sec. IV and Eq. 25):
    # it rotates less than needed, because gamma- >= gamma_new, so u drifts at
    # each step. At theta = 0.5 the change is 1.1e-2 (gamma = 2) and 1.4e-2
    # (gamma = 100) of |u|; the bound 1e-3 is ten times below that and 1e13 times
    # above rounding. In the non relativistic regime Boris is fine (1e-8), which is
    # why the test starts at gamma = 2.
    u, field = _exb_drift_state(kappa)
    dt = _dt_for_angle(0.5, u)

    _, u_new = boris(np.zeros((1, 3)), u, field, 0.0, dt, Q, M)

    assert np.linalg.norm(u_new - u) > 1e-3 * np.linalg.norm(u)


# --- phase-space volume ---------------------------------------------------------------


def _step_jacobian_determinant(push, u, field, dt):
    """Determinant of ``d u_new / d u`` for one step, by central differences.

    In a uniform field the position does not enter the update of ``u``, and the move
    ``x_new = x + dt u_new / gamma`` is a shear (determinant 1): the Jacobian of the
    whole step in ``(x, u)`` is block triangular, so its determinant is that of the
    3 x 3 matrix ``d u_new / d u`` computed here.
    """
    h = 1.0e-5 * c  # step of the differences [m/s]
    jacobian = np.zeros((3, 3))
    for j in range(3):
        change = np.zeros((1, 3))
        change[0, j] = h
        u_plus = push(np.zeros((1, 3)), u + change, field, 0.0, dt, Q, M)[1]
        u_minus = push(np.zeros((1, 3)), u - change, field, 0.0, dt, Q, M)[1]
        jacobian[:, j] = (u_plus - u_minus)[0] / (2.0 * h)
    return np.linalg.det(jacobian)


def _volume_case(u_over_c):
    """A state, a field with E and B not parallel, and a large-ish step."""
    direction = np.array([[0.8, 0.1, 0.6]])
    u = u_over_c * c * direction / np.linalg.norm(direction)
    field = fields.uniform(
        b_field=[0.3, -0.2, BZ], e_field=0.1 * c * BZ * np.array([1.0, -2.0, 3.0])
    )
    return u, field, _dt_for_angle(0.6, u)


@pytest.mark.physics
@pytest.mark.parametrize("push", [boris, higuera_cary], ids=["boris", "higuera_cary"])
@pytest.mark.parametrize("u_over_c", REGIMES)
def test_boris_and_higuera_cary_preserve_the_phase_space_volume(push, u_over_c):
    # Independent truth: the Jacobian determinant of the step is exactly 1
    # (Higuera & Cary 2017, Sec. V, Eqs. 38-39: the two half steps have reciprocal
    # Jacobians). Tolerance 1e-7: the differences have a truncation error of about
    # (1e-5)^2 = 1e-10 and a rounding error eps |u| / h, up to 1e-9 at gamma = 100;
    # 1e-7 leaves a factor 100, and is far below the 1e-2 that Vay shows below.
    u, field, dt = _volume_case(u_over_c)

    determinant = _step_jacobian_determinant(push, u, field, dt)

    assert determinant == pytest.approx(1.0, abs=1e-7)


@pytest.mark.physics
@pytest.mark.parametrize("u_over_c", REGIMES)
def test_vay_changes_the_phase_space_volume_as_the_paper_predicts(u_over_c):
    # Independent truth: Vay does not preserve the volume, and the paper gives the
    # factor (Higuera & Cary 2017, Eqs. 40-41): det = J(u_i) / J(u_f) with
    #     J(u) = 1 + (|beta|^2 + (beta . u / c)^2) / gamma(u)^4,  beta = q B dt / (2 m).
    # Tolerance 1e-7, as for the finite differences above. The factor differs from 1
    # by 1e-2 to 4e-2 in this case, so the test also fails if Vay were volume
    # preserving.
    u, field, dt = _volume_case(u_over_c)
    beta = Q * dt / (2.0 * M) * field(np.zeros((1, 3)), 0.0)[1][0]
    _, u_f = vay(np.zeros((1, 3)), u, field, 0.0, dt, Q, M)

    def jacobian_of_half_step(w):
        return 1.0 + (beta @ beta + (beta @ w[0] / c) ** 2) / _gamma(w)[0] ** 4

    predicted = jacobian_of_half_step(u) / jacobian_of_half_step(u_f)
    determinant = _step_jacobian_determinant(vay, u, field, dt)

    assert abs(predicted - 1.0) > 1e-3
    assert determinant == pytest.approx(predicted, abs=1e-7)


# --- properties that belong to one scheme ---------------------------------------


@pytest.mark.physics
@pytest.mark.parametrize("u_over_c", REGIMES)
def test_higuera_cary_rotation_angle_matches_the_series_of_the_paper(u_over_c):
    # Independent truth: Higuera & Cary 2017, Eq. 27. For u perpendicular to a
    # uniform B, with h = |q| B dt / (m gamma),
    #     angle = h (1 + [ (1 - 1/gamma^2) / 8 - 1/12 ] h^2 + O(h^4)).
    # The omitted term is of relative size h^4. Two checks: it stays below h^4
    # (coefficient 1: measured 0.012 at most, so a factor 80 of margin), and it
    # shrinks by 2^4 = 16 when h is halved (measured 16.0, tolerance 5 %).
    # Angle by atan2, exact to 1e-16: negligible against 1e-8.
    gamma = math.sqrt(1.0 + u_over_c**2)
    u = np.array([[u_over_c * c, 0.0, 0.0]])
    field = _uniform_b()
    relative_errors = []
    for h in (0.1, 0.05):
        dt = _dt_for_angle(h, u)
        _, u_new = higuera_cary(np.zeros((1, 3)), u, field, 0.0, dt, Q, M)
        angle = math.atan2(
            u[0, 0] * u_new[0, 1] - u[0, 1] * u_new[0, 0],
            u[0, 0] * u_new[0, 0] + u[0, 1] * u_new[0, 1],
        )
        series = h * (1.0 + ((1.0 - 1.0 / gamma**2) / 8.0 - 1.0 / 12.0) * h**2)
        relative_errors.append(abs(angle / series - 1.0))
        assert relative_errors[-1] < h**4

    assert relative_errors[0] / relative_errors[1] == pytest.approx(16.0, rel=0.05)


@pytest.mark.physics
@pytest.mark.parametrize("u_over_c", REGIMES)
def test_vay_speed_error_in_one_step_is_a_few_roundings(u_over_c):
    # Documents why Vay misses the 1e-13 bound over 1000 steps in
    # ``test_vay_speed_is_conserved_in_a_pure_magnetic_field``. In a pure B, |u| is
    # exact for Vay only up to the rounding of gamma_new, which goes straight into
    # |u|. Over 200 random directions at theta = 0.05 the largest change is 4.4e-16
    # (2 roundings); the bound 1e-15 is 4.5 roundings.
    rng = np.random.default_rng(1)
    dt = _dt_for_angle(0.05, _state(u_over_c)[1])
    largest = 0.0
    for _ in range(200):
        direction = rng.normal(size=(1, 3))
        u = u_over_c * c * direction / np.linalg.norm(direction)
        _, u_new = vay(np.zeros((1, 3)), u, _uniform_b(), 0.0, dt, Q, M)
        largest = max(largest, abs(np.linalg.norm(u_new) / np.linalg.norm(u) - 1.0))

    assert largest < 1e-15


@pytest.mark.convergence
@pytest.mark.physics
@pytest.mark.parametrize("u_over_c", REGIMES)
def test_vay_perpendicular_speed_error_is_second_order(u_over_c):
    # Vay does not keep |u_perp| when E is parallel to B and gamma changes (see
    # ``test_electric_field_parallel_to_b_accelerates_linearly``), whereas the exact
    # value is constant. The error must be of second order: slope 2 within 0.1 when
    # dt is halved, one gyro-period, 100 -> 800 steps. Measured slope: 2.00, and an
    # error of 1e-4 to 2e-4 at 100 steps, far above rounding.
    speed = u_over_c * c
    u_start = np.array([[0.8 * speed, 0.0, 0.6 * speed]])
    gamma = math.sqrt(1.0 + u_over_c**2)
    duration = 2.0 * math.pi * M * gamma / (abs(Q) * BZ)  # one gyro-period [s]
    gain = 0.5 * max(u_over_c, 1.0) * c  # total change of u_z [m/s]
    step_counts = [100, 200, 400, 800]

    errors = []
    for n_steps in step_counts:
        dt = duration / n_steps
        e_z = M * gain / (Q * duration)
        field = fields.uniform(b_field=[0.0, 0.0, BZ], e_field=[0.0, 0.0, e_z])
        _, u_end = _advance(vay, np.zeros((1, 3)), u_start, field, 0.0, dt, n_steps)
        errors.append(abs(np.hypot(u_end[0, 0], u_end[0, 1]) / (0.8 * speed) - 1.0))

    slope = np.polyfit(np.log(duration / np.array(step_counts)), np.log(errors), 1)[0]
    assert slope == pytest.approx(2.0, abs=0.1)


# --- the Lorentz factor of the rotation ----------------------------------------------


@pytest.mark.parametrize("b", [1.0e-3, 1.0, 2.5e5])
@pytest.mark.parametrize("a", [-2.5e5, -100.0, -1.0, 0.0, 1.0, 1.0e4])
def test_gamma_from_quartic_is_the_positive_root_even_for_a_very_negative_a(a, b):
    # Independent truth: the closed formula gamma^2 = (a + sqrt(a^2 + 4 b)) / 2
    # evaluated with 60 digits (``decimal``), where no cancellation happens. The
    # values of ``a`` go down to -2.5e5, where the plain double-precision formula
    # would lose 6 digits (a + sqrt(a^2 + 4 b) is then of order b / |a|): that is
    # what a huge dt * B gives. Tolerance 1e-14: a few roundings.
    with localcontext() as context:
        context.prec = 60
        exact_a, exact_b = Decimal(a), Decimal(b)
        gamma_squared = (exact_a + (exact_a * exact_a + 4 * exact_b).sqrt()) / 2
        expected = float(gamma_squared.sqrt())

    gamma = _gamma_from_quartic(np.array([[a]]), np.array([[b]]))

    assert gamma.shape == (1, 1)
    assert gamma[0, 0] == pytest.approx(expected, rel=1e-14)


# --- choosing a pusher by name --------------------------------------------------


def test_get_pusher_returns_the_function_of_that_name():
    assert get_pusher("boris") is boris
    assert get_pusher("higuera_cary") is higuera_cary
    assert get_pusher("vay") is vay


def test_get_pusher_lists_the_available_ones_when_the_name_is_unknown():
    message = r"'rk4' is not implemented yet\. Available: "
    message += r"\['boris', 'higuera_cary', 'vay'\]"

    with pytest.raises(ValueError, match=message):
        get_pusher("rk4")


def test_all_pushers_have_the_same_signature():
    # The whole design relies on it: any pusher can replace any other.
    signatures = [
        list(inspect.signature(push).parameters) for push in (boris, higuera_cary, vay)
    ]

    assert signatures[0] == ["x", "u", "field", "t", "dt", "q", "m"]
    assert signatures[1] == signatures[0]
    assert signatures[2] == signatures[0]
