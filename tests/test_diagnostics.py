"""Tests of ``ppf_c5_runaways.diagnostics`` (``energy_error``, ``drift_velocity``)."""

import math

import numpy as np
import pytest

from ppf_c5_runaways import fields
from ppf_c5_runaways.constants import c, e, m_e
from ppf_c5_runaways.diagnostics import (
    drift_velocity,
    energy_error,
    gyration_phase,
    gyration_phase_error,
    magnetic_moment,
    phase_error_per_turn,
    pitch_angle,
    predicted_gyration_phase_error,
    rotation_angle_per_step,
    step_jacobian_determinant,
)
from ppf_c5_runaways.integrate import integrate
from ppf_c5_runaways.pushers import boris, higuera_cary, vay

Q = -e  # signed charge of an electron [C]
M = m_e  # [kg]
BZ = 2.5  # [T]

# |u| / c for the three regimes: gamma = sqrt(1 + (u/c)^2) is about 1, 2 and 100.
REGIMES = [
    pytest.param(1.0e-3, id="non_relativistic"),
    pytest.param(math.sqrt(3.0), id="gamma_2"),
    pytest.param(math.sqrt(9999.0), id="gamma_100"),
]
all_pushers = pytest.mark.parametrize(
    "push", [boris, higuera_cary, vay], ids=["boris", "higuera_cary", "vay"]
)


def _velocities(*u_over_c):
    """One particle along x with the given ``|u| / c`` at successive times.

    Shape ``(len(u_over_c), 1, 3)`` [m/s].
    """
    u = np.zeros((len(u_over_c), 1, 3))
    u[:, 0, 0] = np.array(u_over_c) * c
    return u


# --- the formula ------------------------------------------------------------------


def test_energy_error_of_hand_computed_states():
    # Independent truth: gamma = 2 (|u|/c = sqrt 3) then gamma = 3 (|u|/c = sqrt 8),
    # so K goes from m c^2 to 2 m c^2: +100 %. Then back to gamma = 2: 0 %.
    # Tolerance 1e-14: sqrt(3)^2 and sqrt(8)^2 are exact to a few 1e-16.
    u = _velocities(math.sqrt(3.0), math.sqrt(8.0), math.sqrt(3.0))

    error = energy_error(u)

    assert error.shape == (3, 1)
    np.testing.assert_allclose(error[:, 0], [0.0, 1.0, 0.0], rtol=0.0, atol=1e-14)


def test_energy_error_keeps_its_digits_in_the_non_relativistic_regime():
    # Independent truth: the series gamma - 1 = x^2/2 - x^4/8 + ..., x = |u|/c.
    # Doubling x multiplies K by 4 (1 - 3 x^2 / 4 + ...): the change is +300 %, to
    # 1e-16 at x = 1e-8. There, sqrt(1 + x^2) rounds to exactly 1, so the direct
    # formula gamma - 1 would give 0 / 0; the formula without subtraction must not.
    u = _velocities(1.0e-8, 2.0e-8)

    error = energy_error(u)

    assert error[1, 0] == pytest.approx(3.0, rel=1e-14)


def test_each_particle_is_compared_with_its_own_initial_energy():
    u = np.concatenate(
        [_velocities(math.sqrt(3.0), math.sqrt(8.0)), _velocities(1.0e-3, 1.0e-3)],
        axis=1,
    )  # particle 0: +100 %, particle 1: no change

    error = energy_error(u)

    np.testing.assert_allclose(error[1], [1.0, 0.0], rtol=0.0, atol=1e-14)


@pytest.mark.parametrize(
    ("u", "message"),
    [
        (np.zeros((5, 3)), "shape"),
        (np.zeros((5, 2, 2)), "shape"),
        (_velocities(0.0, 1.0e-3), r"Particle\(s\) \[0\] are at rest"),
    ],
    ids=["missing_particle_axis", "two_components", "at_rest"],
)
def test_wrong_input_is_refused_with_a_clear_message(u, message):
    with pytest.raises(ValueError, match=message):
        energy_error(u)


# --- what it shows about the pushers ---------------------------------------------


@pytest.mark.physics
@pytest.mark.parametrize(
    ("push", "bound"),
    [(boris, 1e-13), (higuera_cary, 1e-13), (vay, 1e-11)],
    ids=["boris", "higuera_cary", "vay"],
)
@pytest.mark.parametrize("u_over_c", REGIMES)
def test_energy_is_kept_in_a_pure_magnetic_field(push, bound, u_over_c):
    # Independent truth: in a static B alone, K is a constant of motion.
    # 10 000 steps at theta = 0.1. Boris and Higuera-Cary keep |u^{n+1/2}| exactly by
    # construction, and in a uniform B the rotation angle is the same at every step,
    # so |u^n| is constant too: only rounding is left (observed <= 8e-15, bound
    # 1e-13). Vay does not keep |u| exactly: the rounding of gamma_new goes into |u|
    # at each step and adds up as a random walk (observed up to 5.8e-13 at
    # u/c = 1e-3, bound 1e-11; see also the Vay xfail of test_pushers).
    gamma = math.sqrt(1.0 + u_over_c**2)
    dt = 0.1 * gamma * M / (e * BZ)
    field = fields.uniform(b_field=[0.0, 0.0, BZ], e_field=[0.0, 0.0, 0.0])
    u0 = u_over_c * c * np.array([[0.8, 0.0, 0.6]])

    _, _, u = integrate(np.zeros((1, 3)), u0, field, push, dt, 10_000, 10, Q, M)

    assert np.max(np.abs(energy_error(u))) < bound


def _drift_case(gamma_drift):
    """A particle moving exactly at the E x B drift velocity: ``(u0, field, dt)``.

    ``E = kappa c B x`` and ``B = BZ z`` give ``v = E x B / B^2`` of speed
    ``kappa c``: the force ``q (E + v x B)`` is zero, the motion is uniform and K is
    constant. ``dt`` is set for a rotation angle theta = 0.5 per step.
    """
    kappa = math.sqrt(1.0 - 1.0 / gamma_drift**2)
    e_vector = np.array([kappa * c * BZ, 0.0, 0.0])
    b_vector = np.array([0.0, 0.0, BZ])
    v_drift = np.cross(e_vector, b_vector) / (b_vector @ b_vector)
    u0 = gamma_drift * v_drift[np.newaxis, :]
    dt = 0.5 * gamma_drift * M / (e * BZ)
    return u0, fields.uniform(b_field=b_vector, e_field=e_vector), dt


@pytest.mark.physics
@pytest.mark.parametrize("push", [higuera_cary, vay], ids=["higuera_cary", "vay"])
@pytest.mark.parametrize("gamma_drift", [2.0, 100.0], ids=["gamma_2", "gamma_100"])
def test_vay_and_higuera_cary_keep_the_energy_of_the_exb_drift(push, gamma_drift):
    # Independent truth: at the drift velocity the motion is uniform, K is constant.
    # Both schemes keep the drift exactly (Higuera & Cary 2017, Eqs. 28-29), so only
    # rounding is left after 1000 steps: observed 0 at gamma = 2 and 2.6e-12 at
    # gamma = 100, where E and v x B nearly cancel and amplify it. Bound 1e-10.
    u0, field, dt = _drift_case(gamma_drift)

    _, _, u = integrate(np.zeros((1, 3)), u0, field, push, dt, 1000, 10, Q, M)

    assert np.max(np.abs(energy_error(u))) < 1e-10


@pytest.mark.physics
@pytest.mark.parametrize("gamma_drift", [2.0, 100.0], ids=["gamma_2", "gamma_100"])
def test_boris_does_not_keep_the_energy_of_the_relativistic_exb_drift(gamma_drift):
    # Documents the known weakness of Boris (Higuera & Cary 2017, Sec. IV): it does
    # not keep the relativistic drift, so the particle starts to gyrate about a
    # wrong drift and its energy changes. Observed over 1000 steps: 28 % at
    # gamma = 2 and a factor 31 at gamma = 100. The bound 1e-2 is well below both,
    # and 1e8 times above the rounding level of the two other schemes.
    u0, field, dt = _drift_case(gamma_drift)

    _, _, u = integrate(np.zeros((1, 3)), u0, field, boris, dt, 1000, 10, Q, M)

    assert np.max(np.abs(energy_error(u))) > 1e-2


# --- drift_velocity -----------------------------------------------------------------


def test_drift_velocity_of_a_uniform_motion_is_exact():
    # Independent truth: x = x0 + v t is a straight line, the fit must give v back
    # for each particle, to rounding (1e-12 of |v| over 11 points).
    t = np.linspace(0.0, 1.0e-9, 11)
    v = np.array([[1.0e6, -2.0e6, 3.0e5], [0.0, 0.0, -4.0e7]])  # 2 particles [m/s]
    x0 = np.array([[1.0e-3, 0.0, 0.0], [0.0, 2.0e-3, 0.0]])
    x = x0 + v * t[:, np.newaxis, np.newaxis]

    np.testing.assert_allclose(drift_velocity(t, x), v, rtol=1e-12, atol=1e-6)


def test_gyration_biases_the_drift_as_predicted():
    # Independent truth: the bias of a least-squares slope on r cos(omega t + phi)
    # over a whole number of turns and a duration L, 12 r sin(phi) / (omega L^2),
    # derived by hand by parts (see the Notes of drift_velocity). Continuous
    # limit: with 200 001 points the sum differs from the integral by 1.5e-5 in
    # relative terms (observed), tolerance 1e-4. Doubling L divides it by 4.
    omega, radius, phase = 2.0, 0.3, 1.0
    for n_turns in (10, 20):
        duration = n_turns * 2.0 * np.pi / omega
        t = np.linspace(0.0, duration, 200_001)
        x = np.zeros((t.size, 1, 3))
        x[:, 0, 0] = radius * np.cos(omega * t + phase)

        bias = drift_velocity(t, x)[0, 0]

        expected = 12.0 * radius * np.sin(phase) / (omega * duration**2)
        assert bias == pytest.approx(expected, rel=1e-4)


@pytest.mark.parametrize(
    ("t", "x", "message"),
    [
        (np.zeros(3), np.zeros((3, 3)), "shape"),
        (np.zeros(3), np.zeros((4, 1, 3)), "same n_saved"),
        (np.full(3, 1.0e-9), np.zeros((3, 1, 3)), "two different times"),
    ],
    ids=["missing_particle_axis", "different_lengths", "one_time_only"],
)
def test_drift_velocity_refuses_wrong_input_with_a_clear_message(t, x, message):
    with pytest.raises(ValueError, match=message):
        drift_velocity(t, x)


def _launched_at_drift(u_over_c):
    """Ripperda et al. (2018, Sec. 3.1.4) fields, E = (E0, 0, 0), B = (0, 0, BZ).

    The particle starts at the exact drift velocity ``v_E = E x B / B^2`` of
    Lorentz factor ``kappa = sqrt(1 + (u_E/c)^2)``, instead of at rest as in the
    paper, so that it does not gyrate. Returns ``(u0, field, v_drift, dt)`` with
    ``dt`` giving a rotation of 0.5 rad per step at the drift Lorentz factor.
    """
    kappa = math.sqrt(1.0 + u_over_c**2)
    e_vector = np.array([u_over_c / kappa * c * BZ, 0.0, 0.0])
    b_vector = np.array([0.0, 0.0, BZ])
    v_drift = np.cross(e_vector, b_vector) / (b_vector @ b_vector)
    dt = 0.5 * kappa * M / (e * BZ)
    field = fields.uniform(b_field=b_vector, e_field=e_vector)
    return kappa * v_drift[np.newaxis, :], field, v_drift, dt


@pytest.mark.physics
@pytest.mark.parametrize("push", [higuera_cary, vay], ids=["higuera_cary", "vay"])
@pytest.mark.parametrize("u_over_c", REGIMES)
def test_vay_and_higuera_cary_drift_at_the_exact_exb_velocity(push, u_over_c):
    # Independent truth: v_E = E x B / B^2, exact at any speed below c. Both
    # schemes keep it exactly (Vay 2008; Higuera & Cary 2017, Eqs. 28-29): over
    # 2000 steps of 0.5 rad the fitted drift differs from v_E by rounding only,
    # observed below 6e-14 of |v_E|, bound 1e-12.
    u0, field, v_drift, dt = _launched_at_drift(u_over_c)

    t, x, _ = integrate(np.zeros((1, 3)), u0, field, push, dt, 2000, 1, Q, M)

    error = np.linalg.norm(drift_velocity(t, x)[0] - v_drift) / np.linalg.norm(v_drift)
    assert error < 1e-12


@pytest.mark.physics
@pytest.mark.parametrize("u_over_c", REGIMES[1:])
def test_boris_drifts_at_a_wrong_velocity_in_the_relativistic_regime(u_over_c):
    # Documents the known weakness of Boris (Higuera & Cary 2017, Sec. IV): the
    # rotation uses gamma- >= gamma_new, so E and v x B do not cancel and the
    # particle drifts at a wrong velocity. Observed with 2000 steps of 0.5 rad:
    # 2.0e-2 of |v_E| at gamma = 2 and 4.5e-2 at gamma = 100, bound 1e-3. In the
    # non-relativistic regime Boris is fine (3e-8), which is why it is not tested.
    u0, field, v_drift, dt = _launched_at_drift(u_over_c)

    t, x, _ = integrate(np.zeros((1, 3)), u0, field, boris, dt, 2000, 1, Q, M)

    error = np.linalg.norm(drift_velocity(t, x)[0] - v_drift) / np.linalg.norm(v_drift)
    assert error > 1e-3


# --- gyration_phase -----------------------------------------------------------------


def test_gyration_phase_recovers_known_angles_about_any_direction():
    # Independent truth: velocities built by hand at known angles in the plane
    # perpendicular to b = (1, 1, 1), in the right-handed basis (a, b_hat x a),
    # with a component along b that must be ignored. The angles go beyond 2 pi, so
    # the unwrapping is tested too. Only differences of phase are compared: the
    # origin of the angle depends on the basis chosen inside the function.
    b_hat = np.array([1.0, 1.0, 1.0]) / math.sqrt(3.0)
    a = np.array([1.0, -1.0, 0.0]) / math.sqrt(2.0)
    angles = np.linspace(0.0, 3.0 * 2.0 * np.pi, 61)  # steps of 0.31 rad < pi
    u = np.zeros((angles.size, 1, 3))
    u[:, 0, :] = 1.0e7 * (
        np.cos(angles)[:, np.newaxis] * a
        + np.sin(angles)[:, np.newaxis] * np.cross(b_hat, a)
        + 0.5 * b_hat
    )

    phase = gyration_phase(u, 2.0 * b_hat)[:, 0]

    np.testing.assert_allclose(phase - phase[0], angles, rtol=0.0, atol=1e-12)


@pytest.mark.parametrize(
    ("u", "b", "message"),
    [
        (np.ones((2, 1, 3)), np.zeros(3), "zero vector"),
        (np.array([[[0.0, 0.0, 1.0e7]]]), np.array([0.0, 0.0, 2.5]), "parallel"),
        (np.ones((2, 3)), np.array([0.0, 0.0, 1.0]), "shape"),
    ],
    ids=["zero_field", "velocity_along_field", "wrong_shape"],
)
def test_gyration_phase_refuses_undefined_cases(u, b, message):
    with pytest.raises(ValueError, match=message):
        gyration_phase(u, b)


def _phase_error_after(push, gamma, n_turns=100, steps_per_turn=100):
    """Phase error of ``push`` after ``n_turns`` exact turns, and the angle theta.

    Ripperda et al. (2018, Sec. 3.1.2): velocity perpendicular to a uniform B, one
    turn resolved by 100 steps. The error is the change of phase of u
    minus the exact one, omega t, so that a constant offset at t = 0 cancels.
    """
    u_perp = math.sqrt(gamma**2 - 1.0) * c
    omega = e * BZ / (gamma * M)
    dt = 2.0 * math.pi / omega / steps_per_turn
    field = fields.uniform(b_field=[0.0, 0.0, BZ], e_field=[0.0, 0.0, 0.0])
    u0 = np.array([[0.0, -u_perp, 0.0]])
    n_steps = n_turns * steps_per_turn

    t, _, u = integrate(np.zeros((1, 3)), u0, field, push, dt, n_steps, 10, Q, M)

    phase = gyration_phase(u, [0.0, 0.0, 1.0])[:, 0]
    return (phase[-1] - phase[0]) - omega * t[-1], omega * dt, n_steps


@pytest.mark.physics
@pytest.mark.parametrize("push", [boris, vay], ids=["boris", "vay"])
@pytest.mark.parametrize("gamma", [1.0 + 5.0e-7, 2.0, 1.0e6], ids=["~1", "2", "1e6"])
def test_boris_and_vay_lag_by_the_exact_boris_angle(push, gamma):
    # Independent truth: the Boris rotation turns u by 2 arctan(theta / 2) per step
    # instead of theta (Zenitani & Umeda 2018, Eq. 12), so after N steps the lag is
    # N (2 arctan(theta / 2) - theta) = -0.2066 rad for 100 turns of 100 steps:
    # the "about 0.2 rad" of Ripperda et al. (2018, Fig. 4). Vay turns by the same
    # angle when u is perpendicular to B, as Ripperda et al. observe. Tolerance
    # 1e-9 rad: observed below 1.2e-11 (rounding of 1e4 steps).
    error, theta, n_steps = _phase_error_after(push, gamma)

    expected = n_steps * (2.0 * math.atan(theta / 2.0) - theta)
    assert error == pytest.approx(expected, abs=1e-9)


@pytest.mark.physics
def test_higuera_cary_leads_by_half_the_boris_lag_in_the_ultra_relativistic_limit():
    # Independent truth: for u perpendicular to B and gamma >> 1, the Lorentz
    # factor of the Higuera-Cary rotation (Higuera & Cary 2017, Eq. 20) tends to
    # gamma sqrt(1 - theta^2 / 4), so the angle per step is 2 arcsin(theta / 2):
    # a lead of N (2 arcsin(theta / 2) - theta) = +0.1034 rad, half the Boris lag and
    # of opposite sign, as Ripperda et al. (2018, Fig. 4) report ("roughly half").
    # At gamma = 1e6 the limit is reached to 1e-12; observed difference 2e-12,
    # tolerance 1e-9 rad.
    error, theta, n_steps = _phase_error_after(higuera_cary, 1.0e6)

    expected = n_steps * (2.0 * math.asin(theta / 2.0) - theta)
    assert error == pytest.approx(expected, abs=1e-9)


# --- rotation angle per step and phase error prediction -----------------------------


def _pitched_velocity(gamma, pitch_degrees):
    """``u`` of shape (1, 3) [m/s] of Lorentz factor ``gamma``, pitch angle to z."""
    speed = math.sqrt(gamma**2 - 1.0) * c
    pitch = math.radians(pitch_degrees)
    return np.array([[0.0, -speed * math.sin(pitch), speed * math.cos(pitch)]])


def _dt_for_exact_angle(theta, gamma):
    """Time step [s] for an exact angle ``theta`` per step at ``gamma``, in BZ."""
    return theta * gamma * M / (e * BZ)


B_ALONG_Z = np.array([0.0, 0.0, BZ])


@pytest.mark.parametrize("theta", [0.01, 0.5, 2.0])
def test_boris_angle_is_twice_arctan_of_half_the_exact_angle(theta):
    # Independent truth: the Boris rotation turns by 2 arctan(theta / 2)
    # (Zenitani & Umeda 2018, Eq. 12); at theta = 2 that is pi / 2 exactly. For an
    # electron and dt > 0 the angle is positive; it changes sign with q or dt.
    # Tolerance: rounding.
    u = _pitched_velocity(2.0, 90.0)
    dt = _dt_for_exact_angle(theta, 2.0)

    angle = rotation_angle_per_step("boris", u, B_ALONG_Z, dt, Q, M)[0]

    assert angle == pytest.approx(2.0 * math.atan(theta / 2.0), rel=1e-14)
    assert rotation_angle_per_step("boris", u, B_ALONG_Z, dt, -Q, M)[0] == -angle
    assert rotation_angle_per_step("boris", u, B_ALONG_Z, -dt, Q, M)[0] == -angle


@pytest.mark.parametrize("pitch", [90.0, 45.0, 10.0])
@pytest.mark.parametrize("gamma", [1.0 + 5.0e-7, 2.0, 1.0e6], ids=["~1", "2", "1e6"])
def test_vay_turns_by_the_boris_angle_at_any_pitch(gamma, pitch):
    # Independent truth: without E, gamma(u) solves Vay's quartic (Vay 2008,
    # Eq. 11), so Vay's step is the Boris rotation, whatever the pitch angle
    # (derivation in rotation_angle_per_step). Checked here on the real pusher:
    # the angle between u and the pushed u, about z, for one step of 0.3 rad.
    # Tolerance 1e-12 rad: rounding of the angles.
    u = _pitched_velocity(gamma, pitch)
    dt = _dt_for_exact_angle(0.3, gamma)
    field = fields.uniform(b_field=B_ALONG_Z, e_field=[0.0, 0.0, 0.0])

    _, u_new = vay(np.zeros((1, 3)), u, field, 0.0, dt, Q, M)

    turned = math.atan2(
        u[0, 0] * u_new[0, 1] - u[0, 1] * u_new[0, 0], u[0, :2] @ u_new[0, :2]
    )
    predicted = rotation_angle_per_step("vay", u, B_ALONG_Z, dt, Q, M)[0]
    assert predicted == pytest.approx(2.0 * math.atan(0.15), abs=1e-14)
    assert turned == pytest.approx(predicted, abs=1e-12)


@pytest.mark.parametrize("pitch", [90.0, 54.7356, 45.0, 10.0])
def test_higuera_cary_angle_matches_its_ultra_relativistic_limit(pitch):
    # Independent truth: for gamma >> 1, Eq. 20 of Higuera & Cary (2017) with
    # |beta| / gamma = theta / 2 and beta . u / (c gamma^2) -> (theta / 2) cos psi
    # gives gamma_r^2 / gamma^2 = x,
    #     x = [(1 - theta^2/4) + sqrt((1 - theta^2/4)^2 + theta^2 cos^2 psi)] / 2,
    # so the angle is 2 arctan((theta / 2) / sqrt(x)), derived by hand. At
    # gamma = 1e8 the neglected terms are of relative size 1 / gamma^2 = 1e-16.
    # Tolerance 1e-13 of the angle.
    theta, gamma = 0.5, 1.0e8
    u = _pitched_velocity(gamma, pitch)
    dt = _dt_for_exact_angle(theta, gamma)
    cos_squared = math.cos(math.radians(pitch)) ** 2

    angle = rotation_angle_per_step("higuera_cary", u, B_ALONG_Z, dt, Q, M)[0]

    a = 1.0 - theta**2 / 4.0
    x = 0.5 * (a + math.sqrt(a**2 + theta**2 * cos_squared))
    assert angle == pytest.approx(
        2.0 * math.atan(theta / 2.0 / math.sqrt(x)), rel=1e-13
    )


@pytest.mark.parametrize("pitch", [90.0, 54.7356, 45.0, 10.0])
def test_higuera_cary_angle_error_has_the_third_order_term_derived_by_hand(pitch):
    # Independent truth: the series of the limit above,
    # alpha - theta = theta^3 (3 sin^2 psi - 2) / 24 + O(theta^5). At theta = 1e-2
    # the next term is about theta^5 = 1e-10, 1e-4 of the theta^3 term: tolerance
    # 1e-3 of the theta^3 coefficient, i.e. 1e-3 * theta^3 / 24 absolute. At the
    # magic pitch (sin^2 psi = 2/3) the theta^3 term vanishes.
    theta, gamma = 1.0e-2, 1.0e8
    u = _pitched_velocity(gamma, pitch)
    dt = _dt_for_exact_angle(theta, gamma)
    sin_squared = math.sin(math.radians(pitch)) ** 2

    angle = rotation_angle_per_step("higuera_cary", u, B_ALONG_Z, dt, Q, M)[0]

    expected = theta**3 * (3.0 * sin_squared - 2.0) / 24.0
    assert angle - theta == pytest.approx(expected, abs=1e-3 * theta**3 / 24.0)


def test_higuera_cary_angle_is_the_boris_angle_in_the_non_relativistic_limit():
    # Independent truth: for |u| << c, gamma_r -> gamma in Eq. 20 (beta . u / c and
    # |beta|^2 / gamma^2 are then the same terms as in Boris to order (u/c)^2).
    # At u/c = 1e-3 the angles differ by about (u/c)^2 theta^3 = 1e-7 theta^3;
    # tolerance 1e-6 of the angle.
    u = _pitched_velocity(math.sqrt(1.0 + 1.0e-6), 45.0)
    dt = _dt_for_exact_angle(0.5, math.sqrt(1.0 + 1.0e-6))

    hc = rotation_angle_per_step("higuera_cary", u, B_ALONG_Z, dt, Q, M)[0]
    boris_angle = rotation_angle_per_step("boris", u, B_ALONG_Z, dt, Q, M)[0]

    assert hc == pytest.approx(boris_angle, rel=1e-6)


def test_rotation_angle_refuses_an_unknown_scheme():
    with pytest.raises(ValueError, match="scheme must be"):
        rotation_angle_per_step("rk4", np.ones((1, 3)), B_ALONG_Z, 1e-12, Q, M)


def test_phase_error_of_a_known_rotation_across_the_atan2_cut():
    # Independent truth: velocities turned by hand at the exact electron rate,
    # plus a made-up error of -1e-6 rad at t = 0 and a lag of 0.01 rad per turn.
    # u0 = (0, -|u|, 0) is on the cut of atan2, and the -1e-6 moves the first
    # stored velocity to the other side: no spurious 2 pi may appear. Tolerance
    # 1e-10 rad: rounding of the angles over 3 turns.
    gamma = 2.0
    speed = math.sqrt(gamma**2 - 1.0) * c
    omega = e * BZ / (gamma * M)
    t = np.linspace(0.0, 3.0 * 2.0 * math.pi / omega, 301)
    made_up = -1.0e-6 - 0.01 * t * omega / (2.0 * math.pi)
    angle = omega * t + made_up
    u = np.zeros((t.size, 1, 3))
    u[:, 0, 0] = speed * np.sin(angle)  # (0, -speed, 0) turned by +angle about z
    u[:, 0, 1] = -speed * np.cos(angle)

    error = gyration_phase_error(t, u, np.array([[0.0, -speed, 0.0]]), B_ALONG_Z, Q, M)

    np.testing.assert_allclose(error[:, 0], made_up, rtol=0.0, atol=1e-10)


def test_phase_error_refuses_states_saved_too_far_apart():
    # 0.8 exact turn between saved states: the phase cannot be followed.
    speed = math.sqrt(3.0) * c
    period = 2.0 * math.pi * 2.0 * M / (e * BZ)
    t = 0.8 * period * np.arange(4)
    u = np.tile([0.0, -speed, 0.0], (4, 1, 1))

    with pytest.raises(ValueError, match="save more often"):
        gyration_phase_error(t, u, u[0], B_ALONG_Z, Q, M)


@pytest.mark.physics
@all_pushers
@pytest.mark.parametrize("pitch", [90.0, 45.0, 10.0])
@pytest.mark.parametrize("gamma", [1.0 + 5.0e-7, 2.0, 1.0e6], ids=["~1", "2", "1e6"])
def test_measured_phase_error_is_the_predicted_curve(push, gamma, pitch):
    # Independent truth: the whole curve e_n = n (alpha - theta), derived from the
    # rotation angle per step of each scheme (see predicted_gyration_phase_error),
    # with the angles taken from the papers (Boris 1970; Vay 2008, Eq. 11; Higuera
    # & Cary 2017, Eq. 20). 10 turns of 100 steps, saved every 10 steps. The error
    # is exactly zero at t = 0 and reaches 2.6e-3 to 2.1e-2 rad after 10 turns.
    # Observed agreement below 1e-12 rad over the whole curve; tolerance 1e-9 rad.
    u0 = _pitched_velocity(gamma, pitch)
    dt = _dt_for_exact_angle(2.0 * math.pi / 100, gamma)
    field = fields.uniform(b_field=B_ALONG_Z, e_field=[0.0, 0.0, 0.0])

    t, _, u = integrate(np.zeros((1, 3)), u0, field, push, dt, 1000, 10, Q, M)

    measured = gyration_phase_error(t, u, u0, B_ALONG_Z, Q, M)
    predicted = predicted_gyration_phase_error(
        push.__name__, t, u0, B_ALONG_Z, dt, Q, M
    )
    np.testing.assert_allclose(measured, predicted, rtol=0.0, atol=1e-9)


@pytest.mark.physics
def test_higuera_cary_has_no_third_order_phase_error_at_the_magic_pitch():
    # Consequence of the ultra-relativistic angle above: the theta^3 term
    # theta^3 (3 sin^2 psi - 2) / 24 vanishes at sin^2 psi = 2 / 3 (psi = 54.7
    # degrees), leaving an error of order theta^5 per step. After 100 turns of 100
    # steps at gamma = 1e6 Boris lags by 0.207 rad; Higuera-Cary must stay below
    # 1e-3 rad, a factor 200 smaller (observed: -4.8e-5 rad, 4300 times smaller).
    pitch = math.degrees(math.asin(math.sqrt(2.0 / 3.0)))
    u0 = _pitched_velocity(1.0e6, pitch)
    dt = _dt_for_exact_angle(2.0 * math.pi / 100, 1.0e6)
    field = fields.uniform(b_field=B_ALONG_Z, e_field=[0.0, 0.0, 0.0])

    t, _, u = integrate(np.zeros((1, 3)), u0, field, higuera_cary, dt, 10_000, 10, Q, M)

    error = gyration_phase_error(t, u, u0, B_ALONG_Z, Q, M)
    assert abs(error[-1, 0]) < 1e-3


# --- pitch angle and phase error per turn -----------------------------------------


def test_pitch_angle_of_hand_computed_velocities():
    # Independent truth: along B 0, perpendicular pi / 2, against B pi, and 45
    # degrees for (1, 0, 1) in a field along z; the length of B does not matter.
    u = np.array([[0, 0, 3.0], [2.0, 0, 0], [0, 0, -1.0], [1.0, 0, 1.0]]) * 1.0e7

    psi = pitch_angle(u, np.array([0.0, 0.0, 5.0]))

    np.testing.assert_allclose(psi, [0.0, np.pi / 2, np.pi, np.pi / 4], atol=1e-15)


def test_pitch_angle_refuses_a_zero_velocity():
    with pytest.raises(ValueError, match="non-zero"):
        pitch_angle(np.zeros((1, 3)), np.array([0.0, 0.0, 1.0]))


def test_phase_error_per_turn_is_the_slope_whatever_the_offset():
    # Independent truth: a made-up error offset + slope * t, per particle; the fit
    # must give slope * period back and ignore the offset. Tolerance: rounding.
    t = np.linspace(0.0, 5.0e-9, 51)
    slopes = np.array([-3.0e6, 2.0e5])  # [rad/s]
    offsets = np.array([1.0e-5, -2.0])  # [rad]
    period = np.array([1.0e-10, 2.0e-10])  # [s]
    error = offsets + slopes * t[:, np.newaxis]

    per_turn = phase_error_per_turn(t, error, period)

    np.testing.assert_allclose(per_turn, slopes * period, rtol=1e-12)


@pytest.mark.physics
@pytest.mark.parametrize("gamma", [1.0 + 5.0e-7, 2.0, 1.0e6], ids=["~1", "2", "1e6"])
def test_boris_and_higuera_cary_errors_per_turn_over_pitch_angles(gamma):
    # Independent truths, per exact turn of 100 steps (theta = 2 pi / 100), with
    # the pitch angles 10, 45, 54.7356 and 90 degrees in one run:
    # * Boris: (2 pi / theta)(2 arctan(theta / 2) - theta) at every pitch angle
    #   (Zenitani & Umeda 2018, Eq. 12); tolerance 1e-9 rad per turn (rounding).
    # * Higuera-Cary at gamma = 1e6: the ratio to the Boris lag is
    #   -(3 sin^2 psi - 2) / 2 to order theta^2 (series of rotation_angle_per_step,
    #   derived by hand); theta^2 = 4e-3, tolerance 5e-3 on the ratio.
    pitches = np.radians([10.0, 45.0, 54.7356, 90.0])
    speed = math.sqrt(gamma**2 - 1.0) * c
    u0 = np.stack(
        [np.zeros(4), -speed * np.sin(pitches), speed * np.cos(pitches)], axis=1
    )
    omega = e * BZ / (gamma * M)
    theta = 2.0 * math.pi / 100
    dt = theta / omega
    field = fields.uniform(b_field=B_ALONG_Z, e_field=[0.0, 0.0, 0.0])
    period = np.full(4, 2.0 * math.pi / omega)
    boris_per_turn = 100 * (2.0 * math.atan(theta / 2.0) - theta)

    per_turn = {}
    for push in (boris, higuera_cary):
        t, _, u = integrate(np.zeros((4, 3)), u0, field, push, dt, 1000, 10, Q, M)
        error = gyration_phase_error(t, u, u0, B_ALONG_Z, Q, M)
        per_turn[push] = phase_error_per_turn(t, error, period)

    np.testing.assert_allclose(per_turn[boris], boris_per_turn, rtol=0.0, atol=1e-9)
    if gamma == 1.0e6:
        ratio = per_turn[higuera_cary] / abs(boris_per_turn)
        expected = (3.0 * np.sin(pitches) ** 2 - 2.0) / 2.0
        np.testing.assert_allclose(ratio, expected, rtol=0.0, atol=5e-3)


# --- magnetic_moment ----------------------------------------------------------------


def test_magnetic_moment_of_a_hand_computed_state():
    # Independent truth: u = (3, 4, 12) 1e6 m/s in B = 2 T along z: u_perp^2 =
    # 25e12 m^2/s^2, so mu = m u_perp^2 / (2 B) = m 6.25e12. The same state with B
    # along x has u_perp^2 = (16 + 144) 1e12, mu = m 40e12. Tolerance: rounding.
    u = np.array([[[3.0e6, 4.0e6, 12.0e6], [3.0e6, 4.0e6, 12.0e6]]])
    b_field = np.array([[[0.0, 0.0, 2.0], [2.0, 0.0, 0.0]]])

    mu = magnetic_moment(u, b_field, M)

    np.testing.assert_allclose(mu[0], [M * 6.25e12, M * 40.0e12], rtol=1e-14)


@pytest.mark.parametrize(
    ("u", "b", "m", "message"),
    [
        (np.ones((2, 1, 3)), np.ones((2, 2, 3)), M, "same shape"),
        (np.ones((2, 1, 3)), np.zeros((2, 1, 3)), M, r"\|B\| = 0"),
        (np.ones((2, 1, 3)), np.ones((2, 1, 3)), 0.0, "strictly positive"),
    ],
    ids=["shapes_differ", "zero_field", "zero_mass"],
)
def test_magnetic_moment_refuses_wrong_input(u, b, m, message):
    with pytest.raises(ValueError, match=message):
        magnetic_moment(u, b, m)


def _bottle_run(push, epsilon=0.01, steps_per_turn=50):
    """One bounce in the magnetic bottle of Ripperda et al. (2018, Sec. 3.2.1).

    Electron of gamma = 100 and pitch angle 45 degrees, started on its gyration
    circle about the axis at z = 0; ``epsilon = r_L / L``. Returns
    ``(x, u, b_field, length, r_larmor)``, saved at every step.
    """
    gamma = 100.0
    u_perp = math.sqrt((gamma**2 - 1.0) / 2.0) * c
    r_larmor = M * u_perp / (e * BZ)
    length = r_larmor / epsilon
    field = fields.magnetic_mirror(b0=BZ, length=length)
    dt = 2.0 * math.pi * gamma * M / (e * BZ) / steps_per_turn
    # Adiabatic bounce, harmonic in this field: period 2 pi L / v_perp.
    bounce_period = 2.0 * math.pi * length / (u_perp / gamma)
    n_steps = round(bounce_period / dt)
    x0 = np.array([[r_larmor, 0.0, 0.0]])
    u0 = np.array([[0.0, u_perp, u_perp]])

    t, x, u = integrate(x0, u0, field, push, dt, n_steps, 1, Q, M)

    b_field = np.array([field(x[k], t[k])[1] for k in range(t.size)])
    return x, u, b_field, length, r_larmor


@pytest.mark.physics
@all_pushers
def test_particle_is_reflected_at_the_predicted_mirror_point(push):
    # Independent truth: with mu conserved, a particle of pitch angle 45 degrees is
    # reflected where |B| = 2 |B(x0)| (Ripperda et al. 2018, Eq. 61): z_max = L for
    # a start on the axis, slightly more here because |B(x0)| = B0 (1 + O(eps^2))
    # off the axis, which is included. The adiabatic theory is exact to order eps
    # = 1e-2; observed agreement 2.1e-7 of z_max, tolerance 1e-4.
    x, _, b_field, length, _ = _bottle_run(push)

    z_max = length * math.sqrt(2.0 * np.linalg.norm(b_field[0, 0]) / BZ - 1.0)
    assert np.max(x[:, 0, 2]) == pytest.approx(z_max, rel=1e-4)
    assert np.min(x[:, 0, 2]) == pytest.approx(-z_max, rel=1e-4)


@pytest.mark.physics
@all_pushers
def test_magnetic_moment_is_an_adiabatic_invariant_in_the_bottle(push):
    # Independent truth: mu is conserved to order eps = r_L / L = 1e-2 in a slowly
    # varying field. Computed with u^n as kept by ``integrate``, it varies by 7.4e-6
    # over a bounce (observed, far below eps: the guiding centre stays on the axis,
    # where the field seen along the circle is uniform to order eps^2). Bound
    # eps^2 / 10 = 1e-5 would be too tight for the scheme, eps / 100 = 1e-4 is used.
    _, u, b_field, _, _ = _bottle_run(push)

    mu = magnetic_moment(u, b_field, M)[:, 0]

    assert np.max(np.abs(mu / mu[0] - 1.0)) < 1e-4


# --- step_jacobian_determinant ------------------------------------------------------


def _vay_volume_factor(u_before, u_after, b_vector, dt):
    """Higuera & Cary (2017, Eqs. 40-41): Vay's determinant J(u_i) / J(u_f)."""
    beta = Q * dt / (2.0 * M) * b_vector

    def j_of(u):
        gamma = math.sqrt(1.0 + (u @ u) / c**2)
        return 1.0 + (beta @ beta + (beta @ u / c) ** 2) / gamma**4

    return j_of(u_before) / j_of(u_after)


@pytest.mark.physics
@all_pushers
@pytest.mark.parametrize("u_over_c", REGIMES)
def test_step_determinant_matches_the_paper_in_crossed_fields(push, u_over_c):
    # Independent truth: Higuera & Cary (2017, Sec. V): det = 1 for Boris and
    # Higuera-Cary, J(u_i) / J(u_f) for Vay. Uniform E and B not parallel, 0.6 rad
    # per step. Tolerance 1e-8: the differences have a truncation error of about
    # (1e-6)^2 and a rounding error of about 2e-10 per entry (observed <= 4.5e-10).
    b_vector = np.array([0.3, -0.2, BZ])
    field = fields.uniform(
        b_field=b_vector, e_field=0.01 * c * BZ * np.array([1.0, -2.0, 3.0])
    )
    direction = np.array([0.8, 0.1, 0.6]) / np.linalg.norm([0.8, 0.1, 0.6])
    u = u_over_c * c * direction[np.newaxis, :]
    dt = 0.6 * math.sqrt(1.0 + u_over_c**2) * M / (e * BZ)
    x = np.zeros((1, 3))

    measured = step_jacobian_determinant(
        push, field, np.zeros(1), x[np.newaxis], u[np.newaxis], dt, Q, M
    )

    expected = 1.0
    if push is vay:
        _, u_after = vay(x, u, field, 0.0, dt, Q, M)
        expected = _vay_volume_factor(u[0], u_after[0], b_vector, dt)
        # The test can tell Vay from a volume-preserving scheme: its factor differs
        # from 1 by 5e-5 to 3e-4 here (observed), the guard 1e-6 is 100 times the
        # tolerance below.
        assert abs(expected - 1.0) > 1e-6
    assert measured[0, 0] == pytest.approx(expected, abs=1e-8)


@pytest.mark.physics
@all_pushers
def test_every_scheme_preserves_the_volume_in_a_static_magnetic_bottle(push):
    # Independent truth: without electric field the rotation keeps |u| and u . B,
    # so Vay's factor J(u_i) / J(u_f) is exactly 1, and Boris and Higuera-Cary are
    # volume preserving in any field (Higuera & Cary 2017, Sec. V). The field
    # depends on x here, so the whole 6 x 6 Jacobian matters. Measured along the
    # first 200 steps of a bounce; tolerance 1e-8 per step, as above.
    x, u, _, length, _ = _bottle_run(push)
    field = fields.magnetic_mirror(b0=BZ, length=length)
    dt = 2.0 * math.pi * 100.0 * M / (e * BZ) / 50

    measured = step_jacobian_determinant(
        push, field, np.zeros(200), x[:200], u[:200], dt, Q, M
    )

    np.testing.assert_allclose(measured, 1.0, rtol=0.0, atol=1e-8)


def test_step_determinant_refuses_wrong_shapes():
    field = fields.uniform(b_field=[0.0, 0.0, BZ], e_field=[0.0, 0.0, 0.0])

    with pytest.raises(ValueError, match="shape"):
        step_jacobian_determinant(
            boris,
            field,
            np.zeros(2),
            np.zeros((3, 1, 3)),
            np.ones((3, 1, 3)),
            1e-12,
            Q,
            M,
        )
