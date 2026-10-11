"""Tests of ``ppf_c5_runaways.run`` (``pusher``)."""

import math

import numpy as np
import pytest

from ppf_c5_runaways import run
from ppf_c5_runaways.config import RunConfig, load_config, save_config
from ppf_c5_runaways.constants import c, e, m_e
from ppf_c5_runaways.io import (
    CONFIG_FILENAME,
    DATA_FILENAME,
    list_results,
    read_result,
)

BZ = 2.5  # [T]
GAMMA = 2.0
OMEGA = e * BZ / (m_e * GAMMA)  # gyration frequency of the electron below [rad/s]
SPEED = math.sqrt(3.0) * c  # |u| for gamma = 2


def _config(**change) -> RunConfig:
    """One relativistic electron in a uniform B, doing one gyration in 200 steps."""
    fields = {
        "name": "uniform_b",
        "pusher": "boris",
        "dt": 2.0 * math.pi / OMEGA / 200,
        "n_steps": 200,
        "save_every": 10,
        "seed": 42,
        "q": -e,
        "m": m_e,
        "field": {
            "kind": "uniform_cartesian",
            "b_field": [0, 0, BZ],
            "e_field": [0, 0, 0],
        },
        "initial": {"x0": [[0, 0, 0]], "u0": [[0.8 * SPEED, 0, 0.6 * SPEED]]},
        **change,
    }
    return RunConfig(**fields)


def _write_yaml(directory, config: RunConfig, filename="run.yaml"):
    path = directory / filename
    save_config(config, path)
    return path


def test_run_creates_a_folder_with_the_config_and_the_trajectories(tmp_path):
    # Independent truth: the exact orbit. After exactly one gyro-period the electron
    # is back at x = y = 0 and has moved by u_z * T / gamma along z. Tolerance:
    # theta^2 * r_L with theta = 2 pi / 200 (see the same test in test_integrate).
    config = _config()
    yaml_path = _write_yaml(tmp_path, config)

    result_dir = run.pusher(yaml_path, base_dir=tmp_path / "results")

    assert result_dir.parent == tmp_path / "results"
    assert load_config(result_dir / CONFIG_FILENAME) == config  # the case is recorded
    assert (result_dir / DATA_FILENAME).is_file()
    result = read_result(result_dir, "boris")
    assert result["x"].shape == (21, 1, 3)
    x_end = result["x"].values[-1, 0]
    r_larmor = m_e * 0.8 * SPEED / (e * BZ)
    theta = OMEGA * config.dt
    assert np.hypot(x_end[0], x_end[1]) < theta**2 * r_larmor
    assert x_end[2] == pytest.approx(
        0.6 * SPEED * (2 * math.pi / OMEGA) / GAMMA, rel=1e-12
    )


def test_run_refuses_a_different_case_in_an_existing_folder(tmp_path):
    result_dir = run.pusher(_write_yaml(tmp_path, _config()), base_dir=tmp_path)
    other = _write_yaml(
        tmp_path, _config(n_steps=400, save_every=10), filename="other.yaml"
    )

    with pytest.raises(ValueError, match=r"different: \['n_steps'\]"):
        run.pusher(other, result_dir=result_dir)

    assert (result_dir / DATA_FILENAME).is_file()  # the existing result is untouched


def test_run_refuses_to_run_boris_twice_in_the_same_folder(tmp_path):
    yaml_path = _write_yaml(tmp_path, _config())
    result_dir = run.pusher(yaml_path, base_dir=tmp_path)

    with pytest.raises(FileExistsError, match="never overwritten"):
        run.pusher(yaml_path, result_dir=result_dir)


def test_run_says_clearly_that_a_pusher_is_not_available_yet(tmp_path):
    yaml_path = _write_yaml(tmp_path, _config(pusher="rk4"))

    with pytest.raises(
        ValueError,
        match=r"'rk4' is not implemented yet.*'boris', 'higuera_cary', 'vay'",
    ):
        run.pusher(yaml_path, base_dir=tmp_path / "results")

    assert not (tmp_path / "results").exists()  # nothing was created


@pytest.mark.parametrize("pusher", ["boris", "higuera_cary", "vay"])
def test_run_each_pusher_stores_its_own_orbit(tmp_path, pusher):
    # Independent truth: the exact orbit, as in the first test. After exactly one
    # gyro-period the electron is back at x = y = 0 and has moved by u_z * T / gamma
    # along z. Tolerance: theta^2 * r_L with theta = 2 pi / 200. It holds for the
    # three schemes, which are all second order (the angle of Vay, Higuera-Cary and
    # Boris differ from theta by terms of order theta^3, less than theta^2).
    config = _config(pusher=pusher)

    result_dir = run.pusher(_write_yaml(tmp_path, config), base_dir=tmp_path)

    result = read_result(result_dir, pusher)
    assert result.attrs["pusher"] == pusher
    x_end = result["x"].values[-1, 0]
    r_larmor = m_e * 0.8 * SPEED / (e * BZ)
    theta = OMEGA * config.dt
    assert np.hypot(x_end[0], x_end[1]) < theta**2 * r_larmor
    assert x_end[2] == pytest.approx(
        0.6 * SPEED * (2 * math.pi / OMEGA) / GAMMA, rel=1e-12
    )


@pytest.mark.physics
@pytest.mark.parametrize("pusher", ["boris", "higuera_cary", "vay"])
def test_run_stores_the_velocity_at_the_times_of_the_positions(tmp_path, pusher):
    # Independent truth: the exact velocity of the gyrating electron,
    # u_x = u_perp cos(omega t), u_y = u_perp sin(omega t) (an electron turns
    # counter-clockwise about +z), u_z constant. The stored u is the velocity at the
    # times of x: its error is of order theta^2 (observed 1.2e-4 to 4.1e-4 of |u|),
    # below the tolerance theta^2 |u| = 1e-3 |u| for theta = 2 pi / 200. A velocity stored half a step
    # early would be off by about theta / 2 * u_perp = 1.3e-2 |u|: checking that the
    # exact velocity at t - dt / 2 fails the same tolerance proves that the test can
    # tell the two apart.
    config = _config(pusher=pusher)

    result_dir = run.pusher(_write_yaml(tmp_path, config), base_dir=tmp_path)

    result = read_result(result_dir, pusher)

    def exact_velocity(time):
        u_exact = np.zeros((time.size, 3))
        u_exact[:, 0] = 0.8 * SPEED * np.cos(OMEGA * time)
        u_exact[:, 1] = 0.8 * SPEED * np.sin(OMEGA * time)
        u_exact[:, 2] = 0.6 * SPEED
        return u_exact

    time = result["time"].values
    u_stored = result["u"].values[:, 0]
    tolerance = (OMEGA * config.dt) ** 2 * SPEED
    error = np.linalg.norm(u_stored - exact_velocity(time), axis=-1)
    error_if_half_a_step_early = np.linalg.norm(
        u_stored - exact_velocity(time - config.dt / 2.0), axis=-1
    )
    assert result["u"].dims == result["x"].dims == ("time", "particle", "axis")
    assert np.max(error) < tolerance
    assert np.max(error_if_half_a_step_early[1:]) > 10.0 * tolerance


def test_run_puts_the_three_pushers_in_the_same_folder_to_compare_them(tmp_path):
    # The workflow of ``main.py``: one folder, one case, one run per pusher.
    boris_yaml = _write_yaml(tmp_path, _config(pusher="boris"), "boris.yaml")
    hc_yaml = _write_yaml(
        tmp_path, _config(pusher="higuera_cary", name="uniform_b_hc"), "hc.yaml"
    )
    vay_yaml = _write_yaml(tmp_path, _config(pusher="vay"), "vay.yaml")

    result_dir = run.pusher(boris_yaml, base_dir=tmp_path / "results")
    assert run.pusher(hc_yaml, result_dir=result_dir) == result_dir
    assert run.pusher(vay_yaml, result_dir=result_dir) == result_dir

    assert list_results(result_dir) == ["boris", "higuera_cary", "vay"]
    # Same case, same times: the three curves can be drawn on the same axes.
    times = [
        read_result(result_dir, name)["time"].values
        for name in list_results(result_dir)
    ]
    np.testing.assert_array_equal(times[0], times[1])
    np.testing.assert_array_equal(times[0], times[2])


def test_run_stores_the_magnetic_field_felt_along_a_mirror_orbit(tmp_path):
    # Independent truth: Ripperda et al. (2018, Eq. 59) evaluated at the stored
    # positions, written out here: B = B0 (-x z / L^2, -y z / L^2, 1 + z^2 / L^2).
    # Tolerance: rounding of the formula (1e-14 of B0).
    b0, length = 2.5, 5.0
    config = _config(
        name="bottle",
        field={"kind": "mirror", "b0": b0, "length": length},
        initial={"x0": [[0.05, 0, 0]], "u0": [[0, 0.8 * SPEED, 0.6 * SPEED]]},
    )

    result_dir = run.pusher(_write_yaml(tmp_path, config), base_dir=tmp_path)

    result = read_result(result_dir, "boris")
    x = result["x"].values[:, 0]
    expected = np.empty_like(x)
    expected[:, 0] = -b0 * x[:, 0] * x[:, 2] / length**2
    expected[:, 1] = -b0 * x[:, 1] * x[:, 2] / length**2
    expected[:, 2] = b0 * (1.0 + x[:, 2] ** 2 / length**2)
    np.testing.assert_allclose(
        result["b_field"].values[:, 0], expected, rtol=0.0, atol=1e-14 * b0
    )


def test_run_measures_the_phase_space_volume_only_when_asked(tmp_path):
    # Truth: Boris preserves the phase-space volume exactly (Higuera & Cary 2017,
    # Sec. V), so every stored step determinant is 1 to the finite-difference
    # accuracy (1e-8, see test_diagnostics). Without the option nothing is stored.
    config = _config(save_every=1)
    yaml_path = _write_yaml(tmp_path, config)

    with_volume = run.pusher(
        yaml_path, base_dir=tmp_path / "with", phase_space_volume=True
    )
    without = run.pusher(yaml_path, base_dir=tmp_path / "without")

    determinants = read_result(with_volume, "boris")["step_jacobian_determinant"]
    np.testing.assert_allclose(determinants.values, 1.0, rtol=0.0, atol=1e-8)
    assert "step_jacobian_determinant" not in read_result(without, "boris")


def test_run_refuses_the_volume_measure_when_steps_are_skipped(tmp_path):
    yaml_path = _write_yaml(tmp_path, _config(save_every=10))

    with pytest.raises(ValueError, match="save_every: 1"):
        run.pusher(yaml_path, base_dir=tmp_path / "results", phase_space_volume=True)

    assert not (tmp_path / "results").exists()  # checked before any computation
