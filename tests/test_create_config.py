import numpy as np
import pytest

from ppf_c5_runaways.config import InitialState
from ppf_c5_runaways.constants import MeV_inJ, c, m_e
from ppf_c5_runaways.create_config import new_config, u_from_theta


def test_u_from_theta_calculates_cartesian_velocity_for_each_particle():
    x0 = [[1.0, 7.2, 0.0], [1.0, 7.2, 0.0]]

    u0 = u_from_theta(
        energy=[40.0, 40.0],
        theta=[57.0, 80.0],
        m=m_e,
        x0=x0,
        alpha=0.0,
    )

    assert np.asarray(u0).shape == (2, 3)
    InitialState(x0=x0, u0=u0)

    expected_magnitude = c * np.sqrt(
        (40.0 * MeV_inJ / (m_e * c**2)) ** 2 - 1
    )
    np.testing.assert_allclose(np.linalg.norm(u0, axis=1), expected_magnitude)
    assert not np.allclose(u0[0], u0[1])


def test_u_from_theta_broadcasts_scalar_inputs_across_positions():
    x0 = [[1.0, 7.2, 0.0], [1.0, 7.2, 0.0]]

    u0 = u_from_theta(energy=40.0, theta=57.0, x0=x0)

    assert np.asarray(u0).shape == (2, 3)
    np.testing.assert_allclose(u0[0], u0[1])


def test_u_from_theta_requires_one_list_value_per_particle():
    with pytest.raises(ValueError, match="theta must be a scalar or have one value"):
        u_from_theta(
            energy=[40.0, 40.0],
            theta=[57.0],
            x0=[[1.0, 7.2, 0.0], [1.0, 7.2, 0.0]],
        )


def test_new_config_writes_block_field_and_flow_style_particle_rows(
    tmp_path, monkeypatch
):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "configs").mkdir()

    new_config(
        name="tokamak_format",
        b_field=[
            5.3,
            6.2,
            4,
            1.6,
            "0",
            "b_theta_expr",
            "b_phi_expr",
        ],
        x0=[[1, 7.2, 0], [2, 8.2, 1]],
        u0=[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]],
    )

    yaml_text = (tmp_path / "configs" / "tokamak_format.yaml").read_text()

    assert "field:\n  kind: tokamak\n" in yaml_text
    assert "field: {" not in yaml_text
    assert "  x0: [[1, 7.2, 0], [2, 8.2, 1]]" in yaml_text
    assert "  u0: [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]" in yaml_text
    assert "\n    - " not in yaml_text
