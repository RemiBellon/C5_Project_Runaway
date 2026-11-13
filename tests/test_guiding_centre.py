import numpy as np
import pytest

from ppf_c5_runaways.guiding_centre import guiding_centre_from_position
from ppf_c5_runaways.particles import position_from_guiding_centre


def test_guiding_centre_is_inverse_of_particle_position_transform():
    centre = np.array([[1.0, 2.0, 3.0], [-4.0, 5.0, -6.0]])
    u = np.array([[2.0, 3.0, 5.0], [-1.0, 4.0, 2.0]])
    b_field = np.array([[0.0, 0.0, 2.5], [0.0, 0.0, 1.5]])
    q, m = -2.0, 3.0

    position = position_from_guiding_centre(centre, u, b_field, q, m)

    estimated = guiding_centre_from_position(position, u, b_field, q, m)

    np.testing.assert_allclose(estimated, centre, rtol=1e-15, atol=1e-15)


def test_guiding_centre_broadcasts_single_vectors():
    position = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    u = np.array([0.0, 1.0, 0.0])
    b_field = np.array([0.0, 0.0, 2.0])

    estimated = guiding_centre_from_position(position, u, b_field, -1.0, 1.0)

    assert estimated.shape == (2, 3)
    np.testing.assert_allclose(estimated, [[0.5, 2.0, 3.0], [3.5, 5.0, 6.0]])


def test_guiding_centre_accepts_time_particle_vector_arrays():
    centre = np.array(
        [
            [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]],
            [[-1.0, -2.0, -3.0], [-4.0, -5.0, -6.0]],
        ]
    )
    u = np.broadcast_to([0.0, 1.0, 0.0], centre.shape)
    b_field = np.broadcast_to([0.0, 0.0, 2.0], centre.shape)
    q, m = -1.0, 1.0
    b_squared = np.sum(b_field**2, axis=-1, keepdims=True)
    larmor_vector = m * np.cross(b_field, u) / (q * b_squared)
    position = centre + larmor_vector

    estimated = guiding_centre_from_position(position, u, b_field, q, m)

    assert estimated.shape == (2, 2, 3)
    np.testing.assert_allclose(estimated, centre, rtol=1e-15, atol=1e-15)


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"q": 0.0, "m": 1.0}, "q must be"),
        ({"q": -1.0, "m": 0.0}, "m must be"),
    ],
)
def test_guiding_centre_rejects_invalid_charge_or_mass(kwargs, message):
    with pytest.raises(ValueError, match=message):
        guiding_centre_from_position(
            [0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 1.0], **kwargs
        )


def test_guiding_centre_rejects_zero_magnetic_field():
    with pytest.raises(ValueError, match="b_field must not be the zero vector"):
        guiding_centre_from_position(
            [0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0], -1.0, 1.0
        )


def test_guiding_centre_rejects_incompatible_particle_counts():
    with pytest.raises(ValueError, match="broadcastable leading dimensions"):
        guiding_centre_from_position(
            [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]],
            [[0.0, 0.0, 0.0], [0.0, 1.0, 0.0], [1.0, 0.0, 0.0]],
            [0.0, 0.0, 1.0],
            -1.0,
            1.0,
        )
