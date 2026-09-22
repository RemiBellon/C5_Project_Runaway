"""Tests of ``ppf_c5_runaways.fields`` (``uniform``)."""

import numpy as np
import pytest

from ppf_c5_runaways.fields import uniform

B = [0.0, 0.0, 2.5]  # T
E = [1.0e3, -2.0e3, 3.0e3]  # V/m


# --- values and shapes ---------------------------------------------------------


@pytest.mark.parametrize("n_part", [1, 2, 1000])
def test_field_gives_the_same_vectors_for_every_particle(n_part):
    # Independent truth: the input vectors themselves, repeated n_part times with
    # plain Python (not with the np.tile used by the code). Exact equality: nothing
    # is computed, values are only copied.
    field = uniform(b_field=B, e_field=E)

    e_out, b_out = field(np.zeros((n_part, 3)), 0.0)

    np.testing.assert_array_equal(e_out, np.array([E] * n_part))
    np.testing.assert_array_equal(b_out, np.array([B] * n_part))


@pytest.mark.parametrize("n_part", [1, 7])
def test_outputs_are_float64_c_ordered_arrays_of_shape_n_part_3(n_part):
    e_out, b_out = uniform(b_field=B, e_field=E)(np.zeros((n_part, 3)), 0.0)

    for array in (e_out, b_out):
        assert array.shape == (n_part, 3)
        assert array.dtype == np.float64
        assert array.flags["C_CONTIGUOUS"]


def test_no_particle_gives_empty_outputs():
    e_out, b_out = uniform(b_field=B, e_field=E)(np.zeros((0, 3)), 0.0)

    assert e_out.shape == b_out.shape == (0, 3)


def test_field_does_not_depend_on_position_or_time():
    field = uniform(b_field=B, e_field=E)
    positions = np.random.default_rng(0).normal(scale=1.0e3, size=(5, 3))
    reference = field(np.zeros((5, 3)), 0.0)

    for t in (0.0, 1.0e-9, -3.0, 1.0e6):
        e_out, b_out = field(positions, t)

        np.testing.assert_array_equal(e_out, reference[0])
        np.testing.assert_array_equal(b_out, reference[1])


def test_field_does_not_modify_the_positions():
    positions = np.arange(6.0).reshape(2, 3)
    before = positions.copy()

    uniform(b_field=B, e_field=E)(positions, 0.0)

    np.testing.assert_array_equal(positions, before)


# --- accepted inputs -----------------------------------------------------------


@pytest.mark.parametrize(
    "as_type",
    [list, tuple, np.array, lambda v: np.array(v, dtype=np.float32)],
    ids=["list", "tuple", "ndarray", "float32"],
)
def test_vectors_can_be_given_as_list_tuple_or_array(as_type):
    # The values chosen are exactly representable in float32, so the comparison stays
    # exact for every input type.
    b_exact, e_exact = [0.0, 0.5, 2.0], [0.25, 0.0, -4.0]

    e_out, b_out = uniform(b_field=as_type(b_exact), e_field=as_type(e_exact))(
        np.zeros((1, 3)), 0.0
    )

    np.testing.assert_array_equal(b_out[0], b_exact)
    np.testing.assert_array_equal(e_out[0], e_exact)


def test_integers_are_accepted_and_stored_as_floats():
    e_out, b_out = uniform(b_field=[0, 0, 2], e_field=[0, 0, 0])(np.zeros((1, 3)), 0.0)

    assert b_out.dtype == e_out.dtype == np.float64
    assert b_out[0, 2] == 2.0


# --- independence: nothing is shared between the field, its inputs and its outputs


def test_changing_the_inputs_after_creation_does_not_change_the_field():
    b_list, e_array = [0.0, 0.0, 2.5], np.array([1.0, 2.0, 3.0])
    field = uniform(b_field=b_list, e_field=e_array)

    b_list[2] = 99.0
    e_array[:] = 99.0
    e_out, b_out = field(np.zeros((1, 3)), 0.0)

    np.testing.assert_array_equal(b_out[0], [0.0, 0.0, 2.5])
    np.testing.assert_array_equal(e_out[0], [1.0, 2.0, 3.0])


def test_changing_the_outputs_does_not_change_the_next_call():
    field = uniform(b_field=B, e_field=E)
    positions = np.zeros((2, 3))

    e_first, b_first = field(positions, 0.0)
    e_first[:] = 99.0
    b_first[:] = 99.0
    e_second, b_second = field(positions, 0.0)

    np.testing.assert_array_equal(e_second, np.array([E] * 2))
    np.testing.assert_array_equal(b_second, np.array([B] * 2))
    assert not np.shares_memory(e_second, b_second)


# --- errors at creation --------------------------------------------------------


@pytest.mark.parametrize("name", ["b_field", "e_field"])
@pytest.mark.parametrize(
    "bad",
    [
        [0.0, 0.0],  # 2 components
        [0.0, 0.0, 1.0, 1.0],  # 4 components
        2.5,  # a scalar, not a vector
        [[0.0, 0.0, 1.0]],  # shape (1, 3): a list of vectors, not a vector
        [0.0, [0.0, 1.0], 1.0],  # ragged
        ["0", "0", "1"],  # strings are not converted silently
        [True, False, True],  # booleans are not numbers here
        [0.0, 0.0, None],
        [0.0, 0.0, 1.0 + 1.0j],
        None,
    ],
)
def test_wrong_vector_is_refused_with_a_message_naming_it(name, bad):
    other = "e_field" if name == "b_field" else "b_field"
    kwargs = {name: bad, other: [0.0, 0.0, 1.0]}

    with pytest.raises(ValueError, match=rf"{name} must be exactly 3 numbers"):
        uniform(**kwargs)


@pytest.mark.parametrize("name", ["b_field", "e_field"])
@pytest.mark.parametrize("bad_value", [float("nan"), float("inf"), float("-inf")])
def test_non_finite_vector_is_refused(name, bad_value):
    other = "e_field" if name == "b_field" else "b_field"
    kwargs = {name: [0.0, 0.0, bad_value], other: [0.0, 0.0, 1.0]}

    with pytest.raises(ValueError, match=rf"{name} must be finite"):
        uniform(**kwargs)


def test_vectors_must_be_given_by_name():
    with pytest.raises(TypeError):
        uniform(B, E)  # type: ignore[misc]


def test_both_vectors_are_required():
    with pytest.raises(TypeError, match="e_field"):
        uniform(b_field=B)  # type: ignore[call-arg]


# --- errors at call ------------------------------------------------------------


@pytest.mark.parametrize(
    "bad_x",
    [
        np.zeros(3),  # one position instead of a list of positions: the classic
        np.zeros((4, 2)),
        np.zeros((4, 3, 1)),
        np.float64(1.0),
    ],
    ids=["single_position", "two_columns", "three_axes", "scalar"],
)
def test_positions_of_wrong_shape_are_refused_with_a_hint(bad_x):
    field = uniform(b_field=B, e_field=E)

    with pytest.raises(ValueError, match=r"shape \(n_part, 3\)") as excinfo:
        field(bad_x, 0.0)

    assert "(1, 3)" in str(excinfo.value)  # the message tells how to fix it
