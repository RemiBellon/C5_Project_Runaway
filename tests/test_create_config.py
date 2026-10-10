"""Tests of ``ppf_c5_runaways.create_config`` (``new_config``).

The reference is always a ``RunConfig`` written out by hand, value by value, so the
tests check what ends up in the file, not how ``new_config`` builds it.
"""

import numpy as np
import pydantic
import pytest

from ppf_c5_runaways import run
from ppf_c5_runaways.config import RunConfig, load_config
from ppf_c5_runaways.create_config import new_config
from ppf_c5_runaways.io import CONFIG_FILENAME

UNIFORM_B = {"kind": "uniform_cartesian", "b_field": [0, 0, 2.5], "e_field": [0, 0, 0]}


def _new(tmp_path, **change):
    """``new_config`` with the smallest valid set of arguments, written in tmp_path."""
    arguments = {
        "name": "gyration",
        "field": UNIFORM_B,
        "u0": [4.0e8, 0.0, 3.0e8],
        "dt": 1.0e-13,
        "n_steps": 100,
        "directory": tmp_path,
        **change,
    }
    return new_config(**arguments)


def test_file_holds_exactly_the_values_given_and_the_electron_defaults(tmp_path):
    # Independent truth: the configuration written by hand. The defaults must be an
    # electron (CODATA 2018 values, written here as literals), at the origin, with
    # every step kept, the boris pusher and the seed 42.
    expected = RunConfig(
        name="gyration",
        pusher="boris",
        dt=1.0e-13,
        n_steps=100,
        save_every=1,
        seed=42,
        q=-1.602176634e-19,
        m=9.1093837139e-31,
        field=UNIFORM_B,
        initial={"x0": [[0.0, 0.0, 0.0]], "u0": [[4.0e8, 0.0, 3.0e8]]},
    )

    path = _new(tmp_path)

    assert path == tmp_path / "gyration_boris.yaml"
    assert load_config(path) == expected


def test_one_vector_is_one_particle_and_a_table_is_one_row_per_particle(tmp_path):
    one = load_config(_new(tmp_path, name="one", u0=[1.0e7, 0, 0]))
    three = load_config(
        _new(tmp_path, name="three", u0=[[1.0e7, 0, 0], [0, 1.0e7, 0], [0, 0, 1.0e7]])
    )

    assert one.initial.u0 == ((1.0e7, 0.0, 0.0),)
    assert three.initial.x0 == ((0.0, 0.0, 0.0),) * 3  # one origin per particle


def test_numpy_arrays_are_accepted_for_positions_and_velocities(tmp_path):
    u0 = np.array([[4.0e8, 0.0, 3.0e8], [1.0e7, 2.0e7, 0.0]])
    x0 = np.arange(6, dtype=np.int64).reshape(2, 3)  # numpy integers, not floats

    config = load_config(_new(tmp_path, u0=u0, x0=x0))

    assert config.initial.u0 == ((4.0e8, 0.0, 3.0e8), (1.0e7, 2.0e7, 0.0))
    assert config.initial.x0 == ((0.0, 1.0, 2.0), (3.0, 4.0, 5.0))


def test_one_file_per_pusher_for_the_same_case(tmp_path):
    # The same name with several pushers must give several files, so that one
    # physical case can be run with each pusher and compared in one results folder.
    paths = {_new(tmp_path, pusher=name) for name in ("boris", "vay", "higuera_cary")}

    assert paths == {
        tmp_path / "gyration_boris.yaml",
        tmp_path / "gyration_vay.yaml",
        tmp_path / "gyration_higuera_cary.yaml",
    }


def test_missing_directory_is_created(tmp_path):
    path = _new(tmp_path, directory=tmp_path / "configs" / "scan")

    assert path.is_file()


@pytest.mark.parametrize(
    ("change", "message"),
    [
        ({"field": {**UNIFORM_B, "kind": "uniform"}}, "uniform_cartesian"),
        ({"field": {**UNIFORM_B, "b_feild": [0, 0, 1]}}, "b_feild"),
        ({"pusher": "leapfrog"}, "pusher"),
        ({"save_every": 3}, "must divide"),
        ({"name": "../outside"}, "name"),
        ({"u0": [True, 0.0, 0.0]}, "u0"),
        ({"u0": np.array([True, False, False])}, "u0"),
        ({"u0": [[1.0e7, 0, 0], [1.0e7, 0]]}, "u0"),
        ({"u0": [[1.0e7, 0, 0]], "x0": [[0, 0, 0], [1, 0, 0]]}, "u0"),
    ],
    ids=[
        "old_kind",
        "misspelled_key",
        "unknown_pusher",
        "save_every",
        "unsafe_name",
        "boolean_in_list",
        "boolean_array",
        "row_of_2_numbers",
        "x0_and_u0_of_different_lengths",
    ],
)
def test_invalid_configuration_is_refused_and_nothing_is_written(
    tmp_path, change, message
):
    with pytest.raises(pydantic.ValidationError, match=message):
        _new(tmp_path, **change)

    assert list(tmp_path.iterdir()) == []


def test_existing_file_is_never_overwritten(tmp_path):
    path = _new(tmp_path)
    before = path.read_text()

    with pytest.raises(FileExistsError):
        _new(tmp_path, dt=2.0e-13)

    assert path.read_text() == before


def test_the_file_runs_as_is(tmp_path):
    # End to end: what new_config writes is what run.pusher expects.
    path = _new(tmp_path, directory=tmp_path / "configs")

    result_dir = run.pusher(path, base_dir=tmp_path / "results")

    assert load_config(result_dir / CONFIG_FILENAME) == load_config(path)
