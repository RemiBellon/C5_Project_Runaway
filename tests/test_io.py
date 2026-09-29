"""Tests of ``ppf_c5_runaways.io``."""

import json
import re
import sys
import tomllib
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pytest
from pydantic import ValidationError

import ppf_c5_runaways.io as results_io
from ppf_c5_runaways.config import RunConfig, load_config
from ppf_c5_runaways.io import (
    CONFIG_FILENAME,
    DATA_FILENAME,
    METADATA_FILENAME,
    create_result_dir,
    list_results,
    read_result,
    write_result,
)

# A fixed moment, with a non-trivial UTC offset (+02:00) so that a bug mixing local
# time and UTC would show up.
_NOW = datetime(2026, 9, 20, 22, 30, 5, tzinfo=timezone(timedelta(hours=2)))


def _config() -> RunConfig:
    """Return a valid configuration."""
    return RunConfig(
        name="uniform_b",
        pusher="boris",
        dt=1.0e-11,
        n_steps=1000,
        save_every=10,
        seed=42,
        q=-1.602176634e-19,
        m=9.1093837139e-31,
        field={"kind": "uniform", "b_field": [0, 0, 2.5], "e_field": [0, 0, 0]},
        initial={"x0": [[0, 0, 0]], "u0": [[1.0e7, 0, 1.0e6]]},
    )


def test_folder_name_is_name_then_timestamp(tmp_path):
    # Independent truth: the name written by hand for the fixed moment _NOW, in the
    # format aaaammjj_hhmmss (2026-09-20 22:30:05 -> 20260920_223005).
    run_dir = create_result_dir(tmp_path, _config(), now=_NOW)

    assert run_dir == tmp_path / "uniform_b_20260920_223005"
    assert run_dir.is_dir()


def test_folder_contains_exactly_the_config_and_the_metadata(tmp_path):
    run_dir = create_result_dir(tmp_path, _config(), now=_NOW)

    assert sorted(path.name for path in run_dir.iterdir()) == [
        CONFIG_FILENAME,
        METADATA_FILENAME,
    ]


def test_saved_config_is_read_back_equal(tmp_path):
    config = _config()

    run_dir = create_result_dir(tmp_path, config, now=_NOW)

    assert load_config(run_dir / CONFIG_FILENAME) == config


def test_metadata_content(tmp_path):
    # Independent truth: values written by hand for the fixed moment _NOW and for
    # the config of ``_config``; versions come from the interpreter, NumPy and
    # ``pyproject.toml`` rather than from the function under test.
    pyproject = Path(__file__).parents[1] / "pyproject.toml"
    package_version = tomllib.loads(pyproject.read_text("utf-8"))["project"]["version"]

    run_dir = create_result_dir(tmp_path, _config(), now=_NOW)
    metadata = json.loads((run_dir / METADATA_FILENAME).read_text("utf-8"))

    assert metadata == {
        "schema_version": 1,
        "run_id": "uniform_b_20260920_223005",
        "name": "uniform_b",
        "created_at": "2026-09-20T22:30:05+02:00",
        "seed": 42,
        "versions": {
            "ppf_c5_runaways": package_version,
            "python": ".".join(str(part) for part in sys.version_info[:3]),
            "numpy": np.__version__,
        },
    }


def test_sub_second_precision_is_dropped(tmp_path):
    # The name and created_at must tell the same second, whatever the microseconds.
    moment = _NOW.replace(microsecond=999_999)

    run_dir = create_result_dir(tmp_path, _config(), now=moment)
    metadata = json.loads((run_dir / METADATA_FILENAME).read_text("utf-8"))

    assert run_dir.name == "uniform_b_20260920_223005"
    assert metadata["created_at"] == "2026-09-20T22:30:05+02:00"


def test_default_moment_is_the_current_local_time(tmp_path):
    # Bounds taken around the call: the timestamp of the folder must fall between.
    before = datetime.now().replace(microsecond=0)

    run_dir = create_result_dir(tmp_path, _config())

    after = datetime.now()
    assert re.fullmatch(r"uniform_b_\d{8}_\d{6}", run_dir.name)
    stamp = datetime.strptime(run_dir.name.removeprefix("uniform_b_"), "%Y%m%d_%H%M%S")
    assert before <= stamp <= after
    metadata = json.loads((run_dir / METADATA_FILENAME).read_text("utf-8"))
    assert datetime.fromisoformat(metadata["created_at"]).utcoffset() is not None


def test_missing_base_directory_is_created_with_its_parents(tmp_path):
    base = tmp_path / "results" / "batch_1"

    run_dir = create_result_dir(base, _config(), now=_NOW)

    assert run_dir.parent == base
    assert run_dir.is_dir()


def test_two_runs_in_different_seconds_get_different_folders(tmp_path):
    first = create_result_dir(tmp_path, _config(), now=_NOW)
    second = create_result_dir(tmp_path, _config(), now=_NOW + timedelta(seconds=1))

    assert first != second
    assert first.is_dir()
    assert second.is_dir()


def test_existing_folder_is_never_reused_or_overwritten(tmp_path):
    first = create_result_dir(tmp_path, _config(), now=_NOW)
    metadata_before = (first / METADATA_FILENAME).read_text("utf-8")

    with pytest.raises(FileExistsError, match="uniform_b_20260920_223005"):
        create_result_dir(tmp_path, _config(), now=_NOW)  # same name, same second

    assert (first / METADATA_FILENAME).read_text("utf-8") == metadata_before


def test_naive_moment_is_rejected_and_nothing_is_created(tmp_path):
    base = tmp_path / "results"

    with pytest.raises(ValueError, match="timezone-aware"):
        create_result_dir(base, _config(), now=datetime(2026, 9, 20, 22, 30, 5))

    assert not base.exists()


@pytest.mark.parametrize("name", ["../escape", "a/b", ".hidden"])
def test_unsafe_name_is_rejected_before_touching_the_disk(tmp_path, name):
    # ``model_copy(update=...)`` skips validation: the way to hold such a config.
    # The check must happen before any directory is created, otherwise ``../escape``
    # would create a folder outside ``base``.
    unsafe = _config().model_copy(update={"name": name})
    base = tmp_path / "results"

    with pytest.raises(ValidationError, match="name"):
        create_result_dir(base, unsafe, now=_NOW)

    assert list(tmp_path.iterdir()) == []  # not even ``base`` was created


def test_folder_is_removed_when_a_step_fails(tmp_path, monkeypatch):
    # A folder that exists must be complete: if writing fails half-way, the folder
    # created just before is removed (the base directory is kept).
    def failing_save_config(config, path):
        raise OSError("disk full")

    monkeypatch.setattr(results_io, "save_config", failing_save_config)
    base = tmp_path / "results"

    with pytest.raises(OSError, match="disk full"):
        create_result_dir(base, _config(), now=_NOW)

    assert base.is_dir()
    assert list(base.iterdir()) == []


# --- trajectories: write_result / read_result -----------------------------------


def _trajectories(seed: int = 0):
    """Made-up trajectories: 4 saved states of 2 particles, with time step 1e-12 s."""
    rng = np.random.default_rng(seed)
    t = np.array([0.0, 2.0e-12, 4.0e-12, 6.0e-12])
    return t, rng.normal(size=(4, 2, 3)), rng.normal(size=(4, 2, 3)), 1.0e-12


def test_written_trajectories_are_read_back_exactly(tmp_path):
    # Independent truth: the arrays given to write_result. Exact equality, since
    # netCDF stores float64 without loss. The time of u is the time of x minus dt/2.
    run_dir = create_result_dir(tmp_path, _config(), now=_NOW)
    t, x, u, dt = _trajectories()

    write_result(run_dir, "boris", t, x, u, dt)
    result = read_result(run_dir, "boris")

    np.testing.assert_array_equal(result["x"].values, x)
    np.testing.assert_array_equal(result["u"].values, u)
    np.testing.assert_array_equal(result["time"].values, t)
    np.testing.assert_allclose(result["time_half"].values, t - 0.5e-12, rtol=1e-15)
    assert result["x"].dims == ("time", "particle", "axis")
    assert result.attrs["pusher"] == "boris"
    assert result.attrs["dt"] == dt


def test_a_second_pusher_is_added_to_the_same_folder_without_touching_the_first(
    tmp_path,
):
    # The reason for one group per pusher: run Boris today, add Vay tomorrow.
    run_dir = create_result_dir(tmp_path, _config(), now=_NOW)
    t, x_boris, u_boris, dt = _trajectories(seed=1)
    _, x_vay, u_vay, _ = _trajectories(seed=2)

    assert list_results(run_dir) == []  # nothing written yet
    write_result(run_dir, "boris", t, x_boris, u_boris, dt)
    write_result(run_dir, "vay", t, x_vay, u_vay, dt)

    assert list_results(run_dir) == ["boris", "vay"]
    np.testing.assert_array_equal(read_result(run_dir, "boris")["x"].values, x_boris)
    np.testing.assert_array_equal(read_result(run_dir, "vay")["x"].values, x_vay)


def test_an_existing_result_is_never_overwritten(tmp_path):
    run_dir = create_result_dir(tmp_path, _config(), now=_NOW)
    t, x, u, dt = _trajectories(seed=1)
    write_result(run_dir, "boris", t, x, u, dt)

    with pytest.raises(FileExistsError, match="never overwritten"):
        write_result(run_dir, "boris", t, x + 1.0, u, dt)

    np.testing.assert_array_equal(read_result(run_dir, "boris")["x"].values, x)


def test_reading_a_missing_pusher_lists_the_available_ones(tmp_path):
    run_dir = create_result_dir(tmp_path, _config(), now=_NOW)
    write_result(run_dir, "boris", *_trajectories())

    with pytest.raises(ValueError, match=r"'vay'.*Available: \['boris'\]"):
        read_result(run_dir, "vay")


def test_reading_a_folder_without_data_says_so(tmp_path):
    run_dir = create_result_dir(tmp_path, _config(), now=_NOW)

    with pytest.raises(FileNotFoundError, match=DATA_FILENAME):
        read_result(run_dir, "boris")


@pytest.mark.parametrize(
    ("change", "message"),
    [
        ({"pusher_name": "../boris"}, "pusher_name"),
        ({"dt": 0.0}, "dt must be"),
        ({"x": np.zeros((4, 3))}, "x must have shape"),
        ({"u": np.zeros((4, 1, 3))}, "u must have the same shape"),
    ],
)
def test_wrong_arguments_are_refused_and_nothing_is_written(tmp_path, change, message):
    run_dir = create_result_dir(tmp_path, _config(), now=_NOW)
    t, x, u, dt = _trajectories()
    arguments = {
        "pusher_name": "boris",
        "t": t,
        "x": x,
        "u": u,
        "dt": dt,
        **change,
    }

    with pytest.raises(ValueError, match=message):
        write_result(run_dir, **arguments)

    assert not (run_dir / DATA_FILENAME).exists()
