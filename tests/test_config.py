"""Tests of ``ppf_c5_runaways.config`` (``RunConfig`` and ``load_config``)."""

import math
import tempfile
from pathlib import Path

import pytest
import yaml
from hypothesis import given, settings
from hypothesis import strategies as st
from pydantic import ValidationError

from ppf_c5_runaways.config import RunConfig, load_config, save_config


def _valid() -> dict:
    """Return a valid set of fields (fresh dict at each call)."""
    return {
        "name": "uniform_b",
        "pusher": "boris",
        "dt": 1.0e-11,
        "n_steps": 1000,
        "save_every": 10,
        "seed": 42,
        "q": -1.602176634e-19,
        "m": 9.1093837139e-31,
        "field": {
            "kind": "uniform",
            "b_field": [0, 0, 2.5],
            "e_field": [0, 0, 0],
        },
        "initial": {"x0": [[0, 0, 0]], "u0": [[1.0e7, 0, 1.0e6]]},
    }


def _nested(section: str, **change: object) -> dict:
    """Return ``{section: <valid section with some keys replaced>}``.

    Meant for the parametrised cases: ``_nested("field", b_field=[0, 0])`` is a
    ``change`` that breaks one key of the ``field`` block and nothing else.
    """
    return {section: {**_valid()[section], **change}}


def _yaml_text(**raw: str) -> str:
    """Return the text of a valid .yaml file, with some values replaced.

    The replacements are *raw YAML* (``n_steps='"1000"'`` writes a quoted string),
    so that a test controls exactly what is in the file, unlike ``yaml.safe_dump``.
    ``field`` and ``initial`` are written in one line each (YAML flow style).
    """
    values = {
        "name": "uniform_b",
        "pusher": "boris",
        "dt": "1.0e-11",
        "n_steps": "1000",
        "save_every": "10",
        "seed": "42",
        "q": "-1.602176634e-19",
        "m": "9.1093837139e-31",
        "field": "{kind: uniform, b_field: [0, 0, 2.5], e_field: [0, 0, 0]}",
        "initial": "{x0: [[0, 0, 0]], u0: [[1.0e7, 0, 1.0e6]]}",
        **raw,
    }
    return "".join(f"{key}: {value}\n" for key, value in values.items())


def _write(directory: Path, content: str | bytes) -> Path:
    """Write content/data to run.yaml in directory and return its path."""
    path = directory / "run.yaml"
    if isinstance(content, bytes):
        path.write_bytes(content)
    else:
        path.write_text(content, encoding="utf-8")
    return path


# --- RunConfig -----------------------------------------------------------------


def test_yaml_round_trip_is_exact():
    config = RunConfig(**_valid())

    text = yaml.safe_dump(config.model_dump())
    config_back = RunConfig(**yaml.safe_load(text))

    assert config_back == config
    assert config_back.dt == config.dt


@pytest.mark.parametrize(
    ("change", "message"),
    [
        # dt: positive, finite, a real number (not a string, not a bool)
        ({"dt": 0.0}, "dt"),
        ({"dt": -1.0e-11}, "dt"),
        ({"dt": float("inf")}, "dt"),
        ({"dt": float("nan")}, "dt"),
        ({"dt": "1e-11"}, "dt"),
        ({"dt": True}, "dt"),
        # integers: strictly positive, and really integers
        ({"n_steps": 0}, "n_steps"),
        ({"n_steps": "1000"}, "n_steps"),
        ({"n_steps": 1000.0}, "n_steps"),
        ({"n_steps": True}, "n_steps"),
        ({"save_every": 0}, "save_every"),
        ({"n_steps": 1001}, "must divide"),  # 1001 % 10 != 0
        # seed: NumPy rejects negative seeds; a bool is not a seed
        ({"seed": -1}, "seed"),
        ({"seed": True}, "seed"),
        # name: becomes a folder name, so nothing that can leave the results folder
        ({"name": ""}, "name"),
        ({"name": "   "}, "name"),
        ({"name": "../escape"}, "name"),
        ({"name": "a/b"}, "name"),
        ({"name": "a\\b"}, "name"),
        ({"name": ".hidden"}, "name"),
        ({"name": "-leading_dash"}, "name"),
        ({"name": "a b"}, "name"),
        ({"name": "a\n"}, "name"),  # a trailing newline must not slip through
        ({"name": "a" * 65}, "name"),  # 64 characters at most
        # the rest
        ({"pusher": "euler"}, "pusher"),
        ({"pusher": "Boris"}, "pusher"),  # case matters: no silent normalisation
        ({"unknown_key": 1}, "unknown_key"),  # typo in a .yaml must not pass
        # q: a real, finite, non-zero number (the sign is the charge, not an error)
        ({"q": 0.0}, "non-zero"),
        ({"q": float("nan")}, "q"),
        ({"q": float("inf")}, "q"),
        ({"q": "-1.6e-19"}, "q"),
        ({"q": True}, "q"),
        # m: strictly positive, finite, a real number
        ({"m": 0.0}, "m"),
        ({"m": -9.1e-31}, "m"),
        ({"m": float("inf")}, "m"),
        ({"m": "9.1e-31"}, "m"),
        # field: known kind, exactly 3 finite numbers per vector, no stray key
        (_nested("field", kind="tokamak"), "kind"),
        (_nested("field", b_field=[0, 0]), "b_field"),  # 2 components
        (_nested("field", b_field=[0, 0, 1, 1]), "b_field"),  # 4 components
        (_nested("field", b_field=[0, 0, float("nan")]), "b_field"),
        (_nested("field", e_field=[0, 0, float("inf")]), "e_field"),
        (_nested("field", b_field=[0, 0, "2.5"]), "b_field"),  # a string
        (_nested("field", b_field=2.5), "b_field"),  # a scalar, not a vector
        (_nested("field", b_z=1.0), "b_z"),  # typo in a sub-block must not pass
        # initial: at least one particle, 3 finite numbers per row, same count
        (_nested("initial", x0=[]), "x0"),
        (_nested("initial", x0=[[0, 0]]), "x0"),
        (_nested("initial", u0=[[0, 0, float("nan")]]), "u0"),
        (_nested("initial", u0=[[0, 0, 1], [0, 0, 2]]), "one row per particle"),
        (_nested("initial", x0=[[0, 0, 0], [1, 1, 1]]), "one row per particle"),
        (_nested("initial", x0=[0, 0, 0]), "x0"),  # one vector, not a list of them
    ],
)
def test_invalid_values_are_rejected(change, message):
    fields = {**_valid(), **change}

    with pytest.raises(ValidationError, match=message):
        RunConfig(**fields)


@pytest.mark.parametrize("name", ["a", "run-01_A", "9lives", "a" * 64])
def test_valid_names_are_accepted(name):
    assert RunConfig(**{**_valid(), "name": name}).name == name


def test_boundary_values_are_accepted():
    # seed = 0 is a legitimate seed; an int is a legitimate float (dt = 1 s).
    config = RunConfig(**{**_valid(), "seed": 0, "dt": 1, "save_every": 1000})

    assert config.seed == 0
    assert config.dt == 1.0


def test_missing_key_is_rejected():
    fields = _valid()
    del fields["seed"]  # the seed must always be explicit (reproducibility)

    with pytest.raises(ValidationError, match="seed"):
        RunConfig(**fields)


@pytest.mark.parametrize(
    ("section", "key"),
    [
        ("run", "q"),
        ("run", "m"),
        ("run", "field"),
        ("run", "initial"),
        ("field", "e_field"),  # "no electric field" must be written [0, 0, 0]
        ("initial", "u0"),
    ],
)
def test_missing_physics_key_is_rejected(section, key):
    fields = _valid()
    target = fields if section == "run" else fields[section]
    del target[key]

    with pytest.raises(ValidationError, match=key):
        RunConfig(**fields)


def test_config_is_immutable():
    config = RunConfig(**_valid())

    with pytest.raises(ValidationError, match="frozen"):
        config.dt = 2.0e-11
    with pytest.raises(ValidationError, match="frozen"):
        config.field.b_field = (0.0, 0.0, 1.0)


def test_vectors_are_stored_as_floats_in_immutable_tuples():
    # A list (what YAML gives) and ints are accepted, but what is stored is a tuple
    # of floats: nothing in a config can be modified after validation.
    config = RunConfig(**_valid())

    assert config.field.b_field == (0.0, 0.0, 2.5)
    assert isinstance(config.field.b_field, tuple)
    assert all(isinstance(value, float) for value in config.field.b_field)
    assert config.initial.x0 == ((0.0, 0.0, 0.0),)
    assert isinstance(config.initial.x0[0], tuple)


def test_several_particles_are_accepted():
    config = RunConfig(
        **{
            **_valid(),
            **_nested("initial", x0=[[0, 0, 0], [1, 0, 0]], u0=[[1, 0, 0], [2, 0, 0]]),
        }
    )

    assert len(config.initial.x0) == len(config.initial.u0) == 2


def test_positive_charge_is_accepted():
    # The sign of q is physics (sense of gyration): a proton or a positron is legal.
    assert RunConfig(**{**_valid(), "q": 1.602176634e-19}).q > 0


# --- load_config: reading ------------------------------------------------------


@pytest.mark.parametrize("as_type", [Path, str])
def test_load_config_reads_file_back_exactly(tmp_path, as_type):
    # Independent truth: the dict written to disk. Exact equality (no tolerance):
    # PyYAML writes floats with repr(), which round-trips float64 bit for bit.
    fields = _valid()
    path = _write(tmp_path, yaml.safe_dump(fields))

    config = load_config(as_type(path))

    assert config == RunConfig(**fields)
    assert config.dt == fields["dt"]


@pytest.mark.parametrize(
    "literal",
    ["1.0e-11", "1e-11", "1E-11", "+1e-11", "10e-12", ".1e-10", "0.00000000001"],
)
def test_load_config_parses_hand_written_floats(tmp_path, literal):
    # Independent truth: the Python literal 1.0e-11. Exact equality, because every
    # spelling is the same decimal number, and the decimal -> float64 conversion is
    # correctly rounded. Plain YAML 1.1 would read ``1e-11`` as a string.
    path = _write(tmp_path, _yaml_text(dt=literal))

    assert load_config(path).dt == 1.0e-11


def test_load_config_accepts_byte_order_mark(tmp_path):
    # Some Windows editors prepend a BOM; it must not turn ``name`` into an
    # unknown key made of the BOM followed by ``name``.
    path = _write(tmp_path, b"\xef\xbb\xbf" + _yaml_text().encode("utf-8"))

    assert load_config(path) == RunConfig(**_valid())


def test_load_config_accepts_windows_line_endings(tmp_path):
    path = _write(tmp_path, _yaml_text().replace("\n", "\r\n").encode("utf-8"))

    assert load_config(path) == RunConfig(**_valid())


# --- load_config: errors on the file itself ------------------------------------


def test_load_config_missing_file_raises(tmp_path):
    with pytest.raises(FileNotFoundError, match=r"no_such_run\.yaml"):
        load_config(tmp_path / "no_such_run.yaml")


def test_load_config_directory_is_not_a_file(tmp_path):
    with pytest.raises(FileNotFoundError, match="not found"):
        load_config(tmp_path)


def test_load_config_non_utf8_file_raises_with_path(tmp_path):
    path = _write(tmp_path, b"name: \xff\xfe\n")  # not valid UTF-8

    with pytest.raises(ValueError, match=r"run\.yaml is not valid UTF-8"):
        load_config(path)


def test_load_config_malformed_yaml_raises_value_error(tmp_path):
    path = _write(tmp_path, "name: [unclosed\n")

    with pytest.raises(ValueError, match="Invalid YAML"):
        load_config(path)


def test_load_config_several_documents_raises(tmp_path):
    # ``---`` starts a second document: the second one would be silently ignored.
    path = _write(tmp_path, _yaml_text() + "---\n" + _yaml_text())

    with pytest.raises(ValueError, match="Invalid YAML"):
        load_config(path)


def test_load_config_refuses_python_specific_tags(tmp_path):
    # Safe loading: a file can never build a Python object, hence never run code.
    path = _write(tmp_path, _yaml_text(name="!!python/object/apply:os.getcwd []"))

    with pytest.raises(ValueError, match="Invalid YAML"):
        load_config(path)


def test_load_config_repeated_key_raises(tmp_path):
    # PyYAML alone keeps the *last* value and says nothing: two ``dt`` lines would
    # silently run with the second one.
    path = _write(tmp_path, _yaml_text() + "dt: 5.0e-11\n")

    with pytest.raises(ValueError, match=r"duplicate key\(s\) \['dt'\]"):
        load_config(path)


@pytest.mark.parametrize("text", ["", "- boris\n- vay\n", "42\n"])
def test_load_config_non_mapping_top_level_raises(tmp_path, text):
    # Empty file -> None, list -> list, bare scalar -> int: none is a mapping.
    path = _write(tmp_path, text)

    with pytest.raises(ValueError, match="top level must be a mapping"):
        load_config(path)


# --- load_config: validation of the content ------------------------------------


@pytest.mark.parametrize(
    ("raw", "message"),
    [
        ({"dt": "-1.0e-11"}, "dt"),
        ({"dt": ".inf"}, "dt"),
        ({"dt": ".nan"}, "dt"),
        ({"dt": "true"}, "dt"),
        ({"dt": '"1e-11"'}, "dt"),  # quoted: an explicit string, not a number
        ({"n_steps": "1001"}, "must divide"),  # 1001 % 10 != 0
        ({"n_steps": '"1000"'}, "n_steps"),  # quoted string
        ({"n_steps": "1000.0"}, "n_steps"),  # float
        ({"n_steps": "1e3"}, "n_steps"),  # exponent form is a float, not an int
        ({"seed": "-1"}, "seed"),
        ({"seed": "true"}, "seed"),
        ({"name": '"../escape"'}, "name"),
        ({"pusher": "euler"}, "pusher"),
        ({"unknown_key": "1"}, "unknown_key"),  # typo in a .yaml must not pass
        ({"q": "0"}, "non-zero"),
        ({"m": "-1.0e-30"}, "m"),
        ({"field": "{kind: uniform, b_field: [0, 0], e_field: [0, 0, 0]}"}, "b_field"),
        (
            {"field": "{kind: uniform, b_field: [0, 0, .nan], e_field: [0, 0, 0]}"},
            "b_field",
        ),
        ({"field": "{kind: uniform, b_field: [0, 0, 1]}"}, "e_field"),  # missing
        ({"initial": "{x0: [[0, 0, 0]], u0: []}"}, "u0"),
        ({"initial": "{x0: [[0, 0, 0]], u0: [[1, 0, 0], [2, 0, 0]]}"}, "one row"),
    ],
)
def test_load_config_applies_run_config_validation(tmp_path, raw, message):
    path = _write(tmp_path, _yaml_text(**raw))

    with pytest.raises(ValidationError, match=message):
        load_config(path)


def test_load_config_reads_a_hand_written_multi_line_file(tmp_path):
    # The file of the RunConfig docstring, as a physicist would write it: block
    # style, comments, ints where floats are expected, exponent form without sign.
    text = """\
name: uniform_b
pusher: boris
dt: 1.0e-13
n_steps: 1000
save_every: 10
seed: 42
q: -1.602176634e-19       # signed charge [C]
m: 9.1093837139e-31       # mass [kg]
field:
  kind: uniform
  b_field: [0, 0, 2.5]    # [T]
  e_field: [0, 0, 0]      # [V/m]
initial:
  x0: [[0, 0, 0]]
  u0: [[1.0e7, 0, 1.0e6]]
"""
    path = _write(tmp_path, text)

    config = load_config(path)

    assert config.field.b_field == (0.0, 0.0, 2.5)
    assert config.field.e_field == (0.0, 0.0, 0.0)
    assert config.initial.x0 == ((0.0, 0.0, 0.0),)
    assert config.initial.u0 == ((1.0e7, 0.0, 1.0e6),)
    # The docstring claims a gyro-period of 1.43e-11 s, i.e. 143 steps per turn.
    # Independent truth: T_c = 2 pi m / (|q| B). Tolerance 0.5 %: the docstring
    # rounds 1.4290e-11 to three digits (relative rounding error <= 0.35 %).
    period = 2 * math.pi * config.m / (abs(config.q) * config.field.b_field[2])
    assert period == pytest.approx(1.43e-11, rel=5e-3)
    assert round(period / config.dt) == 143


def test_load_config_non_string_key_raises(tmp_path):
    path = _write(tmp_path, _yaml_text() + "1: x\n")

    with pytest.raises(ValidationError):
        load_config(path)


def test_load_config_missing_key_raises(tmp_path):
    text = "".join(line for line in _yaml_text().splitlines(True) if "seed" not in line)
    path = _write(tmp_path, text)

    with pytest.raises(ValidationError, match="seed"):
        load_config(path)


def test_load_config_validation_error_names_the_file(tmp_path):
    path = _write(tmp_path, _yaml_text(dt="-1.0"))

    with pytest.raises(ValidationError) as excinfo:
        load_config(path)

    assert any(str(path) in note for note in excinfo.value.__notes__)


# --- save_config ---------------------------------------------------------------


def test_save_config_writes_the_expected_text(tmp_path):
    # Independent truth: the text written by hand below (keys in the order of
    # ``RunConfig``, floats as PyYAML writes them, each vector on one line so that
    # the file stays readable and editable by hand).
    expected = """\
name: uniform_b
pusher: boris
dt: 1.0e-11
n_steps: 1000
save_every: 10
seed: 42
q: -1.602176634e-19
m: 9.1093837139e-31
field:
  kind: uniform
  b_field: [0.0, 0.0, 2.5]
  e_field: [0.0, 0.0, 0.0]
initial:
  x0:
  - [0.0, 0.0, 0.0]
  u0:
  - [10000000.0, 0.0, 1000000.0]
"""
    path = tmp_path / "run.yaml"

    save_config(RunConfig(**_valid()), path)

    assert path.read_text(encoding="utf-8") == expected


@pytest.mark.parametrize("as_type", [Path, str])
def test_save_then_load_gives_back_an_equal_config(tmp_path, as_type):
    # Exact equality (no tolerance): PyYAML writes floats with repr(), which
    # round-trips float64 bit for bit. This is the stage-0 acceptance criterion.
    config = RunConfig(**_valid())
    path = as_type(tmp_path / "run.yaml")

    save_config(config, path)

    assert load_config(path) == config


@pytest.mark.parametrize(
    "name", ["1e3", "1E5", "123", "true", "null", "2024-01-01", "1_000", "0x1F"]
)
def test_save_then_load_keeps_names_that_look_like_numbers(tmp_path, name):
    # Valid names that YAML could read as a number, a bool, null or a date: they
    # must be quoted on writing. ``1e3`` is the one that plain PyYAML does not quote
    # but ``load_config`` would read as a float.
    config = RunConfig(**{**_valid(), "name": name})
    path = tmp_path / "run.yaml"

    save_config(config, path)

    assert load_config(path).name == name


_FINITE = st.floats(allow_nan=False, allow_infinity=False)
_VECTOR3 = st.tuples(_FINITE, _FINITE, _FINITE)


@settings(deadline=None)  # disk access: the default 200 ms deadline is not relevant
@given(
    name=st.from_regex(r"[A-Za-z0-9][A-Za-z0-9_-]{0,63}", fullmatch=True),
    pusher=st.sampled_from(["boris", "vay", "higuera_cary", "rk4"]),
    dt=st.floats(
        min_value=0.0, exclude_min=True, allow_nan=False, allow_infinity=False
    ),
    save_every=st.integers(min_value=1, max_value=10_000),
    n_saves=st.integers(min_value=1, max_value=10_000),
    seed=st.integers(min_value=0, max_value=2**64),
    q=st.floats(allow_nan=False, allow_infinity=False).filter(lambda q: q != 0.0),
    m=st.floats(min_value=0.0, exclude_min=True, allow_nan=False, allow_infinity=False),
    b_field=_VECTOR3,
    e_field=_VECTOR3,
    particles=st.lists(st.tuples(_VECTOR3, _VECTOR3), min_size=1, max_size=4),
)
def test_save_then_load_is_identity_for_any_valid_config(
    name, pusher, dt, save_every, n_saves, seed, q, m, b_field, e_field, particles
):
    # Property: for every valid config, load(save(c)) == c, exactly. Hypothesis
    # explores subnormal and huge numbers, names that look like numbers, big seeds,
    # several particles.
    config = RunConfig(
        name=name,
        pusher=pusher,
        dt=dt,
        n_steps=save_every * n_saves,  # a multiple of save_every by construction
        save_every=save_every,
        seed=seed,
        q=q,
        m=m,
        field={"kind": "uniform", "b_field": b_field, "e_field": e_field},
        initial={
            "x0": [x for x, _ in particles],
            "u0": [u for _, u in particles],
        },
    )
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / "run.yaml"

        save_config(config, path)

        assert load_config(path) == config


def test_save_config_never_overwrites_an_existing_file(tmp_path):
    path = _write(tmp_path, "precious: hand-written\n")

    with pytest.raises(FileExistsError, match=r"run\.yaml"):
        save_config(RunConfig(**_valid()), path)

    assert path.read_text(encoding="utf-8") == "precious: hand-written\n"


def test_save_config_needs_an_existing_directory(tmp_path):
    path = tmp_path / "missing_dir" / "run.yaml"

    with pytest.raises(FileNotFoundError, match="missing_dir"):
        save_config(RunConfig(**_valid()), path)

    assert not path.parent.exists()  # it did not create the directory either


@pytest.mark.parametrize(
    ("change", "message"),
    [
        ({"dt": -1.0}, "dt"),
        ({"dt": float("inf")}, "dt"),
        ({"name": "../escape"}, "name"),
        ({"n_steps": 1001}, "must divide"),
    ],
)
def test_save_config_refuses_an_invalid_config_and_writes_nothing(
    tmp_path, change, message
):
    # ``model_copy(update=...)`` skips validation: the way to hold an invalid config.
    invalid = RunConfig(**_valid()).model_copy(update=change)
    path = tmp_path / "run.yaml"

    with pytest.raises(ValidationError, match=message):
        save_config(invalid, path)

    assert not path.exists()
