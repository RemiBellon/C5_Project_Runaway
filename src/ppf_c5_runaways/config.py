"""Validated configuration of a simulation run.

This module depends on nothing else in the project (pydantic and PyYAML only). It
describes what to run (``RunConfig``, made of a ``FieldConfig`` and an
``InitialState``) and reads it from a .yaml file (``load_config``), and writes it
back (save_config). save_config is the only place that writes, and only to the path
it is given: choosing where results go is the job of ``run``. All physical
quantities are in SI units.
"""

import re
from collections.abc import Hashable
from pathlib import Path
from typing import Annotated, Any, Literal, Self

import yaml
from pydantic import (
    BaseModel,
    BeforeValidator,
    ConfigDict,
    Field,
    FiniteFloat,
    ValidationError,
    field_validator,
    model_validator,
)

# Shared by every model of this module: immutable, unknown keys refused, no implicit
# type conversion. One constant, so that no model can quietly be less strict.
_MODEL_CONFIG = ConfigDict(extra="forbid", frozen=True, strict=True)


def _list_to_tuple(value: Any) -> Any:
    """Turn a list (what YAML gives) into a tuple, and leave anything else as is.

    ``strict`` mode refuses a list where a tuple is expected. Tuples are also what
    keeps a frozen configuration really immutable. Anything that is not a list is
    passed on untouched, so that pydantic still judges (and rejects) it.
    """
    return tuple(value) if isinstance(value, list) else value


# A 3-vector of finite numbers: [x, y, z] in the .yaml file. Exactly 3 components.
_Vector3 = Annotated[
    tuple[FiniteFloat, FiniteFloat, FiniteFloat], BeforeValidator(_list_to_tuple)
]
# One 3-vector per particle, at least one particle: shape (n_part, 3).
_Vectors3 = Annotated[
    tuple[_Vector3, ...], BeforeValidator(_list_to_tuple), Field(min_length=1)
]


class FieldConfig(BaseModel):
    """Electromagnetic field in which the particles move.

    kind : {"uniform"}
        Shape of the field. Only a field that is the same everywhere and at every
        time for now; other kinds (E x B, mirror, tokamak) come with stage 1 to 3.
    b_field : tuple of 3 floats
        Magnetic field ``(Bx, By, Bz)`` [T].
    e_field : tuple of 3 floats
        Electric field ``(Ex, Ey, Ez)`` [V/m]. Write ``[0, 0, 0]`` for no electric
        field: it is never assumed.

    pydantic.ValidationError
        If a vector does not have exactly 3 finite numbers, or a key is unknown or
        missing.
    """

    model_config = _MODEL_CONFIG

    kind: Literal["uniform"]
    b_field: _Vector3
    e_field: _Vector3


class InitialState(BaseModel):
    """Positions and velocities of the particles at ``t = 0``.

    ``n_part``, the number of particles, is the number of rows: it is not a separate
    key, so it can never contradict the data.

    Attributes
    ----------
    x0 : tuple of n_part tuples of 3 floats
        Initial positions ``x^0``, shape ``(n_part, 3)`` [m].
    u0 : tuple of n_part tuples of 3 floats
        Initial *proper* velocities ``u = gamma * v``, shape ``(n_part, 3)`` [m/s].
        It is ``u``, not ``v``: at 20 MeV ``v`` is indistinguishable from ``c``.

    Raises
    ------
    pydantic.ValidationError
        If there is no particle, a row does not have 3 finite numbers, or ``x0`` and
        ``u0`` do not have the same number of particles.
    """

    model_config = _MODEL_CONFIG

    x0: _Vectors3
    u0: _Vectors3

    @model_validator(mode="after")
    def _check_same_number_of_particles(self) -> Self:
        """Require as many rows in ``u0`` as in ``x0``."""
        if len(self.x0) != len(self.u0):
            raise ValueError(
                f"x0 has {len(self.x0)} particle(s) but u0 has {len(self.u0)}: "
                "they must have one row per particle each"
            )
        return self


class RunConfig(BaseModel):
    """Description of one simulation run.

    The instance is immutable (``frozen``), rejects unknown keys
    (``extra="forbid"``) and does no implicit type conversion (``strict``): a typo
    in a ``.yaml`` file, or ``n_steps: "1000"`` (a string), raises an error instead
    of being silently ignored or converted. An ``int`` is still accepted where a
    ``float`` is expected. The same holds for ``field`` and ``initial``.

    A complete file, one electron gyrating in a uniform 2.5 T field (gyro-period
    1.43e-11 s, so 143 steps per turn)::

        name: uniform_b
        pusher: boris
        dt: 1.0e-13
        n_steps: 1000
        save_every: 10
        seed: 42
        q: -1.602176634e-19       # signed charge [C]: negative for an electron
        m: 9.1093837139e-31       # mass [kg]
        field:
          kind: uniform
          b_field: [0, 0, 2.5]    # [T]
          e_field: [0, 0, 0]      # [V/m]
        initial:
          x0: [[0, 0, 0]]         # one row [x, y, z] per particle [m]
          u0: [[1.0e7, 0, 1.0e6]] # one row [ux, uy, uz] per particle [m/s]

    Attributes
    ----------
    name : str
        Short label of the run, used to name the results folder: 1 to 64
        characters among letters, digits, ``_`` and ``-``, starting with a letter
        or a digit. This forbids path separators, ``..`` and hidden names, so the
        label can never point outside the results directory.
    pusher : {"boris", "vay", "higuera_cary", "rk4"}
        Time-integration scheme. The three first are leapfrog pushers, ``"rk4"``
        is the non-structure-preserving reference.
    dt : float
        Fixed time step [s], finite and strictly positive.
    n_steps : int
        Total number of time steps, strictly positive.
    save_every : int
        Store the state every ``save_every`` steps, strictly positive. Must divide
        ``n_steps``, so that the last step is a stored one.
    seed : int
        Seed of the ``numpy.random.Generator`` (used by collisions, stage 4),
        non-negative because NumPy rejects negative seeds. It is always explicit so
        that a run is reproducible.
    q : float
        Signed charge of a particle [C], finite and non-zero: ``-1.602176634e-19``
        for an electron. The sign sets the sense of gyration, so it is never
        replaced by an absolute value.
    m : float
        Rest mass of a particle [kg], finite and strictly positive.
    field : FieldConfig
        Electromagnetic field (see ``FieldConfig``).
    initial : InitialState
        Initial positions and proper velocities (see ``InitialState``).

    Raises
    ------
    pydantic.ValidationError
        If a value has the wrong type or is out of range, a key is unknown or
        missing, or ``n_steps`` is not a multiple of ``save_every``.
    """

    model_config = _MODEL_CONFIG

    name: str = Field(
        min_length=1, max_length=64, pattern=r"^[A-Za-z0-9][A-Za-z0-9_-]*$"
    )
    pusher: Literal["boris", "vay", "higuera_cary", "rk4"]
    dt: float = Field(gt=0.0, allow_inf_nan=False)
    n_steps: int = Field(gt=0)
    save_every: int = Field(gt=0)
    seed: int = Field(ge=0)
    q: FiniteFloat
    m: float = Field(gt=0.0, allow_inf_nan=False)
    field: FieldConfig
    initial: InitialState

    @field_validator("q")
    @classmethod
    def _check_q_is_not_zero(cls, q: float) -> float:
        """Refuse ``q = 0``: it would be a neutral particle, which never gyrates."""
        if q == 0.0:
            raise ValueError(
                "q must be non-zero: it is the signed charge in coulombs "
                "(-1.602176634e-19 for an electron)"
            )
        return q

    @model_validator(mode="after")
    def _check_save_every_divides_n_steps(self) -> Self:
        """Require ``n_steps % save_every == 0`` so the last step is stored."""
        if self.n_steps % self.save_every:
            raise ValueError(
                f"save_every={self.save_every} must divide n_steps={self.n_steps}"
            )
        return self


class _ConfigLoader(yaml.SafeLoader):
    """``SafeLoader`` fixing two PyYAML pitfalls of hand-written configuration files.

    * PyYAML follows YAML 1.1, where a float needs a dot *and* a signed exponent:
      ``1e-11`` would be read as the string ``"1e-11"``. Physicists write it this
      way, so an exponent form without dot or sign is read as a float.
    * A repeated key silently overrides the earlier one. It raises here, for the
      same reason an unknown key does: a config file must mean what it says.

    It stays a ``SafeLoader``: no Python object can be built from a file.
    """

    def construct_mapping(
        self, node: yaml.MappingNode, deep: bool = False
    ) -> dict[Hashable, Any]:
        """Build a mapping, raising if a key appears more than once."""
        mapping = super().construct_mapping(node, deep=deep)
        if len(mapping) != len(node.value):
            keys = [self.construct_object(key_node) for key_node, _ in node.value]
            repeated = sorted({key for key in keys if keys.count(key) > 1}, key=str)
            raise yaml.constructor.ConstructorError(
                None, None, f"found duplicate key(s) {repeated}", node.start_mark
            )
        return mapping


class _ConfigDumper(yaml.SafeDumper):
    """``SafeDumper`` that agrees with ``_ConfigLoader`` on what a float looks like.

    Without it, the name ``"1e3"`` would be written unquoted, and read back as the
    float ``1000.0`` by ``_ConfigLoader``: the file would not survive a round trip.
    """


# Exponent form that the YAML 1.1 float rule misses: 1e-11, 1E5, 2.5e3, .1e-10.
# It requires an exponent, so plain integers and plain floats are unaffected.
# The loader uses it to read such a value as a float, the dumper to quote a string
# that looks like one.
_EXPONENT_FLOAT = re.compile(
    r"^[-+]?(?:[0-9][0-9_]*(?:\.[0-9_]*)?|\.[0-9_]+)[eE][-+]?[0-9]+$"
)
_FLOAT_TAG = "tag:yaml.org,2002:float"
_FLOAT_FIRST_CHARS = list("-+0123456789.")
_ConfigLoader.add_implicit_resolver(_FLOAT_TAG, _EXPONENT_FLOAT, _FLOAT_FIRST_CHARS)
_ConfigDumper.add_implicit_resolver(_FLOAT_TAG, _EXPONENT_FLOAT, _FLOAT_FIRST_CHARS)


def load_config(path: str | Path) -> RunConfig:
    """Read and validate a run configuration from a ``.yaml`` file.

    Values are taken as written in the file, in SI units (``dt`` in seconds):
    no unit conversion is done here. The file is read as UTF-8 (a leading byte
    order mark is accepted), and floats may be written ``1e-11``.

    Parameters
    ----------
    path : str or pathlib.Path
        Location of the ``.yaml`` file.

    Returns
    -------
    RunConfig
        Validated, immutable configuration.

    Raises
    ------
    FileNotFoundError
        If ``path`` is not an existing regular file.
    ValueError
        If the file is not UTF-8 text, is not valid YAML (including a repeated key,
        several documents or a Python-specific tag), or if its top level is not a
        mapping (empty file, list, scalar). The message contains the path.
    pydantic.ValidationError
        If a value has the wrong type or is out of range, a key is unknown or
        missing, or ``n_steps`` is not a multiple of ``save_every``. It is a
        ``ValueError`` subclass; a note added to it (``__notes__``) names the file.
    """
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(
            f"Configuration file not found (or not a regular file): {path}"
        )

    try:
        text = path.read_text(encoding="utf-8-sig")
    except UnicodeDecodeError as error:
        raise ValueError(f"{path} is not valid UTF-8 text: {error}") from error

    try:
        data = yaml.load(text, Loader=_ConfigLoader)
    except yaml.YAMLError as error:
        raise ValueError(f"Invalid YAML in {path}: {error}") from error

    if not isinstance(data, dict):
        raise ValueError(
            f"{path}: the top level must be a mapping of keys to values, "
            f"got {type(data).__name__}"
        )

    try:
        return RunConfig.model_validate(data)
    except ValidationError as error:
        error.add_note(f"Invalid configuration file: {path}")
        raise


def save_config(config: RunConfig, path: str | Path) -> None:
    """Write a run configuration to a new ``.yaml`` file.

    The file is what ``load_config`` reads back: ``load_config(path) == config``
    exactly, floats included (PyYAML writes them with ``repr``, which round-trips
    float64 bit for bit). Keys are written in the field order of ``RunConfig``,
    values in SI units (``dt`` in seconds).

    An existing file is **never overwritten**: a hand-written configuration, or the
    record of an earlier run, must not disappear silently. The parent directory
    must already exist; creating it is the job of the caller.

    Parameters
    ----------
    config : RunConfig
        Configuration to write. It is validated again before anything is written,
        so an instance built around validation (``RunConfig.model_construct``)
        cannot produce a file that ``load_config`` would refuse.
    path : str or pathlib.Path
        Location of the new file.

    Raises
    ------
    pydantic.ValidationError
        If ``config`` does not satisfy the rules of ``RunConfig``. Nothing is
        written.
    FileExistsError
        If ``path`` already exists.
    FileNotFoundError
        If the parent directory of ``path`` does not exist.
    """
    validated = RunConfig.model_validate(config.model_dump())
    # default_flow_style=None: a vector is written on one line, as in
    # ``b_field: [0, 0, 2.5]``, instead of a vertical list: the file stays easy to
    # read and to edit by hand.
    text = yaml.dump(
        validated.model_dump(),
        Dumper=_ConfigDumper,
        sort_keys=False,
        default_flow_style=None,
    )
    # Mode "x": create the file, or fail if it exists (no check-then-write race).
    with Path(path).open("x", encoding="utf-8") as file:
        file.write(text)
