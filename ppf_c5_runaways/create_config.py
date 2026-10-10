"""Write a run configuration (.yaml) from Python, validated before it is written."""

from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import ArrayLike

from ppf_c5_runaways.config import RunConfig, save_config
from ppf_c5_runaways.constants import e, m_e


def new_config(
    name: str,
    *,
    field: Mapping[str, Any],
    u0: ArrayLike,
    dt: float,
    n_steps: int,
    x0: ArrayLike | None = None,
    pusher: str = "boris",
    save_every: int = 1,
    seed: int = 42,
    q: float = -e,
    m: float = m_e,
    directory: str | Path = "configs",
) -> Path:
    """Write a new ``.yaml`` run configuration and return its path.

    The configuration is built as a ``RunConfig`` first, so every rule of
    ``ppf_c5_runaways.config`` is checked **before** anything is written: a wrong
    field ``kind``, a misspelled key or a ``save_every`` that does not divide
    ``n_steps`` raises at once, not when the run starts. The file is written by
    ``save_config``: it is read back exactly by ``load_config`` and an existing file
    is never overwritten.

    Parameters
    ----------
    name : str
        Label of the case: letters, digits, ``_`` and ``-``. It names the results
        folder, and with ``pusher`` the file: ``<directory>/<name>_<pusher>.yaml``.
        Give the same ``name`` to the configurations of one case run with several
        pushers.
    field : mapping
        The ``field:`` block of the ``.yaml`` file, as a dictionary, for example
        ``{"kind": "uniform_cartesian", "b_field": [0, 0, 2.5], "e_field": [0, 0, 0]}``
        (B in T, E in V/m). Every kind accepted by ``config.FieldConfig`` works.
    u0 : array_like, shape (3,) or (n_part, 3)
        Initial proper velocities ``u = gamma * v`` [m/s] at ``t = 0``. One vector
        ``[ux, uy, uz]`` is read as one particle.
    dt : float
        Time step [s]. No default: a good value depends on the field (a small
        fraction of the gyro-period ``2 pi gamma m / (|q| B)``).
    n_steps : int
        Number of time steps. No default, for the same reason.
    x0 : array_like, shape (3,) or (n_part, 3), optional
        Initial positions [m]. By default every particle starts at the origin.
    pusher : {"boris", "vay", "higuera_cary", "rk4"}, default "boris"
        Time-integration scheme.
    save_every : int, default 1
        Keep one state every ``save_every`` steps. Must divide ``n_steps``.
    seed : int, default 42
        Seed of the random generator, stored so that the run is reproducible.
    q : float, default -e
        Signed charge [C]. The default is an electron.
    m : float, default m_e
        Rest mass [kg]. The default is an electron.
    directory : str or pathlib.Path, default "configs"
        Folder of the new file, relative to the current working directory unless
        absolute. It is created if missing.

    Returns
    -------
    pathlib.Path
        Path of the new file, ready for ``run.pusher(path)``.

    Raises
    ------
    pydantic.ValidationError
        If a value breaks a rule of ``RunConfig`` (the message names the key and
        lists the accepted values), including an ``x0`` or ``u0`` that is not a
        table of finite numbers with 3 per row. Nothing is written.
    FileExistsError
        If the file already exists. It is left untouched.

    Examples
    --------
    One electron with ``gamma = 2`` gyrating in 2.5 T, about 140 steps per turn:

    >>> path = new_config(  # doctest: +SKIP
    ...     "gyration_gamma_2",
    ...     field={"kind": "uniform_cartesian",
    ...            "b_field": [0, 0, 2.5], "e_field": [0, 0, 0]},
    ...     u0=[5.19e8, 0, 0],
    ...     dt=2.0e-13,
    ...     n_steps=1400,
    ... )
    >>> path  # doctest: +SKIP
    PosixPath('configs/gyration_gamma_2_boris.yaml')
    """
    # Both tables are given as rows, one per particle. A numpy array becomes nested
    # lists of Python numbers. A list is never passed through numpy: it would turn
    # True into 1.0 in [True, 0.0, 0.0]. Anything wrong (a boolean, a string, a
    # row of 2 numbers) is left as it is, for the strict validation below.
    particles = {}
    for key, value in (("u0", u0), ("x0", x0)):
        if isinstance(value, np.ndarray):
            value = value.tolist()
        is_one_vector = (
            isinstance(value, (list, tuple))
            and len(value) > 0
            and not isinstance(value[0], (list, tuple))
        )
        particles[key] = [value] if is_one_vector else value
    if particles["x0"] is None:
        n_part = len(particles["u0"]) if isinstance(particles["u0"], list) else 1
        particles["x0"] = [[0.0, 0.0, 0.0]] * n_part

    config = RunConfig.model_validate(
        {
            "name": name,
            "pusher": pusher,
            "dt": dt,
            "n_steps": n_steps,
            "save_every": save_every,
            "seed": seed,
            "q": q,
            "m": m,
            "field": dict(field),
            "initial": {"x0": particles["x0"], "u0": particles["u0"]},
        }
    )

    # The path is built only from validated values: ``name`` cannot contain a path
    # separator, so the file stays inside ``directory``.
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"{config.name}_{config.pusher}.yaml"
    save_config(config, path)
    return path
