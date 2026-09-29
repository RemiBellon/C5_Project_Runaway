"""Results folders: one timestamped directory per physical case.

A results folder is self-describing: it holds the configuration that produced it and
a few metadata, so that a run can be recognised and its data reloaded long after it
was computed, without the .yaml file it came from. It also holds the trajectories,
one netCDF *group* per pusher in a single ``data.nc``: this is what lets you add the
result of another pusher to a folder that already exists, to compare them. This
module depends on config.py only. It does not compute anything.

CONFIG_FILENAME : str
    Name of the configuration file inside a results folder.
DATA_FILENAME : str
    Name of the trajectories file inside a results folder.
METADATA_FILENAME : str
    Name of the metadata file inside a results folder.
TIMESTAMP_FORMAT : str
    ``strftime`` format of the timestamp in a folder name: ``aaaammjj_hhmmss``
    (year, month, day, hour, minute, second).
METADATA_SCHEMA_VERSION : int
    Layout version of ``metadata.json``, to be increased when its keys change so
    that a reader can tell which layout it is looking at.
"""

import importlib.metadata
import json
import platform
import re
import shutil
from datetime import datetime
from pathlib import Path
from typing import Final

import h5netcdf
import numpy as np
import xarray as xr
from numpy.typing import NDArray

from ppf_c5_runaways.config import RunConfig, save_config

CONFIG_FILENAME: Final[str] = "config.yaml"
DATA_FILENAME: Final[str] = "data.nc"
METADATA_FILENAME: Final[str] = "metadata.json"
TIMESTAMP_FORMAT: Final[str] = "%Y%m%d_%H%M%S"
METADATA_SCHEMA_VERSION: Final[int] = 1


def create_result_dir(
    base_dir: str | Path, config: RunConfig, *, now: datetime | None = None
) -> Path:
    """Create the results folder of a run and write its configuration and metadata.

    The folder is ``<base_dir>/<config.name>_<aaaammjj_hhmmss>`` and contains::

        config.yaml     the run configuration, as read back by ``load_config``
        metadata.json   run_id, name, created_at, seed, versions, schema_version

    metadata.json is meant to be light: everything about what was computed
    (``pusher``, ``dt``, ``n_steps``...) stays in config.yaml and is not
    duplicated. The seed is repeated in it on purpose, so that the random stream of
    a run can be recovered from the metadata alone.

    The folder is created exclusively: it is never reused or overwritten. If any
    step fails, the folder is removed again, so a folder that exists is complete.

    Parameters
    ----------
    base_dir : str or pathlib.Path
        Directory that will contain the results folder. It is created, with its
        parents, if it does not exist.
    config : RunConfig
        Configuration of the run. It is validated again before anything is created
        on disk, so that ``config.name`` (which becomes a folder name) can never
        point outside ``base_dir``, even for an instance built with
        ``RunConfig.model_construct``.
    now : datetime.datetime, optional
        Moment of creation; it must be timezone-aware. Default: the current local
        time. Its wall-clock time, in its own time zone, gives the timestamp, and
        it is stored in ``created_at`` with its UTC offset, so that the folder name
        is never ambiguous. Sub-second precision is dropped.

    Returns
    -------
    pathlib.Path
        The new results folder (``base_dir / run_id``).

    Raises
    ------
    pydantic.ValidationError
        If ``config`` does not satisfy the rules of ``RunConfig``. Nothing is
        created.
    ValueError
        If ``now`` is naive (has no time zone). Nothing is created.
    FileExistsError
        If the folder already exists, e.g. two runs of the same ``name`` started
        within the same second. Nothing is modified.
    """
    config = RunConfig.model_validate(config.model_dump())
    if now is None:
        now = datetime.now().astimezone()
    elif now.utcoffset() is None:
        raise ValueError(f"now must be timezone-aware, got the naive datetime {now}")
    now = now.replace(microsecond=0)

    run_id = f"{config.name}_{now.strftime(TIMESTAMP_FORMAT)}"
    metadata = {
        "schema_version": METADATA_SCHEMA_VERSION,
        "run_id": run_id,
        "name": config.name,
        "created_at": now.isoformat(),
        "seed": config.seed,
        "versions": {
            "ppf_c5_runaways": importlib.metadata.version("ppf_c5_runaways"),
            "python": platform.python_version(),
            "numpy": np.__version__,
        },
    }

    base_dir = Path(base_dir)
    base_dir.mkdir(parents=True, exist_ok=True)
    run_dir = base_dir / run_id
    run_dir.mkdir()  # no exist_ok: fails if the folder is already there

    try:
        save_config(config, run_dir / CONFIG_FILENAME)
        text = json.dumps(metadata, indent=2) + "\n"
        (run_dir / METADATA_FILENAME).write_text(text, encoding="utf-8")
    except BaseException:
        shutil.rmtree(run_dir)  # a folder that exists must be complete
        raise
    return run_dir


def list_results(run_dir: str | Path) -> list[str]:
    """List the pushers whose trajectories are stored in a results folder.

    Parameters
    ----------
    run_dir : str or pathlib.Path
        Results folder.

    Returns
    -------
    list of str
        Names of the pushers, sorted, for example ``["boris", "vay"]``. Empty if
        nothing has been written in the folder yet.
    """
    data_path = Path(run_dir) / DATA_FILENAME
    if not data_path.is_file():
        return []
    with h5netcdf.File(data_path, "r") as file:
        return sorted(file.groups)


def write_result(
    run_dir: str | Path,
    pusher_name: str,
    t: NDArray[np.float64],
    x: NDArray[np.float64],
    u: NDArray[np.float64],
    dt: float,
) -> None:
    """Store the trajectories of one pusher in the results folder ``run_dir``.

    They go into the group ``pusher_name`` of ``<run_dir>/data.nc``. The file is
    created by the first pusher; the others are *added* to it, without touching what
    is already there. This is how Boris, Vay and Higuera-Cary can be compared in the
    same folder, even if they are run on different days.

    An existing group is **never overwritten**: a result that took hours to compute
    must not disappear silently. To recompute a pusher, use a new results folder.

    The group holds two arrays, with their units and the time convention written in
    the file: ``x`` [m] at times ``time``, and ``u`` [m/s] at times
    ``time_half = time - dt/2`` (leapfrog convention of the project).

    Parameters
    ----------
    run_dir : str or pathlib.Path
        Existing results folder (see ``create_result_dir``).
    pusher_name : str
        Name of the group: letters, digits and ``_`` only, for example ``"boris"``.
    t : ndarray, shape (n_saved,)
        Times of the kept states [s], as returned by ``integrate``.
    x : ndarray, shape (n_saved, n_part, 3)
        Positions [m].
    u : ndarray, shape (n_saved, n_part, 3)
        Proper velocities [m/s], half a step before the positions.
    dt : float
        Time step [s], finite and strictly positive; it gives the times of ``u``.

    Raises
    ------
    FileNotFoundError
        If ``run_dir`` does not exist.
    ValueError
        If ``pusher_name`` is not valid, ``dt`` is not finite and positive, or the
        arrays do not have the shapes above. Nothing is written.
    FileExistsError
        If the group ``pusher_name`` is already in ``data.nc``. Nothing is modified.
    """
    run_dir = Path(run_dir)
    if not run_dir.is_dir():
        raise FileNotFoundError(f"Results folder not found: {run_dir}")
    if not re.fullmatch(r"[A-Za-z0-9_]+", pusher_name):
        raise ValueError(
            "pusher_name must be made of letters, digits and '_' only, "
            f"got {pusher_name!r}"
        )
    if not np.isfinite(dt) or dt <= 0.0:
        raise ValueError(f"dt must be a finite, strictly positive number, got {dt!r}")
    t, x, u = np.asarray(t), np.asarray(x), np.asarray(u)
    if t.ndim != 1 or x.ndim != 3 or x.shape[0] != len(t) or x.shape[2] != 3:
        raise ValueError(
            "x must have shape (n_saved, n_part, 3) with n_saved = len(t); "
            f"got t of shape {t.shape} and x of shape {x.shape}"
        )
    if u.shape != x.shape:
        raise ValueError(
            f"u must have the same shape as x; got x of shape {x.shape} and u of "
            f"shape {u.shape}"
        )

    data_path = run_dir / DATA_FILENAME
    existed = data_path.exists()
    if pusher_name in list_results(run_dir):
        raise FileExistsError(
            f"{data_path} already holds a result for {pusher_name!r}. Results are "
            "never overwritten: run again in a new results folder."
        )

    dataset = xr.Dataset(
        data_vars={
            "x": (("time", "particle", "axis"), x, {"units": "m"}),
            "u": (("time_half", "particle", "axis"), u, {"units": "m/s"}),
        },
        coords={
            "time": ("time", t, {"units": "s"}),
            "time_half": ("time_half", t - dt / 2.0, {"units": "s"}),
            "axis": ["x", "y", "z"],
        },
        attrs={
            "pusher": pusher_name,
            "dt": dt,
            "convention": "x[i] is at time[i]; u[i] = u^(k-1/2) is at time_half[i]",
        },
    )
    try:
        dataset.to_netcdf(data_path, mode="a", group=pusher_name, engine="h5netcdf")
    except BaseException:
        if not existed:
            data_path.unlink(missing_ok=True)  # do not leave a half-written file
        raise


def read_result(run_dir: str | Path, pusher_name: str) -> xr.Dataset:
    """Read the trajectories of one pusher from the results folder ``run_dir``.

    Parameters
    ----------
    run_dir : str or pathlib.Path
        Results folder that contains a ``data.nc``.
    pusher_name : str
        Group to read, for example ``"boris"``.

    Returns
    -------
    xarray.Dataset
        Fully loaded in memory (the file is closed). Variable ``x`` [m], dimensions
        ``(time, particle, axis)``; variable ``u`` [m/s], dimensions
        ``(time_half, particle, axis)``; attributes ``pusher`` and ``dt``.

    Raises
    ------
    FileNotFoundError
        If there is no ``data.nc`` in ``run_dir``.
    ValueError
        If the file has no group ``pusher_name``. The message lists those it has.
    """
    data_path = Path(run_dir) / DATA_FILENAME
    if not data_path.is_file():
        raise FileNotFoundError(
            f"No trajectories file {data_path}: has a simulation been run in this "
            "folder?"
        )
    names = list_results(run_dir)
    if pusher_name not in names:
        raise ValueError(
            f"No result for pusher {pusher_name!r} in {data_path}. Available: {names}"
        )
    with xr.open_dataset(data_path, group=pusher_name, engine="h5netcdf") as dataset:
        return dataset.load()
