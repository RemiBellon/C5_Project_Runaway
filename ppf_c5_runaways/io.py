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
    Layout version of a results folder, written in ``metadata.json``. It is
    increased when the keys of ``metadata.json`` or the variables of ``data.nc``
    change, so that a reader can tell which layout it is looking at. Version 1
    stored the velocities half a step before the positions (``u`` on ``time_half``,
    plus ``u_centered``); version 2 stores ``u`` at the times of ``x``; version 3
    adds to ``metadata.json`` the git state of the code (``git``) and the SciPy
    version, with the same ``data.nc``.
"""

import importlib.metadata
import json
import platform
import re
import shutil
import subprocess
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
METADATA_SCHEMA_VERSION: Final[int] = 3


def _git_revision(directory: Path) -> dict[str, str | bool] | None:
    """Commit of the git repository that holds ``directory``, and whether it is dirty.

    Parameters
    ----------
    directory : pathlib.Path
        A directory inside the repository, for example that of the package.

    Returns
    -------
    dict or None
        ``{"commit": <40-character hash of HEAD>, "dirty": <bool>}``. ``dirty`` is
        True when the working tree differs from that commit: a tracked file is
        modified, staged or deleted, or a file that is not ignored is untracked.
        None when the commit cannot be known: ``directory`` is not inside a git
        repository, the repository has no commit yet, or git is not installed.

    Notes
    -----
    The hash alone does not identify the code that ran when the working tree is
    dirty; ``dirty`` says so, it does not say what differs.
    """
    try:
        head = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=directory,
            capture_output=True,
            text=True,
            check=True,
            timeout=10,
        )
        status = subprocess.run(
            ["git", "status", "--porcelain"],
            cwd=directory,
            capture_output=True,
            text=True,
            check=True,
            timeout=10,
        )
    except (OSError, subprocess.CalledProcessError, subprocess.TimeoutExpired):
        return None
    return {"commit": head.stdout.strip(), "dirty": status.stdout != ""}


def create_result_dir(
    base_dir: str | Path, config: RunConfig, *, now: datetime | None = None
) -> Path:
    """Create the results folder of a run and write its configuration and metadata.

    The folder is ``<base_dir>/<config.name>_<aaaammjj_hhmmss>`` and contains::

        config.yaml     the run configuration, as read back by ``load_config``
        metadata.json   run_id, name, created_at, seed, git, versions,
                        schema_version

    metadata.json is meant to be light: everything about what was computed
    (``pusher``, ``dt``, ``n_steps``...) stays in config.yaml and is not
    duplicated. The seed is repeated in it on purpose, so that the random stream of
    a run can be recovered from the metadata alone. ``git`` is the commit of the
    repository that holds the package and whether its working tree differed from
    it (``_git_revision``), or null when it cannot be known; the run is not
    stopped for it.

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
        "git": _git_revision(Path(__file__).parent),
        "versions": {
            "ppf_c5_runaways": importlib.metadata.version("ppf_c5_runaways"),
            "python": platform.python_version(),
            "numpy": np.__version__,
            "scipy": importlib.metadata.version("scipy"),
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
    *,
    b_field: NDArray[np.float64],
    step_jacobian_determinant: NDArray[np.float64] | None = None,
) -> None:
    """Store the trajectories of one pusher in the results folder ``run_dir``.

    They go into the group ``pusher_name`` of ``<run_dir>/data.nc``. The file is
    created by the first pusher; the others are *added* to it, without touching what
    is already there. This is how Boris, Vay and Higuera-Cary can be compared in the
    same folder, even if they are run on different days.

    An existing group is **never overwritten**: a result that took hours to compute
    must not disappear silently. To recompute a pusher, use a new results folder.

    The group holds three arrays, with their units and the time convention written
    in the file: ``x`` [m], ``u`` [m/s] and ``b_field`` [T], all at the same times
    ``time`` (synchronized convention of the project, see ``integrate``). Any
    diagnostic at these times (energy, drift, invariants) uses ``x`` and ``u`` as
    they are.

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
        Proper velocities ``u = gamma * v`` [m/s] at the times ``t``, as returned
        by ``integrate``.
    dt : float
        Time step of the run [s], finite and strictly positive. Stored as an
        attribute of the group: the kept states can be further apart than ``dt``.
    b_field : ndarray, shape (n_saved, n_part, 3)
        Magnetic field [T] felt by each particle at the positions ``x`` and times
        ``t``. Stored so that diagnostics that need it (magnetic moment, gyration
        phase) can be computed from the results folder alone. Required, and given
        by name, so that no result is stored without it.
    step_jacobian_determinant : ndarray, shape (n_saved, n_part), optional
        Jacobian determinant of the step that starts at each saved state
        (``diagnostics.step_jacobian_determinant``), dimensionless. Stored only if
        given: it is costly and asked for by ``run.pusher(...,
        phase_space_volume=True)``.

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
    t, x, u, b_field = np.asarray(t), np.asarray(x), np.asarray(u), np.asarray(b_field)
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
    if b_field.shape != x.shape:
        raise ValueError(
            f"b_field must have the same shape as x; got x of shape {x.shape} "
            f"and b_field of shape {b_field.shape}"
        )
    if step_jacobian_determinant is not None:
        step_jacobian_determinant = np.asarray(step_jacobian_determinant)
        if step_jacobian_determinant.shape != x.shape[:2]:
            raise ValueError(
                "step_jacobian_determinant must have shape (n_saved, n_part) = "
                f"{x.shape[:2]}; got {step_jacobian_determinant.shape}"
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
            "u": (("time", "particle", "axis"), u, {"units": "m/s"}),
            "b_field": (("time", "particle", "axis"), b_field, {"units": "T"}),
        },
        coords={
            "time": ("time", t, {"units": "s"}),
            "axis": ["x", "y", "z"],
        },
        attrs={
            "pusher": pusher_name,
            "dt": dt,
            "convention": "x[i], u[i] and b_field[i] are all at time[i]",
        },
    )
    if step_jacobian_determinant is not None:
        dataset["step_jacobian_determinant"] = (
            ("time", "particle"),
            step_jacobian_determinant,
            {
                "units": "1",
                "meaning": "determinant of the step (x^n, u^n) -> (x^n+1, u^n+1)",
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
        Fully loaded in memory (the file is closed). Variables ``x`` [m], ``u``
        [m/s] and ``b_field`` [T], all of dimensions ``(time, particle, axis)``: the
        positions, the proper velocities and the magnetic field felt, at the same
        times. Attributes ``pusher``, ``dt`` and ``convention``.

    Raises
    ------
    FileNotFoundError
        If there is no ``data.nc`` in ``run_dir``.
    ValueError
        If the file has no group ``pusher_name`` (the message lists those it has),
        or if this group has the layout of schema version 1, written by the code
        that kept the velocities half a step before the positions. Such a result
        is refused rather than read: its ``u`` is not at the times of its ``x``,
        and every diagnostic would be wrong by an error of order ``dt`` without
        any warning.
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
        result = dataset.load()
    # ``time_half`` is the time axis of the half-step velocities of schema version 1.
    if "time_half" in result.dims:
        raise ValueError(
            f"The result of {pusher_name!r} in {data_path} was written with an older "
            "version of the code (results layout 1): its velocities u are half a "
            "time step before its positions x, which this version cannot use. Run "
            "the case again to get a result in the current layout "
            f"({METADATA_SCHEMA_VERSION})."
        )
    return result
