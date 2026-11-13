"""Figures made from a results folder."""

from collections.abc import Sequence
from itertools import pairwise
from pathlib import Path
from typing import Any, Literal, NamedTuple

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.figure import Figure
from matplotlib.lines import Line2D
from matplotlib.rcsetup import cycler
from matplotlib.ticker import NullFormatter
from matplotlib.typing import LineStyleType, RcKeyType
from numpy.typing import NDArray

import ppf_c5_runaways.config as config
from ppf_c5_runaways import diagnostics, radiation
from ppf_c5_runaways.constants import c, e
from ppf_c5_runaways.guiding_centre import guiding_centre_from_position
from ppf_c5_runaways.io import CONFIG_FILENAME, list_results, read_result

# Lengths are computed in metres, and shown in millimetres because a Larmor radius in
# a tokamak is of that order. The conversion is done here, at the edge, only to draw.
_METRES_TO_MILLIMETRES = 1.0e3

# Momenta are computed as proper velocities [m/s] and shown as p c in MeV, the unit
# of the runaway literature: p c [J] = m u c, and 1 MeV = 1e6 e J.
_JOULES_PER_MEV = 1.0e6 * e

_LINE_COLORS = [
    "royalblue",
    "crimson",
    "limegreen",
    "darkorange",
    "mediumpurple",
    "deepskyblue",
    "black",
]

# One line style per pusher, so that curves lying on top of each other stay visible.
_LINE_STYLES = ("-", "--", ":", "-.")


def figure_style() -> dict[RcKeyType, Any]:
    """Return the matplotlib settings shared by all the figures of the project."""
    return {
        # Square figure. Layout by constrained_layout, so that the export stays square.
        "figure.figsize": (6.0, 6.0),
        "figure.constrained_layout.use": True,
        # text large enough for a printed article.
        "font.family": "serif",
        "font.serif": ["STIXGeneral", "DejaVu Serif"],
        "mathtext.fontset": "stix",
        "axes.labelsize": 16,
        "axes.titlesize": 16,
        "xtick.labelsize": 14,
        "ytick.labelsize": 14,
        "legend.fontsize": 14,
        # full frame, ticks inside on the left and right sides, no grid.
        "axes.spines.top": True,
        "axes.spines.right": True,
        "axes.linewidth": 1.0,
        "axes.grid": False,
        "xtick.direction": "in",
        "ytick.direction": "in",
        "xtick.top": True,
        "ytick.left": True,
        "ytick.right": True,
        "legend.frameon": False,
        "lines.linewidth": 1.8,
        "axes.prop_cycle": cycler(color=_LINE_COLORS),
        # Export.
        "savefig.dpi": 300,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    }


def save_figure(figure: Figure, path: str | Path) -> tuple[Path, Path]:
    """
    Save a figure twice, as a PNG and as a PDF, with the style of the project.

    Parameters
    ----------
    figure : matplotlib.figure.Figure
        The figure to save. It is not modified.
    path : str or pathlib.Path
        File name without extension, for example ``"results/orbit"``. An extension
        ``.png`` or ``.pdf`` is removed, so ``"orbit.png"`` and ``"orbit"`` give the
        same files. Any other dot is kept: ``"run.v2"`` gives ``run.v2.png``. The
        folder must already exist.

    Returns
    -------
    png_path, pdf_path : pathlib.Path
        The two files written. The PNG is a raster image (300 dpi), for the screen
        and for slides; the PDF is vector graphics, for the article.

    Raises
    ------
    TypeError
        If ``figure`` is not a matplotlib ``Figure``.

    Notes
    -----
    The saving settings of ``figure_style`` (300 dpi, embedded TrueType fonts in
    the PDF) are applied here, at the moment of writing, and not when the figure is
    drawn: matplotlib reads them at ``savefig`` time. So the export is the same
    whether the figure was drawn inside a ``figure_style`` block or not. Layout and
    size are those the figure already has.
    """
    if not isinstance(figure, Figure):
        raise TypeError(
            f"figure must be a matplotlib Figure, got {type(figure).__name__}."
        )
    path = Path(path)
    if path.suffix in (".png", ".pdf"):
        path = path.with_suffix("")
    png_path = path.with_name(path.name + ".png")
    pdf_path = path.with_name(path.name + ".pdf")

    with plt.rc_context(figure_style()):
        figure.savefig(png_path)
        figure.savefig(pdf_path)
    return png_path, pdf_path


def orbit_xy(
    result_dir: str | Path,
    pusher: str | Sequence[str] | None = None,
    *,
    show: bool = True,
    save_path: str | Path | None = None,
) -> Figure:
    """
    Draw the trajectory of each particle in the (x, y) plane.

    The figure uses the common style of ``figure_style``. To compare schemes, give
    the folder where several pushers were run (see ``run.pusher``): each pusher is
    drawn on the same axes, with its own line style, and the legend names it.
    Trajectories that lie on top of each other, as is expected when the schemes agree,
    stay distinguishable thanks to the line styles.

    Parameters
    ----------
    result_dir : str or pathlib.Path
        Results folder given by ``run.pusher``.
    pusher : str or sequence of str, optional
        Which pusher(s) of the folder to draw, for example ``"boris"`` or
        ``["boris", "vay"]``. Default: all the pushers stored in the folder.
    show : bool, optional
        Open the figure in a window. Default ``True``. Use ``False`` in a script that
        only saves the figure.
    save_path : str or pathlib.Path, optional
        If given, the figure is also saved as a PNG and as a PDF, with
        ``save_figure``. This is a file name without extension: ``"orbit"`` writes
        ``orbit.png`` and ``orbit.pdf`` (``"orbit.png"`` gives the same two files).

    Returns
    -------
    matplotlib.figure.Figure
        The figure, square, 6 x 6 in.

    Raises
    ------
    FileNotFoundError
        If no pusher is given and the folder holds no trajectories.
    ValueError
        If a pusher is not in the folder. The message lists those it has.
    """
    if pusher is None:
        names = list_results(result_dir)
        if not names:
            raise FileNotFoundError(
                f"No trajectories in {result_dir}: has a simulation been run in this "
                "folder?"
            )
    elif isinstance(pusher, str):
        names = [pusher]
    else:
        names = list(pusher)
    results = {name: read_result(result_dir, name) for name in names}

    # Size, fonts and layout come from the common style; the layout is done by
    # constrained_layout, so there is no figure.tight_layout() here.
    with plt.rc_context(figure_style()):
        figure, axes = plt.subplots()
        for k, (name, result) in enumerate(results.items()):
            x = result["x"].values * _METRES_TO_MILLIMETRES  # (n_saved, n_part, 3)
            n_part = x.shape[1]
            for i in range(n_part):
                if len(results) == 1:
                    label = f"particle {i}"
                elif n_part == 1:
                    label = name
                else:
                    label = f"{name}, particle {i}"
                (line,) = axes.plot(
                    x[:, i, 0],
                    x[:, i, 1],
                    linestyle=_LINE_STYLES[k % len(_LINE_STYLES)],
                    label=label,
                )
                axes.plot(x[0, i, 0], x[0, i, 1], "o", color=line.get_color())
        axes.set_xlabel("x [mm]")
        axes.set_ylabel("y [mm]")
        dt = next(iter(results.values())).attrs["dt"]  # the same for the whole folder
        if len(results) == 1:
            title = f"Orbit seen from above, pusher {names[0]!r}, dt = {dt:.3g} s"
        else:
            title = f"Orbit seen from above, dt = {dt:.3g} s"
        axes.set_title(title)
        # axes.set_aspect("equal")  # a circle must look like a circle
        if len(axes.get_legend_handles_labels()[0]) > 1:
            axes.legend()

    if save_path is not None:
        save_figure(figure, save_path)
    if show:
        plt.show()
    return figure


def orbit_xz(
    result_dir: str | Path,
    pusher: str | Sequence[str] | None = None,
    *,
    show: bool = True,
    save_path: str | Path | None = None,
) -> Figure:
    """
    Draw the trajectory of each particle in the (x, z) plane.

    The figure uses the common style of ``figure_style``. To compare schemes, give
    the folder where several pushers were run (see ``run.pusher``): each pusher is
    drawn on the same axes, with its own line style, and the legend names it.
    Trajectories that lie on top of each other, as is expected when the schemes agree,
    stay distinguishable thanks to the line styles.

    Parameters
    ----------
    result_dir : str or pathlib.Path
        Results folder given by ``run.pusher``.
    pusher : str or sequence of str, optional
        Which pusher(s) of the folder to draw, for example ``"boris"`` or
        ``["boris", "vay"]``. Default: all the pushers stored in the folder.
    show : bool, optional
        Open the figure in a window. Default ``True``. Use ``False`` in a script that
        only saves the figure.
    save_path : str or pathlib.Path, optional
        If given, the figure is also saved as a PNG and as a PDF, with
        ``save_figure``. This is a file name without extension: ``"orbit"`` writes
        ``orbit.png`` and ``orbit.pdf`` (``"orbit.png"`` gives the same two files).

    Returns
    -------
    matplotlib.figure.Figure
        The figure, square, 6 x 6 in.

    Raises
    ------
    FileNotFoundError
        If no pusher is given and the folder holds no trajectories.
    ValueError
        If a pusher is not in the folder. The message lists those it has.
    """
    if pusher is None:
        names = list_results(result_dir)
        if not names:
            raise FileNotFoundError(
                f"No trajectories in {result_dir}: has a simulation been run in this "
                "folder?"
            )
    elif isinstance(pusher, str):
        names = [pusher]
    else:
        names = list(pusher)
    results = {name: read_result(result_dir, name) for name in names}

    # Size, fonts and layout come from the common style; the layout is done by
    # constrained_layout, so there is no figure.tight_layout() here.
    with plt.rc_context(figure_style()):
        figure, axes = plt.subplots()
        for k, (name, result) in enumerate(results.items()):
            x = result["x"].values * _METRES_TO_MILLIMETRES  # (n_saved, n_part, 3)
            n_part = x.shape[1]
            for i in range(n_part):
                if len(results) == 1:
                    label = f"particle {i}"
                elif n_part == 1:
                    label = name
                else:
                    label = f"{name}, particle {i}"
                (line,) = axes.plot(
                    x[:, i, 0],
                    x[:, i, 2],
                    linestyle=_LINE_STYLES[k % len(_LINE_STYLES)],
                    label=label,
                )
                axes.plot(x[0, i, 0], x[0, i, 2], "o", color=line.get_color())
        axes.set_xlabel("x [mm]")
        axes.set_ylabel("z [mm]")
        dt = next(iter(results.values())).attrs["dt"]  # the same for the whole folder
        if len(results) == 1:
            title = f"Orbit seen from above, pusher {names[0]!r}, dt = {dt:.3g} s"
        else:
            title = f"Orbit seen from above, dt = {dt:.3g} s"
        axes.set_title(title)
        axes.set_aspect("equal")  # a circle must look like a circle
        if len(axes.get_legend_handles_labels()[0]) > 1:
            axes.legend()

    if save_path is not None:
        save_figure(figure, save_path)
    if show:
        plt.show()
    return figure


def orbit_3d(
    result_dir: str | Path,
    pusher: str | Sequence[str] | None = None,
    *,
    r0: float | None = None,
    a: float | None = None,
    show: bool = True,
    save_path: str | Path | None = None,
) -> Figure:
    """
    Draw the 3D trajectory of each particle, optionally with a reference torus.

    The figure uses the common style of ``figure_style``. To compare schemes, give
    the folder where several pushers were run (see ``run.pusher``): each pusher is
    drawn on the same axes, with its own line style, and the legend names it.

    Parameters
    ----------
    result_dir : str or pathlib.Path
        Results folder given by ``run.pusher``.
    pusher : str or sequence of str, optional
        Which pusher(s) of the folder to draw, for example ``"boris"`` or
        ``["boris", "vay"]``. Default: all the pushers stored in the folder.
    r0 : float, optional
        Major radius [m] of a reference torus surface to draw alongside the
        trajectories, semi-transparent. Omit to draw the trajectories alone.
    a : float, optional
        Minor radius [m] of the reference torus. Required if ``r0`` is given.
    show : bool, optional
        Open the figure in a window. Default ``True``. Use ``False`` in a script that
        only saves the figure.
    save_path : str or pathlib.Path, optional
        If given, the figure is also saved as a PNG and as a PDF, with
        ``save_figure``. This is a file name without extension: ``"orbit"`` writes
        ``orbit.png`` and ``orbit.pdf`` (``"orbit.png"`` gives the same two files).

    Returns
    -------
    matplotlib.figure.Figure
        The figure, square, 7 x 7 in.

    Raises
    ------
    FileNotFoundError
        If no pusher is given and the folder holds no trajectories.
    ValueError
        If a pusher is not in the folder, or ``a`` is missing while ``r0`` is given.
    """
    if pusher is None:
        names = list_results(result_dir)
        if not names:
            raise FileNotFoundError(
                f"No trajectories in {result_dir}: has a simulation been run in this "
                "folder?"
            )
    elif isinstance(pusher, str):
        names = [pusher]
    else:
        names = list(pusher)
    results = {name: read_result(result_dir, name) for name in names}

    if r0 is not None and a is None:
        raise ValueError("a (minor radius) is required when r0 is given")

    with plt.rc_context(figure_style()):
        figure = plt.figure(figsize=(7, 7))
        axes = figure.add_subplot(111, projection="3d")

        # --- Optional reference torus surface, drawn first so it sits behind ---
        if r0 is not None:
            r0_mm = r0 * _METRES_TO_MILLIMETRES
            a_mm = a * _METRES_TO_MILLIMETRES
            theta_grid, phi_grid = np.meshgrid(
                np.linspace(0, 2 * np.pi, 60), np.linspace(0, 2 * np.pi, 100)
            )
            big_r = r0_mm + a_mm * np.cos(theta_grid)
            xt = big_r * np.cos(phi_grid)
            yt = big_r * np.sin(phi_grid)
            zt = a_mm * np.sin(theta_grid)
            axes.plot_surface(
                xt, yt, zt, color="lightgray", alpha=0.25, linewidth=0, shade=True
            )

        # --- Trajectories, same convention as orbit_xz ---
        all_x = []
        for k, (name, result) in enumerate(results.items()):
            x = result["x"].values * _METRES_TO_MILLIMETRES  # (n_saved, n_part, 3)
            all_x.append(x)
            n_part = x.shape[1]
            for i in range(n_part):
                if len(results) == 1:
                    label = f"particle {i}"
                elif n_part == 1:
                    label = name
                else:
                    label = f"{name}, particle {i}"
                (line,) = axes.plot(
                    x[:, i, 0],
                    x[:, i, 1],
                    x[:, i, 2],
                    linestyle=_LINE_STYLES[k % len(_LINE_STYLES)],
                    label=label,
                )
                axes.plot(
                    [x[0, i, 0]],
                    [x[0, i, 1]],
                    [x[0, i, 2]],
                    "o",
                    color=line.get_color(),
                )
        theta = np.linspace(0, 2 * np.pi, 100)
        axes.plot(r0_mm * np.cos(theta), r0_mm * np.sin(theta), "--")
        axes.set_xlabel("x [mm]")
        axes.set_ylabel("y [mm]")
        axes.set_zlabel("z [mm]")
        dt = next(iter(results.values())).attrs["dt"]
        if len(results) == 1:
            title = f"3D orbit, pusher {names[0]!r}, dt = {dt:.3g} s"
        else:
            title = f"3D orbit, dt = {dt:.3g} s"
        axes.set_title(title)

        # Equal aspect ratio: matplotlib 3D does not do this automatically.
        # stacked = np.concatenate(all_x, axis=0).reshape(-1, 3)
        # centre = (stacked.max(axis=0) + stacked.min(axis=0)) / 2
        # half_range = (stacked.max(axis=0) - stacked.min(axis=0)).max() / 2
        # axes.set_xlim(centre[0] - half_range, centre[0] + half_range)
        # axes.set_ylim(centre[1] - half_range, centre[1] + half_range)
        # axes.set_zlim(centre[2] - half_range, centre[2] + half_range)

        if len(axes.get_legend_handles_labels()[0]) > 1:
            axes.legend()

    if save_path is not None:
        save_figure(figure, save_path)
    if show:
        plt.show()
    return figure


def energy_error(
    result_dir: str | Path,
    pusher: str | Sequence[str] | None = None,
    *,
    show: bool = True,
    save_path: str | Path | None = None,
) -> Figure:
    """
    Draw the relative change of kinetic energy ``|K(t) - K(0)| / K(0)`` in time.

    The energy is computed by ``diagnostics.energy_error`` from the velocities ``u``
    of the results folder, which are at the times of the positions. The vertical
    axis is logarithmic, so that a rounding-level error (Higuera-Cary, Vay in an
    E x B drift) and an error of order 1 (Boris in the same case) can be read on the
    same figure. In a static B alone, and at the exact E x B drift velocity, ``K``
    is constant: the curve is then the error of the scheme.

    Values below the float64 rounding unit ``eps = 2.2e-16`` (including the exact 0
    at ``t = 0``) are drawn at ``eps``, on the grey line labelled "float64
    rounding": a logarithmic axis cannot show 0, and a curve would otherwise vanish.

    Parameters
    ----------
    result_dir : str or pathlib.Path
        Results folder given by ``run.pusher``.
    pusher : str or sequence of str, optional
        Which pusher(s) of the folder to draw, for example ``"boris"`` or
        ``["boris", "vay"]``. Default: all the pushers stored in the folder.
    show : bool, optional
        Open the figure in a window. Default ``True``.
    save_path : str or pathlib.Path, optional
        If given, the figure is also saved as a PNG and as a PDF, with
        ``save_figure`` (file name without extension).

    Returns
    -------
    matplotlib.figure.Figure
        The figure, square, 6 x 6 in.

    Raises
    ------
    FileNotFoundError
        If no pusher is given and the folder holds no trajectories.
    ValueError
        If a pusher is not in the folder, its result was written by an older
        version of the code (see ``io.read_result``), or a particle starts at rest.
    """
    if pusher is None:
        names = list_results(result_dir)
        if not names:
            raise FileNotFoundError(
                f"No trajectories in {result_dir}: has a simulation been run in this "
                "folder?"
            )
    elif isinstance(pusher, str):
        names = [pusher]
    else:
        names = list(pusher)
    results = {name: read_result(result_dir, name) for name in names}

    rounding = np.finfo(np.float64).eps
    with plt.rc_context(figure_style()):
        figure, axes = plt.subplots()
        for k, (name, result) in enumerate(results.items()):
            error = diagnostics.energy_error(result["u"].values)
            shown = np.maximum(np.abs(error), rounding)  # (n_saved, n_part)
            time = result["time"].values
            n_part = shown.shape[1]
            for i in range(n_part):
                if len(results) == 1:
                    label = f"particle {i}"
                elif n_part == 1:
                    label = name
                else:
                    label = f"{name}, particle {i}"
                axes.plot(
                    time,
                    shown[:, i],
                    linestyle=_LINE_STYLES[k % len(_LINE_STYLES)],
                    label=label,
                )
        axes.axhline(rounding, color="0.6", linewidth=1.0, label="float64 rounding")
        axes.set_yscale("log")
        axes.set_xlabel("t [s]")
        axes.set_ylabel(r"$|K(t) - K(0)|\,/\,K(0)$")
        dt = next(iter(results.values())).attrs["dt"]  # the same for the whole folder
        if len(results) == 1:
            title = f"Energy error, pusher {names[0]!r}, dt = {dt:.3g} s"
        else:
            title = f"Energy error, dt = {dt:.3g} s"
        axes.set_title(title)
        axes.legend()

    if save_path is not None:
        save_figure(figure, save_path)
    if show:
        plt.show()
    return figure


def drift_error(
    result_dirs: Sequence[str | Path],
    pusher: str | Sequence[str] | None = None,
    *,
    show: bool = True,
    save_path: str | Path | None = None,
) -> Figure:
    """
    Draw the error on the E x B drift velocity against the drift proper velocity.

    Each results folder is one run of a scan: uniform, crossed ``E`` and ``B``, one
    value of the drift speed per folder. The exact drift ``v_E = E x B / B^2`` is
    computed from the ``config.yaml`` of the folder, the drift of the scheme by
    ``diagnostics.drift_velocity`` from the stored positions. The point drawn is
    ``|v_d - v_E| / |v_E|`` against ``u_E / c = gamma_E v_E / c``, both axes
    logarithmic, one curve per pusher, one point per folder and per particle.

    The measure is meaningful for particles launched at the drift velocity: a
    gyrating particle adds a bias (see ``diagnostics.drift_velocity``). As in
    ``energy_error``, values below the float64 rounding unit are drawn on the grey
    "float64 rounding" line.

    Parameters
    ----------
    result_dirs : sequence of str or pathlib.Path
        Results folders given by ``run.pusher``, one per point of the scan, in any
        order.
    pusher : str or sequence of str, optional
        Which pusher(s) to draw. Default: all the pushers of the first folder.
    show : bool, optional
        Open the figure in a window. Default ``True``.
    save_path : str or pathlib.Path, optional
        If given, the figure is also saved as a PNG and as a PDF, with
        ``save_figure`` (file name without extension).

    Returns
    -------
    matplotlib.figure.Figure
        The figure, square, 6 x 6 in.

    Raises
    ------
    ValueError
        If ``result_dirs`` is empty, if a folder does not hold a uniform field with
        ``E`` perpendicular to ``B`` and ``|E| < c |B|`` (no E x B drift then), or if
        a pusher is not in a folder.
    """
    folders = [Path(directory) for directory in result_dirs]
    if not folders:
        raise ValueError("result_dirs is empty: give at least one results folder")
    if pusher is None:
        names = list_results(folders[0])
    elif isinstance(pusher, str):
        names = [pusher]
    else:
        names = list(pusher)

    # For each pusher, the points (u_E / c, relative drift error) of every folder.
    points: dict[str, list[tuple[float, float]]] = {name: [] for name in names}
    for directory in folders:
        field = config.load_config(directory / CONFIG_FILENAME).field
        if field.kind != "uniform_cartesian":
            raise ValueError(
                f"{directory} holds a {field.kind!r} field: the drift error needs "
                "uniform crossed E and B"
            )
        e_vector = np.array(field.e_field)
        b_vector = np.array(field.b_field)
        v_drift = np.cross(e_vector, b_vector) / (b_vector @ b_vector)
        speed_over_c = np.linalg.norm(v_drift) / c
        if abs(e_vector @ b_vector) > 0.0 or not 0.0 < speed_over_c < 1.0:
            raise ValueError(
                f"{directory}: E must be perpendicular to B, non-zero, and "
                f"|E| < c |B| for an E x B drift; got E = {e_vector} V/m, "
                f"B = {b_vector} T"
            )
        u_drift_over_c = speed_over_c / np.sqrt(1.0 - speed_over_c**2)
        for name in names:
            result = read_result(directory, name)
            v_scheme = diagnostics.drift_velocity(
                result["time"].values, result["x"].values
            )
            for v_particle in v_scheme:
                error = np.linalg.norm(v_particle - v_drift) / np.linalg.norm(v_drift)
                points[name].append((u_drift_over_c, error))

    rounding = np.finfo(np.float64).eps
    with plt.rc_context(figure_style()):
        figure, axes = plt.subplots()
        for k, name in enumerate(names):
            u_over_c, error = np.array(sorted(points[name])).T
            axes.plot(
                u_over_c,
                np.maximum(error, rounding),
                marker="o",
                linestyle=_LINE_STYLES[k % len(_LINE_STYLES)],
                label=name,
            )
        axes.axhline(rounding, color="0.6", linewidth=1.0, label="float64 rounding")
        axes.set_xscale("log")
        axes.set_yscale("log")
        axes.set_xlabel(r"$u_E / c = \gamma_E\, v_E / c$")
        axes.set_ylabel(r"$|v_d - v_E|\,/\,|v_E|$")
        axes.set_title(r"$E \times B$ drift error")
        axes.legend()

    if save_path is not None:
        save_figure(figure, save_path)
    if show:
        plt.show()
    return figure


def _read_selected(
    result_dir: str | Path, pusher: str | Sequence[str] | None
) -> dict[str, Any]:
    """Read the results of the chosen pushers, in order: ``{name: Dataset}``."""
    if pusher is None:
        names = list_results(result_dir)
        if not names:
            raise FileNotFoundError(
                f"No trajectories in {result_dir}: has a simulation been run in this "
                "folder?"
            )
    elif isinstance(pusher, str):
        names = [pusher]
    else:
        names = list(pusher)
    return {name: read_result(result_dir, name) for name in names}


def _curve_label(name: str, particle: int, n_pushers: int, n_part: int) -> str:
    """Legend of one curve: the pusher, the particle, or both."""
    if n_pushers == 1:
        return f"particle {particle}"
    if n_part == 1:
        return name
    return f"{name}, particle {particle}"


def phase_error(
    result_dir: str | Path,
    pusher: str | Sequence[str] | None = None,
    *,
    show: bool = True,
    save_path: str | Path | None = None,
) -> Figure:
    """
    Draw the gyration phase error in a uniform magnetic field, with its prediction.

    This is the test of Ripperda et al. (2018, Sec. 3.1.2, Fig. 4). For each pusher
    the measured error ``diagnostics.gyration_phase_error`` (signed, in radians;
    negative is a lag for an electron) is drawn against ``t / T_c``, the number of
    exact gyrations of the particle. On top of it, a thin black dashed line is the
    exact prediction ``diagnostics.predicted_gyration_phase_error`` of the same
    scheme, which follows from its rotation angle per step alone. The two must
    coincide to rounding: a visible gap means that the scheme or the analysis is
    wrong.

    Parameters
    ----------
    result_dir : str or pathlib.Path
        Results folder of a run in a uniform ``B`` without electric field, saved
        often enough (``diagnostics.gyration_phase_error`` checks it).
    pusher : str or sequence of str, optional
        Which pusher(s) to draw. Default: all the pushers of the folder.
    show : bool, optional
        Open the figure in a window. Default ``True``.
    save_path : str or pathlib.Path, optional
        If given, the figure is also saved as a PNG and as a PDF (no extension).

    Returns
    -------
    matplotlib.figure.Figure
        The figure, square, 6 x 6 in.

    Raises
    ------
    ValueError
        If the field of the folder is not a uniform ``B`` without ``E``, or the
        states were saved too far apart to follow the phase.
    """
    run_config = config.load_config(Path(result_dir) / CONFIG_FILENAME)
    field = run_config.field
    if (
        field.kind != "uniform_cartesian"
        or any(field.e_field)
        or not any(field.b_field)
    ):
        raise ValueError(
            f"{result_dir}: the phase error needs a uniform B and no E, got {field}"
        )
    results = _read_selected(result_dir, pusher)
    b_vector = np.array(field.b_field)
    q, m, dt = run_config.q, run_config.m, run_config.dt
    u0 = np.array(run_config.initial.u0)  # (n_part, 3), exact initial state
    gamma0 = np.sqrt(1.0 + np.sum(u0**2, axis=1) / c**2)
    omega = abs(q) * np.linalg.norm(b_vector) / (gamma0 * m)  # (n_part,) [rad/s]
    period = 2.0 * np.pi / omega  # exact gyration period of each particle [s]

    with plt.rc_context(figure_style()):
        figure, axes = plt.subplots()
        prediction_label: str | None = "prediction"
        for k, (name, result) in enumerate(results.items()):
            time = result["time"].values
            error = diagnostics.gyration_phase_error(
                time, result["u"].values, u0, b_vector, q, m
            )
            predicted = None
            if name in ("boris", "vay", "higuera_cary"):
                predicted = diagnostics.predicted_gyration_phase_error(
                    name, time, u0, b_vector, dt, q, m
                )
            for i in range(error.shape[1]):
                turns = time / period[i]
                axes.plot(
                    turns,
                    error[:, i],
                    linestyle=_LINE_STYLES[k % len(_LINE_STYLES)],
                    linewidth=3.0,
                    label=_curve_label(name, i, len(results), error.shape[1]),
                )
                if predicted is not None:
                    axes.plot(
                        turns,
                        predicted[:, i],
                        color="black",
                        linestyle="--",
                        linewidth=0.8,
                        label=prediction_label,
                    )
                    prediction_label = None  # one legend entry for all of them
        axes.axhline(0.0, color="0.6", linewidth=1.0)
        axes.set_xlabel(r"$t\,/\,T_c$")
        axes.set_ylabel(r"$\theta_c - \theta_{c,\mathrm{exact}}$ [rad]")
        theta = omega[0] * dt  # exact angle per step, of particle 0
        axes.set_title(rf"Gyration phase error, $\omega_c \Delta t$ = {theta:.3g}")
        axes.legend()

    if save_path is not None:
        save_figure(figure, save_path)
    if show:
        plt.show()
    return figure


def mirror_trajectory(
    result_dir: str | Path,
    pusher: str | Sequence[str] | None = None,
    *,
    show: bool = True,
    save_path: str | Path | None = None,
) -> Figure:
    """
    Draw the position along the axis of a magnetic bottle, ``z(t) / L``.

    This is Fig. 10 of Ripperda et al. (2018, Sec. 3.2.1). The dashed lines are the
    mirror points predicted by the conservation of the magnetic moment, where the
    field reaches ``B_m = |B(x0)| |u0|^2 / u_perp0^2``:
    ``z_max = L sqrt(B_m / B0 - 1)`` (their Eq. 61), computed from the initial
    velocity of the configuration and the stored field at the start.

    Parameters
    ----------
    result_dir : str or pathlib.Path
        Results folder of a run in the ``mirror`` field.
    pusher : str or sequence of str, optional
        Which pusher(s) to draw. Default: all the pushers of the folder.
    show : bool, optional
        Open the figure in a window. Default ``True``.
    save_path : str or pathlib.Path, optional
        If given, the figure is also saved as a PNG and as a PDF (no extension).

    Returns
    -------
    matplotlib.figure.Figure
        The figure, square, 6 x 6 in.

    Raises
    ------
    ValueError
        If the field of the folder is not a ``mirror``.
    """
    run_config = config.load_config(Path(result_dir) / CONFIG_FILENAME)
    if run_config.field.kind != "mirror":
        raise ValueError(f"{result_dir}: not a magnetic bottle ({run_config.field})")
    length, b0 = run_config.field.length, run_config.field.b0
    results = _read_selected(result_dir, pusher)

    # Predicted mirror point of each particle, from mu = const.
    u0 = np.array(run_config.initial.u0)
    b_start = next(iter(results.values()))["b_field"].values[0]  # (n_part, 3)
    b_start_norm = np.linalg.norm(b_start, axis=1)
    u_parallel = np.sum(u0 * b_start, axis=1) / b_start_norm
    sin_pitch_squared = 1.0 - u_parallel**2 / np.sum(u0**2, axis=1)

    with plt.rc_context(figure_style()):
        figure, axes = plt.subplots()
        for k, (name, result) in enumerate(results.items()):
            z = result["x"].values[:, :, 2] / length
            for i in range(z.shape[1]):
                axes.plot(
                    result["time"].values,
                    z[:, i],
                    linestyle=_LINE_STYLES[k % len(_LINE_STYLES)],
                    label=_curve_label(name, i, len(results), z.shape[1]),
                )
        for i, sin2 in enumerate(sin_pitch_squared):
            if sin2 > 0.0:  # a particle along B has no mirror point
                z_max = np.sqrt(b_start_norm[i] / sin2 / b0 - 1.0)
                label = r"predicted $\pm z_{max}$" if i == 0 else None
                for sign in (1.0, -1.0):
                    axes.axhline(sign * z_max, color="0.6", linestyle="--", label=label)
                    label = None
        axes.set_xlabel("t [s]")
        axes.set_ylabel(r"$z\,/\,L$")
        axes.set_title(f"Magnetic bottle, L = {length:.3g} m, B0 = {b0:.3g} T")
        axes.legend()

    if save_path is not None:
        save_figure(figure, save_path)
    if show:
        plt.show()
    return figure


def magnetic_moment_error(
    result_dir: str | Path,
    pusher: str | Sequence[str] | None = None,
    *,
    show: bool = True,
    save_path: str | Path | None = None,
) -> Figure:
    """
    Draw the relative change of the magnetic moment ``|mu(t) / mu(0) - 1|``.

    This is Fig. 12 of Ripperda et al. (2018, Sec. 3.2.1). ``mu = p_perp^2 /
    (2 m |B|)`` is computed by ``diagnostics.magnetic_moment`` from the stored
    velocity ``u`` and the stored field at the particle, both at the times of the
    positions. It is an adiabatic invariant: its relative change oscillates
    with an amplitude of order ``eps = r_L / L`` in the exact motion too, so only
    a drift from cycle to cycle is an error of the scheme. Logarithmic axis, values
    below the float64 rounding unit drawn on the grey line.

    Parameters
    ----------
    result_dir : str or pathlib.Path
        Results folder given by ``run.pusher``.
    pusher : str or sequence of str, optional
        Which pusher(s) to draw. Default: all the pushers of the folder.
    show : bool, optional
        Open the figure in a window. Default ``True``.
    save_path : str or pathlib.Path, optional
        If given, the figure is also saved as a PNG and as a PDF (no extension).

    Returns
    -------
    matplotlib.figure.Figure
        The figure, square, 6 x 6 in.

    Raises
    ------
    ValueError
        If a result has no stored magnetic field (written before it was stored:
        run it again), or was written by an older version of the code (see
        ``io.read_result``).
    """
    mass = config.load_config(Path(result_dir) / CONFIG_FILENAME).m
    results = _read_selected(result_dir, pusher)

    rounding = np.finfo(np.float64).eps
    with plt.rc_context(figure_style()):
        figure, axes = plt.subplots()
        for k, (name, result) in enumerate(results.items()):
            if "b_field" not in result:
                raise ValueError(
                    f"The result of {name!r} in {result_dir} has no b_field: run it "
                    "again to get its magnetic moment."
                )
            mu = diagnostics.magnetic_moment(
                result["u"].values, result["b_field"].values, mass
            )
            shown = np.maximum(np.abs(mu / mu[0] - 1.0), rounding)
            for i in range(shown.shape[1]):
                axes.plot(
                    result["time"].values,
                    shown[:, i],
                    linestyle=_LINE_STYLES[k % len(_LINE_STYLES)],
                    label=_curve_label(name, i, len(results), shown.shape[1]),
                )
        axes.axhline(rounding, color="0.6", linewidth=1.0, label="float64 rounding")
        axes.set_yscale("log")
        axes.set_xlabel("t [s]")
        axes.set_ylabel(r"$|\mu(t)\,/\,\mu(0) - 1|$")
        axes.set_title("Magnetic moment")
        axes.legend()

    if save_path is not None:
        save_figure(figure, save_path)
    if show:
        plt.show()
    return figure


def phase_space_volume(
    result_dir: str | Path,
    pusher: str | Sequence[str] | None = None,
    *,
    show: bool = True,
    save_path: str | Path | None = None,
) -> Figure:
    """
    Draw the change of phase-space volume of the run, ``|ln(V(t) / V(0))|``.

    ``ln V(t_n) = sum over k < n of ln det J_k``, with ``det J_k`` the Jacobian
    determinant of step ``k`` measured during the run (``run.pusher(...,
    phase_space_volume=True)``, which needs ``save_every: 1``). Boris and
    Higuera-Cary preserve the volume exactly (Higuera & Cary 2017, Sec. V); the
    curve is then the noise of the finite differences, about ``2e-10`` per step.
    Vay changes it by ``J(u^0) / J(u^n)`` in a uniform field: a bounded
    change, not an accumulation. Logarithmic axis, values below the float64
    rounding unit drawn on the grey line.

    Parameters
    ----------
    result_dir : str or pathlib.Path
        Results folder given by ``run.pusher(..., phase_space_volume=True)``.
    pusher : str or sequence of str, optional
        Which pusher(s) to draw. Default: all the pushers of the folder.
    show : bool, optional
        Open the figure in a window. Default ``True``.
    save_path : str or pathlib.Path, optional
        If given, the figure is also saved as a PNG and as a PDF (no extension).

    Returns
    -------
    matplotlib.figure.Figure
        The figure, square, 6 x 6 in.

    Raises
    ------
    ValueError
        If a result has no stored step determinants, or was not saved at every step.
    """
    results = _read_selected(result_dir, pusher)
    rounding = np.finfo(np.float64).eps
    with plt.rc_context(figure_style()):
        figure, axes = plt.subplots()
        for k, (name, result) in enumerate(results.items()):
            if "step_jacobian_determinant" not in result:
                raise ValueError(
                    f"The result of {name!r} in {result_dir} has no step "
                    "determinants: run it with run.pusher(..., "
                    "phase_space_volume=True)."
                )
            time = result["time"].values
            if not np.allclose(np.diff(time), result.attrs["dt"], rtol=1e-9):
                raise ValueError(
                    f"The result of {name!r} was not saved at every step: the volume "
                    "change of the run is the product over all the steps."
                )
            log_det = np.log(result["step_jacobian_determinant"].values)
            # V(t_0) = 1; V(t_n) = product of the determinants of steps 0 .. n-1.
            log_volume = np.zeros_like(log_det)
            log_volume[1:] = np.cumsum(log_det[:-1], axis=0)
            shown = np.maximum(np.abs(log_volume), rounding)
            for i in range(shown.shape[1]):
                axes.plot(
                    time,
                    shown[:, i],
                    linestyle=_LINE_STYLES[k % len(_LINE_STYLES)],
                    label=_curve_label(name, i, len(results), shown.shape[1]),
                )
        # axes.axhline(rounding, color="0.6", linewidth=1.0, label="float64 rounding")
        axes.set_yscale("log")
        axes.set_xlabel("t [s]")
        axes.set_ylabel(r"$|\ln(V(t)\,/\,V(0))|$")
        axes.set_title("Phase-space volume")
        axes.legend()

    if save_path is not None:
        save_figure(figure, save_path)
    if show:
        plt.show()
    return figure


def _log_edges(centres: NDArray[np.float64]) -> NDArray[np.float64]:
    """Cell edges around sorted positive centres, at geometric midpoints."""
    log_c = np.log10(centres)
    inner = 0.5 * (log_c[1:] + log_c[:-1])
    edges = np.concatenate(
        [[2.0 * log_c[0] - inner[0]], inner, [2.0 * log_c[-1] - inner[-1]]]
    )
    return 10.0**edges


def _linear_edges(centres: NDArray[np.float64]) -> NDArray[np.float64]:
    """Cell edges around sorted centres, at arithmetic midpoints."""
    inner = 0.5 * (centres[1:] + centres[:-1])
    return np.concatenate(
        [[2.0 * centres[0] - inner[0]], inner, [2.0 * centres[-1] - inner[-1]]]
    )


def phase_error_map(
    result_dirs: Sequence[str | Path],
    pusher: str,
    *,
    show: bool = True,
    save_path: str | Path | None = None,
) -> Figure:
    """
    Map the gyration phase error per turn against ``u / c`` and the pitch angle.

    Each results folder is one run of a scan in a uniform ``B`` without ``E``: all
    its particles have the same Lorentz factor, and different pitch angles ``psi``
    (the same set in every folder). For every particle, the drift of the phase
    error per exact turn (``diagnostics.phase_error_per_turn``) is divided by the
    Boris lag per turn at the same angle per step,
    ``(2 pi / theta) |2 arctan(theta / 2) - theta|``. This ratio is -1 everywhere for
    Boris and Vay, and does not depend on ``theta`` to leading order: the map
    compares schemes, not time steps.

    The colour is the measured ratio (blue: lag, red: lead, white: no drift, scale
    fixed to [-1, 1]). The solid black line is the zero of the measured ratio, the
    dashed one the zero predicted by ``diagnostics.rotation_angle_per_step``; for
    Higuera-Cary it tends to ``sin^2 psi = 2 / 3`` (54.7 degrees) at large
    ``gamma``.

    Parameters
    ----------
    result_dirs : sequence of str or pathlib.Path
        At least two results folders, one per Lorentz factor, in any order.
    pusher : str
        The pusher to map, for example ``"higuera_cary"``.
    show : bool, optional
        Open the figure in a window. Default ``True``.
    save_path : str or pathlib.Path, optional
        If given, the figure is also saved as a PNG and as a PDF (no extension).

    Returns
    -------
    matplotlib.figure.Figure
        The figure, square, 6 x 6 in.

    Raises
    ------
    ValueError
        If fewer than two folders or two pitch angles are given, a folder is not a
        uniform ``B`` without ``E``, its particles do not share one Lorentz factor,
        or the folders do not hold the same pitch angles.
    """
    folders = [Path(directory) for directory in result_dirs]
    if len(folders) < 2:
        raise ValueError("a map needs at least two results folders (two gammas)")

    speed_values: list[float] = []
    measured_rows: list[NDArray[np.float64]] = []
    predicted: list[NDArray[np.float64]] = []
    pitches: NDArray[np.float64] | None = None
    for directory in folders:
        run_config = config.load_config(directory / CONFIG_FILENAME)
        field = run_config.field
        if (
            field.kind != "uniform_cartesian"
            or any(field.e_field)
            or not any(field.b_field)
        ):
            raise ValueError(f"{directory}: needs a uniform B and no E, got {field}")
        b_vector = np.array(field.b_field)
        q, m, dt = run_config.q, run_config.m, run_config.dt
        u0 = np.array(run_config.initial.u0)
        u_over_c = np.linalg.norm(u0, axis=1) / c
        if not np.allclose(u_over_c, u_over_c[0], rtol=1e-12, atol=0.0):
            raise ValueError(f"{directory}: the particles must share one gamma")
        folder_pitches = np.degrees(diagnostics.pitch_angle(u0, b_vector))
        if pitches is None:
            pitches = folder_pitches
            if pitches.size < 2 or np.any(np.diff(pitches) <= 0.0):
                raise ValueError(
                    f"{directory}: needs at least two particles, in increasing "
                    f"pitch angle; got {pitches} degrees"
                )
        elif not np.allclose(folder_pitches, pitches, rtol=0.0, atol=1e-9):
            raise ValueError(
                f"{directory} holds the pitch angles {folder_pitches} degrees, not "
                f"{pitches} as the first folder: a map needs the same ones"
            )

        gamma = np.sqrt(1.0 + u_over_c[0] ** 2)
        omega = abs(q) * np.linalg.norm(b_vector) / (gamma * m)
        theta = omega * dt  # exact angle per step
        boris_lag_per_step = abs(2.0 * np.arctan(theta / 2.0) - theta)
        boris_lag_per_turn = 2.0 * np.pi / theta * boris_lag_per_step
        period = np.full(u0.shape[0], 2.0 * np.pi / omega)

        result = read_result(directory, pusher)
        time = result["time"].values
        error = diagnostics.gyration_phase_error(
            time, result["u"].values, u0, b_vector, q, m
        )
        per_turn = diagnostics.phase_error_per_turn(time, error, period)
        speed_values.append(u_over_c[0])
        measured_rows.append(per_turn / boris_lag_per_turn)
        if pusher in ("boris", "vay", "higuera_cary"):
            angle = diagnostics.rotation_angle_per_step(pusher, u0, b_vector, dt, q, m)
            exact = np.sign(-q) * theta
            predicted.append(2.0 * np.pi / theta * (angle - exact) / boris_lag_per_turn)

    # Set by the first folder: there are at least two, checked above.
    assert pitches is not None
    order = np.argsort(speed_values)
    speeds = np.array(speed_values)[order]
    measured = np.array(measured_rows)[order]  # (n_gamma, n_pitch)
    # Lag is negative for an electron (Omega > 0); make "lag" negative for any sign.
    sign = np.sign(-config.load_config(folders[0] / CONFIG_FILENAME).q)
    measured = sign * measured

    with plt.rc_context(figure_style()):
        figure, axes = plt.subplots()
        mesh = axes.pcolormesh(
            _log_edges(speeds),
            _linear_edges(pitches),
            measured.T,
            cmap="RdBu_r",
            vmin=-1.0,
            vmax=1.0,
            shading="flat",
        )
        colorbar = figure.colorbar(mesh, ax=axes)
        colorbar.set_label("phase error per turn / |Boris lag per turn|")
        # Zero lines, named in a legend: when theory and measure agree they lie on
        # top of each other, and the legend says that two lines are there.
        legend_lines = []
        if measured.min() < 0.0 < measured.max():
            axes.contour(speeds, pitches, measured.T, levels=[0.0], colors="black")
            legend_lines.append(Line2D([], [], color="black", label="measured zero"))
        if predicted:
            theory = sign * np.array(predicted)[order]
            if theory.min() < 0.0 < theory.max():
                axes.contour(
                    speeds,
                    pitches,
                    theory.T,
                    levels=[0.0],
                    colors="white",
                    linestyles="--",
                )
                predicted_line = Line2D(
                    [], [], color="white", linestyle="--", label="predicted zero"
                )
                legend_lines.append(predicted_line)
        if legend_lines:
            axes.legend(handles=legend_lines, loc="lower right", frameon=True)
        axes.set_xscale("log")
        axes.set_xlabel(r"$u / c = \gamma v / c$")
        axes.set_ylabel(r"pitch angle $\psi$ [deg]")
        axes.set_title(f"Gyration phase error per turn, {pusher}")

    if save_path is not None:
        save_figure(figure, save_path)
    if show:
        plt.show()
    return figure


def _canonical_momentum_change(
    result: Any, q: float, m: float, *, half_step: bool
) -> NDArray[np.float64]:
    """``|P_zeta / P_zeta0 - 1|`` of the stored states, shape ``(n_saved, n_part)``.

    ``P_zeta0`` is that of the initial state ``(x^0, u^0)``. With ``half_step``,
    each ``P_zeta`` is computed as a staggered code would, from ``x^{n+1/2}`` and
    ``u^n`` (roadmap C5(c)); otherwise from the states ``(x^n, u^n)`` of the scheme.
    The flux ``Psi`` is the one stored by ``run`` at these positions.
    """
    if "poloidal_flux" not in result:
        raise ValueError(
            f"The result of {result.attrs['pusher']!r} holds no poloidal flux: "
            "P_zeta is only defined for the tokamak field ('kind: carbajal')"
        )
    x, u = result["x"].values, result["u"].values
    psi = result["poloidal_flux"].values
    p_zeta0 = diagnostics.canonical_toroidal_momentum(x[:1], u[:1], psi[:1], q, m)[0]
    if half_step:
        x = diagnostics.half_step_position(x, u, result.attrs["dt"])
        psi = result["poloidal_flux_half_step"].values
    p_zeta = diagnostics.canonical_toroidal_momentum(x, u, psi, q, m)
    return np.abs(p_zeta / p_zeta0 - 1.0)


def _pitch_angle_labels(result: Any) -> list[str]:
    """Legend of each particle: its initial pitch angle, in degrees."""
    u0, b0 = result["u"].values[0], result["b_field"].values[0]
    labels = []
    for i in range(u0.shape[0]):  # pitch_angle takes one field for all its particles
        pitch = diagnostics.pitch_angle(u0[i : i + 1], b0[i])[0]
        labels.append(rf"$\theta_0 = {np.degrees(pitch):.1f}^\circ$")
    return labels


def canonical_momentum_error(
    result_dir: str | Path,
    pusher: str | Sequence[str] | None = None,
    *,
    half_step: bool = False,
    show: bool = True,
    save_path: str | Path | None = None,
) -> Figure:
    """
    Draw the change of the canonical toroidal momentum ``|P_zeta / P_zeta0 - 1|``.

    ``P_zeta = R u_zeta - (q / m) Psi`` (Carbajal et al. 2017, Eq. 6, per unit
    mass) is an exact invariant in the static, axisymmetric field of
    ``kind: carbajal`` without radiation: the curve is the error of the scheme, as
    in Fig. 1(b) of the paper. It is computed by
    ``diagnostics.canonical_toroidal_momentum`` from the stored ``x``, ``u`` and
    ``Psi``. With ``half_step=True`` it is computed instead from ``x^{n+1/2}`` and
    ``u^n``, as a staggered code measures it (roadmap C5(c)): an error of order
    ``dt``, much larger. Logarithmic vertical axis; values below the float64
    rounding unit (the exact 0 at ``t = 0``) are drawn at it. One curve per
    particle, labelled with its initial pitch angle.

    Parameters
    ----------
    result_dir : str or pathlib.Path
        Results folder given by ``run.pusher`` for a ``carbajal`` field.
    pusher : str or sequence of str, optional
        Which pusher(s) of the folder to draw. Default: all of them.
    half_step : bool, optional
        Pair ``x^{n+1/2}`` with ``u^n``, as a staggered code. Default ``False``.
    show : bool, optional
        Open the figure in a window. Default ``True``.
    save_path : str or pathlib.Path, optional
        If given, the figure is also saved as a PNG and as a PDF (no extension).

    Returns
    -------
    matplotlib.figure.Figure
        The figure, square, 6 x 6 in.

    Raises
    ------
    FileNotFoundError
        If no pusher is given and the folder holds no trajectories.
    ValueError
        If a pusher is not in the folder, or its result holds no poloidal flux
        (a field other than ``carbajal``).
    """
    results = _read_selected(result_dir, pusher)
    run_config = config.load_config(Path(result_dir) / CONFIG_FILENAME)
    rounding = np.finfo(np.float64).eps
    with plt.rc_context(figure_style()):
        figure, axes = plt.subplots()
        for k, (name, result) in enumerate(results.items()):
            change = _canonical_momentum_change(
                result, run_config.q, run_config.m, half_step=half_step
            )
            labels = _pitch_angle_labels(result)
            for i in range(change.shape[1]):
                label = labels[i] if len(results) == 1 else f"{name}, {labels[i]}"
                axes.plot(
                    result["time"].values,
                    np.maximum(change[:, i], rounding),
                    color=_LINE_COLORS[i % len(_LINE_COLORS)],
                    linestyle=_LINE_STYLES[k % len(_LINE_STYLES)],
                    linewidth=1.0,
                    label=label,
                )
        axes.set_yscale("log")
        axes.set_xlabel("t [s]")
        axes.set_ylabel(r"$|P_\zeta(t) / P_{\zeta 0} - 1|$")
        states = r"$(x^{n+1/2}, u^n)$" if half_step else r"$(x^n, u^n)$"
        dt = next(iter(results.values())).attrs["dt"]
        if len(results) == 1:
            title = f"{next(iter(results))}, {states}, dt = {dt:.3g} s"
        else:
            title = f"{states}, dt = {dt:.3g} s"
        axes.set_title(title, fontsize=13)
        axes.legend(fontsize=11)

    if save_path is not None:
        save_figure(figure, save_path)
    if show:
        plt.show()
    return figure


def canonical_momentum_convergence(
    result_dirs: Sequence[str | Path],
    pusher: str | Sequence[str] | None = None,
    *,
    show: bool = True,
    save_path: str | Path | None = None,
) -> Figure:
    """
    Draw the amplitude of the error on ``P_zeta`` against the time step.

    Each results folder is one run of the same case (field and initial state) with
    its own ``dt``. For each particle, the amplitude is the largest
    ``|P_zeta / P_zeta0 - 1|`` over the run (see ``canonical_momentum_error``),
    computed twice: from the states ``(x^n, u^n)`` of the scheme (filled circles,
    solid line) and from ``x^{n+1/2}`` with ``u^n``, as a staggered code such as
    KORC (open squares, dashed line). This is the test of roadmap C5(b) and (c):
    the first should fall as ``dt^2`` and the second as ``dt``. Grey dotted lines
    of slopes 2 and 1 are drawn for comparison, a factor 3 below the lowest curve
    of each kind at the largest ``dt``. One panel per pusher, logarithmic axes, the
    legend below the panels.

    Parameters
    ----------
    result_dirs : sequence of str or pathlib.Path
        At least two results folders given by ``run.pusher`` for a ``carbajal``
        field, the same case with different time steps, in any order.
    pusher : str or sequence of str, optional
        Which pusher(s) to draw. Default: all the pushers of the first folder.
    show : bool, optional
        Open the figure in a window. Default ``True``.
    save_path : str or pathlib.Path, optional
        If given, the figure is also saved as a PNG and as a PDF (no extension).

    Returns
    -------
    matplotlib.figure.Figure
        The figure, one 5 x 5.5 in panel per pusher.

    Raises
    ------
    ValueError
        If fewer than two folders are given, if they do not hold the same field
        and initial state, or if a result holds no poloidal flux.
    """
    folders = [Path(directory) for directory in result_dirs]
    if len(folders) < 2:
        raise ValueError("give at least two results folders, with different dt")
    configs = [config.load_config(folder / CONFIG_FILENAME) for folder in folders]
    for folder, run_config in zip(folders, configs, strict=True):
        if (run_config.field, run_config.initial) != (
            configs[0].field,
            configs[0].initial,
        ):
            raise ValueError(
                f"{folder} does not hold the same field and initial state as "
                f"{folders[0]}: a convergence study compares one case at several dt"
            )
    if pusher is None:
        names = list_results(folders[0])
    elif isinstance(pusher, str):
        names = [pusher]
    else:
        names = list(pusher)

    rounding = np.finfo(np.float64).eps
    with plt.rc_context(figure_style()):
        figure, panels = plt.subplots(
            1, len(names), figsize=(5.0 * len(names), 5.5), sharey=True, squeeze=False
        )
        for axes, name in zip(panels[0], names, strict=True):
            dts: list[float] = []
            synchronised: list[np.ndarray] = []
            staggered: list[np.ndarray] = []
            for folder, run_config in zip(folders, configs, strict=True):
                result = read_result(folder, name)
                dts.append(result.attrs["dt"])
                for half_step, store in ((False, synchronised), (True, staggered)):
                    change = _canonical_momentum_change(
                        result, run_config.q, run_config.m, half_step=half_step
                    )
                    store.append(np.max(change, axis=0))
            order = np.argsort(dts)
            dt = np.array(dts)[order]
            synchronised_amplitude = np.array(synchronised)[order]  # (n_dt, n_part)
            staggered_amplitude = np.array(staggered)[order]
            labels = _pitch_angle_labels(read_result(folders[0], name))
            for i, label in enumerate(labels):
                color = _LINE_COLORS[i % len(_LINE_COLORS)]
                axes.plot(
                    dt,
                    np.maximum(synchronised_amplitude[:, i], rounding),
                    marker="o",
                    color=color,
                    label=label,
                )
                axes.plot(
                    dt,
                    np.maximum(staggered_amplitude[:, i], rounding),
                    marker="s",
                    markerfacecolor="none",
                    linestyle="--",
                    color=color,
                )
            # Guides of slopes 2 and 1, below the curves they are compared with.
            guides = ((2, synchronised_amplitude), (1, staggered_amplitude))
            for slope, amplitudes in guides:
                start = np.min(amplitudes[-1]) / 3.0
                guide = start * (dt / dt[-1]) ** slope
                axes.plot(dt, guide, color="0.5", linestyle=":", linewidth=1.2)
                axes.annotate(
                    rf"$\propto \Delta t^{slope}$",
                    (dt[1], guide[1]),
                    textcoords="offset points",
                    xytext=(0, -16),
                    color="0.4",
                    fontsize=12,
                )
            axes.set_xscale("log")
            axes.set_yscale("log")
            axes.xaxis.set_minor_formatter(NullFormatter())
            axes.set_xlabel(r"$\Delta t$ [s]")
            axes.set_title(name, fontsize=14)
        panels[0][0].set_ylabel(r"$\max_n\, |P_\zeta / P_{\zeta 0} - 1|$")
        marker_legend = [
            Line2D([], [], color="0.2", marker="o", label=r"$(x^n, u^n)$"),
            Line2D(
                [],
                [],
                color="0.2",
                marker="s",
                markerfacecolor="none",
                linestyle="--",
                label=r"$(x^{n+1/2}, u^n)$",
            ),
        ]
        angle_handles = panels[0][0].get_legend_handles_labels()[0]
        figure.legend(
            handles=angle_handles + marker_legend,
            loc="outside lower center",
            ncol=len(angle_handles) + 2,
            fontsize=12,
        )

    if save_path is not None:
        save_figure(figure, save_path)
    if show:
        plt.show()
    return figure


def orbit_poloidal(
    result_dir: str | Path,
    pusher: str | Sequence[str] | None = None,
    *,
    show: bool = True,
    save_path: str | Path | None = None,
) -> Figure:
    """
    Draw the orbits projected on a poloidal plane: height ``Z`` against ``R``.

    ``R = sqrt(x^2 + y^2)`` is the distance to the vertical axis of the torus and
    ``Z = z``: each point of the orbit is turned about that axis into one plane, as
    in Fig. 2 of Carbajal et al. (2017). A passing electron draws a closed curve
    around the magnetic axis, a trapped one a banana. The start of each orbit is
    marked by a dot, and for a ``carbajal`` field the magnetic axis by a cross at
    ``(R0, 0)``. One colour per particle, labelled with its initial pitch angle;
    one line style per pusher. Equal scales on both axes.

    Parameters
    ----------
    result_dir : str or pathlib.Path
        Results folder given by ``run.pusher``.
    pusher : str or sequence of str, optional
        Which pusher(s) of the folder to draw. Default: all of them.
    show : bool, optional
        Open the figure in a window. Default ``True``.
    save_path : str or pathlib.Path, optional
        If given, the figure is also saved as a PNG and as a PDF (no extension).

    Returns
    -------
    matplotlib.figure.Figure
        The figure, square, 6 x 6 in.

    Raises
    ------
    FileNotFoundError
        If no pusher is given and the folder holds no trajectories.
    ValueError
        If a pusher is not in the folder.
    """
    results = _read_selected(result_dir, pusher)
    run_config = config.load_config(Path(result_dir) / CONFIG_FILENAME)
    with plt.rc_context(figure_style()):
        figure, axes = plt.subplots()
        for k, (name, result) in enumerate(results.items()):
            x = result["x"].values
            major_radius = np.hypot(x[..., 0], x[..., 1])
            labels = _pitch_angle_labels(result)
            for i in range(x.shape[1]):
                color = _LINE_COLORS[i % len(_LINE_COLORS)]
                label = labels[i] if len(results) == 1 else f"{name}, {labels[i]}"
                # Later particles drawn below: with growing pitch angles, the wide
                # band of a large angle does not hide the thin one of a small angle.
                axes.plot(
                    major_radius[:, i],
                    x[:, i, 2],
                    color=color,
                    linestyle=_LINE_STYLES[k % len(_LINE_STYLES)],
                    linewidth=0.5,
                    label=label,
                    zorder=x.shape[1] - i,
                )
                axes.plot(major_radius[0, i], x[0, i, 2], "o", color=color)
        if run_config.field.kind == "carbajal":
            axes.plot(run_config.field.r0, 0.0, "k+", markersize=12)
        axes.set_aspect("equal", adjustable="datalim")
        axes.set_xlabel("R [m]")
        axes.set_ylabel("Z [m]")
        axes.set_title("Orbits in a poloidal plane")
        axes.legend(fontsize=11)

    if save_path is not None:
        save_figure(figure, save_path)
    if show:
        plt.show()
    return figure


def parallel_velocity(
    result_dir: str | Path,
    pusher: str | Sequence[str] | None = None,
    *,
    show: bool = True,
    save_path: str | Path | None = None,
) -> Figure:
    """
    Draw the velocity along the field, ``u_par / |u|``, of each particle against time.

    ``u_par = u . B / |B|``, with ``B`` the field stored by ``run`` at each saved
    position. A passing particle keeps the sign of ``u_par``; a trapped one changes
    it at each mirror point (roadmap C6). Each curve is labelled with the initial
    pitch angle of its particle and with its class given by
    ``diagnostics.is_trapped``, which holds for the duration of the run only. One
    colour per particle, one line style per pusher.

    Parameters
    ----------
    result_dir : str or pathlib.Path
        Results folder given by ``run.pusher``.
    pusher : str or sequence of str, optional
        Which pusher(s) of the folder to draw. Default: all of them.
    show : bool, optional
        Open the figure in a window. Default ``True``.
    save_path : str or pathlib.Path, optional
        If given, the figure is also saved as a PNG and as a PDF (no extension).

    Returns
    -------
    matplotlib.figure.Figure
        The figure, square, 6 x 6 in.

    Raises
    ------
    FileNotFoundError
        If no pusher is given and the folder holds no trajectories.
    ValueError
        If a pusher is not in the folder.
    """
    results = _read_selected(result_dir, pusher)
    with plt.rc_context(figure_style()):
        figure, axes = plt.subplots()
        for k, (name, result) in enumerate(results.items()):
            u = result["u"].values
            b_field = result["b_field"].values
            u_parallel = np.sum(u * b_field, axis=2) / np.linalg.norm(b_field, axis=2)
            ratio = u_parallel / np.linalg.norm(u, axis=2)
            trapped = diagnostics.is_trapped(u, b_field)
            labels = _pitch_angle_labels(result)
            for i in range(u.shape[1]):
                kind = "trapped" if trapped[i] else "passing"
                label = f"{labels[i]}, {kind}"
                if len(results) > 1:
                    label = f"{name}, {label}"
                axes.plot(
                    result["time"].values,
                    ratio[:, i],
                    color=_LINE_COLORS[i % len(_LINE_COLORS)],
                    linestyle=_LINE_STYLES[k % len(_LINE_STYLES)],
                    linewidth=1.0,
                    label=label,
                )
        axes.axhline(0.0, color="0.5", linewidth=0.8)
        axes.set_xlabel("t [s]")
        axes.set_ylabel(r"$u_\parallel / |u|$")
        axes.set_title("Velocity along the field")
        axes.legend(fontsize=11)

    if save_path is not None:
        save_figure(figure, save_path)
    if show:
        plt.show()
    return figure


def poincare_section(
    result_dir: str | Path,
    pusher: str | Sequence[str] | None = None,
    *,
    show: bool = True,
    save_path: str | Path | None = None,
) -> Figure:
    """
    Draw the Poincare section of the orbits, ``u_y / c`` against ``y``, per pusher.

    The points are the states where an orbit crosses the plane ``x = 0`` towards
    ``x > 0``, interpolated linearly between two saved states
    (``diagnostics.poincare_section``), as in Figs. 1 and 2 of Higuera and Cary
    (2017) for roadmap C7. In the field ``fields.higuera_cary_shear``,
    ``u_y = p_y / m``, and an exact orbit draws a closed curve; the curves of
    different particles are nested. One colour per particle (ten colours, then
    repeated), labelled with its initial ``u_y / c``; one panel per pusher. The
    states must be saved at every step (``save_every: 1``) for the interpolation
    error to stay of the order of that of the scheme.

    Parameters
    ----------
    result_dir : str or pathlib.Path
        Results folder given by ``run.pusher``.
    pusher : str or sequence of str, optional
        Which pusher(s) of the folder to draw. Default: all of them.
    show : bool, optional
        Open the figure in a window. Default ``True``.
    save_path : str or pathlib.Path, optional
        If given, the figure is also saved as a PNG and as a PDF (no extension).

    Returns
    -------
    matplotlib.figure.Figure
        The figure, one square panel of 6 x 6 in per pusher, with shared axes.

    Raises
    ------
    FileNotFoundError
        If no pusher is given and the folder holds no trajectories.
    ValueError
        If a pusher is not in the folder.
    """
    results = _read_selected(result_dir, pusher)
    with plt.rc_context(figure_style()):
        figure, panels = plt.subplots(
            1,
            len(results),
            figsize=(6.0 * len(results), 6.0),
            sharex=True,
            sharey=True,
            squeeze=False,
        )
        for axes, (name, result) in zip(panels[0], results.items(), strict=True):
            u = result["u"].values
            particle, _, x_cross, u_cross = diagnostics.poincare_section(
                result["time"].values, result["x"].values, u
            )
            for i in range(u.shape[1]):
                mine = particle == i
                axes.plot(
                    x_cross[mine, 1],
                    u_cross[mine, 1] / c,
                    ".",
                    markersize=1.5,
                    color=plt.get_cmap("tab10")(i % 10),
                    label=rf"$u_{{y,0}} / c = {u[0, i, 1] / c:.3f}$",
                )
            axes.set_xlabel("y [m]")
            axes.set_title(rf"{name}, $\Delta t$ = {result.attrs['dt']:.3g} s")
        panels[0, 0].set_ylabel(r"$u_y / c$")
        panels[0, -1].legend(fontsize=9, markerscale=6)

    if save_path is not None:
        save_figure(figure, save_path)
    if show:
        plt.show()
    return figure


def _radiating_helix_field(run_config: config.RunConfig) -> float:
    """``b_z`` of a run that ``radiation.radiating_helix`` solves exactly.

    That is a uniform ``B = b_z z`` with ``b_z > 0``, no electric field and the
    radiation force on (``rr_scale > 0``).
    """
    field = run_config.field
    if field.kind != "uniform_cartesian":
        raise ValueError(
            f"The exact solution needs a uniform field (kind 'uniform_cartesian'), "
            f"got kind {field.kind!r}"
        )
    if run_config.rr_scale <= 0.0:
        raise ValueError(
            "The run has no radiation (rr_scale: 0): there is no decay to draw"
        )
    if any(component != 0.0 for component in field.e_field):
        raise ValueError(
            f"The exact solution has no electric field, got e_field {field.e_field}"
        )
    b_x, b_y, b_z = field.b_field
    if b_x != 0.0 or b_y != 0.0 or b_z <= 0.0:
        raise ValueError(
            "The exact solution needs B along +z, b_field [0, 0, b_z] with b_z > 0; "
            f"got {field.b_field}"
        )
    return float(b_z)


def radiative_decay(
    result_dir: str | Path,
    pusher: str | Sequence[str] | None = None,
    *,
    show: bool = True,
    save_path: str | Path | None = None,
) -> Figure:
    """
    Draw the Lorentz factor of electrons that radiate in a uniform magnetic field.

    The run is a uniform ``B = b_z z``, without electric field, with the radiation
    force multiplied by ``rr_scale = s``. Its exact solution
    (``radiation.radiating_helix``) is drawn as a black dotted line over each
    computed curve. The parallel velocity ``beta_par`` is constant, and ``gamma``
    decays to ``gamma_par = 1 / sqrt(1 - beta_par^2)``. ``gamma`` depends on time
    only through ``s t``, so the horizontal axis is the physical time ``s t``,
    exactly. Both axes are logarithmic; the state at ``t = 0`` is not drawn. One
    colour per particle, labelled with its initial pitch angle; one line style per
    pusher.

    Parameters
    ----------
    result_dir : str or pathlib.Path
        Results folder given by ``run.pusher`` for such a run.
    pusher : str or sequence of str, optional
        Which pusher(s) of the folder to draw. Default: all of them.
    show : bool, optional
        Open the figure in a window. Default ``True``.
    save_path : str or pathlib.Path, optional
        If given, the figure is also saved as a PNG and as a PDF (no extension).

    Returns
    -------
    matplotlib.figure.Figure
        The figure, square, 6 x 6 in.

    Raises
    ------
    FileNotFoundError
        If no pusher is given and the folder holds no trajectories.
    ValueError
        If a pusher is not in the folder, or the run is not one that
        ``radiation.radiating_helix`` solves: field not uniform, no radiation, an
        electric field, or ``B`` not along ``+z``.
    """
    run_config = config.load_config(Path(result_dir) / CONFIG_FILENAME)
    b_z = _radiating_helix_field(run_config)
    results = _read_selected(result_dir, pusher)
    s = run_config.rr_scale
    u0 = np.array(run_config.initial.u0, dtype=np.float64)

    with plt.rc_context(figure_style()):
        figure, axes = plt.subplots()
        for k, (name, result) in enumerate(results.items()):
            time = result["time"].values
            later = time > 0.0  # a logarithmic axis cannot show t = 0
            gamma = np.sqrt(1.0 + np.sum(result["u"].values ** 2, axis=-1) / c**2)
            labels = _pitch_angle_labels(result)
            for i in range(gamma.shape[1]):
                label = labels[i] if len(results) == 1 else f"{name}, {labels[i]}"
                axes.plot(
                    s * time[later],
                    gamma[later, i],
                    color=_LINE_COLORS[i % len(_LINE_COLORS)],
                    linestyle=_LINE_STYLES[k % len(_LINE_STYLES)],
                    label=label,
                )

        # The pushers of a folder share the case, hence the times and the exact curve.
        time = next(iter(results.values()))["time"].values
        time = time[time > 0.0]
        u_exact = radiation.radiating_helix(
            u0, b_z, time, run_config.q, run_config.m, rr_scale=s
        )
        gamma_exact = np.sqrt(1.0 + np.sum(u_exact**2, axis=-1) / c**2)
        for i in range(gamma_exact.shape[1]):
            axes.plot(
                s * time,
                gamma_exact[:, i],
                color="black",
                linestyle=":",
                linewidth=1.2,
                label="exact solution" if i == 0 else None,
            )

        axes.set_xscale("log")
        axes.set_yscale("log")
        axes.set_xlabel(r"physical time $s\,t$ [s]")
        axes.set_ylabel(r"$\gamma$")
        axes.set_title(rf"$B$ = {b_z:g} T, no $E$, $s$ = {s:.3g}", fontsize=13)
        axes.legend(fontsize=11)

    if save_path is not None:
        save_figure(figure, save_path)
    if show:
        plt.show()
    return figure


# Line style of each case of momentum_plane, after the forces of the run.
_CASE_STYLES: dict[str, LineStyleType] = {
    "E and radiation": "-",
    "E only": "--",
    "radiation only": ":",
}


def _force_case(run_config: config.RunConfig) -> str:
    """Name of the case of a run: which of ``E`` and radiation act.

    For the uniform field and the tokamak field of Carbajal et al., the two kinds
    with an electric field and radiation in the article.
    """
    field = run_config.field
    if field.kind == "uniform_cartesian":
        has_e = any(component != 0.0 for component in field.e_field)
    elif field.kind == "carbajal":
        has_e = field.e0 != 0.0
    else:
        raise ValueError(
            f"The run {run_config.name!r} has a field of kind {field.kind!r}; the "
            "cases are defined for 'uniform_cartesian' and 'carbajal'"
        )
    has_radiation = run_config.rr_scale > 0.0
    if has_e and has_radiation:
        return "E and radiation"
    if has_e:
        return "E only"
    if has_radiation:
        return "radiation only"
    raise ValueError(
        f"The run {run_config.name!r} has neither an electric field nor radiation: "
        "its momentum does not change"
    )


def _parallel_and_perpendicular_mev(
    u: NDArray[np.float64], b_field: NDArray[np.float64], m: float
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """``p_par c`` and ``p_perp c`` [MeV] of the states, shape (n_saved, n_part)."""
    b_unit = b_field / np.linalg.norm(b_field, axis=-1, keepdims=True)
    u_par = np.sum(u * b_unit, axis=-1)
    u_perp = np.linalg.norm(u - u_par[..., None] * b_unit, axis=-1)
    to_mev = m * c / _JOULES_PER_MEV
    return to_mev * u_par, to_mev * u_perp


def _check_momentum_plane_cases(
    run_configs: list[config.RunConfig], time_scale: float
) -> tuple[float, float]:
    """Check that the runs are one uniform ``E || B`` case; return ``(|B|, E/s)``.

    ``E/s`` [V/m] is the physical electric field, 0 if no run has one.
    """
    if not np.isfinite(time_scale) or time_scale <= 0.0:
        raise ValueError(f"time_scale must be finite and > 0, got {time_scale!r}")
    b_norms, e_norms = set(), set()
    for run_config in run_configs:
        field = run_config.field
        if field.kind != "uniform_cartesian":
            raise ValueError(
                f"The run {run_config.name!r} is not in a uniform field (kind "
                f"{field.kind!r}): the balance curve is that of uniform E || B"
            )
        b_field = np.array(field.b_field, dtype=np.float64)
        e_field = np.array(field.e_field, dtype=np.float64)
        b_norm, e_norm = np.linalg.norm(b_field), np.linalg.norm(e_field)
        if np.linalg.norm(np.cross(e_field, b_field)) > 1.0e-12 * e_norm * b_norm:
            raise ValueError(
                f"In the run {run_config.name!r}, E {field.e_field} is not along "
                f"B {field.b_field}"
            )
        if run_config.rr_scale > 0.0 and run_config.rr_scale != time_scale:
            raise ValueError(
                f"The run {run_config.name!r} has rr_scale {run_config.rr_scale}, "
                f"not the time_scale {time_scale}: E and F_R must be scaled together"
            )
        _force_case(run_config)
        b_norms.add(float(b_norm))
        if e_norm > 0.0:
            e_norms.add(float(e_norm))
    if len(b_norms) != 1 or len(e_norms) > 1:
        raise ValueError(
            "The runs must share the same |B| and, when they have one, the same |E|: "
            f"got |B| {sorted(b_norms)} T and |E| {sorted(e_norms)} V/m"
        )
    e_physical = e_norms.pop() / time_scale if e_norms else 0.0
    return b_norms.pop(), e_physical


def momentum_plane(
    result_dirs: Sequence[str | Path],
    time_scale: float,
    pusher: str = "boris",
    *,
    show: bool = True,
    save_path: str | Path | None = None,
) -> Figure:
    """
    Draw the path of each electron in the plane ``(p_par c, p_perp c)``, in MeV.

    The runs are one case in uniform fields ``E || B``, computed with the forces
    multiplied by ``s = time_scale`` (OPEN_POINTS N44): ``E`` (the field of the
    ``.yaml`` is ``s`` times the physical one) and, when ``rr_scale > 0``, the
    radiation force (``rr_scale = s``). Each folder is one choice of forces, read
    from its configuration and drawn with its own line style: ``E`` and
    radiation (solid), ``E`` only (dashed), radiation only (dotted). One colour per
    particle, labelled with its initial pitch angle; a black dot marks the start.

    The grey curve is the balance ``gamma = gamma_max(theta)`` of
    ``radiation.maximum_lorentz_factor``, for the physical field ``|E| / s``. It
    does not depend on ``s``, since the gain goes as ``s E`` and the loss as
    ``s F_R``. It is drawn only if a run has an electric field.

    Parameters
    ----------
    result_dirs : sequence of str or pathlib.Path
        Results folders given by ``run.pusher``, one per choice of forces.
    time_scale : float
        The factor ``s`` on ``E`` and ``F_R`` [dimensionless], > 0. The physical
        time is ``s t``.
    pusher : str, optional
        The pusher drawn, present in every folder. Default ``"boris"``.
    show : bool, optional
        Open the figure in a window. Default ``True``.
    save_path : str or pathlib.Path, optional
        If given, the figure is also saved as a PNG and as a PDF (no extension).

    Returns
    -------
    matplotlib.figure.Figure
        The figure, square, 6 x 6 in.

    Raises
    ------
    ValueError
        If ``time_scale`` is not finite and > 0, or the runs are not one uniform
        ``E || B`` case: a field not uniform, ``E`` not along ``B``, a run with
        ``rr_scale`` other than ``time_scale``, different ``|B|``, different
        ``|E|`` among the runs that have one, different durations, a run without
        any force, or ``pusher`` missing from a folder.
    """
    run_configs = [
        config.load_config(Path(directory) / CONFIG_FILENAME)
        for directory in result_dirs
    ]
    b_norm, e_physical = _check_momentum_plane_cases(run_configs, time_scale)
    results = [read_result(directory, pusher) for directory in result_dirs]
    end_times = {float(result["time"].values[-1]) for result in results}
    if len(end_times) != 1:
        raise ValueError(
            f"The runs must last the same time; they end at {sorted(end_times)} s"
        )
    m = run_configs[0].m

    with plt.rc_context(figure_style()):
        figure, axes = plt.subplots()
        labels = _pitch_angle_labels(results[0])
        for run_config, result in zip(run_configs, results, strict=True):
            case = _force_case(run_config)
            p_par, p_perp = _parallel_and_perpendicular_mev(
                result["u"].values, result["b_field"].values, m
            )
            for i in range(p_par.shape[1]):
                axes.plot(
                    p_par[:, i],
                    p_perp[:, i],
                    color=_LINE_COLORS[i % len(_LINE_COLORS)],
                    linestyle=_CASE_STYLES[case],
                )
                axes.plot(p_par[0, i], p_perp[0, i], "o", color="black", markersize=4)

        if e_physical > 0.0:
            # Drawn over the range of the paths only: gamma_max grows without bound
            # as theta -> 0.
            x_limits, y_limits = axes.get_xlim(), axes.get_ylim()
            theta = np.linspace(0.002, 0.5 * np.pi - 0.002, 2000)
            gamma_max = radiation.maximum_lorentz_factor(
                e_physical, b_norm, theta, run_configs[0].q, m
            )
            p_max = m * c**2 * np.sqrt(gamma_max**2 - 1.0) / _JOULES_PER_MEV
            axes.plot(
                p_max * np.cos(theta),
                p_max * np.sin(theta),
                color="0.6",
                linewidth=3.0,
                zorder=0,
            )
            axes.set_xlim(x_limits)
            axes.set_ylim(y_limits)

        handles = [
            Line2D([], [], color=_LINE_COLORS[i % len(_LINE_COLORS)], label=label)
            for i, label in enumerate(labels)
        ]
        cases = [_force_case(run_config) for run_config in run_configs]
        handles += [
            Line2D([], [], color="black", linestyle=_CASE_STYLES[case], label=case)
            for case in _CASE_STYLES
            if case in cases
        ]
        if e_physical > 0.0:
            handles.append(
                Line2D(
                    [],
                    [],
                    color="0.6",
                    linewidth=3.0,
                    label=r"$\gamma = \gamma_{\max}(\theta)$",
                )
            )
        axes.legend(handles=handles, fontsize=10)
        axes.set_xlabel(r"$p_\parallel c$ [MeV]")
        axes.set_ylabel(r"$p_\perp c$ [MeV]")
        duration = time_scale * end_times.pop()
        axes.set_title(
            rf"$|E|$ = {e_physical:.3g} V/m, $B$ = {b_norm:.3g} T, "
            rf"{duration * 1.0e3:.3g} ms, $s$ = {time_scale:.3g}",
            fontsize=12,
        )

    if save_path is not None:
        save_figure(figure, save_path)
    if show:
        plt.show()
    return figure


# Colour and open marker of each pusher on the convergence figures, the same
# whatever the folders; open markers of different shapes stay visible when the
# errors of two pushers are equal.
_PUSHER_ORDER = ("boris", "vay", "higuera_cary", "rk4")
_PUSHER_MARKERS = {"boris": "o", "vay": "s", "higuera_cary": "^", "rk4": "D"}


def radiative_decay_convergence(
    result_dirs: Sequence[str | Path],
    pusher: str | Sequence[str] | None = None,
    *,
    show: bool = True,
    save_path: str | Path | None = None,
) -> Figure:
    """
    Draw the error on ``gamma`` against the angle turned per step, for each pusher.

    Each folder is the run of ``radiative_decay`` (uniform ``B`` along ``z``,
    radiation only) with one time step ``dt`` and one ``rr_scale = s``. The error
    of a pusher is the largest ``|gamma / gamma_exact - 1|`` over the saved states
    (``t > 0``) and the particles, against ``radiation.radiating_helix``. It is
    drawn against ``theta_end = |q| B dt / (m gamma_end)``, the angle turned per
    step at the end of the run by the particle of smallest exact ``gamma_end``:
    the step resolves the gyration worst there (OPEN_POINTS N45). Logarithmic axes;
    one colour and open marker per pusher (of decreasing size, so that equal
    errors stay visible), one line style per ``s``. The grey guides go as
    ``theta^2`` and ``theta^4``, through the point of largest ``theta`` of the first
    second-order pusher and of ``rk4``.

    Parameters
    ----------
    result_dirs : sequence of str or pathlib.Path
        Results folders given by ``run.pusher``, one per ``(dt, s)``: the same
        particles at ``t = 0``, the same ``B`` and the same physical duration
        ``s t_end``.
    pusher : str or sequence of str, optional
        Which pusher(s) to draw, present in every folder. Default: those of the
        first folder.
    show : bool, optional
        Open the figure in a window. Default ``True``.
    save_path : str or pathlib.Path, optional
        If given, the figure is also saved as a PNG and as a PDF (no extension).

    Returns
    -------
    matplotlib.figure.Figure
        The figure, square, 6 x 6 in.

    Raises
    ------
    ValueError
        If no folder is given, a run is not one that ``radiation.radiating_helix``
        solves, or the runs differ by more than ``dt`` and ``s``: other initial
        velocities, another ``B``, another physical duration.
    """
    if len(result_dirs) == 0:
        raise ValueError("No results folder given")
    if pusher is None:
        names = list_results(result_dirs[0])
    elif isinstance(pusher, str):
        names = [pusher]
    else:
        names = list(pusher)

    points: dict[tuple[str, float], list[tuple[float, float]]] = {}
    cases = set()
    for directory in result_dirs:
        run_config = config.load_config(Path(directory) / CONFIG_FILENAME)
        b_z = _radiating_helix_field(run_config)
        s = run_config.rr_scale
        u0 = np.array(run_config.initial.u0, dtype=np.float64)
        for name in names:
            result = read_result(directory, name)
            time = result["time"].values
            later = time > 0.0
            cases.add((u0.tobytes(), b_z, float(s * time[-1])))
            u_exact = radiation.radiating_helix(
                u0, b_z, time[later], run_config.q, run_config.m, rr_scale=s
            )
            gamma_exact = np.sqrt(1.0 + np.sum(u_exact**2, axis=-1) / c**2)
            u = result["u"].values[later]
            gamma = np.sqrt(1.0 + np.sum(u**2, axis=-1) / c**2)
            error = float(np.max(np.abs(gamma / gamma_exact - 1.0)))
            theta_end = (
                abs(run_config.q)
                * b_z
                * result.attrs["dt"]
                / (run_config.m * gamma_exact[-1].min())
            )
            points.setdefault((name, s), []).append((theta_end, error))
    # Durations compared to rounding: s t_end is a product of two stored numbers.
    durations = sorted(case[2] for case in cases)
    same_duration = np.isclose(durations[0], durations[-1], rtol=1e-12, atol=0.0)
    if len({case[:2] for case in cases}) != 1 or not same_duration:
        raise ValueError(
            "The folders must hold one case run with several dt and s: the same "
            "initial velocities, the same B and the same physical duration s t_end"
        )

    scales = sorted({s for _, s in points})
    with plt.rc_context(figure_style()):
        figure, axes = plt.subplots()
        for (name, s), group in points.items():
            theta, error = np.array(sorted(group)).T
            axes.plot(
                theta,
                error,
                marker=_PUSHER_MARKERS[name],
                markerfacecolor="none",
                markersize=9 - 1.5 * _PUSHER_ORDER.index(name),
                color=_LINE_COLORS[_PUSHER_ORDER.index(name)],
                linestyle=_LINE_STYLES[scales.index(s) % len(_LINE_STYLES)],
                label=f"{name}, $s$ = {s:.0e}",
            )
        # The point of largest theta of the first group of each order.
        second_order = [max(g) for (name, _), g in points.items() if name != "rk4"]
        fourth_order = [max(g) for (name, _), g in points.items() if name == "rk4"]
        for order, anchors in ((2, second_order), (4, fourth_order)):
            if anchors:
                theta_anchor, error_anchor = anchors[0]
                theta_guide = np.geomspace(theta_anchor / 10.0, theta_anchor, 2)
                axes.plot(
                    theta_guide,
                    error_anchor * (theta_guide / theta_anchor) ** order,
                    color="0.6",
                    linewidth=1.0,
                    linestyle="--",
                    label=rf"$\propto \theta^{order}$",
                )
        axes.set_xscale("log")
        axes.set_yscale("log")
        axes.xaxis.set_minor_formatter(NullFormatter())  # labels would overlap
        axes.set_xlabel(r"angle per step at the end, $\theta_{\mathrm{end}}$ [rad]")
        axes.set_ylabel(r"$\max\,|\gamma / \gamma_{\mathrm{exact}} - 1|$")
        axes.legend(fontsize=9)

    if save_path is not None:
        save_figure(figure, save_path)
    if show:
        plt.show()
    return figure


def _physical_case(run_config: config.RunConfig) -> tuple[dict[str, Any], Any]:
    """Split a scaled run into what must match between ``s`` and its ``E / s``.

    Returns the configuration without the keys that change with ``s`` (name,
    number of steps and of saves, ``rr_scale``, the electric field), and the
    physical electric field ``E / s``: ``e_field / s`` [V/m] for a uniform field,
    ``e0 / s`` [V/m] for the field of Carbajal et al.
    """
    s = run_config.rr_scale
    field = run_config.field
    if s <= 0.0:
        raise ValueError(
            f"The run {run_config.name!r} has no radiation (rr_scale: 0): its scale "
            "s is not known"
        )
    if field.kind == "uniform_cartesian":
        e_physical: Any = np.array(field.e_field, dtype=np.float64) / s
        electric_key = "e_field"
    elif field.kind == "carbajal":
        e_physical = field.e0 / s
        electric_key = "e0"
    else:
        raise ValueError(
            f"The run {run_config.name!r} has a field of kind {field.kind!r}, "
            "without electric field to scale"
        )
    case = run_config.model_dump(exclude={"name", "n_steps", "save_every", "rr_scale"})
    del case["field"][electric_key]
    return case, e_physical


def scaling_error(
    result_dirs: Sequence[str | Path],
    pusher: str = "boris",
    *,
    show: bool = True,
    save_path: str | Path | None = None,
) -> Figure:
    """
    Draw how ``gamma`` changes with the common scale ``s`` of ``E`` and ``F_R``.

    Each folder is the same physical case run with ``E`` and the radiation force
    multiplied by its own ``s = rr_scale`` (OPEN_POINTS N44), with the same step
    ``dt`` and its states saved at the same physical times ``s t``. Sorted by
    ``s``, two successive runs ``s < s'`` give, for each particle, the root mean
    square over the saved times of ``gamma_s' - gamma_s``, over its initial
    ``gamma_0``; it is drawn against ``s``. If the error of the scaling is of
    first order, ``gamma_s = gamma_phys + s A(t)``: with ``s' = 2 s`` the points lie
    on the grey guide of slope 1, and ``gamma_2s - gamma_s`` estimates the error of
    the run at ``s``. The pusher's own error, which also differs from one ``s``
    to the next (the same physical time is a different number of steps), is
    included: it must be small for the figure to show the scaling. One colour per
    particle, labelled with its initial pitch angle.

    Parameters
    ----------
    result_dirs : sequence of str or pathlib.Path
        At least two results folders given by ``run.pusher``, one per ``s``.
    pusher : str, optional
        The pusher drawn, present in every folder. Default ``"boris"``.
    show : bool, optional
        Open the figure in a window. Default ``True``.
    save_path : str or pathlib.Path, optional
        If given, the figure is also saved as a PNG and as a PDF (no extension).

    Returns
    -------
    matplotlib.figure.Figure
        The figure, square, 6 x 6 in.

    Raises
    ------
    ValueError
        If there are fewer than two folders, two of them have the same ``s``, a run
        has no radiation or no field to scale, or the runs are not one physical
        case: another configuration (``dt`` included), another physical field
        ``E / s``, other physical output times.
    """
    if len(result_dirs) < 2:
        raise ValueError("Two runs at least are needed to compare two values of s")
    runs = []
    for directory in result_dirs:
        run_config = config.load_config(Path(directory) / CONFIG_FILENAME)
        runs.append((run_config.rr_scale, run_config, read_result(directory, pusher)))
    runs.sort(key=lambda run: run[0])
    scales = [s for s, _, _ in runs]
    if len(set(scales)) != len(scales):
        raise ValueError(f"Two runs have the same s: {scales}")

    reference_case, reference_field = _physical_case(runs[0][1])
    reference_times = scales[0] * runs[0][2]["time"].values
    for s, run_config, result in runs[1:]:
        case, e_physical = _physical_case(run_config)
        times = s * result["time"].values
        # E / s and s t are products of stored numbers: equal to rounding.
        same_field = np.allclose(e_physical, reference_field, rtol=1e-12, atol=0.0)
        same_times = times.shape == reference_times.shape and np.allclose(
            times, reference_times, rtol=1e-12, atol=0.0
        )
        if case != reference_case or not same_field or not same_times:
            raise ValueError(
                f"The run at s = {s:g} is not the case of the run at s = "
                f"{scales[0]:g}: the configuration (dt included), the physical "
                "field E / s and the physical output times s t must be the same"
            )

    gammas = [
        np.sqrt(1.0 + np.sum(result["u"].values ** 2, axis=-1) / c**2)
        for _, _, result in runs
    ]
    gamma_0 = gammas[0][0]
    # rms over time of gamma_s' - gamma_s, shape (n_pairs, n_part).
    differences = np.array(
        [
            np.sqrt(np.mean((later - earlier) ** 2, axis=0)) / gamma_0
            for earlier, later in pairwise(gammas)
        ]
    )
    pair_scales = np.array(scales[:-1])

    with plt.rc_context(figure_style()):
        figure, axes = plt.subplots()
        labels = _pitch_angle_labels(runs[0][2])
        for i in range(differences.shape[1]):
            axes.plot(
                pair_scales,
                differences[:, i],
                marker="o",
                color=_LINE_COLORS[i % len(_LINE_COLORS)],
                label=labels[i],
            )
        guide = differences[0, 0] * pair_scales / pair_scales[0]
        axes.plot(
            pair_scales,
            guide,
            color="0.6",
            linewidth=1.0,
            linestyle="--",
            label=r"$\propto s$",
        )
        axes.set_xscale("log")
        axes.set_yscale("log")
        axes.set_xlabel(r"scale $s$ of $E$ and $F_R$")
        axes.set_ylabel(r"rms$_t(\gamma_{s'} - \gamma_s)\,/\,\gamma_0$")
        duration = reference_times[-1]
        axes.set_title(
            rf"{pusher}, {duration * 1.0e3:.3g} ms, $s'$ = next $s$", fontsize=12
        )
        axes.legend(fontsize=11)

    if save_path is not None:
        save_figure(figure, save_path)
    if show:
        plt.show()
    return figure


# Colour of each case of tokamak_time_traces, where one particle is drawn.
_CASE_COLORS = {
    "E and radiation": "black",
    "E only": "royalblue",
    "radiation only": "crimson",
}


def _check_tokamak_cases(
    run_configs: list[config.RunConfig], time_scale: float
) -> tuple[float, float]:
    """Check that the runs are one tokamak case but for the forces.

    Returns ``(e0, R0)``: ``e0`` [V/m] is the scaled toroidal field on the axis
    of the runs that have one (``s`` times the physical field), 0 if none has;
    ``R0`` [m] is the major radius.
    """
    if not np.isfinite(time_scale) or time_scale <= 0.0:
        raise ValueError(f"time_scale must be finite and > 0, got {time_scale!r}")
    cases, scaled_fields, major_radii = [], set(), set()
    for run_config in run_configs:
        field = run_config.field
        if field.kind != "carbajal":
            raise ValueError(
                f"The run {run_config.name!r} is not in the tokamak field (kind "
                f"{field.kind!r}, not 'carbajal')"
            )
        if run_config.rr_scale > 0.0 and run_config.rr_scale != time_scale:
            raise ValueError(
                f"The run {run_config.name!r} has rr_scale {run_config.rr_scale}, "
                f"not the time_scale {time_scale}: E and F_R must be scaled together"
            )
        _force_case(run_config)
        if field.e0 != 0.0:
            scaled_fields.add(field.e0)
        major_radii.add(field.r0)
        case = run_config.model_dump(exclude={"name", "pusher", "rr_scale"})
        del case["field"]["e0"]
        cases.append(case)
    if len(scaled_fields) > 1 or any(case != cases[0] for case in cases):
        raise ValueError(
            "The runs must be one case but for the forces: the same configuration "
            "(dt, duration, field and initial state), and the same e0 when they have "
            f"one; got e0 {sorted(scaled_fields)} V/m"
        )
    e0 = scaled_fields.pop() if scaled_fields else 0.0
    return e0, major_radii.pop()


def tokamak_time_traces(
    result_dirs: Sequence[str | Path],
    time_scale: float,
    pusher: str = "boris",
    *,
    particle: int = 0,
    show: bool = True,
    save_path: str | Path | None = None,
) -> Figure:
    """
    Draw ``gamma``, the pitch angle and ``P_zeta`` of one electron in the tokamak.

    The runs are one case in the field of Carbajal et al. (2017), computed with the
    forces multiplied by ``s = time_scale`` (OPEN_POINTS N44); each folder is one
    choice of forces, read from its configuration: ``E`` and radiation, ``E``
    only, radiation only (one colour and line style each). Three panels share the
    physical time ``s t``, in ms:

    - the Lorentz factor ``gamma``;
    - the pitch angle ``theta`` between ``u`` and the local ``B``, in degrees; it
      oscillates along the orbit, as ``B`` changes;
    - the change of the canonical toroidal momentum, ``Delta P_zeta / (R0 c)``,
      dimensionless (about the change of ``gamma`` for an electron along ``B``).
      The grey dashed line is the torque of ``E`` alone, ``(q / m) e0 t / c``
      (RAPPORT 5.57, ``e0`` the field of the ``.yaml``); the difference is the
      torque of the radiation force.

    ``P_zeta`` is computed by ``diagnostics.canonical_toroidal_momentum`` from the
    stored ``x``, ``u`` and ``Psi``, at the states ``(x^n, u^n)``.

    Parameters
    ----------
    result_dirs : sequence of str or pathlib.Path
        Results folders given by ``run.pusher`` for a ``carbajal`` field, one per
        choice of forces.
    time_scale : float
        The factor ``s`` on ``E`` and ``F_R`` [dimensionless], > 0.
    pusher : str, optional
        The pusher drawn, present in every folder. Default ``"boris"``.
    particle : int, optional
        Index of the particle drawn. Default 0.
    show : bool, optional
        Open the figure in a window. Default ``True``.
    save_path : str or pathlib.Path, optional
        If given, the figure is also saved as a PNG and as a PDF (no extension).

    Returns
    -------
    matplotlib.figure.Figure
        The figure, square, 6 x 6 in, with three panels.

    Raises
    ------
    ValueError
        If ``time_scale`` is not finite and > 0, ``particle`` is not an index of
        the runs, or the runs are not one tokamak case but for the forces: a field
        other than ``carbajal``, ``rr_scale`` other than ``time_scale``, another
        configuration, another ``e0``, a run without any force, or ``pusher``
        missing from a folder.
    """
    run_configs = [
        config.load_config(Path(directory) / CONFIG_FILENAME)
        for directory in result_dirs
    ]
    e0, r0 = _check_tokamak_cases(run_configs, time_scale)
    n_part = len(run_configs[0].initial.u0)
    if isinstance(particle, bool) or not 0 <= particle < n_part:
        raise ValueError(
            f"particle must be an index in [0, {n_part}), got {particle!r}"
        )
    q, m = run_configs[0].q, run_configs[0].m

    with plt.rc_context(figure_style()):
        figure, panels = plt.subplots(3, 1, sharex=True)
        gamma_axes, pitch_axes, momentum_axes = panels
        for directory, run_config in zip(result_dirs, run_configs, strict=True):
            result = read_result(directory, pusher)
            case = _force_case(run_config)
            style = {
                "color": _CASE_COLORS[case],
                "linestyle": _CASE_STYLES[case],
                "linewidth": 1.0,
            }
            time = result["time"].values
            physical_ms = time_scale * time * 1.0e3
            x, u = result["x"].values, result["u"].values
            gamma = np.sqrt(1.0 + np.sum(u[:, particle] ** 2, axis=-1) / c**2)
            p_par, p_perp = _parallel_and_perpendicular_mev(
                u[:, particle], result["b_field"].values[:, particle], m
            )
            p_zeta = diagnostics.canonical_toroidal_momentum(
                x, u, result["poloidal_flux"].values, q, m
            )[:, particle]
            gamma_axes.plot(physical_ms, gamma, label=case, **style)
            pitch_axes.plot(physical_ms, np.degrees(np.arctan2(p_perp, p_par)), **style)
            momentum_axes.plot(physical_ms, (p_zeta - p_zeta[0]) / (r0 * c), **style)
        if e0 != 0.0:
            momentum_axes.plot(
                physical_ms,
                (q / m) * e0 * time / c,
                color="0.6",
                linestyle="--",
                linewidth=1.0,
                label=r"torque of $E$ alone",
            )
            momentum_axes.legend(fontsize=10)
        gamma_axes.set_ylabel(r"$\gamma$")
        pitch_axes.set_ylabel(r"$\theta$ [deg]")
        momentum_axes.set_ylabel(r"$\Delta P_\zeta / (R_0 c)$")
        momentum_axes.set_xlabel(r"physical time $s\,t$ [ms]")
        gamma_axes.legend(fontsize=10)
        theta_0 = _pitch_angle_labels(result)[particle]
        title = f"{pusher}, {theta_0}, $s$ = {time_scale:.3g}"
        gamma_axes.set_title(title, fontsize=12)

    if save_path is not None:
        save_figure(figure, save_path)
    if show:
        plt.show()
    return figure


# Rows of the ensemble figures, in the order of Carbajal et al. (2017, Fig. 12).
_CASE_ORDER = ("radiation only", "E only", "E and radiation")

# The electrons of one ensemble share one pitch angle and one energy (Carbajal et
# al. 2017, Sec. III.D): built from these two values, they differ by rounding only,
# about 1e-15. 1e-9 (radians, and relative on the energy) is far above it, and far
# below the spacing of the ensembles, several degrees.
_ENSEMBLE_SPREAD_TOLERANCE = 1.0e-9


class _FinalEnsemble(NamedTuple):
    """Final state of one ensemble of electrons under one choice of forces.

    Angles are in degrees, energies and momenta (``p c``) in MeV. The final arrays
    hold the confined electrons only; ``n_lost`` counts the others.
    """

    case: str
    pitch_angle_0: float
    kinetic_energy_0: float
    pitch_angle: NDArray[np.float64]
    kinetic_energy: NDArray[np.float64]
    p_par: NDArray[np.float64]
    p_perp: NDArray[np.float64]
    n_lost: int


def _final_ensembles(
    result_dirs: Sequence[str | Path],
    time_scale: float,
    r_edge: float,
    pusher: str,
) -> list[_FinalEnsemble]:
    """Read the final state of ensembles of electrons run in the tokamak.

    The runs are the plan of Carbajal et al. (2017, Fig. 12): ensembles of one
    pitch angle and one energy, each run with several choices of forces (``E``
    and radiation, ``E`` only, radiation only) multiplied by ``s = time_scale``.
    The runs of one ensemble start from the same state, so that the same electrons
    are compared. The pitch angle is measured from the stored local ``B``. The
    electrons lost at one saved state at least (``diagnostics.is_lost``) are left
    out of the final arrays and counted.

    Parameters
    ----------
    result_dirs : sequence of str or pathlib.Path
        Results folders given by ``run.pusher`` for a ``carbajal`` field.
    time_scale : float
        The factor ``s`` on ``E`` and ``F_R`` [dimensionless], > 0.
    r_edge : float
        Minor radius of the edge of the plasma [m], in ``(0, R0)``.
    pusher : str
        The pusher read, present in every folder.

    Returns
    -------
    list of _FinalEnsemble
        One per folder, sorted by case (radiation only, ``E`` only, both) and then
        by initial pitch angle.

    Raises
    ------
    ValueError
        If no folder is given, ``time_scale`` is not finite and > 0, a run is not
        in the ``carbajal`` field, has ``rr_scale`` other than 0 and
        ``time_scale`` or no force at all, the runs differ by anything but their
        forces, time step and initial state (field, ``e0`` when not 0, particle,
        physical duration ``s t``), one ensemble is run twice with the same forces
        or not with the same choices of forces as the others, an ensemble has more
        than one pitch angle or energy, ``r_edge`` is refused by
        ``diagnostics.is_lost``, or ``pusher`` is missing from a folder.
    """
    if not np.isfinite(time_scale) or time_scale <= 0.0:
        raise ValueError(f"time_scale must be finite and > 0, got {time_scale!r}")
    run_configs = [
        config.load_config(Path(directory) / CONFIG_FILENAME)
        for directory in result_dirs
    ]
    if not run_configs:
        raise ValueError("No results folder given")

    shared_parts, scaled_fields, physical_durations = [], set(), []
    for run_config in run_configs:
        field = run_config.field
        if field.kind != "carbajal":
            raise ValueError(
                f"The run {run_config.name!r} is not in the tokamak field (kind "
                f"{field.kind!r}, not 'carbajal')"
            )
        if run_config.rr_scale > 0.0 and run_config.rr_scale != time_scale:
            raise ValueError(
                f"The run {run_config.name!r} has rr_scale {run_config.rr_scale}, "
                f"not the time_scale {time_scale}: E and F_R must be scaled together"
            )
        _force_case(run_config)
        if field.e0 != 0.0:
            scaled_fields.add(field.e0)
        r0 = field.r0  # the same for all the runs, checked below with the field
        shared = run_config.model_dump(
            exclude={
                "name",
                "pusher",
                "rr_scale",
                "dt",
                "n_steps",
                "save_every",
                "seed",
                "initial",
            }
        )
        del shared["field"]["e0"]
        shared_parts.append(shared)
        physical_durations.append(time_scale * run_config.dt * run_config.n_steps)
    if len(scaled_fields) > 1 or any(part != shared_parts[0] for part in shared_parts):
        raise ValueError(
            "The runs must differ by their forces, time step and initial state only: "
            "the same field, the same e0 when they have one, the same particle; got "
            f"e0 {sorted(scaled_fields)} V/m"
        )
    # Each duration is one product of the .yaml values: rounding, about 1e-16.
    if not np.allclose(physical_durations, physical_durations[0], rtol=1e-12, atol=0):
        raise ValueError(
            "The runs must last the same physical time s t; got "
            f"{sorted(set(physical_durations))} s"
        )

    # One ensemble is one initial state; its runs are indexed by their case.
    ensembles: list[tuple[dict[str, Any], dict[str, int]]] = []
    for index, run_config in enumerate(run_configs):
        initial = run_config.initial.model_dump()
        runs_of_ensemble = [runs for state, runs in ensembles if state == initial]
        if runs_of_ensemble:
            runs = runs_of_ensemble[0]
        else:
            runs = {}
            ensembles.append((initial, runs))
        case = _force_case(run_config)
        if case in runs:
            raise ValueError(
                f"The runs {run_configs[runs[case]].name!r} and {run_config.name!r} "
                f"are the same ensemble with the same forces ({case})"
            )
        runs[case] = index
    cases_run = set(ensembles[0][1])
    if any(set(runs) != cases_run for _, runs in ensembles):
        raise ValueError(
            "Every ensemble must be run with the same choices of forces, from the same "
            f"initial state; got {[sorted(runs) for _, runs in ensembles]}"
        )

    m = run_configs[0].m
    final_ensembles = []
    for index, run_config in enumerate(run_configs):
        result = read_result(result_dirs[index], pusher)
        x, u = result["x"].values, result["u"].values
        b_field = result["b_field"].values
        p_par, p_perp = _parallel_and_perpendicular_mev(u[[0, -1]], b_field[[0, -1]], m)
        pitch_angle = np.arctan2(p_perp, p_par)
        kinetic_energy = diagnostics.kinetic_energy(u[[0, -1]], m) / _JOULES_PER_MEV
        pitch_spread = np.ptp(pitch_angle[0])
        energy_spread = np.ptp(kinetic_energy[0]) / np.max(kinetic_energy[0])
        if max(pitch_spread, energy_spread) > _ENSEMBLE_SPREAD_TOLERANCE:
            raise ValueError(
                f"The run {run_config.name!r} is not one ensemble: its electrons "
                "start with several pitch angles or energies"
            )
        confined = ~diagnostics.is_lost(x, r0, r_edge)
        final_ensembles.append(
            _FinalEnsemble(
                case=_force_case(run_config),
                pitch_angle_0=float(np.degrees(pitch_angle[0, 0])),
                kinetic_energy_0=float(kinetic_energy[0, 0]),
                pitch_angle=np.degrees(pitch_angle[1, confined]),
                kinetic_energy=kinetic_energy[1, confined],
                p_par=p_par[1, confined],
                p_perp=p_perp[1, confined],
                n_lost=int(np.count_nonzero(~confined)),
            )
        )
    final_ensembles.sort(
        key=lambda ensemble: (_CASE_ORDER.index(ensemble.case), ensemble.pitch_angle_0)
    )
    return final_ensembles


def final_distributions(
    result_dirs: Sequence[str | Path],
    time_scale: float,
    r_edge: float,
    quantity: Literal["pitch", "energy"],
    pusher: str = "boris",
    *,
    n_bins: int = 40,
    show: bool = True,
    save_path: str | Path | None = None,
) -> Figure:
    """
    Draw the distribution of the pitch angle or energy of ensembles at the end.

    This is a column of Fig. 12 of Carbajal et al. (2017): ``f(theta)``, their
    Eq. 9 (panels b, e, h), or ``f(E)`` of the kinetic energy, their Eq. 17
    (panels c, f, i). Each folder is one ensemble of electrons of one initial
    pitch angle ``theta_0`` and energy ``E_0`` under one choice of forces,
    multiplied by ``s = time_scale`` (OPEN_POINTS N44). One panel per choice of
    forces, in their order: radiation only, ``E`` only, both. In each, one
    probability density per ensemble, normalised to 1 over its confined
    electrons, with the same bins in every panel; its initial value is dotted,
    in its colour. The lost electrons (``diagnostics.is_lost``, at one saved state
    at least) are left out and counted in the legend.

    The pitch angle is measured from the local ``B`` stored with each state.

    Parameters
    ----------
    result_dirs : sequence of str or pathlib.Path
        Results folders given by ``run.pusher`` for a ``carbajal`` field: each
        ensemble with the same choices of forces, from the same initial state.
    time_scale : float
        The factor ``s`` on ``E`` and ``F_R`` [dimensionless], > 0.
    r_edge : float
        Minor radius of the edge of the plasma [m], in ``(0, R0)``.
    quantity : {"pitch", "energy"}
        The pitch angle ``theta`` [deg] or the kinetic energy [MeV] at the end.
    pusher : str, optional
        The pusher drawn, present in every folder. Default ``"boris"``.
    n_bins : int, optional
        Number of bins, spanning all the final values of all the panels, at
        least 1. Default 40.
    show : bool, optional
        Open the figure in a window. Default ``True``.
    save_path : str or pathlib.Path, optional
        If given, the figure is also saved as a PNG and as a PDF (no extension).

    Returns
    -------
    matplotlib.figure.Figure
        The figure, square, 6 x 6 in, with one panel per choice of forces.

    Raises
    ------
    ValueError
        If ``quantity`` is not ``"pitch"`` or ``"energy"``, ``n_bins`` is not an
        integer >= 1, no electron is confined, or the runs are refused by
        ``_final_ensembles`` (not one plan of ensembles and forces).
    """
    if quantity not in ("pitch", "energy"):
        raise ValueError(f"quantity must be 'pitch' or 'energy', got {quantity!r}")
    if isinstance(n_bins, bool) or not isinstance(n_bins, int) or n_bins < 1:
        raise ValueError(f"n_bins must be an integer >= 1, got {n_bins!r}")
    ensembles = _final_ensembles(result_dirs, time_scale, r_edge, pusher)

    if quantity == "pitch":
        final_values = [ensemble.pitch_angle for ensemble in ensembles]
        initial_values = [ensemble.pitch_angle_0 for ensemble in ensembles]
    else:
        final_values = [ensemble.kinetic_energy for ensemble in ensembles]
        initial_values = [ensemble.kinetic_energy_0 for ensemble in ensembles]
    all_values = np.concatenate(final_values)
    if all_values.size == 0:
        raise ValueError("No electron is confined: there is no distribution to draw")
    edges = np.histogram_bin_edges(all_values, bins=n_bins)
    cases_run = {ensemble.case for ensemble in ensembles}
    cases = [case for case in _CASE_ORDER if case in cases_run]
    pitch_angles_0 = sorted({ensemble.pitch_angle_0 for ensemble in ensembles})

    with plt.rc_context(figure_style()):
        figure, panels = plt.subplots(len(cases), 1, sharex=True, squeeze=False)
        for axes, case in zip(panels[:, 0], cases, strict=True):
            for ensemble, values, value_0 in zip(
                ensembles, final_values, initial_values, strict=True
            ):
                if ensemble.case != case:
                    continue
                color = _LINE_COLORS[pitch_angles_0.index(ensemble.pitch_angle_0)]
                label = rf"$\theta_0 = {ensemble.pitch_angle_0:.0f}^\circ$"
                if ensemble.n_lost > 0:
                    label += f", {ensemble.n_lost} lost"
                if values.size > 0:
                    density, _ = np.histogram(values, bins=edges, density=True)
                    axes.stairs(density, edges, color=color, label=label)
                axes.axvline(value_0, color=color, linestyle=":", linewidth=1.0)
            axes.legend(title=case, fontsize=9, title_fontsize=10)
            if quantity == "pitch":
                axes.set_ylabel(r"$f(\theta)$ [1/deg]")
            else:
                axes.set_ylabel(r"$f(\mathcal{E})$ [1/MeV]")
        if quantity == "pitch":
            panels[-1, 0].set_xlabel(r"final pitch angle $\theta$ [deg]")
        else:
            panels[-1, 0].set_xlabel(r"final kinetic energy $\mathcal{E}$ [MeV]")
        panels[0, 0].set_title(f"{pusher}, $s$ = {time_scale:.3g}", fontsize=12)

    if save_path is not None:
        save_figure(figure, save_path)
    if show:
        plt.show()
    return figure


def final_momentum_cloud(
    result_dirs: Sequence[str | Path],
    time_scale: float,
    r_edge: float,
    pusher: str = "boris",
    *,
    show: bool = True,
    save_path: str | Path | None = None,
) -> Figure:
    """
    Draw the final momenta of ensembles of electrons in the plane ``(p_par, p_perp)``.

    This is the first column of Fig. 12 of Carbajal et al. (2017, panels a, d, g):
    one point per confined electron at the end, ``p_par c`` and ``p_perp c`` in
    MeV, measured from the local ``B`` stored with each state. Each folder is one
    ensemble of electrons of one initial pitch angle ``theta_0`` and energy ``E_0``
    under one choice of forces, multiplied by ``s = time_scale`` (OPEN_POINTS
    N44). One panel per choice of forces, in their order: radiation only, ``E``
    only, both; one colour per ensemble. The grey dotted lines are the arc of the
    initial momentum ``|p| c = p_0 c`` and the rays ``theta = theta_0``: they
    cross at the initial state of each ensemble. Points inside the arc lost
    energy; points below their ray turned towards ``B``. The lost electrons
    (``diagnostics.is_lost``) are left out and counted in the legend.

    Parameters
    ----------
    result_dirs : sequence of str or pathlib.Path
        Results folders given by ``run.pusher`` for a ``carbajal`` field: each
        ensemble with the same choices of forces, from the same initial state.
    time_scale : float
        The factor ``s`` on ``E`` and ``F_R`` [dimensionless], > 0.
    r_edge : float
        Minor radius of the edge of the plasma [m], in ``(0, R0)``.
    pusher : str, optional
        The pusher drawn, present in every folder. Default ``"boris"``.
    show : bool, optional
        Open the figure in a window. Default ``True``.
    save_path : str or pathlib.Path, optional
        If given, the figure is also saved as a PNG and as a PDF (no extension).

    Returns
    -------
    matplotlib.figure.Figure
        The figure, square, 6 x 6 in, with one panel per choice of forces.

    Raises
    ------
    ValueError
        If no electron is confined, or the runs are refused by
        ``_final_ensembles`` (not one plan of ensembles and forces).
    """
    ensembles = _final_ensembles(result_dirs, time_scale, r_edge, pusher)
    if all(ensemble.p_par.size == 0 for ensemble in ensembles):
        raise ValueError("No electron is confined: there is no momentum to draw")
    cases_run = {ensemble.case for ensemble in ensembles}
    cases = [case for case in _CASE_ORDER if case in cases_run]
    pitch_angles_0 = sorted({ensemble.pitch_angle_0 for ensemble in ensembles})
    # p_0 c = sqrt(E_0 (E_0 + 2 m c^2)), with E_0 the kinetic energy (MeV here).
    m = config.load_config(Path(result_dirs[0]) / CONFIG_FILENAME).m
    rest_energy = m * c**2 / _JOULES_PER_MEV
    energies_0 = sorted({ensemble.kinetic_energy_0 for ensemble in ensembles})
    momenta_0 = [
        np.sqrt(energy * (energy + 2.0 * rest_energy)) for energy in energies_0
    ]
    reference_style = {"color": "0.6", "linestyle": ":", "linewidth": 1.0}

    with plt.rc_context(figure_style()):
        figure, panels = plt.subplots(
            len(cases), 1, sharex=True, sharey=True, squeeze=False
        )
        for axes, case in zip(panels[:, 0], cases, strict=True):
            for ensemble in ensembles:
                if ensemble.case != case:
                    continue
                label = rf"$\theta_0 = {ensemble.pitch_angle_0:.0f}^\circ$"
                if ensemble.n_lost > 0:
                    label += f", {ensemble.n_lost} lost"
                axes.scatter(
                    ensemble.p_par,
                    ensemble.p_perp,
                    s=2,
                    color=_LINE_COLORS[pitch_angles_0.index(ensemble.pitch_angle_0)],
                    label=label,
                    rasterized=True,
                )
            axes.legend(title=case, fontsize=9, title_fontsize=10, markerscale=4)
            axes.set_ylabel(r"$p_\perp c$ [MeV]")
        panels[-1, 0].set_xlabel(r"$p_\parallel c$ [MeV]")
        panels[0, 0].set_title(f"{pusher}, $s$ = {time_scale:.3g}", fontsize=12)

        # The references are drawn after the points, then the limits of the points
        # are restored, so that the rays do not stretch the axes.
        x_limits, y_limits = panels[0, 0].get_xlim(), panels[0, 0].get_ylim()
        ray_length = 2.0 * max(abs(np.array([*x_limits, *y_limits])))
        arc_angle = np.linspace(0.0, np.pi / 2.0, 200)
        for axes in panels[:, 0]:
            for momentum_0 in momenta_0:
                axes.plot(
                    momentum_0 * np.cos(arc_angle),
                    momentum_0 * np.sin(arc_angle),
                    **reference_style,
                )
            for pitch_angle_0 in pitch_angles_0:
                direction = np.radians(pitch_angle_0)
                axes.plot(
                    [0.0, ray_length * np.cos(direction)],
                    [0.0, ray_length * np.sin(direction)],
                    **reference_style,
                )
        panels[0, 0].set_xlim(x_limits)
        panels[0, 0].set_ylim(y_limits)

    if save_path is not None:
        save_figure(figure, save_path)
    if show:
        plt.show()
    return figure


def velocity(
    result_dir: str | Path,
    pusher: str = "boris",
    *,
    show: bool = True,
    save_path: str | Path | None = None,
) -> Figure:
    """Draw the physical velocity components of every particle over time."""
    result = read_result(result_dir, pusher)
    proper_velocity = result["u"].values
    #gamma = np.sqrt(1.0 + np.sum(proper_velocity**2, axis=-1) / c**2)
    #velocity = proper_velocity / gamma[:, :, np.newaxis]
    time = result["time"].values
    component_names = ("x", "y", "z")

    with plt.rc_context(figure_style()):
        figure, axes = plt.subplots()
        for particle in range(proper_velocity.shape[1]):
            for component, name in enumerate(component_names):
                label = rf"$u_{name}$"
                if proper_velocity.shape[1] > 1:
                    label += f" (particle {particle + 1})"
                axes.plot(time, proper_velocity[:, particle, component], label=label)
        axes.set_xlabel("Time [s]")
        axes.set_ylabel("Proper Velocity [m/s]")
        axes.set_title(f"Velocity components for {pusher} pusher")
        axes.legend()

    if save_path is not None:
        save_figure(figure, save_path)
    if show:
        plt.show()
    return figure



def orbit_guiding_centre_3D(result_dir: str | Path,
    pusher: str | Sequence[str] | None = None,
    *,
    r0: float | None = None,
    a: float | None = None,
    show: bool = True,
    save_path: str | Path | None = None,
) -> Figure:
    """
    Draw the 3D trajectory of each particle, optionally with a reference torus.

    The figure uses the common style of ``figure_style``. To compare schemes, give
    the folder where several pushers were run (see ``run.pusher``): each pusher is
    drawn on the same axes, with its own line style, and the legend names it.

    Parameters
    ----------
    result_dir : str or pathlib.Path
        Results folder given by ``run.pusher``.
    pusher : str or sequence of str, optional
        Which pusher(s) of the folder to draw, for example ``"boris"`` or
        ``["boris", "vay"]``. Default: all the pushers stored in the folder.
    r0 : float, optional
        Major radius [m] of a reference torus surface to draw alongside the
        trajectories, semi-transparent. Omit to draw the trajectories alone.
    a : float, optional
        Minor radius [m] of the reference torus. Required if ``r0`` is given.
    show : bool, optional
        Open the figure in a window. Default ``True``. Use ``False`` in a script that
        only saves the figure.
    save_path : str or pathlib.Path, optional
        If given, the figure is also saved as a PNG and as a PDF, with
        ``save_figure``. This is a file name without extension: ``"orbit"`` writes
        ``orbit.png`` and ``orbit.pdf`` (``"orbit.png"`` gives the same two files).

    Returns
    -------
    matplotlib.figure.Figure
        The figure, square, 7 x 7 in.

    Raises
    ------
    FileNotFoundError
        If no pusher is given and the folder holds no trajectories.
    ValueError
        If a pusher is not in the folder, or ``a`` is missing while ``r0`` is given.
    """
    if pusher is None:
        names = list_results(result_dir)
        if not names:
            raise FileNotFoundError(
                f"No trajectories in {result_dir}: has a simulation been run in this "
                "folder?"
            )
    elif isinstance(pusher, str):
        names = [pusher]
    else:
        names = list(pusher)
    results = {name: read_result(result_dir, name) for name in names}

    if r0 is not None and a is None:
        raise ValueError("a (minor radius) is required when r0 is given")

    with plt.rc_context(figure_style()):
        figure = plt.figure(figsize=(7, 7))
        axes = figure.add_subplot(111, projection="3d")

        # --- Optional reference torus surface, drawn first so it sits behind ---
        if r0 is not None:
            r0_mm = r0 * _METRES_TO_MILLIMETRES
            a_mm = a * _METRES_TO_MILLIMETRES
            theta_grid, phi_grid = np.meshgrid(
                np.linspace(0, 2 * np.pi, 60), np.linspace(0, 2 * np.pi, 100)
            )
            big_r = r0_mm + a_mm * np.cos(theta_grid)
            xt = big_r * np.cos(phi_grid)
            yt = big_r * np.sin(phi_grid)
            zt = a_mm * np.sin(theta_grid)
            axes.plot_surface(
                xt, yt, zt, color="lightgray", alpha=0.25, linewidth=0, shade=True
            )

        # --- Trajectories, same convention as orbit_xz ---
        all_x = []
        for k, (name, result) in enumerate(results.items()):
            x = result["x"].values  # (n_saved, n_part, 3), in metres
            gc = (
                guiding_centre_from_position(
                    x, result["u"].values, result["b_field"].values
                )
                * _METRES_TO_MILLIMETRES
            )
            all_x.append(x)
            n_part = x.shape[1]
            for i in range(n_part):
                if len(results) == 1:
                    label = f"particle {i}"
                elif n_part == 1:
                    label = name
                else:
                    label = f"{name}, particle {i}"
                (line,) = axes.plot(
                    gc[:, i, 0],
                    gc[:, i, 1],
                    gc[:, i, 2],
                    linestyle=_LINE_STYLES[k % len(_LINE_STYLES)],
                    label=label,
                )
                axes.plot(
                    [gc[0, i, 0]],
                    [gc[0, i, 1]],
                    [gc[0, i, 2]],
                    "o",
                    color=line.get_color(),
                )

        if r0 is not None:
            theta = np.linspace(0, 2 * np.pi, 100)
            axes.plot(r0_mm * np.cos(theta), r0_mm * np.sin(theta), "--")
        axes.set_xlabel("x [mm]")
        axes.set_ylabel("y [mm]")
        axes.set_zlabel("z [mm]")
        dt = next(iter(results.values())).attrs["dt"]
        if len(results) == 1:
            title = f"3D guiding orbit, pusher {names[0]!r}, dt = {dt:.3g} s"
        else:
            title = f"3D guiding orbit, dt = {dt:.3g} s"
        axes.set_title(title)

        # Equal aspect ratio: matplotlib 3D does not do this automatically.
        # stacked = np.concatenate(all_x, axis=0).reshape(-1, 3)
        # centre = (stacked.max(axis=0) + stacked.min(axis=0)) / 2
        # half_range = (stacked.max(axis=0) - stacked.min(axis=0)).max() / 2
        # axes.set_xlim(centre[0] - half_range, centre[0] + half_range)
        # axes.set_ylim(centre[1] - half_range, centre[1] + half_range)
        # axes.set_zlim(centre[2] - half_range, centre[2] + half_range)

        if len(axes.get_legend_handles_labels()[0]) > 1:
            axes.legend()

    if save_path is not None:
        save_figure(figure, save_path)
    if show:
        plt.show()
    return figure
