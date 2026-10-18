"""Figures made from a results folder."""

from collections.abc import Sequence
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.figure import Figure
from matplotlib.lines import Line2D
from matplotlib.rcsetup import cycler
from matplotlib.typing import RcKeyType
from numpy.typing import NDArray

import ppf_c5_runaways.config as config
import ppf_c5_runaways.fields as fields
from ppf_c5_runaways import diagnostics
from ppf_c5_runaways.constants import c
from ppf_c5_runaways.io import CONFIG_FILENAME, list_results, read_result

# Lengths are computed in metres, and shown in millimetres because a Larmor radius in
# a tokamak is of that order. The conversion is done here, at the edge, only to draw.
_METRES_TO_MILLIMETRES = 1.0e3

_LINE_COLORS = [
    "#0072B2",  # blue
    "#D55E00",  # vermillion
    "#009E73",  # bluish green
    "#E69F00",  # orange
    "#56B4E9",  # sky blue
    "#CC79A7",  # reddish purple
    "#000000",  # black
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
        #axes.set_aspect("equal")  # a circle must look like a circle
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
        theta = np.linspace(0,2*np.pi,100)
        axes.plot(r0_mm*np.cos(theta),r0_mm*np.sin(theta),'--')
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
        #stacked = np.concatenate(all_x, axis=0).reshape(-1, 3)
        #centre = (stacked.max(axis=0) + stacked.min(axis=0)) / 2
        #half_range = (stacked.max(axis=0) - stacked.min(axis=0)).max() / 2
        #axes.set_xlim(centre[0] - half_range, centre[0] + half_range)
        #axes.set_ylim(centre[1] - half_range, centre[1] + half_range)
        #axes.set_zlim(centre[2] - half_range, centre[2] + half_range)

        if len(axes.get_legend_handles_labels()[0]) > 1:
            axes.legend()

    if save_path is not None:
        save_figure(figure, save_path)
    if show:
        plt.show()
    return figure

def visufields(
    result_dir: str | Path,
    pusher: str | Sequence[str] | None = None,
    *,
    r0: float | None = None,
    a: float | None = None,
) -> Figure:

    N = 30
    r = np.linspace(0,a,N)
    phi = np.linspace(0,2*np.pi,N//2)

    R, TH = np.meshgrid(r,phi)

    cfg = config.load_config("configs/tokamak_custom_toroidal_boris.yaml")
    field = fields.custom_toroidal(b0=cfg.field.b0, r0=cfg.field.r0,  q0=cfg.field.q0,  lamb=cfg.field.lamb,  b_r=cfg.field.b_r, b_theta=cfg.field.b_theta, b_phi=cfg.field.b_phi)

    X = np.zeros_like(R)
    Y = r0 + R * np.cos(TH)
    Z = R * np.sin(TH)

    x_flat = np.stack([X.ravel(), Y.ravel(), Z.ravel()], axis=-1)
    e_flat, b_flat = field(x_flat, 0.0)   # b_flat.shape == (N*N, 3)

    By_plot = b_flat[:, 1].reshape(N, N//2)
    Bz_plot = b_flat[:, 2].reshape(N, N//2)


    fig, ax = plt.subplots()
    ax.quiver(Y, Z, By_plot, Bz_plot, color="C0", angles='xy',
              scale_units='xy', scale=0.5, width=.005)



    phi_cercle = np.linspace(0,2*np.pi,200)
    ax.plot(r0+(a)*np.cos(phi_cercle),(a)*np.sin(phi_cercle))
    ax.set_aspect("equal")
    plt.show()
    plt.xlim(4,7)
    return fig

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
        axes.axhline(rounding, color="0.6", linewidth=1.0, label="float64 rounding")
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
