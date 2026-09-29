"""Figures made from a results folder."""

from collections.abc import Sequence
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.rcsetup import cycler
from matplotlib.typing import RcKeyType

import numpy as np

from ppf_c5_runaways.io import list_results, read_result
import ppf_c5_runaways.fields as fields
import ppf_c5_runaways.config as config

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
                    [x[0, i, 0]], [x[0, i, 1]], [x[0, i, 2]], "o", color=line.get_color()
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

def visu_toroidal_fields(
    result_dir: str | Path,
    pusher: str | Sequence[str] | None = None,
    *,
    r0: float | None = None,
    a: float | None = None,
) -> Figure:

    cfg = config.load_config("configs/tokamak_custom_toroidal_boris.yaml")
    field = fields.custom_toroidal(
        b0=cfg.field.b0, r0=cfg.field.r0, q0=cfg.field.q0,
        lamb=cfg.field.lamb, b_r=cfg.field.b_r,
        b_theta=cfg.field.b_theta, b_phi=cfg.field.b_phi,
    )

    N = 20
    r = np.linspace(0, a, N)
    theta = np.linspace(0, 2*np.pi, N)
    R, TH = np.meshgrid(r, theta)

    # côté phi = 0 : tube "proche" (Y > 0)
    X_near = np.zeros_like(R)
    Y_near = r0 + R * np.cos(TH)
    Z_near = R * np.sin(TH)

    # côté phi = pi : tube "opposé" (Y < 0)
    X_far = np.zeros_like(R)
    Y_far = -(r0 + R * np.cos(TH))
    Z_far = R * np.sin(TH)

    def get_B(X, Y, Z):
        x_flat = np.stack([X.ravel(), Y.ravel(), Z.ravel()], axis=-1)
        _, b_flat = field(x_flat, 0.0)
        Bx = b_flat[:, 0].reshape(X.shape)
        By = b_flat[:, 1].reshape(X.shape)
        Bz = b_flat[:, 2].reshape(X.shape)
        return Bx, By, Bz

    Bx_near, By_near, Bz_near = get_B(X_near, Y_near, Z_near)
    Bx_far,  By_far,  Bz_far  = get_B(X_far, Y_far, Z_far)

    fig, ax = plt.subplots()

    ax.quiver(Y_near, Z_near, By_near, Bz_near, color="C0", angles='xy',
              scale_units='xy', scale=5, width=.005, label="phi = 0")
    ax.quiver(Y_far, Z_far, By_far, Bz_far, color="C1", angles='xy',
              scale_units='xy', scale=5, width=.005, label="phi = pi")

    theta_cercle = np.linspace(0, 2*np.pi, 200)
    ax.plot(r0 + a*np.cos(theta_cercle), a*np.sin(theta_cercle), color="C0")
    ax.plot(-(r0 + a*np.cos(theta_cercle)), a*np.sin(theta_cercle), color="C1")

    ax.axvline(0, color="grey", linestyle="--", linewidth=0.8)  # axe du tokamak
    ax.set_xlabel("Y (m)")
    ax.set_ylabel("Z (m)")
    ax.set_aspect("equal")
    ax.legend()
    ax.set_title("Coupe poloïdale complète (plan X=0)")


    fig, ax = plt.subplots()
    r = np.linspace(-r0 - a, r0 + a, 100)
    plt.plot(r , np.sqrt(get_B(r,np.zeros(len(r)),np.zeros(len(r)))[0]**2 + get_B(r,np.zeros(len(r)),np.zeros(len(r)))[1]**2 + get_B(r,np.zeros(len(r)),np.zeros(len(r)))[2]**2))
    plt.ylim(3,8)
    plt.xlim(-r0 - a, r0 + a)
    plt.axvline(x=-r0 - a, color="grey", linestyle="--")
    plt.axvline(x=-r0 + a, color="grey", linestyle="--")
    plt.axvline(x=r0 -a , color="grey", linestyle="--")
    plt.axvline(x=r0 + a, color="grey", linestyle="--")
    ax.axvspan(xmin=-r0 + a, xmax=r0 - a, hatch="//", facecolor="none", edgecolor="grey")
    plt.show()
    return fig