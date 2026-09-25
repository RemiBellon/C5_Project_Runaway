"""Figures made from a results folder."""

from collections.abc import Sequence
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.rcsetup import cycler
from matplotlib.typing import RcKeyType

from ppf_c5_runaways.io import list_results, read_result

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
        axes.set_aspect("equal")  # a circle must look like a circle
        if len(axes.get_legend_handles_labels()[0]) > 1:
            axes.legend()

    if save_path is not None:
        save_figure(figure, save_path)
    if show:
        plt.show()
    return figure


def orbit_yz(
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
                    x[:, i, 1],
                    x[:, i, 2],
                    linestyle=_LINE_STYLES[k % len(_LINE_STYLES)],
                    label=label,
                )
                axes.plot(x[0, i, 1], x[0, i, 2], "o", color=line.get_color())
        axes.set_xlabel("y [mm]")
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


def orbit_3D(
    result_dir: str | Path,
    pusher: str | Sequence[str] | None = None,
    *,
    show: bool = True,
    save_path: str | Path | None = None,
) -> Figure:
    """
    Draw the trajectory of each particle in 3D.

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
            figure = plt.figure()
            axes = plt.axes(projection='3d')  
            
            for k, (name, result) in enumerate(results.items()):
                x = result["x"].values * _METRES_TO_MILLIMETRES  # (n_saved, n_part, 3)
    
                print(result["x"].shape)
    
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
                        label=label
                    )
                    axes.plot(x[0, i, 0], x[0, i, 1], x[0, i, 2], "o", color=line.get_color())
            axes.set_xlabel("x [mm]")
            axes.set_ylabel("y [mm]")
            axes.set_zlabel("z [mm]")
            dt = next(iter(results.values())).attrs["dt"]  # the same for the whole folder
            if len(results) == 1:
                title = f"3D Orbit, pusher {names[0]!r}, dt = {dt:.3g} s"
            else:
                title = f"3D Orbit, dt = {dt:.3g} s"
            axes.set_title(title)
            axes.set_aspect("equal")  # a circle must look like a circle
            if len(axes.get_legend_handles_labels()[0]) > 1:
                axes.legend()
    
    if save_path is not None:
            save_figure(figure, save_path)
    if show:
            plt.show()
    return figure






