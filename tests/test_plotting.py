"""Tests of ``ppf_c5_runaways.plotting``."""

import matplotlib
import numpy as np
import pytest

matplotlib.use("Agg")  # draw in memory: no window opens during the tests

import matplotlib.pyplot as plt

from ppf_c5_runaways import plotting
from ppf_c5_runaways.config import RunConfig, load_config
from ppf_c5_runaways.io import CONFIG_FILENAME, create_result_dir, write_result


@pytest.fixture
def result_dir(tmp_path):
    """A results folder holding a made-up circular orbit of radius 1 mm."""
    config = RunConfig(
        name="circle",
        pusher="boris",
        dt=1.0e-12,
        n_steps=100,
        save_every=10,
        seed=0,
        q=-1.602176634e-19,
        m=9.1093837139e-31,
        field={"kind": "uniform", "b_field": [0, 0, 2.5], "e_field": [0, 0, 0]},
        initial={"x0": [[1.0e-3, 0, 0]], "u0": [[0, 1.0e7, 0]]},
    )
    directory = create_result_dir(tmp_path, config)
    angle = np.linspace(0.0, 2.0 * np.pi, 11)
    x = np.zeros((11, 1, 3))
    x[:, 0, 0], x[:, 0, 1] = 1.0e-3 * np.cos(angle), 1.0e-3 * np.sin(angle)
    write_result(
        directory, "boris", 1.0e-11 * np.arange(11), x, np.zeros_like(x), 1e-12
    )
    return directory, x


def test_orbit_draws_the_stored_positions_in_millimetres(result_dir):
    # Independent truth: the positions written by hand in the fixture, in metres.
    # The figure shows millimetres, so the plotted data must be 1000 times larger.
    directory, x = result_dir

    figure = plotting.orbit_xy(directory, show=False)

    line = figure.axes[0].lines[0]
    np.testing.assert_allclose(line.get_xdata(), x[:, 0, 0] * 1.0e3, rtol=1e-15)
    np.testing.assert_allclose(line.get_ydata(), x[:, 0, 1] * 1.0e3, rtol=1e-15)
    assert figure.axes[0].get_xlabel() == "x [mm]"
    plt.close(figure)


def test_orbit_is_saved_as_a_png_and_a_pdf(result_dir, tmp_path):
    directory, _ = result_dir

    figure = plotting.orbit_xy(directory, show=False, save_path=tmp_path / "orbit")

    assert (tmp_path / "orbit.png").stat().st_size > 0
    assert (tmp_path / "orbit.pdf").stat().st_size > 0
    plt.close(figure)


def test_orbit_uses_the_common_style(result_dir, tmp_path):
    # Truth: the specification of the style, 6 x 6 in figure, 300 dpi export, labels
    # at least 14 pt, full frame, ticks on the left and right sides.
    # 1800 pixels = 6 in x 300 dpi, by hand.
    directory, _ = result_dir

    figure = plotting.orbit_xy(directory, show=False, save_path=tmp_path / "orbit")

    axes = figure.axes[0]
    assert tuple(figure.get_size_inches()) == (6.0, 6.0)
    assert axes.xaxis.label.get_fontsize() >= 14
    assert axes.yaxis.label.get_fontsize() >= 14
    assert all(spine.get_visible() for spine in axes.spines.values())
    # tick1 is the left tick of the y axis, tick2 the right one.
    y_tick = axes.yaxis.get_major_ticks()[0]
    assert y_tick.tick1line.get_visible()
    assert y_tick.tick2line.get_visible()
    assert plt.imread(tmp_path / "orbit.png").shape[:2] == (1800, 1800)
    plt.close(figure)


def test_style_makes_square_figures():
    # Truth: the specification, "figures must be square".
    with plt.rc_context(plotting.figure_style()):
        figure = plt.figure()
    width, height = figure.get_size_inches()
    assert width == height  # both set to 6.0 in: exact, no computation involved
    plt.close(figure)


def test_style_uses_large_text():
    # Truth: the specification. 14 pt is the smallest size that stays readable when a
    # figure is reduced to one column of an article.
    with plt.rc_context(plotting.figure_style()):
        sizes = [
            plt.rcParams["axes.labelsize"],
            plt.rcParams["xtick.labelsize"],
            plt.rcParams["ytick.labelsize"],
            plt.rcParams["legend.fontsize"],
        ]
    assert min(sizes) >= 14


def test_style_does_not_leak_into_the_global_settings():
    # Truth: matplotlib's own rcParams, read before and after. The style must apply
    # only inside the ``with`` block.
    before = dict(plt.rcParams)

    with plt.rc_context(plotting.figure_style()):
        pass

    assert dict(plt.rcParams) == before


def test_style_returns_a_new_dictionary_each_time():
    first = plotting.figure_style()
    first["axes.labelsize"] = 1

    second = plotting.figure_style()

    assert second["axes.labelsize"] != 1


def test_style_saves_a_square_png_of_1800_pixels(tmp_path):
    # Truth: 6 in x 300 dpi = 1800 pixels, computed by hand. Exact equality: pixel
    # counts are integers. A cropped export (bbox_inches="tight") would fail here.
    picture = tmp_path / "square.png"
    with plt.rc_context(plotting.figure_style()):
        figure, axes = plt.subplots()
        axes.plot([0, 1], [0, 1])
        axes.set_xlabel("x [mm]")
        figure.savefig(picture)
    plt.close(figure)

    height, width = plt.imread(picture).shape[:2]

    assert (height, width) == (1800, 1800)


@pytest.fixture
def styled_figure():
    """A square figure drawn with the project style, with some text on it."""
    with plt.rc_context(plotting.figure_style()):
        figure, axes = plt.subplots()
        axes.plot([0, 1], [0, 1])
        axes.set_xlabel("x [mm]")
    yield figure
    plt.close(figure)


def test_save_figure_writes_a_png_and_a_pdf(styled_figure, tmp_path):
    # Truth: the file formats themselves. A PNG starts with the 8 signature bytes
    # of the PNG standard, a PDF with "%PDF-".
    png_path, pdf_path = plotting.save_figure(styled_figure, tmp_path / "orbit")

    assert (png_path, pdf_path) == (tmp_path / "orbit.png", tmp_path / "orbit.pdf")
    assert png_path.read_bytes().startswith(b"\x89PNG\r\n\x1a\n")
    assert pdf_path.read_bytes().startswith(b"%PDF-")


def test_save_figure_png_is_1800_pixels_and_pdf_is_432_points(styled_figure, tmp_path):
    # Truth, by hand: 6 in x 300 dpi = 1800 pixels; 6 in x 72 pt/in = 432 pt.
    # Exact equality: pixel counts are integers, and the PDF page size is written
    # as an integer number of points.
    png_path, pdf_path = plotting.save_figure(styled_figure, tmp_path / "orbit")

    height, width = plt.imread(png_path).shape[:2]
    assert (height, width) == (1800, 1800)
    assert b"/MediaBox [ 0 0 432 432 ]" in pdf_path.read_bytes()


def test_save_figure_applies_the_saving_settings_outside_the_style_block(tmp_path):
    # The figure is drawn with the plain matplotlib defaults (dpi 100, Type 3 fonts
    # in PDF); the saving settings of the project must still apply at the export.
    # Truth: 6 in x 300 dpi = 1800 pixels; a PDF with embedded TrueType fonts holds
    # a /FontFile2 stream and no /Type3 font (matplotlib documentation of fonttype).
    figure, axes = plt.subplots(figsize=(6, 6))
    axes.set_xlabel("x [mm]")

    png_path, pdf_path = plotting.save_figure(figure, tmp_path / "plain")

    plt.close(figure)
    assert plt.imread(png_path).shape[:2] == (1800, 1800)
    pdf_bytes = pdf_path.read_bytes()
    assert b"/FontFile2" in pdf_bytes
    assert b"/Type3" not in pdf_bytes


def test_save_figure_removes_only_a_png_or_pdf_extension(styled_figure, tmp_path):
    png_path, pdf_path = plotting.save_figure(styled_figure, tmp_path / "orbit.png")
    assert (png_path.name, pdf_path.name) == ("orbit.png", "orbit.pdf")

    png_path, pdf_path = plotting.save_figure(styled_figure, tmp_path / "run.v2")
    assert (png_path.name, pdf_path.name) == ("run.v2.png", "run.v2.pdf")


def test_save_figure_does_not_change_the_global_settings(styled_figure, tmp_path):
    before = dict(plt.rcParams)

    plotting.save_figure(styled_figure, tmp_path / "orbit")

    assert dict(plt.rcParams) == before


def test_save_figure_refuses_something_that_is_not_a_figure(tmp_path):
    with pytest.raises(TypeError, match="must be a matplotlib Figure, got str"):
        plotting.save_figure("not a figure", tmp_path / "orbit")  # type: ignore[arg-type]


@pytest.fixture
def three_pushers_dir(result_dir):
    """The folder above, where two more pushers were added with other radii.

    The orbit of ``higuera_cary`` is 1.1 times that of ``boris``, and the orbit of
    ``vay`` is 1.2 times: made up, so that each curve can be recognised.
    """
    directory, x = result_dir
    for name, scale in (("higuera_cary", 1.1), ("vay", 1.2)):
        write_result(
            directory, name, 1.0e-11 * np.arange(11), scale * x, np.zeros_like(x), 1e-12
        )
    return directory, x


def _named_lines(axes):
    """The curves that have a legend entry (not the markers of the start points)."""
    return [line for line in axes.lines if not line.get_label().startswith("_")]


def test_orbit_draws_every_pusher_of_the_folder_with_a_clear_legend(three_pushers_dir):
    # Independent truth: the positions written by hand in the fixture (metres), times
    # the made-up scale of each pusher, times 1000 for millimetres. This is the
    # workflow of main.py: run three pushers in one folder, then draw one figure.
    directory, x = three_pushers_dir

    figure = plotting.orbit_xy(directory, show=False)

    axes = figure.axes[0]
    legend = [text.get_text() for text in axes.get_legend().get_texts()]
    assert legend == ["boris", "higuera_cary", "vay"]
    lines = _named_lines(axes)
    assert [line.get_label() for line in lines] == legend
    for line, scale in zip(lines, (1.0, 1.1, 1.2), strict=True):
        np.testing.assert_allclose(
            line.get_xdata(), scale * x[:, 0, 0] * 1.0e3, rtol=1e-15
        )
    # Curves that lie on top of each other must stay distinguishable.
    assert len({line.get_linestyle() for line in lines}) == 3
    assert len({line.get_color() for line in lines}) == 3
    plt.close(figure)


def test_orbit_draws_only_the_asked_pushers_in_the_order_given(three_pushers_dir):
    directory, _ = three_pushers_dir

    figure = plotting.orbit_xy(directory, pusher=["vay", "boris"], show=False)

    legend = [text.get_text() for text in figure.axes[0].get_legend().get_texts()]
    assert legend == ["vay", "boris"]
    plt.close(figure)


def test_orbit_of_one_pusher_names_it_in_the_title_and_has_no_legend(
    three_pushers_dir,
):
    directory, _ = three_pushers_dir

    figure = plotting.orbit_xy(directory, pusher="vay", show=False)

    assert "'vay'" in figure.axes[0].get_title()
    assert figure.axes[0].get_legend() is None  # one curve: nothing to tell apart
    plt.close(figure)


def test_orbit_of_a_folder_without_trajectories_says_so(result_dir, tmp_path):
    directory, _ = result_dir
    config = load_config(directory / CONFIG_FILENAME)
    empty = create_result_dir(tmp_path / "empty", config)

    with pytest.raises(FileNotFoundError, match="No trajectories"):
        plotting.orbit_xy(empty, show=False)


def test_orbit_of_a_missing_pusher_lists_the_available_ones(result_dir):
    directory, _ = result_dir

    with pytest.raises(ValueError, match=r"'vay'.*Available: \['boris'\]"):
        plotting.orbit_xy(directory, pusher="vay", show=False)
