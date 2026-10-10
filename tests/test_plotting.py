"""Tests of ``ppf_c5_runaways.plotting``."""

import math

import matplotlib
import numpy as np
import pytest
import xarray as xr

matplotlib.use("Agg")  # draw in memory: no window opens during the tests

import matplotlib.pyplot as plt

from ppf_c5_runaways import plotting
from ppf_c5_runaways.config import RunConfig, load_config
from ppf_c5_runaways.constants import c
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
        field={
            "kind": "uniform_cartesian",
            "b_field": [0, 0, 2.5],
            "e_field": [0, 0, 0],
        },
        initial={"x0": [[1.0e-3, 0, 0]], "u0": [[0, 1.0e7, 0]]},
    )
    directory = create_result_dir(tmp_path, config)
    angle = np.linspace(0.0, 2.0 * np.pi, 11)
    x = np.zeros((11, 1, 3))
    x[:, 0, 0], x[:, 0, 1] = 1.0e-3 * np.cos(angle), 1.0e-3 * np.sin(angle)
    velocity = np.zeros_like(x)
    write_result(
        directory,
        "boris",
        1.0e-11 * np.arange(11),
        x,
        velocity,
        1e-12,
        u_centered=velocity,
        b_field=velocity,
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
        velocity = np.zeros_like(x)
        write_result(
            directory,
            name,
            1.0e-11 * np.arange(11),
            scale * x,
            velocity,
            1e-12,
            u_centered=velocity,
            b_field=velocity,
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


# --- energy_error ---------------------------------------------------------------------


@pytest.fixture
def energy_dir(result_dir):
    """A results folder with two made-up pushers of known energy error.

    ``boris``: gamma = 2, 3, 2 (|u|/c = sqrt 3, sqrt 8, sqrt 3), so K = m c^2,
    2 m c^2, m c^2 and the relative error is 0, 1, 0. ``vay``: gamma = 2 at the
    three times, error exactly 0.
    """
    directory, _ = result_dir  # holds "boris" already: use a new folder
    config = load_config(directory / CONFIG_FILENAME)
    energy_directory = create_result_dir(directory.parent / "energy", config)
    t = 1.0e-11 * np.arange(3)
    x = np.zeros((3, 1, 3))
    for name, u_over_c in (
        ("boris", [np.sqrt(3.0), np.sqrt(8.0), np.sqrt(3.0)]),
        ("vay", [np.sqrt(3.0)] * 3),
    ):
        u = np.zeros((3, 1, 3))
        u[:, 0, 0] = np.array(u_over_c) * c
        write_result(energy_directory, name, t, x, u, 1e-12, u_centered=u, b_field=u)
    return energy_directory


def test_energy_error_draws_the_known_errors_on_a_log_axis(energy_dir):
    # Independent truth: the errors computed by hand in the fixture. The exact zeros
    # are drawn at the float64 rounding unit, where the grey line is, so that the
    # curve of a perfect scheme stays visible on the logarithmic axis. Tolerance
    # 1e-14: sqrt(3)^2 and sqrt(8)^2 are exact to a few 1e-16.
    rounding = np.finfo(np.float64).eps

    figure = plotting.energy_error(energy_dir, show=False)

    axes = figure.axes[0]
    assert axes.get_yscale() == "log"
    lines = {line.get_label(): line for line in _named_lines(axes)}
    assert list(lines) == ["boris", "vay", "float64 rounding"]
    np.testing.assert_allclose(
        lines["boris"].get_ydata(), [rounding, 1.0, rounding], rtol=1e-14
    )
    np.testing.assert_array_equal(lines["vay"].get_ydata(), [rounding] * 3)
    np.testing.assert_array_equal(lines["boris"].get_xdata(), 1.0e-11 * np.arange(3))
    assert axes.get_xlabel() == "t [s]"


def test_energy_error_of_one_pusher_names_it_in_the_title(energy_dir):
    figure = plotting.energy_error(energy_dir, pusher="vay", show=False)

    assert "'vay'" in figure.axes[0].get_title()


def test_energy_error_is_saved_as_a_png_and_a_pdf(energy_dir, tmp_path):
    plotting.energy_error(energy_dir, show=False, save_path=tmp_path / "energy")

    assert (tmp_path / "energy.png").is_file()
    assert (tmp_path / "energy.pdf").is_file()


def test_energy_error_of_a_result_without_u_centered_says_to_run_again(tmp_path):
    # A folder written before u_centered was stored: only x and u in the group.
    directory = tmp_path / "old"
    directory.mkdir()
    old = xr.Dataset(
        {
            "x": (("time", "particle", "axis"), np.zeros((2, 1, 3))),
            "u": (("time_half", "particle", "axis"), np.ones((2, 1, 3))),
        },
        attrs={"pusher": "boris", "dt": 1e-12},
    )
    old.to_netcdf(directory / "data.nc", group="boris", engine="h5netcdf")

    with pytest.raises(ValueError, match=r"no u_centered.*Run it again"):
        plotting.energy_error(directory, show=False)


# --- drift_error ----------------------------------------------------------------------


def _drift_dir(base, name, u_over_c, factors):
    """A folder of uniform crossed fields with drift proper velocity ``u_over_c c``.

    ``factors`` maps a pusher to a made-up drift: the stored positions are
    ``factor * v_E * t``, so the relative drift error is exactly ``|factor - 1|``.
    """
    kappa = np.sqrt(1.0 + u_over_c**2)
    b_z = 2.5
    config = RunConfig(
        name=name,
        pusher="boris",
        dt=1.0e-12,
        n_steps=10,
        save_every=1,
        seed=0,
        q=-1.602176634e-19,
        m=9.1093837139e-31,
        field={
            "kind": "uniform_cartesian",
            "b_field": [0.0, 0.0, b_z],
            "e_field": [u_over_c / kappa * c * b_z, 0.0, 0.0],
        },
        initial={"x0": [[0, 0, 0]], "u0": [[0, -u_over_c * c, 0]]},
    )
    directory = create_result_dir(base, config)
    t = 1.0e-12 * np.arange(11)
    v_drift = np.array([0.0, -u_over_c / kappa * c, 0.0])  # E x B / B^2, along -y
    for pusher, factor in factors.items():
        x = np.zeros((11, 1, 3))
        x[:, 0, :] = factor * v_drift * t[:, np.newaxis]
        u = np.ones_like(x)
        write_result(directory, pusher, t, x, u, 1.0e-12, u_centered=u, b_field=u)
    return directory


def test_drift_error_draws_the_known_errors_against_the_drift_speed(tmp_path):
    # Independent truth: the made-up drifts of the folders, 10 % and 1 % too fast for
    # "boris", exact for "vay" (drawn on the rounding line). The points are sorted
    # by u_E / c whatever the order of the folders. Tolerance 1e-10: the fit of a
    # straight line over 11 points is exact to rounding.
    fast = _drift_dir(tmp_path, "fast", 100.0, {"boris": 1.01, "vay": 1.0})
    slow = _drift_dir(tmp_path, "slow", 1.0e-3, {"boris": 1.1, "vay": 1.0})

    figure = plotting.drift_error([fast, slow], show=False)

    axes = figure.axes[0]
    assert (axes.get_xscale(), axes.get_yscale()) == ("log", "log")
    lines = {line.get_label(): line for line in _named_lines(axes)}
    assert list(lines) == ["boris", "vay", "float64 rounding"]
    np.testing.assert_allclose(lines["boris"].get_xdata(), [1.0e-3, 100.0], rtol=1e-12)
    np.testing.assert_allclose(lines["boris"].get_ydata(), [0.1, 0.01], rtol=1e-10)
    assert np.all(lines["vay"].get_ydata() <= 1e-15)


def test_drift_error_refuses_a_folder_without_crossed_fields(result_dir):
    directory, _ = result_dir  # uniform B alone: no E x B drift

    with pytest.raises(ValueError, match="perpendicular to B"):
        plotting.drift_error([directory], show=False)


def test_drift_error_needs_at_least_one_folder():
    with pytest.raises(ValueError, match="empty"):
        plotting.drift_error([], show=False)


# --- figures of the verification cases 11, 14, 17, 18 ------------------------------


def _folder(tmp_path, name, field, u0, x0=((0.0, 0.0, 0.0),)):
    """An empty results folder for a made-up electron run in ``field``."""
    config = RunConfig(
        name=name,
        pusher="boris",
        dt=1.0e-12,
        n_steps=10,
        save_every=1,
        seed=0,
        q=-1.602176634e-19,
        m=9.1093837139e-31,
        field=field,
        initial={"x0": [list(row) for row in x0], "u0": [list(row) for row in u0]},
    )
    return create_result_dir(tmp_path, config)


def test_phase_error_draws_a_known_lag(tmp_path):
    # Independent truth: velocities turned by hand at the exact electron rate
    # +e B / (gamma m) about z, plus a made-up error: -1e-6 rad at t = 0, then a lag
    # of 0.01 rad per exact gyration. The figure must show that error against
    # t / T_c. u0 = (0, -|u|, 0) lies on the cut of atan2 (read as -pi), and the
    # -1e-6 puts the first stored velocity on the other side (read as pi - 1e-6):
    # a spurious 2 pi must not appear. Tolerance 1e-10 rad: rounding of the angles.
    b_z, gamma = 2.5, 2.0
    speed = np.sqrt(gamma**2 - 1.0) * c
    directory = _folder(
        tmp_path,
        "phase",
        {"kind": "uniform_cartesian", "b_field": [0, 0, b_z], "e_field": [0, 0, 0]},
        [(0.0, -speed, 0.0)],
    )
    rate = 1.602176634e-19 * b_z / (gamma * 9.1093837139e-31)
    period = 2.0 * np.pi / rate
    t = np.linspace(0.0, 3.0 * period, 301)  # 0.063 rad between points
    lag = -1.0e-6 - 0.01 * t / period
    angle = rate * t + lag
    u = np.zeros((t.size, 1, 3))
    u[:, 0, 0] = speed * np.sin(angle)  # rotation of (0, -speed, 0) by +angle
    u[:, 0, 1] = -speed * np.cos(angle)
    write_result(
        directory, "boris", t, np.zeros_like(u), u, 1e-12, u_centered=u, b_field=u
    )

    figure = plotting.phase_error(directory, show=False)

    lines = {line.get_label(): line for line in _named_lines(figure.axes[0])}
    assert list(lines) == ["particle 0", "prediction"]
    np.testing.assert_allclose(lines["particle 0"].get_xdata(), t / period, rtol=1e-12)
    np.testing.assert_allclose(
        lines["particle 0"].get_ydata(), lag, rtol=0.0, atol=1e-10
    )
    # The prediction for "boris" with the dt of the configuration (1e-12 s):
    # offset arctan(theta / 2) - 2 arctan(theta / 4), then 2 arctan(theta / 2) -
    # theta per step (Zenitani & Umeda 2018, Eq. 12), written out here. Tolerance
    # 1e-12 relative: the figure computes the angle as 2 arctan(|beta| / gamma), in
    # another order, so the rounding differs by a few eps per step, times up to 90
    # steps (observed 3e-14).
    theta = rate * 1.0e-12
    steps = np.round(t / 1.0e-12)
    expected = (math.atan(theta / 2.0) - 2.0 * math.atan(theta / 4.0)) + steps * (
        2.0 * math.atan(theta / 2.0) - theta
    )
    np.testing.assert_allclose(
        lines["prediction"].get_ydata(), expected, rtol=1e-12, atol=0.0
    )


def test_phase_error_refuses_a_field_with_an_electric_part(tmp_path):
    directory = _folder(
        tmp_path,
        "exb",
        {"kind": "uniform_cartesian", "b_field": [0, 0, 2.5], "e_field": [1, 0, 0]},
        [(1.0e7, 0.0, 0.0)],
    )

    with pytest.raises(ValueError, match="uniform B and no E"):
        plotting.phase_error(directory, show=False)


def test_mirror_trajectory_draws_z_over_l_and_the_predicted_mirror_points(tmp_path):
    # Independent truth: pitch angle 45 degrees at a point where |B| = B0, so the
    # mirror field is 2 B0 and z_max = L sqrt(2 - 1) = L (Ripperda et al. 2018,
    # Eq. 61): dashed lines at z / L = +1 and -1.
    length = 10.0
    directory = _folder(
        tmp_path,
        "bottle",
        {"kind": "mirror", "b0": 2.5, "length": length},
        [(0.0, 1.0e8, 1.0e8)],
    )
    t = 1.0e-9 * np.arange(5)
    x = np.zeros((5, 1, 3))
    x[:, 0, 2] = [0.0, 5.0, 10.0, 5.0, 0.0]
    b_field = np.zeros_like(x)
    b_field[:, 0, 2] = 2.5
    write_result(directory, "boris", t, x, x, 1e-12, u_centered=x, b_field=b_field)

    figure = plotting.mirror_trajectory(directory, show=False)

    axes = figure.axes[0]
    lines = {line.get_label(): line for line in _named_lines(axes)}
    np.testing.assert_allclose(lines["particle 0"].get_ydata(), [0, 0.5, 1, 0.5, 0])
    dashed = [
        line.get_ydata()[0] for line in axes.lines if line.get_linestyle() == "--"
    ]
    np.testing.assert_allclose(sorted(dashed), [-1.0, 1.0], rtol=1e-12)


def test_magnetic_moment_error_uses_the_half_step_velocity_by_default(tmp_path):
    # Independent truth: mu = m u_perp^2 / (2 |B|). The half-step velocities keep
    # u_perp and B fixed except at the second time, where u_perp^2 is 10 % larger:
    # errors eps (drawn on the rounding line), 0.1, eps. u_centered is made
    # different on purpose, so that using it would give other numbers.
    directory = _folder(
        tmp_path,
        "mu",
        {"kind": "mirror", "b0": 2.5, "length": 10.0},
        [(1.0e7, 0.0, 1.0e7)],
    )
    t = 1.0e-9 * np.arange(3)
    u = np.zeros((3, 1, 3))
    u[:, 0, 0] = 1.0e7 * np.sqrt([1.0, 1.1, 1.0])
    u[:, 0, 2] = 5.0e6
    b_field = np.zeros_like(u)
    b_field[:, 0, 2] = 2.5
    write_result(
        directory,
        "boris",
        t,
        np.zeros_like(u),
        u,
        1e-12,
        u_centered=2 * u,
        b_field=b_field,
    )

    figure = plotting.magnetic_moment_error(directory, show=False)

    (line, _) = _named_lines(figure.axes[0])  # the particle, then the rounding line
    rounding = np.finfo(np.float64).eps
    np.testing.assert_allclose(line.get_ydata(), [rounding, 0.1, rounding], rtol=1e-12)


def test_phase_space_volume_is_the_product_of_the_step_determinants(tmp_path):
    # Independent truth: three steps that each multiply the volume by 1.1, then one
    # of determinant 1: ln V = 0, ln 1.1, 2 ln 1.1, 3 ln 1.1 at the four times.
    directory = _folder(
        tmp_path,
        "volume",
        {"kind": "uniform_cartesian", "b_field": [0, 0, 2.5], "e_field": [0, 0, 0]},
        [(1.0e7, 0.0, 0.0)],
    )
    t = 1.0e-12 * np.arange(4)
    u = np.ones((4, 1, 3))
    determinants = np.array([[1.1], [1.1], [1.1], [1.0]])
    write_result(
        directory,
        "vay",
        t,
        u,
        u,
        1.0e-12,
        u_centered=u,
        b_field=u,
        step_jacobian_determinant=determinants,
    )

    figure = plotting.phase_space_volume(directory, show=False)

    lines = {line.get_label(): line for line in _named_lines(figure.axes[0])}
    rounding = np.finfo(np.float64).eps
    expected = [rounding, np.log(1.1), 2 * np.log(1.1), 3 * np.log(1.1)]
    np.testing.assert_allclose(lines["particle 0"].get_ydata(), expected, rtol=1e-12)


def test_phase_space_volume_needs_the_stored_determinants(result_dir):
    directory, _ = result_dir

    with pytest.raises(ValueError, match="phase_space_volume=True"):
        plotting.phase_space_volume(directory, show=False)


def test_energy_error_from_the_half_step_velocity_uses_its_own_times(energy_dir):
    figure = plotting.energy_error(energy_dir, pusher="boris", velocity="u", show=False)

    (line, _) = _named_lines(figure.axes[0])
    np.testing.assert_allclose(
        line.get_xdata(), 1.0e-11 * np.arange(3) - 0.5e-12, rtol=1e-12
    )


def _phase_scan_folder(tmp_path, name, gamma, pitches_degrees, drifts_per_turn):
    """A made-up folder: electrons of one gamma, several pitch angles, in B = 2.5 z.

    The stored u_centered turns at the exact rate plus ``drifts_per_turn[i]`` rad
    per exact turn, so the phase error per turn of particle i is known exactly.
    """
    b_z = 2.5
    speed = np.sqrt(gamma**2 - 1.0) * c
    pitch = np.radians(pitches_degrees)
    u0 = np.stack(
        [np.zeros(pitch.size), -speed * np.sin(pitch), speed * np.cos(pitch)], axis=1
    )
    directory = _folder(
        tmp_path,
        name,
        {"kind": "uniform_cartesian", "b_field": [0, 0, b_z], "e_field": [0, 0, 0]},
        [tuple(row) for row in u0],
        x0=[(0.0, 0.0, 0.0)] * pitch.size,
    )
    rate = 1.602176634e-19 * b_z / (gamma * 9.1093837139e-31)
    period = 2.0 * np.pi / rate
    t = np.linspace(0.0, 2.0 * period, 201)
    angle = (
        rate * t[:, np.newaxis] + np.array(drifts_per_turn) * t[:, np.newaxis] / period
    )
    u = np.zeros((t.size, pitch.size, 3))
    u[:, :, 0] = speed * np.sin(pitch) * np.sin(angle)  # u0 turned by +angle about z
    u[:, :, 1] = -speed * np.sin(pitch) * np.cos(angle)
    u[:, :, 2] = speed * np.cos(pitch)
    write_result(
        directory,
        "higuera_cary",
        t,
        np.zeros_like(u),
        u,
        1e-12,
        u_centered=u,
        b_field=u,
    )
    return directory, rate


def test_phase_error_map_shows_the_known_ratios_in_the_right_cells(tmp_path):
    # Independent truth: made-up drifts per turn divided by the Boris lag per turn
    # (2 pi / theta)|2 arctan(theta / 2) - theta|, theta = omega dt with the dt of
    # the configuration (1e-12 s), written out here. Rows are pitch angles, columns
    # u / c in increasing order whatever the order of the folders. Tolerance 1e-9
    # relative, and 1e-9 absolute for the zero cell: the rounding of the phase
    # (1e-16 rad) is divided by the Boris lag per turn, only 1e-5 rad here at
    # gamma = 100 and dt = 1e-12 s (observed 1.2e-11).
    pitches = [30.0, 60.0, 90.0]
    fast, fast_rate = _phase_scan_folder(
        tmp_path, "fast", 100.0, pitches, [-1e-3, 0.0, 2e-3]
    )
    slow, slow_rate = _phase_scan_folder(
        tmp_path, "slow", 2.0, pitches, [-4e-3, -2e-3, 1e-3]
    )

    figure = plotting.phase_error_map([fast, slow], "higuera_cary", show=False)

    def boris_lag_per_turn(rate):
        theta = rate * 1.0e-12
        return 2.0 * np.pi / theta * abs(2.0 * np.arctan(theta / 2.0) - theta)

    expected = np.array(
        [
            np.array([-4e-3, -2e-3, 1e-3]) / boris_lag_per_turn(slow_rate),
            np.array([-1e-3, 0.0, 2e-3]) / boris_lag_per_turn(fast_rate),
        ]
    ).T  # (pitch, u / c)
    mesh = figure.axes[0].collections[0]
    np.testing.assert_allclose(
        np.asarray(mesh.get_array()).reshape(3, 2), expected, rtol=1e-9, atol=1e-9
    )
    assert figure.axes[0].get_xscale() == "log"


def test_phase_error_map_refuses_folders_with_other_pitch_angles(tmp_path):
    first, _ = _phase_scan_folder(tmp_path, "a", 2.0, [30.0, 60.0], [0.0, 0.0])
    second, _ = _phase_scan_folder(tmp_path, "b", 10.0, [30.0, 70.0], [0.0, 0.0])

    with pytest.raises(ValueError, match="same ones"):
        plotting.phase_error_map([first, second], "higuera_cary", show=False)


def test_phase_error_map_needs_two_folders(tmp_path):
    only, _ = _phase_scan_folder(tmp_path, "a", 2.0, [30.0, 60.0], [0.0, 0.0])

    with pytest.raises(ValueError, match="at least two"):
        plotting.phase_error_map([only], "higuera_cary", show=False)
