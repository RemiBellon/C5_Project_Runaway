"""Milestone G2: every run and every verification figure, one step at a time.

Run one step at a time:

    uv run python main_g2.py --list          # the steps, their cost, their figures
    uv run python main_g2.py gyration        # one step
    uv run python main_g2.py gyration --show # same, and open the figures

Each step runs its case with ``run.pusher``: one new, timestamped results folder per
case in ``results/``, the three pushers side by side in it, and the figures of the
step saved in that folder (PNG and PDF). Nothing is ever overwritten. Steps marked
"long" are the runs at the resolution of Ripperda et al. (2018); they also print the
measured quantity next to its reference.

The cases are the ``.yaml`` files of ``configs/``, named ``<case>_<pusher>.yaml``.
Two cases (``mirror_bottle``, ``crossed_fields_volume``) are written to ``configs/``
by this file the first time they are needed, from the parameters below; an existing
file is checked against them and never rewritten.

The tests themselves are run with pytest. Whole suite, then one file at a time:

    uv run pytest -m "not slow"              # fast suite, about 30 s
    uv run pytest                            # complete suite, about 70 s
    uv run pytest tests/test_pushers.py -v   # likewise for each file of tests/

One roadmap identifier at a time (``-v`` lists each case):

    A1, A2  uv run pytest tests/test_pushers.py -v -k "rotation_keeps_the_norm_and_obeys_its_defining_identity"
    A3      uv run pytest tests/test_pushers.py -v -k "without_magnetic_field_the_two_half_kicks_add_up"
    A4      uv run pytest tests/test_pushers.py -v -k "energy_changes_by_the_work_of_the_electric_field"
    A5      uv run pytest tests/test_pushers.py -v -k "higuera_cary_rotates_with_the_gamma_of_the_mean_momentum"
    A6      uv run pytest tests/test_pushers.py -v -k "vay_gamma_formula_is_the_gamma_of_the_new_velocity"
    A7      uv run pytest tests/test_pushers.py -v -k "vay_is_boris_in_a_pure_magnetic_field"
    A8      uv run pytest tests/test_pushers.py -v -k "the_three_pushers_are_boris_in_the_non_relativistic_limit"
    A9      uv run pytest tests -v -k "step_determinant_matches_the_paper_in_crossed_fields or preserves_the_volume_in_a_static_magnetic_bottle or preserve_the_phase_space_volume"
    A10     uv run pytest tests/test_pushers.py -v -k "exb_fixed_point_in_the_configuration_of_the_roadmap"
    A11     uv run pytest tests/test_pushers.py -v -k "one_step_forward_then_backward_gives_back_the_start"
    A12     uv run pytest tests/test_diagnostics.py -v -k "kinetic_energy_keeps_its_digits_from_rest_to_ultra_relativistic"
    A13     uv run pytest tests/test_particles.py -v -k "energy_and_pitch_angle_are_given_back or one_larmor_radius_from_its_guiding_centre"
    B1      uv run pytest tests/test_diagnostics.py -v -k "lag_by_the_exact_boris_angle or leads_by_half_the_boris_lag or phase_advance_is_the_truncated_formula or gamma_is_kept_within_the_rounding_bound"
    B2      uv run pytest tests/test_pushers.py -v -k "relativistic_acceleration_error_decreases or gamma_is_exact_when_accelerated_from_rest or position_error_from_rest_decreases or start_from_rest_misses_the_first_half_step"
    B3      uv run pytest tests/test_diagnostics.py -v -k "drift_frame"
    B4      uv run pytest tests/test_pushers.py -v -k "force_free_line"
    B5      uv run pytest tests/test_pushers.py -v -k "phase_error_decreases_as_dt_squared_when_e_is_parallel_to_b"
    B6      uv run pytest tests/test_integrate.py -v -k "position_and_velocity_converge_as_dt_squared or order_depends_on_where_the_field_is_read"
    S1      uv run pytest tests/test_diagnostics.py -v -k "spectrum_holds_the_ultra_relativistic_power or exceeds_the_power_of_equation_15"
    S2      uv run pytest tests/test_diagnostics.py -v -k "power_below_the_critical_wavelength_is_the_exact_fraction or exact_fraction_is_not_one_half"
    S3      uv run pytest tests/test_diagnostics.py -v -k "kernel_integral"
    S4      uv run pytest tests/test_diagnostics.py -v -k "curvature_matches_the_shape or curvature_with_an_electric_field"
"""  # noqa: E501

import argparse
import math
from collections.abc import Callable
from pathlib import Path

import numpy as np

from ppf_c5_runaways import plotting, run
from ppf_c5_runaways.config import RunConfig, load_config, save_config
from ppf_c5_runaways.constants import c, e, m_e
from ppf_c5_runaways.diagnostics import boost_proper_velocity
from ppf_c5_runaways.io import CONFIG_FILENAME, read_result

CONFIGS = Path("configs")
PUSHERS = ("boris", "higuera_cary", "vay")
B_Z = 2.5  # [T], the field of every electron case of G2


# --- running a case ---------------------------------------------------------------


def run_case(case: str, *, phase_space_volume: bool = False) -> Path:
    """Run ``configs/<case>_<pusher>.yaml`` for the three pushers, in one new folder."""
    result_dir = None
    for pusher in PUSHERS:
        config_path = CONFIGS / f"{case}_{pusher}.yaml"
        print(f"  running {config_path} ...", flush=True)
        if result_dir is None:
            result_dir = run.pusher(config_path, phase_space_volume=phase_space_volume)
        else:
            run.pusher(
                config_path,
                result_dir=result_dir,
                phase_space_volume=phase_space_volume,
            )
    print(f"  results in {result_dir}")
    return result_dir


def ensure_config(config: RunConfig) -> None:
    """Write ``config`` to ``configs/`` if it is not there; check it if it is."""
    path = CONFIGS / f"{config.name}_{config.pusher}.yaml"
    if not path.exists():
        save_config(config, path)
        print(f"  wrote {path}")
    elif load_config(path) != config:
        raise ValueError(
            f"{path} differs from the parameters of main_g2.py: remove it or fix "
            "the parameters, so that the figures match what is described."
        )


# --- the two cases written by this file -------------------------------------------


def mirror_bottle_configs() -> list[RunConfig]:
    """Magnetic bottle of Ripperda et al. (2018, Sec. 3.2.1), three bounces.

    Electron of gamma = 100 and pitch angle 45 degrees, started on its gyration
    circle about the axis at z = 0, with r_L / L = 1e-2 and 50 steps per turn: the
    case of the tests of ``diagnostics.magnetic_moment`` (test_diagnostics.py).
    """
    gamma = 100.0
    u_perp = math.sqrt((gamma**2 - 1.0) / 2.0) * c
    r_larmor = m_e * u_perp / (e * B_Z)
    length = r_larmor / 1.0e-2
    dt = 2.0 * math.pi * gamma * m_e / (e * B_Z) / 50
    bounce_period = 2.0 * math.pi * length / (u_perp / gamma)
    save_every = 5
    n_steps = save_every * round(3 * bounce_period / dt / save_every)
    return [
        RunConfig(
            name="mirror_bottle",
            pusher=pusher,
            dt=dt,
            n_steps=n_steps,
            save_every=save_every,
            seed=42,
            q=-e,
            m=m_e,
            field={"kind": "mirror", "b0": B_Z, "length": length},
            initial={"x0": [[r_larmor, 0.0, 0.0]], "u0": [[0.0, u_perp, u_perp]]},
        )
        for pusher in PUSHERS
    ]


def crossed_fields_volume_configs() -> list[RunConfig]:
    """Electron from rest in crossed fields, E = 0.9 c B: a relativistic gyration.

    The drift has the Lorentz factor 2.29; 0.1 / Omega_0 per step, about 760 steps
    per gyration, 2000 steps. Every step is kept (``save_every: 1``), as
    ``phase_space_volume=True`` needs.
    """
    dt = 0.1 * m_e / (e * B_Z)
    return [
        RunConfig(
            name="crossed_fields_volume",
            pusher=pusher,
            dt=dt,
            n_steps=2000,
            save_every=1,
            seed=42,
            q=-e,
            m=m_e,
            field={
                "kind": "uniform_cartesian",
                "b_field": [0.0, 0.0, B_Z],
                "e_field": [0.9 * c * B_Z, 0.0, 0.0],
            },
            initial={"x0": [[0.0, 0.0, 0.0]], "u0": [[0.0, 0.0, 0.0]]},
        )
        for pusher in PUSHERS
    ]


# --- the steps --------------------------------------------------------------------


def step_gyration(show: bool) -> None:
    """Gyration in a uniform B: orbits, and the energy kept by the three pushers."""
    result_dir = run_case("uniform_b")
    plotting.orbit_xy(result_dir, show=show, save_path=result_dir / "orbit_xy")
    plotting.orbit_xz(result_dir, show=show, save_path=result_dir / "orbit_xz")
    plotting.energy_error(result_dir, show=show, save_path=result_dir / "energy_error")


def step_exb_gamma_100(show: bool) -> None:
    """Relativistic E x B drift at gamma = 100: energy error of each pusher."""
    result_dir = run_case("exb_drift_gamma_100")
    plotting.energy_error(result_dir, show=show, save_path=result_dir / "energy_error")
    plotting.orbit_xy(result_dir, show=show, save_path=result_dir / "orbit_xy")


def step_exb_scan(show: bool) -> None:
    """E x B drift scan over ten drift speeds: drift velocity error (Ripperda)."""
    result_dirs = [run_case(f"exb_drift_scan_{index:02d}") for index in range(10)]
    save_path = result_dirs[0] / "exb_drift_error"
    plotting.drift_error(result_dirs, show=show, save_path=save_path)
    print(f"  figure in {result_dirs[0]}")


def step_phase_gamma_1e6(show: bool) -> None:
    """Gyration phase error at gamma = 1e6 (Ripperda et al. 2018, Fig. 4)."""
    result_dir = run_case("gyration_gamma_1e6")
    plotting.phase_error(result_dir, show=show, save_path=result_dir / "phase_error")


def step_phase_map(show: bool) -> None:
    """Phase error per turn against u / c and pitch angle, one map per pusher."""
    result_dirs = [run_case(f"phase_map_{index:02d}") for index in range(13)]
    for pusher in PUSHERS:
        save_path = result_dirs[0] / f"phase_error_map_{pusher}"
        plotting.phase_error_map(result_dirs, pusher, show=show, save_path=save_path)
    print(f"  figures in {result_dirs[0]}")


def step_mirror(show: bool) -> None:
    """Magnetic bottle: z(t) / L and the change of the magnetic moment."""
    for config in mirror_bottle_configs():
        ensure_config(config)
    result_dir = run_case("mirror_bottle")
    plotting.mirror_trajectory(
        result_dir, show=show, save_path=result_dir / "mirror_trajectory"
    )
    plotting.magnetic_moment_error(
        result_dir, show=show, save_path=result_dir / "magnetic_moment_error"
    )


def step_volume(show: bool) -> None:
    """Phase-space volume in crossed fields, every step measured."""
    for config in crossed_fields_volume_configs():
        ensure_config(config)
    result_dir = run_case("crossed_fields_volume", phase_space_volume=True)
    plotting.phase_space_volume(
        result_dir, show=show, save_path=result_dir / "phase_space_volume"
    )


def report_force_free(result_dir: Path) -> None:
    """Print the distance to the force-free line against its references (N25).

    Boris: the formula of test_pushers.py, ``Omega_0^3 dt^2 T^2 v^3 / (16 gamma^3
    c^2)``, and Table 2 of Ripperda et al. divided by 100. Vay and Higuera-Cary:
    the bound of four roundings, ``4 eps c Omega_0 T^2 / (2 gamma)``.
    """
    config = load_config(result_dir / CONFIG_FILENAME)
    gamma = 1.0e6
    duration = config.dt * config.n_steps  # [s], Omega_0 = 1 rad/s in this case
    v = c * math.sqrt(1.0 - 1.0 / gamma**2)
    formula = config.dt**2 * duration**2 * v**3 / (16.0 * gamma**3 * c**2)
    bound = 4.0 * np.finfo(np.float64).eps * c * duration**2 / (2.0 * gamma)
    table_over_100 = {1.0: 0.187705, 0.1: 0.047705, 0.01: 1.23e-3, 0.001: 2.52e-4}
    print(f"  dt = {config.dt} s, T = {duration:.0f} s")
    for pusher in PUSHERS:
        x_end = abs(read_result(result_dir, pusher)["x"].values[-1, 0, 0])
        if pusher == "boris":
            published = table_over_100.get(config.dt, float("nan"))
            print(
                f"  boris: |x| = {x_end:.4e} m, formula {formula:.4e} m, "
                f"Table 2 / 100 = {published:.4e} m"
            )
        else:
            print(f"  {pusher}: |x| = {x_end:.4e} m, bound {bound:.4e} m")


def force_free_step(dt_label: str) -> Callable[[bool], None]:
    """Build the step that runs the force-free case at one of its time steps."""

    def step(show: bool) -> None:
        """Run the force-free case and print the distances to the line."""
        result_dir = run_case(f"ripperda_force_free_dt_{dt_label}")
        report_force_free(result_dir)

    return step


def report_drift_frame(result_dir: Path) -> None:
    """Print the error on gamma' in the drifting frame, as in test B3 (N24).

    ``gamma'`` must equal the Lorentz factor of the drift, ``kappa``. Its largest
    relative error, and the growth from the first half of the run to the second
    (1 means no secular drift), for each pusher.
    """
    config = load_config(result_dir / CONFIG_FILENAME)
    e_field = np.array(config.field.e_field)
    b_field = np.array(config.field.b_field)
    drift_velocity = np.cross(e_field, b_field) / (b_field @ b_field)
    kappa = 1.0 / math.sqrt(1.0 - drift_velocity @ drift_velocity / c**2)
    print(f"  kappa = {kappa:.4f}, dt Omega_0 = {config.dt * e * B_Z / m_e:.1e}")
    for pusher in PUSHERS:
        u = read_result(result_dir, pusher)["u"].values[:, 0]
        u_drift_frame = boost_proper_velocity(u, drift_velocity)
        speed_over_c = np.linalg.norm(u_drift_frame, axis=1) / c
        gamma_error = np.abs(np.sqrt(1.0 + speed_over_c**2) / kappa - 1.0)
        half = gamma_error.size // 2
        growth = np.max(gamma_error[half:]) / np.max(gamma_error[:half])
        print(
            f"  {pusher}: largest |gamma'/kappa - 1| = {np.max(gamma_error):.3e}, "
            f"growth second half / first half = {growth:.5f}"
        )


def step_exb_kappa10(show: bool) -> None:
    """B3 at the time step of the paper, and the error on gamma'."""
    result_dir = run_case("ripperda_exb_kappa10")
    report_drift_frame(result_dir)


# name: (function, cost on the user's laptop, what it shows)
STEPS: dict[str, tuple[Callable[[bool], None], str, str]] = {
    "gyration": (step_gyration, "seconds", "orbits and energy in a uniform B"),
    "exb_gamma_100": (
        step_exb_gamma_100,
        "seconds",
        "energy of the relativistic E x B drift (Boris does not keep it)",
    ),
    "exb_scan": (step_exb_scan, "seconds", "E x B drift error vs speed"),
    "phase_gamma_1e6": (
        step_phase_gamma_1e6,
        "seconds",
        "gyration phase at gamma = 1e6 (Ripperda Fig. 4)",
    ),
    "phase_map": (
        step_phase_map,
        "seconds",
        "phase error per turn vs u/c and pitch angle",
    ),
    "mirror": (
        step_mirror,
        "seconds",
        "bounces and magnetic moment in a bottle (Ripperda Fig. 10)",
    ),
    "volume": (
        step_volume,
        "seconds",
        "phase-space volume in crossed fields (Vay is not volume preserving)",
    ),
    "force_free_dt_1": (
        force_free_step("1"),
        "seconds",
        "B4 at dt = 1 (also a pytest test)",
    ),
    "force_free_dt_0p1": (
        force_free_step("0p1"),
        "long: about 30 s per pusher",
        "B4 at dt = 0.1, settles the value 4.7705 of Table 2 (N25)",
    ),
    "force_free_dt_0p01": (
        force_free_step("0p01"),
        "long: about 5 minutes per pusher",
        "B4 at dt = 0.01 (N25)",
    ),
    "force_free_dt_0p001": (
        force_free_step("0p001"),
        "long: about an hour per pusher",
        "B4 at dt = 0.001 (N25)",
    ),
    "exb_kappa10": (
        step_exb_kappa10,
        "long: about 70 minutes per pusher",
        "B3 at the time step of the paper (N24)",
    ),
}


def main() -> None:
    """Run the step named on the command line, or list the steps."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("step", nargs="?", choices=STEPS, help="the step to run")
    parser.add_argument("--list", action="store_true", help="list the steps")
    parser.add_argument("--show", action="store_true", help="open the figures")
    arguments = parser.parse_args()
    if arguments.list or arguments.step is None:
        for name, (_, cost, purpose) in STEPS.items():
            print(f"{name:22s} {cost:34s} {purpose}")
        return
    step, _, purpose = STEPS[arguments.step]
    print(f"{arguments.step}: {purpose}")
    step(arguments.show)


if __name__ == "__main__":
    main()
