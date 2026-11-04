"""Milestone G3: the runs and verification figures of the tokamak, one step at a time.

Run one step at a time:

    uv run python main_g3.py --list          # the steps, their cost, their figures
    uv run python main_g3.py c5_short        # one step
    uv run python main_g3.py c5_short --show # same, and open the figures

Each step runs its cases with ``run.pusher``: one new, timestamped results folder
per case in ``results/``, the pushers side by side in it, and the figures of the
step saved in the first folder (PNG and PDF). Nothing is ever overwritten. The
cases are the ``.yaml`` files of ``configs/``, named ``<case>_<pusher>.yaml``; they
are written there by this file the first time they are needed, from the parameters
below, and an existing file is checked against them and never rewritten.

The case is that of Carbajal et al. (2017, Fig. 1) and of roadmap C5: DIII-D with
q_edge = 2 (B0 = 2.19 T, R0 = 1.5 m, q0 = 1.6, lambda = 1 m), no electric field,
four electrons of 20 MeV with the pitch angles 10, 30, 50 and 70 degrees, from
r = 0.2 m on the outer midplane (the paper does not give the starting point; the
70-degree electron is trapped there, the others pass). dt = tau_e / N, with
tau_e = 2 pi gamma m / (e B0) = 6.55e-10 s the gyration period. Step c6 uses
the same fields for ITER and DIII-D. Step c7 is the Poincare section of Higuera &
Cary (2017, Sec. VI), in their sheared field (see ``c7_configs``).

The tests of G3 are run with pytest (``-m slow`` for the long ones):

    C1      uv run pytest tests/test_diagnostics.py -v -k "ten_bounces or rk4_loses_the_magnetic_moment"
    C2      uv run pytest tests/test_diagnostics.py -v -k "gradient_drift"
    C3      uv run pytest tests/test_diagnostics.py -v -k "toroidal_drift"
    C4      uv run pytest tests/test_fields.py -v -k "zero_divergence_to_fourth_order or wind_as_q_over or psi_derivative"
    C5      uv run pytest tests/test_diagnostics.py -v -k "p_zeta_error or energy_error_in_the_tokamak"
    B6      uv run pytest tests/test_integrate.py -v -k "order_depends_on_where_the_field_is_read"
    C6      uv run pytest tests/test_diagnostics.py -v -k "is_trapped or strongest_field or classified_as_mu or orbit_width_moves"
    C7      uv run pytest tests/test_diagnostics.py -v -k "poincare or shear or i_y or nested"
    U1-U8   uv run pytest tests/test_fields.py tests/test_diagnostics.py -v -k "toroidal or carbajal or canonical"
"""  # noqa: E501

import argparse
import math
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np

from ppf_c5_runaways import fields, plotting, run
from ppf_c5_runaways.config import RunConfig, load_config, save_config
from ppf_c5_runaways.constants import c, e, m_e
from ppf_c5_runaways.particles import proper_velocity

CONFIGS = Path("configs")
STRUCTURE_PRESERVING = ("boris", "higuera_cary", "vay")

DIII_D = {"kind": "carbajal", "b0": 2.19, "r0": 1.5, "q0": 1.6, "lam": 1.0, "e0": 0.0}
# ITER of Carbajal et al. (Sec. III): r_edge = 2 m, q_edge = 3, q_edge / q0 = 1.25.
ITER = {"kind": "carbajal", "b0": 5.3, "r0": 6.2, "q0": 2.4, "lam": 4.0, "e0": 0.0}
KINETIC_ENERGY = 20.0e6 * e  # [J]
PITCH_ANGLES_DEG = (10.0, 30.0, 50.0, 70.0)
START_MINOR_RADIUS = 0.2  # [m], on the outer midplane
C7_P_Y0 = (0.4, 0.8, 1.2, 1.5, 1.625, 1.775, 2.0, 2.5, 3.0, 3.5)  # u_y0 / c
GAMMA = 1.0 + KINETIC_ENERGY / (m_e * c**2)
GYRATION_PERIOD = 2.0 * math.pi * GAMMA * m_e / (e * DIII_D["b0"])  # tau_e [s]


# --- running a case ---------------------------------------------------------------


def run_case(case: str, pushers: tuple[str, ...]) -> Path:
    """Run ``configs/<case>_<pusher>.yaml`` for each pusher, in one new folder."""
    result_dir = None
    for pusher in pushers:
        config_path = CONFIGS / f"{case}_{pusher}.yaml"
        print(f"  running {config_path} ...", flush=True)
        if result_dir is None:
            result_dir = run.pusher(config_path)
        else:
            run.pusher(config_path, result_dir=result_dir)
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
            f"{path} differs from the parameters of main_g3.py: remove it or fix "
            "the parameters, so that the figures match what is described."
        )


def c5_configs(
    name: str,
    steps_per_period: int,
    periods: int,
    save_every: int,
    pushers: tuple[str, ...],
) -> list[RunConfig]:
    """Build the configurations of C5 at ``dt = tau_e / steps_per_period``."""
    field = fields.carbajal(**{key: DIII_D[key] for key in DIII_D if key != "kind"})
    x0 = np.repeat(
        fields.toroidal_to_cartesian(START_MINOR_RADIUS, 0.0, 0.0, DIII_D["r0"])[
            np.newaxis
        ],
        len(PITCH_ANGLES_DEG),
        axis=0,
    )
    b_direction = field(x0, 0.0)[1]
    u0 = proper_velocity(
        KINETIC_ENERGY, np.radians(PITCH_ANGLES_DEG), 0.0, b_direction, m_e
    )
    return [
        RunConfig(
            name=name,
            pusher=pusher,
            dt=GYRATION_PERIOD / steps_per_period,
            n_steps=periods * steps_per_period,
            save_every=save_every,
            seed=42,
            q=-e,
            m=m_e,
            field=DIII_D,
            initial={"x0": x0.tolist(), "u0": u0.tolist()},
        )
        for pusher in pushers
    ]


def c6_configs(
    name: str,
    machine: dict[str, Any],
    start_minor_radius: float,
    energy_mev: float,
    offsets_deg: tuple[float, ...],
    periods: int,
) -> list[RunConfig]:
    """Build the configurations of C6: pitch angles about the boundary of mu alone.

    The electrons start on the outer midplane at ``start_minor_radius`` with the
    pitch angles ``theta_c + offsets_deg``, where ``sin^2 theta_c = |B(x0)| /
    B_max`` and ``B_max`` is on the inner midplane (roadmap C6). dt = tau_e / 50;
    ten states kept per gyration, so that the drawn orbits show the gyration (one
    state per gyration, as in the test of C6, would draw it as a broken line).
    """
    parameters = {key: machine[key] for key in machine if key != "kind"}
    field = fields.carbajal(**parameters)
    x_start = fields.toroidal_to_cartesian(start_minor_radius, 0.0, 0.0, machine["r0"])
    x_inner = fields.toroidal_to_cartesian(
        start_minor_radius, math.pi, 0.0, machine["r0"]
    )
    b_start = field(x_start[np.newaxis], 0.0)[1]
    b_max = np.linalg.norm(field(x_inner[np.newaxis], 0.0)[1])
    theta_c = math.asin(math.sqrt(np.linalg.norm(b_start) / b_max))
    pitch_angles = theta_c + np.radians(offsets_deg)
    x0 = np.repeat(x_start[np.newaxis], len(offsets_deg), axis=0)
    kinetic_energy = energy_mev * 1.0e6 * e
    u0 = proper_velocity(kinetic_energy, pitch_angles, 0.0, b_start, m_e)
    gamma = 1.0 + kinetic_energy / (m_e * c**2)
    period = 2.0 * math.pi * gamma * m_e / (e * machine["b0"])
    return [
        RunConfig(
            name=name,
            pusher=pusher,
            dt=period / 50,
            n_steps=periods * 50,
            save_every=5,
            seed=42,
            q=-e,
            m=m_e,
            field=machine,
            initial={"x0": x0.tolist(), "u0": u0.tolist()},
        )
        for pusher in STRUCTURE_PRESERVING
    ]


def c7_configs(name: str, dt_in_paper_units: float, n_steps: int) -> list[RunConfig]:
    """Build the configurations of C7: ten electrons at H = 4 in the sheared field.

    The units of Higuera & Cary (q = m = c = 1) are converted with the unit of
    length 1 m and the unit of time 1 m / c: positions in metres and u / c are the
    numbers of the paper. a = 1 and b = 2 give a = m c^2 / q and b = 2 m c / q for
    an electron. The electrons start at the origin with u / c = (sqrt(15 - p_y0^2),
    p_y0, 0), so that H = gamma + a x^2 / 2 = 4 and p_z = 0, with the grid of p_y0
    of the test of C7 (1.625 and 1.775 are those of Fig. 3 of the paper). Every
    state is kept, for the interpolation of the crossings of x = 0.
    """
    p_y0 = np.array(C7_P_Y0)
    x0 = np.zeros((p_y0.size, 3))
    u0 = np.stack([np.sqrt(15.0 - p_y0**2), p_y0, np.zeros_like(p_y0)], axis=1) * c
    field = {
        "kind": "higuera_cary_shear",
        "a": m_e * c**2 / -e,
        "b": 2.0 * m_e * c / -e,
    }
    return [
        RunConfig(
            name=name,
            pusher=pusher,
            dt=dt_in_paper_units / c,
            n_steps=n_steps,
            save_every=1,
            seed=42,
            q=-e,
            m=m_e,
            field=field,
            initial={"x0": x0.tolist(), "u0": u0.tolist()},
        )
        for pusher in STRUCTURE_PRESERVING
    ]


# --- the steps --------------------------------------------------------------------


def step_c5_short(show: bool) -> None:
    """C5 short: P_zeta and energy at four time steps, 1000 tau_e, four pushers."""
    pushers = (*STRUCTURE_PRESERVING, "rk4")
    result_dirs = []
    for n in (50, 100, 200, 400):
        case = f"c5_short_n{n:03d}"
        # 50 states kept per gyration whatever N, as in the test of C5.
        for config in c5_configs(case, n, 1000, n // 50, pushers):
            ensure_config(config)
        result_dirs.append(run_case(case, pushers))
    reference = result_dirs[1]  # N = 100, the step of the paper
    plotting.canonical_momentum_convergence(
        result_dirs,
        pushers,
        show=show,
        save_path=reference / "p_zeta_convergence",
    )
    for pusher in pushers:
        plotting.canonical_momentum_error(
            reference, pusher, show=show, save_path=reference / f"p_zeta_{pusher}"
        )
        plotting.canonical_momentum_error(
            reference,
            pusher,
            half_step=True,
            show=show,
            save_path=reference / f"p_zeta_half_step_{pusher}",
        )
    plotting.energy_error(reference, show=show, save_path=reference / "energy_error")
    plotting.orbit_poloidal(
        reference, "boris", show=show, save_path=reference / "orbit_poloidal"
    )
    print(f"  figures in {reference}")


def step_c5_paper(show: bool) -> None:
    """C5 at the duration of Carbajal et al. (Fig. 1): 1e-3 s at dt = tau_e / 100."""
    # One state kept every 101 steps, 1.01 gyration: the kept states sweep the
    # gyration phase every 100 of them, so the oscillation of P_zeta within a
    # gyration is sampled; one state per gyration would always see the same phase.
    # 1.5e6 kept states, about 0.5 GB per pusher. The number of gyrations is a
    # multiple of 101 so that save_every divides n_steps.
    save_every = 101
    periods = save_every * round(1.0e-3 / GYRATION_PERIOD / save_every)  # 1.53e6
    for config in c5_configs(
        "c5_paper", 100, periods, save_every, STRUCTURE_PRESERVING
    ):
        ensure_config(config)
    result_dir = run_case("c5_paper", STRUCTURE_PRESERVING)
    for pusher in STRUCTURE_PRESERVING:
        plotting.canonical_momentum_error(
            result_dir, pusher, show=show, save_path=result_dir / f"p_zeta_{pusher}"
        )
    plotting.energy_error(result_dir, show=show, save_path=result_dir / "energy_error")


def step_c6(show: bool) -> None:
    """C6: trapped and passing electrons about the boundary predicted by mu."""
    cases = (
        # ITER, 10 MeV, r = 0.8 m: theta_c = 61.44 degrees. The numerical boundary
        # lies between theta_c - 0.6 and theta_c - 0.5 (test of C6): the two middle
        # electrons are on each side of it.
        ("c6_iter_10mev", ITER, 0.8, 10.0, (-4.0, -0.6, -0.5, 4.0), 12_000),
        # DIII-D, 40 MeV, r = 0.2 m: theta_c = 60.98 degrees; the orbit width
        # moves the boundary to between theta_c - 9.2 and theta_c - 9.0.
        ("c6_diii_d_40mev", DIII_D, 0.2, 40.0, (-12.0, -9.2, -9.0, 0.0), 800),
    )
    for name, machine, start, energy, offsets, periods in cases:
        for config in c6_configs(name, machine, start, energy, offsets, periods):
            ensure_config(config)
        result_dir = run_case(name, STRUCTURE_PRESERVING)
        plotting.orbit_poloidal(
            result_dir, "boris", show=show, save_path=result_dir / "orbit_poloidal"
        )
        plotting.parallel_velocity(
            result_dir, "boris", show=show, save_path=result_dir / "parallel_velocity"
        )
        print(f"  figures in {result_dir}")


def step_c7(show: bool) -> None:
    """C7: Poincare sections at dt = 1/40 and 1/10, 1e6 steps (Higuera & Cary)."""
    for name, dt_in_paper_units in (("c7_dt040", 1.0 / 40.0), ("c7_dt010", 1.0 / 10.0)):
        for config in c7_configs(name, dt_in_paper_units, 1_000_000):
            ensure_config(config)
        result_dir = run_case(name, STRUCTURE_PRESERVING)
        plotting.poincare_section(
            result_dir, show=show, save_path=result_dir / "poincare_section"
        )
        print(f"  figures in {result_dir}")


# name: (function, cost on the user's laptop, what it shows)
STEPS: dict[str, tuple[Callable[[bool], None], str, str]] = {
    "c5_short": (
        step_c5_short,
        "about 4 minutes",
        "C5: P_zeta of order 2, its staggered measure of order 1 (R11), energy",
    ),
    "c5_paper": (
        step_c5_paper,
        "long: 2 hours and 0.5 GB per pusher",
        "C5 at the duration of Carbajal Fig. 1, 1.5e8 steps (R11, N9)",
    ),
    "c6": (
        step_c6,
        "about 2 minutes",
        "C6: trapped and passing electrons about theta_c, ITER and DIII-D",
    ),
    "c7": (
        step_c7,
        "estimated: 5 minutes, 4.3 GB on disk",
        "C7: Poincare sections, islands of Vay at dt = 1/10 (Higuera & Cary)",
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
            print(f"{name:12s} {cost:32s} {purpose}")
        return
    step, _, purpose = STEPS[arguments.step]
    print(f"{arguments.step}: {purpose}")
    step(arguments.show)


if __name__ == "__main__":
    main()
