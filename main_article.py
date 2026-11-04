"""The article: its runs and figures, one step at a time (OPEN_POINTS N44, N48).

Run one step at a time:

    uv run python main_article.py --list           # the steps, their cost, figures
    uv run python main_article.py fig1             # one step
    uv run python main_article.py fig1 --show      # same, and open the figures
    uv run python main_article.py fig3 --half-dt   # the same runs at dt / 2 (N46)
    uv run python main_article.py fig4 --electrons 200 --root preview
                                                   # a smaller ensemble, elsewhere

Each step builds its cases, writes them as ``.yaml`` files in ``<root>/configs/``
(``root`` is the current folder by default; an existing file is checked against
the parameters below and never rewritten), runs them with ``run.pusher`` into new,
timestamped folders of ``<root>/results/``, and saves its figures (PNG and PDF) in
the first folder of the step. Nothing is ever overwritten.

The article follows electrons of 40 MeV for 10 ms, with the pitch angles 10, 30
and 50 degrees, in the DIII-D field of Carbajal et al. (2017, q_edge = 2: B0 =
2.19 T, R0 = 1.5 m, q0 = 1.6, lambda = 1 m) or in a uniform B0. The electric
field, 4 V/m, and the radiation reaction force are multiplied by the same s =
1e3, so that 10 ms of physical time take 1e-5 s of computed time (N44, RAPPORT
5.51); the figures show the physical time s t. The time step is a fraction of
the shortest gyration period expected in the run (N45). Every run uses Boris,
but those of Fig. 2(a), which compare the four pushers.

    fig1         uniform fields: (a) decay of gamma in B alone, s = 1e7, against
                 the exact solution; (b) paths in (p_par, p_perp) under E || B,
                 three choices of forces (RAPPORT 5.58, 5.59)
    fig2         errors: (a) the decay against the exact solution at five dt,
                 four pushers, s = 1e6 and 1e7; (b) the change of gamma with s in
                 the tokamak (RAPPORT 5.60, 5.61)
    fig3         one electron at 30 degrees in the tokamak, three choices of
                 forces: poloidal orbit and time traces (RAPPORT 5.62)
    fig4         three ensembles of electrons in the tokamak, three choices of
                 forces, the analogue of Fig. 12 of Carbajal et al.: final pitch
                 angles, energies and momenta
    fig4_checks  the first 1000 electrons of fig4 at dt / 2 (N46) and at s = 2e3
                 (N48 d), drawn as fig4

The tests behind the figures are run with pytest (``-m slow`` for the long ones):

    Fig. 1  uv run pytest tests/test_radiation.py tests/test_plotting.py -v -k "radiating_helix or radiative_decay or momentum_plane"
    Fig. 2  uv run pytest tests/test_pushers.py tests/test_plotting.py -v -k "radiative_decay_convergence or scaling_error or n44 or scale"
    Fig. 3  uv run pytest tests/test_pushers.py tests/test_plotting.py -v -k "canonical_momentum_follows or tokamak_time_traces or orbit_poloidal"
    Fig. 4  uv run pytest tests/test_particles.py tests/test_diagnostics.py tests/test_plotting.py -v -k "disc or ensemble or is_lost or final_"
"""  # noqa: E501

import argparse
import math
from collections.abc import Callable
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

from ppf_c5_runaways import fields, plotting, run
from ppf_c5_runaways.config import RunConfig, load_config, save_config
from ppf_c5_runaways.constants import c, e, m_e
from ppf_c5_runaways.particles import poloidal_disc_ensemble, proper_velocity
from ppf_c5_runaways.radiation import radiating_helix

PITCH_ANGLES_DEG = (10.0, 30.0, 50.0)
KINETIC_ENERGY = 40.0e6 * e  # [J]
GAMMA_0 = 1.0 + KINETIC_ENERGY / (m_e * c**2)  # 79.3
DIII_D = {"b0": 2.19, "r0": 1.5, "q0": 1.6, "lam": 1.0}  # Carbajal et al., Sec. III
B_STRONGEST = 3.3  # [T], about |B| on the inner side of the plasma, R = 1 m
E_PHYSICAL = 4.0  # [V/m], E0 of Carbajal et al. (Sec. III.D)
SCALE = 1.0e3  # s, the factor on E and F_R
DURATION = 1.0e-2  # [s] of physical time, that of Fig. 12 of Carbajal et al.
START_MINOR_RADIUS = 0.05  # [m], one electron on the outer midplane (Fig. 2b, 3)
DISC_RADIUS = 0.05  # [m], the disc of the ensembles (N48 b, RAPPORT 5.56)
R_EDGE = 0.5  # [m], the edge of the DIII-D plasma
N_ELECTRONS = 3000  # per ensemble, as Carbajal et al.
N_CHECK = 1000  # first electrons of an ensemble for the checks (N48 d)
SEED = 42
# One state kept every FIG4_SAVE_EVERY steps in fig4: the figures need the final
# state, and the losses are read on the kept states (N49). Measured on the first
# 200 electrons of each ensemble, 10 ms, every step (OPEN_POINTS N49): the largest
# r read on one state in 1000 is short of the true one by 11.1 mm at most, and no
# electron comes nearer than 85 mm to the edge.
FIG4_SAVE_EVERY = 1000
# Smallest gamma of the ensembles at the end: 45.9, measured on the same 200
# electrons (radiation only, 50 degrees); 44 leaves room for the other 2800.
FIG4_SMALLEST_GAMMA = 44.0

# name in the .yaml: (electric field on, radiation on)
FORCE_CASES = {
    "both": (True, True),
    "e_only": (True, False),
    "radiation_only": (False, True),
}


def gyration_period(gamma: float, b: float) -> float:
    """Gyration period ``2 pi gamma m / (e B)`` [s] of an electron, B in T."""
    return 2.0 * math.pi * gamma * m_e / (e * b)


# --- writing and running the cases -------------------------------------------------


def ensure_config(config: RunConfig, root: Path) -> Path:
    """Write ``config`` to ``<root>/configs/`` if it is not there; check it if it is."""
    path = root / "configs" / f"{config.name}_{config.pusher}.yaml"
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        save_config(config, path)
        print(f"  wrote {path}")
    elif load_config(path) != config:
        raise ValueError(
            f"{path} differs from the parameters of main_article.py: remove it or fix "
            "the parameters, so that the figures match what is described."
        )
    return path


def run_case(configs: list[RunConfig], root: Path) -> Path:
    """Run the configurations of one case (one per pusher) into one new folder."""
    first, *others = configs
    config_path = ensure_config(first, root)
    print(f"  running {config_path} ...", flush=True)
    result_dir = run.pusher(config_path, base_dir=root / "results")
    for config in others:
        config_path = ensure_config(config, root)
        print(f"  running {config_path} ...", flush=True)
        run.pusher(config_path, result_dir)
    print(f"  results in {result_dir}")
    return result_dir


def case_config(
    name: str,
    field: dict[str, object],
    x0: NDArray[np.float64],
    u0: NDArray[np.float64],
    dt: float,
    n_steps: int,
    save_every: int,
    rr_scale: float,
    pusher: str = "boris",
) -> RunConfig:
    """Build the configuration of one run of an electron of the article."""
    return RunConfig.model_validate(
        {
            "name": name,
            "pusher": pusher,
            "dt": dt,
            "n_steps": n_steps,
            "save_every": save_every,
            "seed": SEED,
            "q": -e,
            "m": m_e,
            "rr_scale": rr_scale,
            "field": field,
            "initial": {"x0": x0.tolist(), "u0": u0.tolist()},
        }
    )


def n_steps_for(duration: float, dt: float, save_every: int) -> int:
    """Count the steps of ``dt`` in ``duration`` [s], a multiple of ``save_every``.

    Rounded down, so the run lasts at most ``duration``, short of it by less than
    ``save_every`` steps. Halving ``dt`` and doubling ``save_every`` doubles it
    exactly, so that a run at ``dt / 2`` lasts the same time.
    """
    return save_every * int(duration / dt / save_every)


# --- the electrons -------------------------------------------------------------------


def uniform_start() -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Three electrons at the origin, 40 MeV and 10, 30, 50 degrees from B along z."""
    pitch = np.radians(PITCH_ANGLES_DEG)
    momentum = math.sqrt(GAMMA_0**2 - 1.0) * c
    u0 = momentum * np.stack(
        [np.sin(pitch), np.zeros(pitch.size), np.cos(pitch)], axis=1
    )
    return np.zeros_like(u0), u0


def tokamak_start(
    pitch_angles_deg: tuple[float, ...],
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Electrons at r = 0.05 m on the outer midplane, 40 MeV, gyration phase 0."""
    n = len(pitch_angles_deg)
    x0 = fields.toroidal_to_cartesian(
        np.full(n, START_MINOR_RADIUS), np.zeros(n), 0.0, DIII_D["r0"]
    )
    b_start = fields.carbajal(**DIII_D, e0=0.0)(x0, 0.0)[1]
    u0 = proper_velocity(
        np.full(n, KINETIC_ENERGY),
        np.radians(pitch_angles_deg),
        np.zeros(n),
        b_start,
        m_e,
    )
    return x0, u0


def uniform_field(electric: bool, scale: float) -> dict[str, object]:
    """Uniform B0 along z and, if asked, the scaled E along -z (it pushes along B)."""
    e_z = -E_PHYSICAL * scale if electric else 0.0
    return {
        "kind": "uniform_cartesian",
        "b_field": [0.0, 0.0, DIII_D["b0"]],
        "e_field": [0.0, 0.0, e_z],
    }


def tokamak_field(electric: bool, scale: float) -> dict[str, object]:
    """DIII-D field of Carbajal et al. with, if asked, the scaled E0 (e0 < 0)."""
    e0 = -E_PHYSICAL * scale if electric else 0.0
    return {"kind": "carbajal", **DIII_D, "e0": e0}


# --- the steps -----------------------------------------------------------------------


def step_fig1(root: Path, show: bool, refinement: int) -> None:
    """Fig. 1: radiative decay in a uniform B, and E || B, three choices of forces."""
    x0, u0 = uniform_start()
    suffix = "" if refinement == 1 else f"_dt{refinement}"

    # (a) s = 1e7: 15 s of physical time, gamma down to gamma_par (5.74 to 1.31).
    # dt from the slowest final gyration, that of the smallest gamma_par.
    scale = 1.0e7
    gamma_par = 1.0 / np.sqrt(1.0 - (u0[:, 2] / (GAMMA_0 * c)) ** 2)
    dt = gyration_period(gamma_par.min(), DIII_D["b0"]) / 100.0 / refinement
    save_every = 100 * refinement
    config = case_config(
        f"article_fig1a{suffix}",
        uniform_field(False, scale),
        x0,
        u0,
        dt,
        n_steps_for(15.0 / scale, dt, save_every),
        save_every,
        scale,
    )
    decay_dir = run_case([config], root)
    plotting.radiative_decay(
        decay_dir, "boris", show=show, save_path=decay_dir / "fig1a_radiative_decay"
    )

    # (b) s = 1e3, 10 ms; gamma stays above 50 (55.3 measured, RAPPORT 5.59).
    dt = gyration_period(50.0, DIII_D["b0"]) / 100.0 / refinement
    save_every = 1000 * refinement
    plane_dirs = []
    for case, (electric, radiation) in FORCE_CASES.items():
        config = case_config(
            f"article_fig1b_{case}{suffix}",
            uniform_field(electric, SCALE),
            x0,
            u0,
            dt,
            n_steps_for(DURATION / SCALE, dt, save_every),
            save_every,
            SCALE if radiation else 0.0,
        )
        plane_dirs.append(run_case([config], root))
    plotting.momentum_plane(
        plane_dirs, SCALE, show=show, save_path=plane_dirs[0] / "fig1b_momentum_plane"
    )
    print(f"  figures in {decay_dir} and {plane_dirs[0]}")


def step_fig2(root: Path, show: bool, refinement: int) -> None:
    """Fig. 2: errors against the exact decay, and the change of gamma with s."""
    if refinement != 1:
        raise ValueError("fig2 is already a study of dt: it has no --half-dt run")
    x0, u0 = uniform_start()
    pushers = ("boris", "vay", "higuera_cary", "rk4")

    # (a) 0.05 s of physical time; dt from the final gyration period of the slowest
    # electron, n_turn steps per period; 100 states kept.
    convergence_dirs = []
    for scale in (1.0e6, 1.0e7):
        duration = 0.05 / scale
        u_end = radiating_helix(u0, DIII_D["b0"], np.array([duration]), -e, m_e, scale)
        gamma_end = np.sqrt(1.0 + np.sum(u_end**2, axis=2) / c**2)[0]
        period_end = gyration_period(gamma_end.min(), DIII_D["b0"])
        for n_turn in (25, 50, 100, 200, 400):
            n_steps = 100 * math.ceil(duration / (period_end / n_turn) / 100)
            configs = [
                case_config(
                    f"article_fig2a_s1e{round(math.log10(scale))}_n{n_turn:03d}",
                    uniform_field(False, scale),
                    x0,
                    u0,
                    duration / n_steps,
                    n_steps,
                    n_steps // 100,
                    scale,
                    pusher,
                )
                for pusher in pushers
            ]
            convergence_dirs.append(run_case(configs, root))
    plotting.radiative_decay_convergence(
        convergence_dirs,
        show=show,
        save_path=convergence_dirs[0] / "fig2a_radiative_decay_convergence",
    )

    # (b) 1 ms of physical time at s = 1e3 to 8e3, with E and radiation; the same
    # dt for all, 400 steps per gyration at 40 MeV and 3.3 T at s = 8e3.
    x0, u0 = tokamak_start(PITCH_ANGLES_DEG)
    duration, largest_scale, n_kept = 1.0e-3, 8.0e3, 400
    period = gyration_period(GAMMA_0, B_STRONGEST)
    n_largest = n_kept * math.ceil(duration / largest_scale / (period / 400.0) / n_kept)
    dt = duration / largest_scale / n_largest
    scaling_dirs = []
    for scale in (1.0e3, 2.0e3, 4.0e3, 8.0e3):
        n_steps = round(n_largest * largest_scale / scale)
        config = case_config(
            f"article_fig2b_s{round(scale):05d}",
            tokamak_field(True, scale),
            x0,
            u0,
            dt,
            n_steps,
            n_steps // n_kept,
            scale,
        )
        scaling_dirs.append(run_case([config], root))
    plotting.scaling_error(
        scaling_dirs, show=show, save_path=scaling_dirs[0] / "fig2b_scaling_error"
    )
    print(f"  figures in {convergence_dirs[0]} and {scaling_dirs[0]}")


def step_fig3(root: Path, show: bool, refinement: int) -> None:
    """Fig. 3: one electron at 30 degrees in the tokamak, three choices of forces."""
    x0, u0 = tokamak_start((30.0,))
    suffix = "" if refinement == 1 else f"_dt{refinement}"
    # gamma stays above 60 (65.7 measured, RAPPORT 5.62). 10 states kept per
    # gyration, so that the orbit shows it (2.5 drew it as a broken line).
    dt = gyration_period(60.0, B_STRONGEST) / 100.0 / refinement
    save_every = 10 * refinement
    result_dirs = []
    for case, (electric, radiation) in FORCE_CASES.items():
        config = case_config(
            f"article_fig3_{case}{suffix}",
            tokamak_field(electric, SCALE),
            x0,
            u0,
            dt,
            n_steps_for(DURATION / SCALE, dt, save_every),
            save_every,
            SCALE if radiation else 0.0,
        )
        result_dirs.append(run_case([config], root))
    plotting.orbit_poloidal(
        result_dirs[0], "boris", show=show, save_path=result_dirs[0] / "fig3a_orbit"
    )
    plotting.tokamak_time_traces(
        result_dirs, SCALE, show=show, save_path=result_dirs[0] / "fig3b_time_traces"
    )
    print(f"  figures in {result_dirs[0]}")


def ensemble_runs(
    root: Path, label: str, n_electrons: int, scale: float, refinement: int
) -> list[Path]:
    """Run the three ensembles of the disc with the three choices of forces."""
    dt = gyration_period(FIG4_SMALLEST_GAMMA, B_STRONGEST) / 100.0 / refinement
    save_every = FIG4_SAVE_EVERY * refinement
    start_field = fields.carbajal(**DIII_D, e0=0.0)
    result_dirs = []
    for pitch_deg in PITCH_ANGLES_DEG:
        x0, u0 = poloidal_disc_ensemble(
            n_electrons,
            DISC_RADIUS,
            KINETIC_ENERGY,
            math.radians(pitch_deg),
            start_field,
            DIII_D["r0"],
            m_e,
            SEED,
        )
        for case, (electric, radiation) in FORCE_CASES.items():
            config = case_config(
                f"article_{label}_theta{pitch_deg:02.0f}_{case}_n{n_electrons}",
                tokamak_field(electric, scale),
                x0,
                u0,
                dt,
                n_steps_for(DURATION / scale, dt, save_every),
                save_every,
                scale if radiation else 0.0,
            )
            result_dirs.append(run_case([config], root))
    return result_dirs


def draw_ensembles(result_dirs: list[Path], scale: float, show: bool) -> None:
    """Draw the three panels of Fig. 4 in the first folder."""
    first = result_dirs[0]
    plotting.final_momentum_cloud(
        result_dirs, scale, R_EDGE, show=show, save_path=first / "fig4a_momenta"
    )
    plotting.final_distributions(
        result_dirs, scale, R_EDGE, "pitch", show=show, save_path=first / "fig4b_pitch"
    )
    plotting.final_distributions(
        result_dirs,
        scale,
        R_EDGE,
        "energy",
        show=show,
        save_path=first / "fig4c_energy",
    )
    print(f"  figures in {first}")


def step_fig4(root: Path, show: bool, refinement: int, n_electrons: int) -> None:
    """Fig. 4: three ensembles in the tokamak, three choices of forces."""
    if refinement != 1:
        raise ValueError("the dt / 2 runs of fig4 are those of fig4_checks")
    draw_ensembles(ensemble_runs(root, "fig4", n_electrons, SCALE, 1), SCALE, show)


def step_fig4_checks(root: Path, show: bool, refinement: int, n_electrons: int) -> None:
    """Run the first electrons of fig4 at dt / 2 (N46) and at s = 2e3 (N48 d)."""
    if refinement != 1:
        raise ValueError("fig4_checks holds its own dt / 2 runs")
    n_check = min(n_electrons, N_CHECK)
    draw_ensembles(ensemble_runs(root, "fig4_dt2", n_check, SCALE, 2), SCALE, show)
    scale = 2.0 * SCALE
    draw_ensembles(ensemble_runs(root, "fig4_s2e3", n_check, scale, 1), scale, show)


# name: (function, cost on the user's laptop, what it shows)
STEPS: dict[str, tuple[Callable[..., None], str, str]] = {
    # fig1 to fig3 measured on the user's laptop on 2026-11-03; fig4 estimated
    # from 5.5e6 particle-steps per second with radiation (N48).
    "fig1": (step_fig1, "measured: 8.5 minutes", "uniform fields: decay, E || B"),
    "fig2": (step_fig2, "measured: 2 minutes", "errors in dt and in s"),
    "fig3": (step_fig3, "measured: 5 minutes", "one electron in the tokamak"),
    "fig4": (
        step_fig4,
        "long, estimated: 2.5 hours and 7 GB",
        "three ensembles of electrons (3000 by default) in the tokamak",
    ),
    "fig4_checks": (
        step_fig4_checks,
        "long, estimated: 2 hours and 4 GB",
        "1000 electrons of fig4 at dt / 2 and at s = 2e3",
    ),
}


def main() -> None:
    """Run the step named on the command line, or list the steps."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("step", nargs="?", choices=STEPS, help="the step to run")
    parser.add_argument("--list", action="store_true", help="list the steps")
    parser.add_argument("--show", action="store_true", help="open the figures")
    parser.add_argument(
        "--half-dt", action="store_true", help="run fig1 or fig3 at dt / 2 (N46)"
    )
    parser.add_argument(
        "--electrons",
        type=int,
        default=N_ELECTRONS,
        help=f"electrons per ensemble in fig4 (default {N_ELECTRONS})",
    )
    parser.add_argument(
        "--root",
        type=Path,
        default=Path("."),
        help="folder of configs/ and results/ (default: the current folder)",
    )
    arguments = parser.parse_args()
    if arguments.list or arguments.step is None:
        for name, (_, cost, purpose) in STEPS.items():
            print(f"{name:12s} {cost:38s} {purpose}")
        return
    if arguments.electrons < 1:
        parser.error(f"--electrons must be >= 1, got {arguments.electrons}")
    step, _, purpose = STEPS[arguments.step]
    refinement = 2 if arguments.half_dt else 1
    print(f"{arguments.step}: {purpose}")
    if arguments.step in ("fig4", "fig4_checks"):
        step(arguments.root, arguments.show, refinement, arguments.electrons)
    else:
        if arguments.electrons != N_ELECTRONS:
            parser.error("--electrons is for fig4 and fig4_checks only")
        step(arguments.root, arguments.show, refinement)


if __name__ == "__main__":
    main()
