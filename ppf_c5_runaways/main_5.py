"""Gyration phase error per turn against u / c and the pitch angle, uniform B."""

import math
from pathlib import Path

import numpy as np

from ppf_c5_runaways import plotting, run
from ppf_c5_runaways.constants import c, e, m_e
from ppf_c5_runaways.create_config import new_config

# Gyration phase error per turn against u / c and the pitch angle, uniform B.
# One run per Lorentz factor (all particles share it), one particle per pitch angle.
b_z = 2.5  # [T]
steps_per_turn = 100  # theta = 2 pi / 100, as in Ripperda et al. (2018, Sec. 3.1.2)
n_turns = 10
pitch_angles = np.radians(np.arange(5.0, 91.0, 5.0))  # 5 to 90 degrees
proper_speeds = np.logspace(-2.0, 4.0, 13)  # u / c, gamma from 1.00005 to 1e4

result_dirs = []
for index, u_over_c in enumerate(proper_speeds):
    gamma = math.sqrt(1.0 + u_over_c**2)
    omega = e * b_z / (gamma * m_e)  # exact gyration frequency [rad/s]
    speed = u_over_c * c
    u0 = [[0.0, -speed * math.sin(p), speed * math.cos(p)] for p in pitch_angles]
    name = f"phase_map_{index:02d}"
    result_dir = None
    for pusher in ("boris", "higuera_cary", "vay"):
        config_path = Path("configs") / f"{name}_{pusher}.yaml"
        if not config_path.exists():  # new_config never overwrites a file
            new_config(
                name,
                field={
                    "kind": "uniform_cartesian",
                    "b_field": [0, 0, b_z],
                    "e_field": [0, 0, 0],
                },
                u0=u0,
                dt=2.0 * math.pi / omega / steps_per_turn,
                n_steps=n_turns * steps_per_turn,
                save_every=10,  # 0.63 rad between saved states, below pi / 2
                pusher=pusher,
            )
        if result_dir is None:
            result_dir = run.pusher(config_path)
        else:
            run.pusher(config_path, result_dir=result_dir)
    result_dirs.append(result_dir)
plotting.phase_error_map(
    result_dirs, "higuera_cary", save_path=Path("results") / "phase_error_map_hc"
)
