"""Gyration phase in a uniform B: Ripperda et al. (2018), Sec. 3.1.2, Fig. 4."""

import math
from pathlib import Path

from ppf_c5_runaways import plotting, run
from ppf_c5_runaways.constants import c, e, m_e
from ppf_c5_runaways.create_config import new_config

# Gyration phase in a uniform B: Ripperda et al. (2018), Sec. 3.1.2, Fig. 4.
# gamma = 1e6, velocity perpendicular to B, 100 steps per turn, 100 turns.
name = "gyration_gamma_1e6"
b_z = 2.5  # [T]
gamma = 1.0e6
omega = e * b_z / (gamma * m_e)  # exact gyration frequency [rad/s]
field = {"kind": "uniform_cartesian", "b_field": [0, 0, b_z], "e_field": [0, 0, 0]}
u0 = [0.0, -math.sqrt(gamma**2 - 1.0) * c, 0.0]  # perpendicular to B

result_dir = None
for pusher in ("boris", "higuera_cary", "vay"):
    config_path = Path("configs") / f"{name}_{pusher}.yaml"
    if not config_path.exists():  # new_config never overwrites a file
        new_config(
            name,
            field=field,
            u0=u0,
            dt=2.0 * math.pi / omega / 100,  # 100 steps per turn
            n_steps=100 * 100,  # 100 turns
            save_every=10,  # 0.63 rad between saved states, below pi / 2
            pusher=pusher,
        )
    if result_dir is None:
        result_dir = run.pusher(config_path)
    else:
        run.pusher(config_path, result_dir=result_dir)

plotting.phase_error(result_dir, save_path=result_dir / "phase_error")
