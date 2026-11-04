import math
from pathlib import Path

from ppf_c5_runaways import plotting, run
from ppf_c5_runaways.constants import c, e, m_e
from ppf_c5_runaways.create_config import new_config

name = "exb_drift_gamma_100"
gamma = 100.0
b_z = 2.5  # [T]
kappa = math.sqrt(1.0 - 1.0 / gamma**2)  # v_drift / c = E / (c B)

field = {
    "kind": "uniform_cartesian",
    "b_field": [0.0, 0.0, b_z],  # [T]
    "e_field": [kappa * c * b_z, 0.0, 0.0],  # [V/m]
}
u0 = [0.0, -gamma * kappa * c, 0.0]  # u = gamma v, with v = E x B / B^2 along -y
dt = 0.5 * gamma * m_e / (e * b_z)  # the velocity turns by 0.5 rad per step

result_dir = None
for pusher in ("boris", "higuera_cary", "vay"):
    config_path = Path("configs") / f"{name}_{pusher}.yaml"
    if not config_path.exists():  # new_config never overwrites a file
        new_config(name, field=field, u0=u0, dt=dt, n_steps=1000, pusher=pusher)
    if result_dir is None:
        result_dir = run.pusher(config_path)  # new results folder
    else:
        run.pusher(config_path, result_dir=result_dir)  # same case, same folder

plotting.energy_error(result_dir, save_path=result_dir / "energy_error")
