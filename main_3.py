import math
from pathlib import Path

from ppf_c5_runaways import plotting, run
from ppf_c5_runaways.constants import c, e, m_e
from ppf_c5_runaways.create_config import new_config

# E x B drift scan, fields of Ripperda et al. (2018, Sec. 3.1.3 and 3.1.4):
# E = (E0, 0, 0), B = (0, 0, b_z). One results folder per drift speed (gamma fixed
# per run), the three pushers in each folder. The electron is launched exactly at
# the drift velocity v_E = E x B / B^2, so it must not gyrate.
b_z = 2.5  # [T]
theta = 0.5  # rotation angle per step [rad], at the drift Lorentz factor
n_steps = 200000
drift_proper_speeds = [1e-3, 1e-2, 1e-1, 1.0, 3.0, 10.0, 30.0, 100.0, 300.0, 1000.0]

result_dirs = []
for index, u_over_c in enumerate(drift_proper_speeds):
    kappa = math.sqrt(1.0 + u_over_c**2)  # Lorentz factor of the drift
    field = {
        "kind": "uniform_cartesian",
        "b_field": [0.0, 0.0, b_z],  # [T]
        "e_field": [u_over_c / kappa * c * b_z, 0.0, 0.0],  # [V/m], |E| < c B
    }
    u0 = [0.0, -u_over_c * c, 0.0]  # u = kappa v_E, v_E along -y
    dt = theta * kappa * m_e / (e * b_z)
    name = f"exb_drift_scan_{index:02d}"
    result_dir = None
    for pusher in ("boris", "higuera_cary", "vay"):
        config_path = Path("configs") / f"{name}_{pusher}.yaml"
        if not config_path.exists():  # new_config never overwrites a file
            new_config(name, field=field, u0=u0, dt=dt, n_steps=n_steps, pusher=pusher)
        if result_dir is None:
            result_dir = run.pusher(config_path)
        else:
            run.pusher(config_path, result_dir=result_dir)
    result_dirs.append(result_dir)

plotting.drift_error(result_dirs, save_path=Path("results") / "exb_drift_error")
