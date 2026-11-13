from ppf_c5_runaways import plotting 
import math
import numpy as np
from ppf_c5_runaways import fields, run
from ppf_c5_runaways.constants import e, m_e
from ppf_c5_runaways.create_config import new_config
from ppf_c5_runaways.particles import proper_velocity

ITER = {"b0": 5.3, "r0": 6.2, "q0": 1.6, "lam": 4.0}
s = 1.0e3                                  # facteur commun sur E et F_R
e0 = -4.0 * s
field_cfg = {"kind": "carbajal", **ITER, "e0": e0}

#  2 électrons à r = 5 cm, plan médian extérieur, 10 MeV, 50° et 80° (pour voir la différence de précession toroidale)
x0 = fields.toroidal_to_cartesian(np.array([0.8]), np.array([0.0]), 0.0, ITER["r0"])
b_start = fields.carbajal(**ITER, e0=e0)(x0, 0.0)[1]
u0 = proper_velocity(10e6 * e, math.radians(50.0), 0.0, b_start, m_e)

x1 = fields.toroidal_to_cartesian(np.array([0.8]), np.array([0.0]), 0.0, ITER["r0"])
b_start1 = fields.carbajal(**ITER, e0=e0)(x1, 0.0)[1]
u1 = proper_velocity(10e6 * e, math.radians(80.0), 0.0, b_start1, m_e)

x = np.array([x0[0], x1[0]])
u = np.array([u0[0], u1[0]])


# Pas de temps : 1/100 de la giration la plus rapide
gamma_min, b_max = 44.0, 3.3               # b_max ≈ b0 R0 / (R0 - 0.5) côté intérieur
dt = 2 * math.pi * gamma_min * m_e / (e * b_max) / 100
duration = 1e-2 / s                        # 10 ms physiques = 10 µs calculées pour gagné du temps de simulation
save_every = 1000
#n_steps=60000
n_steps = save_every * int(duration / dt / save_every)

path = new_config("ITER_deg50_80_10MeV_yesE", field=field_cfg, x0=x, u0=u, dt=dt,
                  n_steps=n_steps, save_every=save_every, rr_scale=s, pusher="vay")



#result_dir = run.pusher(path)              # lance le calcul, écrit dans results/



#print(f"Results saved in: {result_dir}")

#plotting.visufields(result_dir,pusher="vay",r0=6.2,a=2.0)
#plotting.orbit_3d(result_dir, pusher="vay",r0=6.2, a=2.0) 