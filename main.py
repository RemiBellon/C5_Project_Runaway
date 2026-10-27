from ppf_c5_runaways import plotting, run
from ppf_c5_runaways.estimations import *
from ppf_c5_runaways.io import list_results, read_result
from ppf_c5_runaways.create_config import new_config, u_from_theta
from ppf_c5_runaways.constants import *
#"""
result_dir = run.pusher("configs/tokamak_theta57_80_alpha10.yaml")
##result_dir = run.pusher("configs/uniform_b_vay.yaml")
#result_dir = run.pusher("configs/uniform_b_higuera_cary.yaml")


#result_dir = "results/tokamak_20261025_192231"

#print(f"Results saved in: {result_dir}")

#plotting.orbit_3d(result_dir, pusher="boris", r0=6.2, a= 2, show=True)
#plotting.orbit_yz(result_dir, pusher="boris")
#result_dir = "results/tokamak_theta57_80_alpha90_20261026_banana_orbit"
# To redraw a figure later WITHOUT computing again, give the folder by hand:
plotting.orbit_rz(result_dir, pusher="boris", r0=6.2, a= 2, show=True)
#plotting.plot_energy(result_dir, pusher="boris")
#plotting.orbit_3d(result_dir, pusher="boris", r0=6.2, a= 2, show=True)



#results = read_result(result_dir, "boris") 
#print(f"Results loaded from: {result_dir}")
#print("x =", results["x"].values)
#print(f"Results: {results['B'].values}, {results['E'].values}")



"""


##how to create a new config file for a tokamak with custom parameters

x_init = [[1, 7.2, 0], [1, 5, 0]]  # initial positions for two particles

b_field_tokamak = [5.3, 6.2, 4, 1.6, "0", "b0 * r * q0 * (1 + r**2/lamb**2) * 1 / r0*(1 + (r/r0)*cos(theta))","b0 * 1 / (1 + (r/r0)*cos(theta))"]
#e_field_tokamak = [-4, 0,0, "E0/(1+ r/r0 *cos(theta))"] 
e_field_tokamak = None #no electric field
theta = [57, 57]  # angle in degrees
energies = [40, 80]  # energy in MeV
u_init = u_from_theta(
    energy=energies, theta=theta, m=m_e,
    x0=x_init, alpha=0, b_field=b_field_tokamak
)
#print(f"Initial velocity u0: {u_init}")

new_config(name="tokamak_Etot_40_80", kind="tokamak", b_field=b_field_tokamak, 
           e_field=e_field_tokamak, steps=10000, pusher='boris', dt=1e-11, 
           save_every=5, seed=42, q=-1.602176634e-19, m=9.1093837139e-31, 
           x0=x_init, u0=u_init)
"""