from ppf_c5_runaways import plotting, run

result_dir = run.pusher("configs/tokamak_custom_toroidal_boris.yaml")
#result_dir = run.pusher("configs/uniform_b_vay.yaml")
#result_dir = run.pusher("configs/uniform_b_higuera_cary.yaml")

print(f"Results saved in: {result_dir}")

plotting.orbit_xz(result_dir, pusher="boris")
plotting.orbit_xy(result_dir, pusher="boris")

#plotting.visufields(result_dir,pusher="boris",r0=6.2,a=2.0)
plotting.orbit_3d(result_dir, pusher="boris",r0=6.2, a=2.0) 


# To redraw a figure later WITHOUT computing again, give the folder by hand:
# plotting.orbit_xy("results/uniform_b_20260921_120000", pusher="boris")
