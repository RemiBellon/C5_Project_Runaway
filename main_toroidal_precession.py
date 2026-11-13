from ppf_c5_runaways import plotting 
from ppf_c5_runaways import run

config = "configs/ITER_deg50_80_10MeV_yesE_vay.yaml"

result_dir = run.pusher(config)              # lance le calcul, écrit dans results/


print(f"Results saved in: {result_dir}")


plotting.orbit_poloidal(result_dir, pusher="vay",show=True)
plotting.orbit_guiding_centre_3D(result_dir, pusher="vay",r0=6.2, a=2.0) 
