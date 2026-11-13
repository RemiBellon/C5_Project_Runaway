from ppf_c5_runaways import plotting, run


#result_dir = "results/ITER_toroidal_precession_65deg_40MeV_20261113_111118" #with boris
#result_dir = "results/ITER_65deg_10MeV_20261113_120030" #with vay
#result_dir = "results/ITER_65deg_10MeV_noE_20261113_124653"
result_dir = "results/ITER_deg50_80_10MeV_noE_20261113_145518"
#result_dir = "results/ITER_deg50_80_10MeV_yesE_20261113_154508"
#result_dir = run.pusher("configs/ITER_65deg_10MeV_quickvay.yaml")              # lance le calcul, écrit dans results/

#plotting.orbit_3d(result_dir, pusher="boris",r0=6.2, a=2.0) 

#plotting.orbit_poloidal(result_dir, pusher="vay",show=True)
#plotting.orbit_3d(result_dir, pusher="vay",r0=6.2, a=2.0) 

#plotting.orbit_guiding_centre_3D(result_dir, pusher="vay",r0=6.2, a=2.0, show=True, one_particle=1)
#plotting.orbit_poloidal_gc(result_dir, pusher="vay",show=True)#, one_particle=1)
config = "configs/ITER_deg50_80_10MeV_noE_vay.yaml"
plotting.visu_toroidal_fields(config, r0=6.2, a=2.0)