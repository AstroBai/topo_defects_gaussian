import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from theory.WLDefects import DefectDensity
from GenMapData import GenMap
from DefectFinder import DefectFinder
import numpy as np
import pyccl as ccl
from tqdm import tqdm
from healpy.sphtfunc import smoothing
import glass

Oms = np.array([0.1, 0.2, 0.3, 0.4, 0.5])
sigma8s = np.array([0.4, 0.6, 0.8, 1.0, 1.2])

density_sim_list_Om = []
density_sim_std_list_Om = []
density_finder_list_Om = []
density_finder_std_list_Om = []

DATA_DIR = "/cephfs/jbai/DefectValidation"
pixel_size = np.sqrt(4*np.pi * (180/np.pi)**2 / (256*256) / 12) * 60 
sigma = 10 * pixel_size
sigma = np.radians(sigma / 60)

z_source = np.linspace(0., 2.0, 100)
dndz_source = np.zeros_like(z_source)
# set all sources at z=1 for simplicity
dndz_source[np.argmin(np.abs(z_source - 1.0))] = 1.0

density_theory_list_Om = []
for Om in tqdm(Oms):
    cosmo_params = {
        'Oc': Om - 0.05,
        'Ob': 0.05,
        'h': 0.7,
        'sigma8': 0.8
    }
    
    # Theory calculation
    cosmo_theory = ccl.Cosmology(Omega_c=Om-0.05, Omega_b=0.05, h=0.7, sigma8=0.8, n_s=0.96)
    density_theory = DefectDensity(cosmo_theory, (z_source, dndz_source), lmax=256*3, filter='gaussian', filter_angle=sigma, include_pixel_window=False, nside=256)
    density_theory_list_Om.append(density_theory)

    density_sim_ = []
    density_finder_ = []
    for seed in range(10):
        # os.makedirs(DATA_DIR, exist_ok=True)
        # kappa_bar_unsmoothed, kappa_bar, gamm1_bar, gamm2_bar, phi_bar = GenMap(cosmo_params, nside=256, lmax=256*3, Plot=False, Savepath=f"outfiles/shear_field_patch_hires_Om{Om}.png", random_seed=seed)
        # np.savez(f"{DATA_DIR}/ShearConvergenceMapData_Om{Om}_{seed}.npz",
        #         kappa_bar_unsmoothed=kappa_bar_unsmoothed,
        #         kappa_bar=kappa_bar,
        #         gamm1_bar=gamm1_bar,
        #         gamm2_bar=gamm2_bar,
        #         phi_bar=phi_bar)
        df = DefectFinder(f"{DATA_DIR}/ShearConvergenceMapData_Om{Om}_{seed}.npz")
        
        df.kappa_bar = smoothing(df.kappa_bar_unsmoothed, sigma=sigma, lmax=256*3)
        df.gamm1_bar, df.gamm2_bar = glass.shear_from_convergence(df.kappa_bar, lmax=256*3)
        df.shear_amp = np.sqrt(df.gamm1_bar**2 + df.gamm2_bar**2)
        df.phi_bar = 0.5 * np.arctan2(df.gamm2_bar, df.gamm1_bar)
        
        df.find_defects(savepath=f"{DATA_DIR}/defects_data_Om{Om}_{seed}.npz", cluster=True)
        df.load_defects(f"{DATA_DIR}/defects_data_Om{Om}_{seed}.npz")
        df.save_defects(savepath=f"{DATA_DIR}/defects_data_Om{Om}_{seed}")
        
        density_sim = df.number_estimation(lmax=256*3, psi_method='from_kappa_cl')
        density_finder, pos, neg = df.number()
        print(pos, neg)
        density_sim_.append(density_sim)
        density_finder_.append(density_finder)

    density_sim_list_Om.append(np.mean(density_sim_))
    density_finder_list_Om.append(np.mean(density_finder_))
    density_sim_std_list_Om.append(np.std(density_sim_))
    density_finder_std_list_Om.append(np.std(density_finder_))


density_sim_list_sigma8 = []
density_sim_std_list_sigma8 = []
density_finder_list_sigma8 = []
density_finder_std_list_sigma8 = []

density_theory_list_sigma8 = []
for sigma8 in tqdm(sigma8s):
    cosmo_params = {
        'Oc': 0.25,
        'Ob': 0.05,
        'h': 0.7,
        'sigma8': sigma8
    }

    # Theory calculation
    cosmo_theory = ccl.Cosmology(Omega_c=0.25, Omega_b=0.05, h=0.7, sigma8=sigma8, n_s=0.96)
    density_theory = DefectDensity(cosmo_theory, (z_source, dndz_source), lmax=256*3, filter='gaussian', filter_angle=sigma, include_pixel_window=False, nside=256)
    density_theory_list_sigma8.append(density_theory)

    density_sim_ = []
    density_finder_ = []
    for seed in range(10):
        # os.makedirs(DATA_DIR, exist_ok=True)
        # kappa_bar_unsmoothed, kappa_bar, gamm1_bar, gamm2_bar, phi_bar = GenMap(cosmo_params, nside=256, lmax=256*3, Plot=False, Savepath=f"outfiles/shear_field_patch_hires_sigma8{sigma8}.png", random_seed=seed)
        # np.savez(f"{DATA_DIR}/ShearConvergenceMapData_sigma8{sigma8}_{seed}.npz",
        #         kappa_bar_unsmoothed=kappa_bar_unsmoothed,
        #         kappa_bar=kappa_bar,
        #         gamm1_bar=gamm1_bar,
        #         gamm2_bar=gamm2_bar,
        #         phi_bar=phi_bar)
        df = DefectFinder(f"{DATA_DIR}/ShearConvergenceMapData_sigma8{sigma8}_{seed}.npz")
        
        df.kappa_bar = smoothing(df.kappa_bar_unsmoothed, sigma=sigma, lmax=256*3)
        df.gamm1_bar, df.gamm2_bar = glass.shear_from_convergence(df.kappa_bar, lmax=256*3)
        df.shear_amp = np.sqrt(df.gamm1_bar**2 + df.gamm2_bar**2)
        df.phi_bar = 0.5 * np.arctan2(df.gamm2_bar, df.gamm1_bar)

        df.find_defects(savepath=f"{DATA_DIR}/defects_data_sigma8{sigma8}_{seed}.npz", cluster=True)
        df.load_defects(f"{DATA_DIR}/defects_data_sigma8{sigma8}_{seed}.npz")
        df.save_defects(savepath=f"{DATA_DIR}/defects_data_sigma8{sigma8}_{seed}")
        
        density_sim = df.number_estimation(lmax=256*3, psi_method='from_kappa_cl')
        density_finder, pos, neg = df.number()
        print(pos, neg)
        density_sim_.append(density_sim)
        density_finder_.append(density_finder)

    density_sim_list_sigma8.append(np.mean(density_sim_))
    density_finder_list_sigma8.append(np.mean(density_finder_))
    density_sim_std_list_sigma8.append(np.std(density_sim_))
    density_finder_std_list_sigma8.append(np.std(density_finder_))





import matplotlib.pyplot as plt
import matplotlib as mpl
mpl.rcParams['xtick.direction'] = 'in'
mpl.rcParams['ytick.direction'] = 'in'
mpl.rcParams['font.size'] = 8
color_map = plt.get_cmap('tab10')


fig, ax1 = plt.subplots(figsize=(4, 3))

# Plot Omega_m on bottom axis
ax1.plot(Oms, density_theory_list_Om, label=r'Theory ($\Omega_{\rm m}$)', ls='--', color=color_map(0))
ax1.fill_between(Oms, np.array(density_sim_list_Om) - np.array(density_sim_std_list_Om), np.array(density_sim_list_Om) + np.array(density_sim_std_list_Om), alpha=0.3, label=r'Expected ($\Omega_{\rm m}$)', color=color_map(0))
ax1.errorbar(Oms, density_finder_list_Om, yerr=density_finder_std_list_Om, label=r'Measured ($\Omega_{\rm m}$)', fmt='o', markersize=3, color=color_map(0))
ax1.set_xlabel(r'$\Omega_{\rm m}$')
ax1.set_ylabel('Defect Density')
ax1.set_xticks(Oms)

# Plot sigma8 on top axis
ax2 = ax1.twiny()
ax2.plot(sigma8s, density_theory_list_sigma8, label=r'Theory ($\sigma_{\rm 8}$)', ls='--', color=color_map(1))
ax2.fill_between(sigma8s, np.array(density_sim_list_sigma8) - np.array(density_sim_std_list_sigma8), np.array(density_sim_list_sigma8) + np.array(density_sim_std_list_sigma8), alpha=0.3, label=r'Expected ($\sigma_{\rm 8}$)', color=color_map(1))
ax2.errorbar(sigma8s, density_finder_list_sigma8, yerr=density_finder_std_list_sigma8, label=r'Measured ($\sigma_{\rm 8}$)', fmt='o', markersize=3, color=color_map(1))
ax2.set_xlabel(r'$\sigma_{\rm 8}$')
ax2.set_xticks(sigma8s)

# Combine legends
lines1, labels1 = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax1.legend(lines1 + lines2, labels1 + labels2, loc='best')

plt.tight_layout()
plt.savefig("outfiles/defect_density_vs_Om.png", dpi=300)

#===========================================================================

# import matplotlib.pyplot as plt
# import matplotlib as mpl
# mpl.rcParams['xtick.direction'] = 'in'
# mpl.rcParams['ytick.direction'] = 'in'
# mpl.rcParams['font.size'] = 8
# color_map = plt.get_cmap('tab10')



# from scipy.ndimage import gaussian_filter
# lmaxs_ratio = np.linspace(1/4, 3, 20)
# lmaxs = (256 * lmaxs_ratio).astype(int)
# smoothing_scales = np.array([0, 2, 4, 8]) # in units of pixel

# density_sim_list_lmax = []
# density_sim_std_list_lmax = []
# density_finder_list_lmax = []
# density_finder_std_list_lmax = []

# density_sim_list_smoothing = []
# density_sim_std_list_smoothing = []
# density_finder_list_smoothing = []
# density_finder_std_list_smoothing = []

# for smoothing_scale in smoothing_scales:
#     if smoothing_scale == 0:
#         for lmax in tqdm(lmaxs):
#             cosmo_params = {
#                 'Oc': 0.25,
#                 'Ob': 0.05,
#                 'h': 0.7,
#                 'sigma8': 0.8
#             }
#             density_sim_ = []
#             density_finder_ = []
#             for seed in range(10):
#                 os.makedirs(DATA_DIR, exist_ok=True)
#                 # kappa_bar_unsmoothed, kappa_bar, gamm1_bar, gamm2_bar, phi_bar = GenMap(cosmo_params, nside=256, lmax=int(lmax), Plot=False, Savepath=f"outfiles/shear_field_patch_hires_lmax{lmax}_seed{seed}.png", random_seed=seed)
#                 # np.savez(f"{DATA_DIR}/ShearConvergenceMapData_lmax{lmax}_{seed}.npz",
#                 #         kappa_bar_unsmoothed=kappa_bar_unsmoothed,
#                 #         kappa_bar=kappa_bar,
#                 #         gamm1_bar=gamm1_bar,
#                 #         gamm2_bar=gamm2_bar,
#                 #         phi_bar=phi_bar)
#                 df = DefectFinder(f"{DATA_DIR}/ShearConvergenceMapData_lmax{lmax}_{seed}.npz")
#                 df.find_defects(savepath=f"{DATA_DIR}/defects_data_lmax{lmax}_{seed}.npz", cluster=True)
#                 df.load_defects(f"{DATA_DIR}/defects_data_lmax{lmax}_{seed}.npz")
#                 df.save_defects(savepath=f"{DATA_DIR}/defects_data_lmax{lmax}_{seed}")
#                 density_sim = df.number_estimation(lmax=int(lmax))
#                 density_finder, pos, neg = df.number()
#                 density_sim_.append(density_sim)
#                 density_finder_.append(density_finder)

#             density_sim_list_lmax.append(np.mean(density_sim_))
#             density_finder_list_lmax.append(np.mean(density_finder_))
#             density_sim_std_list_lmax.append(np.std(density_sim_))
#             density_finder_std_list_lmax.append(np.std(density_finder_))
#     else:
#         for lmax in tqdm(lmaxs):
#             for seed in range(10):
#                 df = DefectFinder(f"{DATA_DIR}/ShearConvergenceMapData_lmax{lmax}_{seed}.npz")
#                 df.kappa_bar_unsmoothed = gaussian_filter(df.kappa_bar_unsmoothed, sigma=smoothing_scale)
#                 df.kappa_bar = gaussian_filter(df.kappa_bar, sigma=smoothing_scale)
#                 df.gamm1_bar = gaussian_filter(df.gamm1_bar, sigma=smoothing_scale)
#                 df.gamm2_bar = gaussian_filter(df.gamm2_bar, sigma=smoothing_scale)
#                 df.phi_bar = np.arctan2(df.gamm2_bar, df.gamm1_bar) / 2
#                 df.find_defects(savepath=f"{DATA_DIR}/defects_data_lmax{lmax}_smoothing{smoothing_scale}_seed{seed}.npz", cluster=True)
#                 df.load_defects(f"{DATA_DIR}/defects_data_lmax{lmax}_smoothing{smoothing_scale}_seed{seed}.npz")
#                 df.save_defects(savepath=f"{DATA_DIR}/defects_data_lmax{lmax}_smoothing{smoothing_scale}_seed{seed}")
#                 density_sim = df.number_estimation(lmax=int(lmax))
#                 density_finder, pos, neg = df.number()
#                 density_sim_.append(density_sim)
#                 density_finder_.append(density_finder)  
#             density_sim_list_smoothing.append(np.mean(density_sim_))
#             density_finder_list_smoothing.append(np.mean(density_finder_))
#             density_sim_std_list_smoothing.append(np.std(density_sim_))
#             density_finder_std_list_smoothing.append(np.std(density_finder_))

    

# plt.figure(figsize=(4, 3))
# plt.fill_between(lmaxs_ratio, np.array(density_sim_list_lmax) - np.array(density_sim_std_list_lmax), np.array(density_sim_list_lmax) + np.array(density_sim_std_list_lmax), alpha=0.3, label='Expected', color=color_map(0))
# plt.errorbar(lmaxs_ratio, density_finder_list_lmax, yerr=density_finder_std_list_lmax, label='Measured', fmt='o', markersize=3, color=color_map(0))
# plt.fill_between(lmaxs_ratio, np.array(density_sim_list_smoothing)[:20] - np.array(density_sim_std_list_smoothing)[:20], np.array(density_sim_list_smoothing)[:20] + np.array(density_sim_std_list_smoothing)[:20], alpha=0.3, label='Expected (Smoothed 2 pixels)', color=color_map(1))
# plt.errorbar(lmaxs_ratio, density_finder_list_smoothing[:20], yerr=density_finder_std_list_smoothing[:20], label='Measured (Smoothed 2 pixels)', fmt='o', markersize=3, color=color_map(1))
# plt.fill_between(lmaxs_ratio, np.array(density_sim_list_smoothing)[20:40] - np.array(density_sim_std_list_smoothing)[20:40], np.array(density_sim_list_smoothing)[20:40] + np.array(density_sim_std_list_smoothing)[20:40], alpha=0.3, label='Expected (Smoothed 4 pixels)', color=color_map(2))
# plt.errorbar(lmaxs_ratio, density_finder_list_smoothing[20:40], yerr=density_finder_std_list_smoothing[20:40], label='Measured (Smoothed 4 pixels)', fmt='o', markersize=3, color=color_map(2))
# plt.fill_between(lmaxs_ratio, np.array(density_sim_list_smoothing)[40:60] - np.array(density_sim_std_list_smoothing)[40:60], np.array(density_sim_list_smoothing)[40:60] + np.array(density_sim_std_list_smoothing)[40:60], alpha=0.3, label='Expected (Smoothed 8 pixels)', color=color_map(3))
# plt.errorbar(lmaxs_ratio, density_finder_list_smoothing[40:60], yerr=density_finder_std_list_smoothing[40:60], label='Measured (Smoothed 8 pixels)', fmt='o', markersize=3, color=color_map(3))
# # plt.xscale('log')
# # plt.yscale('log')
# plt.xlabel(r'$\ell_{\rm max}$')
# plt.ylabel('Defect Density')
# plt.legend()
# plt.tight_layout()
# plt.savefig("outfiles/defect_density_vs_lmax.png", dpi=300)