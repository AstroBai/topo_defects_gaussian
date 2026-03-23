import sys
import os

import glass
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from theory.WLDefects import DefectDensity
from GenMapData import GenMap
from DefectFinder import DefectFinder
import numpy as np
import pyccl as ccl
from tqdm import tqdm
from healpy.sphtfunc import smoothing

# Oms = np.array([0.1, 0.2, 0.3, 0.4, 0.5])
# sigma8s = np.array([0.4, 0.6, 0.8, 1.0, 1.2])

# density_sim_list_Om = []
# density_sim_std_list_Om = []
# density_finder_list_Om = []
# density_finder_std_list_Om = []

DATA_DIR = "/cephfs/jbai/DefectValidation"
# for Om in tqdm(Oms):
#     cosmo_params = {
#         'Oc': Om - 0.05,
#         'Ob': 0.05,
#         'h': 0.7,
#         'sigma8': 0.8
#     }
#     density_sim_ = []
#     density_finder_ = []
#     for seed in range(10):
#         os.makedirs(DATA_DIR, exist_ok=True)
#         # kappa_bar_unsmoothed, kappa_bar, gamm1_bar, gamm2_bar, phi_bar = GenMap(cosmo_params, nside=256, lmax=64, Plot=False, Savepath=f"outfiles/shear_field_patch_hires_Om{Om}.png", random_seed=seed)
#         # np.savez(f"{DATA_DIR}/ShearConvergenceMapData_Om{Om}_{seed}.npz",
#         #         kappa_bar_unsmoothed=kappa_bar_unsmoothed,
#         #         kappa_bar=kappa_bar,
#         #         gamm1_bar=gamm1_bar,
#         #         gamm2_bar=gamm2_bar,
#         #         phi_bar=phi_bar)
#         df = DefectFinder(f"{DATA_DIR}/ShearConvergenceMapData_Om{Om}_{seed}.npz")
#         # df.find_defects(savepath=f"{DATA_DIR}/defects_data_Om{Om}_{seed}.npz", cluster=True)
#         df.load_defects(f"{DATA_DIR}/defects_data_Om{Om}_{seed}.npz")
#         df.save_defects(savepath=f"{DATA_DIR}/defects_data_Om{Om}_{seed}")
        
#         density_sim = df.number_estimation()
#         density_finder, pos, neg = df.number()
#         print(pos, neg)
#         density_sim_.append(density_sim)
#         density_finder_.append(density_finder)

#     density_sim_list_Om.append(np.mean(density_sim_))
#     density_finder_list_Om.append(np.mean(density_finder_))
#     density_sim_std_list_Om.append(np.std(density_sim_))
#     density_finder_std_list_Om.append(np.std(density_finder_))


# density_sim_list_sigma8 = []
# density_sim_std_list_sigma8 = []
# density_finder_list_sigma8 = []
# density_finder_std_list_sigma8 = []

# for sigma8 in tqdm(sigma8s):
#     cosmo_params = {
#         'Oc': 0.25,
#         'Ob': 0.05,
#         'h': 0.7,
#         'sigma8': sigma8
#     }
#     density_sim_ = []
#     density_finder_ = []
#     for seed in range(10):
#         os.makedirs(DATA_DIR, exist_ok=True)
#         # kappa_bar_unsmoothed, kappa_bar, gamm1_bar, gamm2_bar, phi_bar = GenMap(cosmo_params, nside=256, lmax=64, Plot=False, Savepath=f"outfiles/shear_field_patch_hires_sigma8{sigma8}.png", random_seed=seed)
#         # np.savez(f"{DATA_DIR}/ShearConvergenceMapData_sigma8{sigma8}_{seed}.npz",
#         #         kappa_bar_unsmoothed=kappa_bar_unsmoothed,
#         #         kappa_bar=kappa_bar,
#         #         gamm1_bar=gamm1_bar,
#         #         gamm2_bar=gamm2_bar,
#         #         phi_bar=phi_bar)
#         df = DefectFinder(f"{DATA_DIR}/ShearConvergenceMapData_sigma8{sigma8}_{seed}.npz")
#         # df.find_defects(savepath=f"{DATA_DIR}/defects_data_sigma8{sigma8}_{seed}.npz", cluster=True)
#         df.load_defects(f"{DATA_DIR}/defects_data_sigma8{sigma8}_{seed}.npz")
#         df.save_defects(savepath=f"{DATA_DIR}/defects_data_sigma8{sigma8}_{seed}")
        
#         density_sim = df.number_estimation()
#         density_finder, pos, neg = df.number()
#         print(pos, neg)
#         density_sim_.append(density_sim)
#         density_finder_.append(density_finder)

#     density_sim_list_sigma8.append(np.mean(density_sim_))
#     density_finder_list_sigma8.append(np.mean(density_finder_))
#     density_sim_std_list_sigma8.append(np.std(density_sim_))
#     density_finder_std_list_sigma8.append(np.std(density_finder_))





# import matplotlib.pyplot as plt
# import matplotlib as mpl
# mpl.rcParams['xtick.direction'] = 'in'
# mpl.rcParams['ytick.direction'] = 'in'
# mpl.rcParams['font.size'] = 8
# color_map = plt.get_cmap('tab10')


# fig, ax1 = plt.subplots(figsize=(4, 3))

# # Plot Omega_m on bottom axis
# ax1.fill_between(Oms, np.array(density_sim_list_Om) - np.array(density_sim_std_list_Om), np.array(density_sim_list_Om) + np.array(density_sim_std_list_Om), alpha=0.3, label=r'Expected ($\Omega_{\rm m}$)', color=color_map(0))
# ax1.errorbar(Oms, density_finder_list_Om, yerr=density_finder_std_list_Om, label=r'Measured ($\Omega_{\rm m}$)', fmt='o', markersize=3, color=color_map(0))
# ax1.set_xlabel(r'$\Omega_{\rm m}$')
# ax1.set_ylabel('Defect Density')
# ax1.set_xticks(Oms)

# # Plot sigma8 on top axis
# ax2 = ax1.twiny()
# ax2.fill_between(sigma8s, np.array(density_sim_list_sigma8) - np.array(density_sim_std_list_sigma8), np.array(density_sim_list_sigma8) + np.array(density_sim_std_list_sigma8), alpha=0.3, label=r'Expected ($\sigma_{\rm 8}$)', color=color_map(1))
# ax2.errorbar(sigma8s, density_finder_list_sigma8, yerr=density_finder_std_list_sigma8, label=r'Measured ($\sigma_{\rm 8}$)', fmt='o', markersize=3, color=color_map(1))
# ax2.set_xlabel(r'$\sigma_{\rm 8}$')
# ax2.set_xticks(sigma8s)

# # Combine legends
# lines1, labels1 = ax1.get_legend_handles_labels()
# lines2, labels2 = ax2.get_legend_handles_labels()
# ax1.legend(lines1 + lines2, labels1 + labels2, loc='best')

# plt.tight_layout()
# plt.savefig("outfiles/defect_density_vs_Om.png", dpi=300)

#===========================================================================

import matplotlib.pyplot as plt
import matplotlib as mpl
mpl.rcParams['xtick.direction'] = 'in'
mpl.rcParams['ytick.direction'] = 'in'
mpl.rcParams['font.size'] = 8
color_map = plt.get_cmap('tab10')

# density_sim_list_lmax = []
# density_sim_std_list_lmax = []
# density_finder_list_lmax = []
# density_finder_std_list_lmax = []

# lmaxs_ratio = np.linspace(1/4, 3, 20)
# lmaxs = (256 * lmaxs_ratio).astype(int)
# for lmax in tqdm(lmaxs):
#     cosmo_params = {
#         'Oc': 0.25,
#         'Ob': 0.05,
#         'h': 0.7,
#         'sigma8': 0.8
#     }
#     density_sim_ = []
#     density_finder_ = []
#     for seed in range(10):
#         # os.makedirs(DATA_DIR, exist_ok=True)
#         # kappa_bar_unsmoothed, kappa_bar, gamm1_bar, gamm2_bar, phi_bar = GenMap(cosmo_params, nside=256, lmax=int(lmax), Plot=False, Savepath=f"outfiles/shear_field_patch_hires_lmax{lmax}_seed{seed}.png", random_seed=seed)
#         # np.savez(f"{DATA_DIR}/ShearConvergenceMapData_lmax{lmax}_{seed}.npz",
#         #         kappa_bar_unsmoothed=kappa_bar_unsmoothed,
#         #         kappa_bar=kappa_bar,
#         #         gamm1_bar=gamm1_bar,
#         #         gamm2_bar=gamm2_bar,
#         #         phi_bar=phi_bar)
#         df = DefectFinder(f"{DATA_DIR}/ShearConvergenceMapData_lmax{lmax}_{seed}.npz")
#         # df.find_defects(savepath=f"{DATA_DIR}/defects_data_lmax{lmax}_{seed}.npz", cluster=True)
#         df.load_defects(f"{DATA_DIR}/defects_data_lmax{lmax}_{seed}.npz")
#         df.save_defects(savepath=f"{DATA_DIR}/defects_data_lmax{lmax}_{seed}")
#         density_sim = df.number_estimation(lmax=int(lmax))
#         print(density_sim)
#         density_finder, pos, neg = df.number()
#         print(density_finder)
#         density_sim_.append(density_sim)
#         density_finder_.append(density_finder)

#     density_sim_list_lmax.append(np.mean(density_sim_))
#     density_finder_list_lmax.append(np.mean(density_finder_))
#     density_sim_std_list_lmax.append(np.std(density_sim_))
#     density_finder_std_list_lmax.append(np.std(density_finder_))

# density_sim_list_lmax = np.array(density_sim_list_lmax)
# density_finder_list_lmax = np.array(density_finder_list_lmax)
# density_sim_std_list_lmax = np.array(density_sim_std_list_lmax)
# density_finder_std_list_lmax = np.array(density_finder_std_list_lmax)

# # save results
# np.savez(f"{DATA_DIR}/defect_density_vs_lmax.npz",
#          lmaxs_ratio=lmaxs_ratio,
#          density_sim_list_lmax=density_sim_list_lmax,
#          density_sim_std_list_lmax=density_sim_std_list_lmax,
#          density_finder_list_lmax=density_finder_list_lmax,
#          density_finder_std_list_lmax=density_finder_std_list_lmax)

# # read results
# data = np.load(f"{DATA_DIR}/defect_density_vs_lmax.npz")
# lmaxs_ratio = data['lmaxs_ratio'][:-1]
# density_sim_list_lmax = data['density_sim_list_lmax'][:-1]
# density_sim_std_list_lmax = data['density_sim_std_list_lmax'][:-1]
# density_finder_list_lmax = data['density_finder_list_lmax'][:-1]
# density_finder_std_list_lmax = data['density_finder_std_list_lmax'][:-1]

# # 2plots, upper: density vs lmax_ratio, lower: relative error vs lmax_ratio, sharex
# fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(4, 3), sharex=True, height_ratios=[2, 1])
# ax1.fill_between(lmaxs_ratio, np.array(density_sim_list_lmax) - np.array(density_sim_std_list_lmax), np.array(density_sim_list_lmax) + np.array(density_sim_std_list_lmax), alpha=0.3, label='Expected', color=color_map(0))
# ax1.errorbar(lmaxs_ratio, density_finder_list_lmax, yerr=density_finder_std_list_lmax, label='Measured', fmt='o', markersize=3, color=color_map(0))
# # plt.xscale('log')
# ax1.set_yscale('log')
# # ax1.set_xlabel(r'$\ell_{\rm max} / N_{\rm side}$')
# ax1.set_ylabel('Defect Density')
# ax1.legend()
# relative_error = (np.array(density_finder_list_lmax) - np.array(density_sim_list_lmax)) / np.array(density_sim_list_lmax)
# ax2.errorbar(lmaxs_ratio, relative_error, yerr=np.array(density_finder_std_list_lmax) / np.array(density_sim_list_lmax), fmt='o', markersize=3, color=color_map(0))
# ax2.axhline(0, color='gray', linestyle='--')
# ax2.set_xlabel(r'$\ell_{\rm max} / N_{\rm side}$')
# ax2.set_ylabel('Relative Error')
# # no padding between subplots
# plt.subplots_adjust(hspace=0)
# plt.savefig("outfiles/defect_density_vs_lmax.png", dpi=300)


# # fix nside and lmax=3*nside, vary smoothing scale
# def gaussian_kernel(size, sigma):
#     """Generates a 2D Gaussian kernel."""
#     ax = np.arange(-size // 2 + 1., size // 2 + 1.)
#     xx, yy = np.meshgrid(ax, ax)
#     kernel = np.exp(-(xx**2 + yy**2) / (2. * sigma**2))
#     return kernel / np.sum(kernel)


density_sim_list_smoothing = []
density_sim_std_list_smoothing = []
density_theory_direct_list_smoothing = []
density_finder_list_smoothing = []
density_finder_std_list_smoothing = []
smoothing_sizes = np.linspace(1, 16, 16)
pixel_size = np.sqrt(4*np.pi * (180/np.pi)**2 / (256*256) / 12) * 60 
smoothing_sizes_arcmin = smoothing_sizes * pixel_size

cosmo_theory = ccl.Cosmology(Omega_c=0.25, Omega_b=0.05, h=0.7, sigma8=0.8, n_s=0.965)
z_source = np.linspace(0., 2.0, 100)
dndz_source = np.zeros_like(z_source)
# set all sources at z=1 for simplicity
dndz_source[np.argmin(np.abs(z_source - 1.0))] = 1.0
print(dndz_source)

# generate maps
# first make 10 maps with same lmax and nside
# for seed in tqdm(range(10)):
#     cosmo_params = {
#         'Oc': 0.25,
#         'Ob': 0.05,
#         'h': 0.7,
#         'sigma8': 0.8
#     }
#     os.makedirs(DATA_DIR, exist_ok=True)
#     kappa_bar_unsmoothed, kappa_bar, gamm1_bar, gamm2_bar, phi_bar = GenMap(cosmo_params, nside=256, lmax=256*3, Plot=False, Savepath=f"outfiles/shear_field_patch_hires_smoothing_seed{seed}.png", random_seed=seed)
#     np.savez(f"{DATA_DIR}/ShearConvergenceMapData_smoothing_{seed}.npz",
#             kappa_bar_unsmoothed=kappa_bar_unsmoothed,
#             kappa_bar=kappa_bar,
#             gamm1_bar=gamm1_bar,
#             gamm2_bar=gamm2_bar,
#             phi_bar=phi_bar)
    
# then apply different smoothing to the same map and find defects


for sigma in tqdm(smoothing_sizes_arcmin):
    sigma = np.radians(sigma / 60)
    density_theory_direct = DefectDensity(
        cosmo_theory,
        (z_source, dndz_source),
        lmax=256*3,
        filter='gaussian',
        filter_angle=sigma,
        include_pixel_window=False,
        nside=256
    )
    print(density_theory_direct)
    density_theory_direct_list_smoothing.append(density_theory_direct)
    cosmo_params = {
        'Oc': 0.25,
        'Ob': 0.05,
        'h': 0.7,
        'sigma8': 0.8
    }
    density_sim_ = []
    density_finder_ = []
    for seed in range(10):
        df = DefectFinder(f"{DATA_DIR}/ShearConvergenceMapData_smoothing_{seed}.npz")
        df.kappa_bar = smoothing(df.kappa_bar_unsmoothed, sigma=sigma, lmax=256*3)
        df.gamm1_bar, df.gamm2_bar = glass.shear_from_convergence(df.kappa_bar, lmax=256*3)
        df.shear_amp = np.sqrt(df.gamm1_bar**2 + df.gamm2_bar**2)
        df.phi_bar = 0.5 * np.arctan2(df.gamm2_bar, df.gamm1_bar)
        df.find_defects(savepath=f"{DATA_DIR}/defects_data_smoothing_sigma{sigma}_{seed}.npz", cluster=True)
        df.load_defects(f"{DATA_DIR}/defects_data_smoothing_sigma{sigma}_{seed}.npz")
        df.save_defects(savepath=f"{DATA_DIR}/defects_data_smoothing_sigma{sigma}_{seed}")
        density_sim = df.number_estimation(lmax=256*3, psi_method='from_kappa_cl')
        print(density_sim)
        density_finder, pos, neg = df.number()
        print(density_finder)
        density_sim_.append(density_sim)
        density_finder_.append(density_finder)

    density_sim_list_smoothing.append(np.mean(density_sim_))
    density_finder_list_smoothing.append(np.mean(density_finder_))
    density_sim_std_list_smoothing.append(np.std(density_sim_))
    density_finder_std_list_smoothing.append(np.std(density_finder_))

np.savez(f"{DATA_DIR}/defect_density_vs_smoothing.npz",
         smoothing_sizes_arcmin=smoothing_sizes_arcmin,
         density_sim_list_smoothing=density_sim_list_smoothing,
         density_sim_std_list_smoothing=density_sim_std_list_smoothing,
         density_theory_direct_list_smoothing=density_theory_direct_list_smoothing,
         density_finder_list_smoothing=density_finder_list_smoothing,
         density_finder_std_list_smoothing=density_finder_std_list_smoothing)

data = np.load(f"{DATA_DIR}/defect_density_vs_smoothing.npz")
smoothing_sizes_arcmin = data['smoothing_sizes_arcmin']
density_sim_list_smoothing = data['density_sim_list_smoothing']
density_sim_std_list_smoothing = data['density_sim_std_list_smoothing']
density_theory_direct_list_smoothing = data['density_theory_direct_list_smoothing']
density_finder_list_smoothing = data['density_finder_list_smoothing']
density_finder_std_list_smoothing = data['density_finder_std_list_smoothing']
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(4, 3), sharex=True, height_ratios=[2, 1])
ax1.plot(smoothing_sizes, density_theory_direct_list_smoothing, label='Theory', ls='--', color=color_map(1))
ax1.fill_between(smoothing_sizes, np.array(density_sim_list_smoothing) - np.array(density_sim_std_list_smoothing), np.array(density_sim_list_smoothing) + np.array(density_sim_std_list_smoothing), alpha=0.3, label='Expected', color=color_map(0))
ax1.errorbar(smoothing_sizes, density_finder_list_smoothing, yerr=density_finder_std_list_smoothing, label='Measured', fmt='o', markersize=3, color=color_map(0))
# ax1.set_xscale('log')
ax1.set_yscale('log')
ax1.set_xlabel(r'$\theta_{\rm s} / \theta_{\rm pix}$')
ax1.set_ylabel('Defect Density')
ax1.legend()
relative_error = (np.array(density_finder_list_smoothing) - np.array(density_theory_direct_list_smoothing)) / np.array(density_theory_direct_list_smoothing)
relative_error_direct = (np.array(density_sim_list_smoothing) - np.array(density_theory_direct_list_smoothing)) / np.array(density_theory_direct_list_smoothing)
ax2.fill_between(smoothing_sizes, relative_error_direct - np.array(density_sim_std_list_smoothing) / np.array(density_theory_direct_list_smoothing), relative_error_direct + np.array(density_sim_std_list_smoothing) / np.array(density_theory_direct_list_smoothing), alpha=0.3, color=color_map(0))    
ax2.errorbar(smoothing_sizes, relative_error, yerr=np.array(density_finder_std_list_smoothing) / np.array(density_theory_direct_list_smoothing), fmt='o', markersize=3, color=color_map(0))

ax2.axhline(0, color='gray', linestyle='-.')
# ax2.set_xscale('log')
# ax2.set_yscale('log')
ax2.set_xlabel(r'$\theta_{\rm s} / \theta_{\rm pix}$')
ax2.set_ylabel('Relative Error')
plt.subplots_adjust(hspace=0)
plt.savefig("outfiles/defect_density_vs_smoothing.png", dpi=300)
