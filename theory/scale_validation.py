import sys
import os

import glass
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from theory.WLDefects import DefectDensity, DefectCF
from GenMapData import GenMap
from DefectFinder import DefectFinder
import numpy as np
import pyccl as ccl
import healpy as hp
from tqdm import tqdm
from healpy.sphtfunc import smoothing
from scipy.interpolate import interp1d
from scipy.special import jvp

# Oms = np.array([0.1, 0.2, 0.3, 0.4, 0.5])
# sigma8s = np.array([0.4, 0.6, 0.8, 1.0, 1.2])

# density_sim_list_Om = []
# density_sim_std_list_Om = []
# density_finder_list_Om = []
# density_finder_std_list_Om = []

DATA_DIR = "/home/jbai/cephfs_copy/cephfs/jbai/DefectValidation"
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

# # generate maps
# # first make 10 maps with same lmax and nside
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


# for sigma in tqdm(smoothing_sizes_arcmin):
#     sigma = np.radians(sigma / 60)
#     density_theory_direct = DefectDensity(
#         cosmo_theory,
#         (z_source, dndz_source),
#         lmax=256*3,
#         filter='gaussian',
#         filter_angle=sigma,
#         include_pixel_window=False,
#         nside=256
#     )
#     print(density_theory_direct)
#     density_theory_direct_list_smoothing.append(density_theory_direct)
#     cosmo_params = {
#         'Oc': 0.25,
#         'Ob': 0.05,
#         'h': 0.7,
#         'sigma8': 0.8
#     }
#     density_sim_ = []
#     density_finder_ = []
#     for seed in range(10):
#         df = DefectFinder(f"{DATA_DIR}/ShearConvergenceMapData_smoothing_{seed}.npz")
#         df.kappa_bar = smoothing(df.kappa_bar_unsmoothed, sigma=sigma, lmax=256*3)
#         df.gamm1_bar, df.gamm2_bar = glass.shear_from_convergence(df.kappa_bar, lmax=256*3)
#         df.shear_amp = np.sqrt(df.gamm1_bar**2 + df.gamm2_bar**2)
#         df.phi_bar = 0.5 * np.arctan2(df.gamm2_bar, df.gamm1_bar)
#         df.find_defects(savepath=f"{DATA_DIR}/defects_data_smoothing_sigma{sigma}_{seed}.npz", cluster=True)
#         df.load_defects(f"{DATA_DIR}/defects_data_smoothing_sigma{sigma}_{seed}.npz")
#         df.save_defects(savepath=f"{DATA_DIR}/defects_data_smoothing_sigma{sigma}_{seed}")
#         density_sim = df.number_estimation(lmax=256*3, psi_method='from_kappa_cl')
#         print(density_sim)
#         density_finder, pos, neg = df.number()
#         print(density_finder)
#         print(pos, neg)
#         density_sim_.append(density_sim)
#         density_finder_.append(density_finder)

#     density_sim_list_smoothing.append(np.mean(density_sim_))
#     density_finder_list_smoothing.append(np.mean(density_finder_))
#     density_sim_std_list_smoothing.append(np.std(density_sim_))
#     density_finder_std_list_smoothing.append(np.std(density_finder_))

# np.savez(f"{DATA_DIR}/defect_density_vs_smoothing.npz",
#          smoothing_sizes_arcmin=smoothing_sizes_arcmin,
#          density_sim_list_smoothing=density_sim_list_smoothing,
#          density_sim_std_list_smoothing=density_sim_std_list_smoothing,
#          density_theory_direct_list_smoothing=density_theory_direct_list_smoothing,
#          density_finder_list_smoothing=density_finder_list_smoothing,
#          density_finder_std_list_smoothing=density_finder_std_list_smoothing)

data = np.load(f"{DATA_DIR}/defect_density_vs_smoothing.npz")
smoothing_sizes_arcmin = data['smoothing_sizes_arcmin']
density_sim_list_smoothing = data['density_sim_list_smoothing']
density_sim_std_list_smoothing = data['density_sim_std_list_smoothing']
density_theory_direct_list_smoothing = data['density_theory_direct_list_smoothing']
density_finder_list_smoothing = data['density_finder_list_smoothing']
density_finder_std_list_smoothing = data['density_finder_std_list_smoothing']
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(4.6, 3.2), sharex=True, height_ratios=[2, 1])
ax1.plot(smoothing_sizes, density_theory_direct_list_smoothing, label='Theory', ls='--', color='k')
ax1.fill_between(smoothing_sizes, np.array(density_sim_list_smoothing) - np.array(density_sim_std_list_smoothing)/np.sqrt(10), np.array(density_sim_list_smoothing) + np.array(density_sim_std_list_smoothing)/np.sqrt(10), alpha=0.3, label='Expected', color=color_map(0))
ax1.errorbar(smoothing_sizes, density_finder_list_smoothing, yerr=density_finder_std_list_smoothing/np.sqrt(10), label='Measured', fmt='o', markersize=3, color=color_map(3))
# ax1.set_xscale('log')
ax1.set_yscale('log')
ax1.set_xlabel(r'$\theta_{\rm G} / \theta_{\rm pix}$')
ax1.set_ylabel('Defect Density')
ax1.legend()
relative_error = (np.array(density_finder_list_smoothing) - np.array(density_theory_direct_list_smoothing)) / np.array(density_theory_direct_list_smoothing)
relative_error_direct = (np.array(density_sim_list_smoothing) - np.array(density_theory_direct_list_smoothing)) / np.array(density_theory_direct_list_smoothing)
ax2.fill_between(smoothing_sizes, relative_error_direct - np.array(density_sim_std_list_smoothing)/np.sqrt(10) / np.array(density_theory_direct_list_smoothing), relative_error_direct + np.array(density_sim_std_list_smoothing)/np.sqrt(10) / np.array(density_theory_direct_list_smoothing), alpha=0.3, color=color_map(0))    
ax2.errorbar(smoothing_sizes, relative_error, yerr=np.array(density_finder_std_list_smoothing)/np.sqrt(10) / np.array(density_theory_direct_list_smoothing), fmt='o', markersize=3, color=color_map(3))
ax2.set_ylim(-0.4, 0.4)
ax2.axhline(0, color='gray', linestyle='-.')
# ax2.set_xscale('log')
# ax2.set_yscale('log')
ax2.set_xlabel(r'$\theta_{\rm G} / \theta_{\rm pix}$')
ax2.set_ylabel('Relative Error')
plt.subplots_adjust(hspace=0)
plt.savefig("outfiles/defect_density_vs_smoothing.png", dpi=300)


# ===========================================================================
# Charge correlation validation vs smoothing (theory / simulated Cl / finder)


def cl_psi_from_kappa(cl_kappa):
    ells = np.arange(len(cl_kappa))
    cl_psi = np.zeros_like(cl_kappa)
    valid = ells >= 2
    denom = (ells[valid] ** 2) * ((ells[valid] + 1) ** 2)
    cl_psi[valid] = 4.0 * cl_kappa[valid] / denom
    return cl_psi


def corr_theta_derivative_from_cl_bessel(cl_psi, theta, order):
    n = int(order)
    ells = np.arange(len(cl_psi), dtype=np.float64)
    pref = (ells / (2.0 * np.pi)) * cl_psi
    x = np.outer(ells, theta)
    kernel = jvp(0, x, n=n)
    return np.sum((pref * (ells ** n))[:, None] * kernel, axis=0)


def compute_c0_moments(cl_psi):
    ells = np.arange(len(cl_psi), dtype=np.float64)
    pref = (ells / (2.0 * np.pi)) * cl_psi
    c4_0 = np.sum(pref * (3.0 / 8.0) * ells**4)
    c6_0 = np.sum(pref * (-5.0 / 16.0) * ells**6)
    return c4_0, c6_0


def compute_c0_from_moment(cl_field):
    ells = np.arange(len(cl_field), dtype=np.float64)
    pref = (ells / (2.0 * np.pi)) * cl_field
    return np.sum(pref)


def compute_h_u_and_derivative_analytic(theta, c1, c2, c3, c4, c5, c6, c4_0):
    r = theta
    l0 = c4_0 / 3.0
    l = c3 / r - 2.0 * c2 / (r**2) + 2.0 * c1 / (r**3)
    m = c4
    n = 3.0 * c2 / (r**2) - 3.0 * c1 / (r**3)
    p = -c5
    q = -c4 / r + 3.0 * c3 / (r**2) - 6.0 * c2 / (r**3) + 6.0 * c1 / (r**4)
    rr = -3.0 * c3 / (r**2) + 9.0 * c2 / (r**3) - 9.0 * c1 / (r**4)

    scomb = m + n - 2.0 * l
    a = l0**2 - (scomb**2) / 16.0
    b = l0**2 - l**2
    detk = a * b

    l1 = c4 / r - 3.0 * c3 / (r**2) + 6.0 * c2 / (r**3) - 6.0 * c1 / (r**4)
    l2 = c5 / r - 4.0 * c4 / (r**2) + 12.0 * c3 / (r**3) - 24.0 * c2 / (r**4) + 24.0 * c1 / (r**5)
    m1 = c5
    m2 = c6
    n1 = 3.0 * c3 / (r**2) - 9.0 * c2 / (r**3) + 9.0 * c1 / (r**4)
    n2 = 3.0 * c4 / (r**2) - 15.0 * c3 / (r**3) + 36.0 * c2 / (r**4) - 36.0 * c1 / (r**5)
    p1 = -c6
    q1 = -c5 / r + 4.0 * c4 / (r**2) - 12.0 * c3 / (r**3) + 24.0 * c2 / (r**4) - 24.0 * c1 / (r**5)
    q2 = -c6 / r + 5.0 * c5 / (r**2) - 20.0 * c4 / (r**3) + 60.0 * c3 / (r**4) - 120.0 * c2 / (r**5) + 120.0 * c1 / (r**6)
    rr1 = -3.0 * c4 / (r**2) + 15.0 * c3 / (r**3) - 36.0 * c2 / (r**4) + 36.0 * c1 / (r**5)
    rr2 = -3.0 * c5 / (r**2) + 21.0 * c4 / (r**3) - 81.0 * c3 / (r**4) + 180.0 * c2 / (r**5) - 180.0 * c1 / (r**6)

    s1 = m1 + n1 - 2.0 * l1
    s2 = m2 + n2 - 2.0 * l2
    a1 = -0.125 * scomb * s1
    a2 = -0.125 * (s1**2 + scomb * s2)
    b1 = -2.0 * l * l1
    b2 = -2.0 * (l1**2 + l * l2)
    detk1 = a1 * b + a * b1
    detk2 = a2 * b + 2.0 * a1 * b1 + a * b2

    s = np.sqrt(detk)
    invs = 1.0 / s
    s1 = detk1 / (2.0 * s)
    s2 = detk2 / (2.0 * s) - (detk1**2) / (4.0 * s**3)
    invs1 = -s1 * invs**2
    invs2 = -s2 * invs**2 + 2.0 * (s1**2) * invs**3

    t = q - rr
    t1 = q1 - rr1
    t2 = q2 - rr2
    g = t**2
    g1 = 2.0 * t * t1
    g2 = 2.0 * (t1**2 + t * t2)

    aa1 = 0.25 * (g * invs + r * g1 * invs + r * g * invs1)
    aa2 = 0.25 * (2.0 * g1 * invs + 2.0 * g * invs1 + r * g2 * invs + 2.0 * r * g1 * invs1 + r * g * invs2)
    mm = q * (p + rr - 2.0 * q)
    mm1 = q1 * (p + rr - 2.0 * q) + q * (p1 + rr1 - 2.0 * q1)
    bb = 0.25 * mm * invs
    bb1 = 0.25 * (mm1 * invs + mm * invs1)
    h_u = aa1 + bb
    dh_u = aa2 + bb1
    return h_u, dh_u


def charge_correlation_from_cl_kappa(cl_kappa, theta_rad):
    theta_rad = np.asarray(theta_rad, dtype=np.float64)
    cl_psi = cl_psi_from_kappa(cl_kappa)

    c0_raw = compute_c0_from_moment(cl_psi)
    if not np.isfinite(c0_raw) or np.abs(c0_raw) < np.finfo(np.float64).tiny:
        return np.full_like(theta_rad, np.nan)
    cl_psi /= c0_raw

    c1 = corr_theta_derivative_from_cl_bessel(cl_psi, theta_rad, 1)
    c2 = corr_theta_derivative_from_cl_bessel(cl_psi, theta_rad, 2)
    c3 = corr_theta_derivative_from_cl_bessel(cl_psi, theta_rad, 3)
    c4 = corr_theta_derivative_from_cl_bessel(cl_psi, theta_rad, 4)
    c5 = corr_theta_derivative_from_cl_bessel(cl_psi, theta_rad, 5)
    c6 = corr_theta_derivative_from_cl_bessel(cl_psi, theta_rad, 6)

    c4_0, c6_0 = compute_c0_moments(cl_psi)
    _, dh_u = compute_h_u_and_derivative_analytic(theta_rad, c1, c2, c3, c4, c5, c6, c4_0)
    d_u = np.abs(3.0 * c6_0 / (10.0 * np.pi * c4_0)) + 1/np.pi
    if not np.isfinite(d_u) or np.abs(d_u) < np.finfo(np.float64).tiny:
        return np.full_like(theta_rad, np.nan)

    return dh_u / (4.0 * np.pi**2 * d_u**2 * theta_rad)


theta_cf_arcmin = np.logspace(1, 3, 10)
theta_cf_rad = theta_cf_arcmin * (np.pi / 180.0) / 60.0

charge_corr_theory_list_smoothing = []
charge_corr_sim_list_smoothing = []
charge_corr_sim_std_list_smoothing = []
charge_corr_finder_list_smoothing = []
charge_corr_finder_std_list_smoothing = []

for sigma in tqdm(smoothing_sizes_arcmin, desc='Charge correlation vs smoothing'):
    sigma_rad = np.radians(sigma / 60.0)

    gq_theory = DefectCF(cosmo_theory, (z_source, dndz_source), lmax=256*3,
                         theta=theta_cf_rad, filter='gaussian', filter_angle=sigma_rad)
    charge_corr_theory_list_smoothing.append(gq_theory)

    gq_sim_ = []
    gq_finder_ = []
    for seed in range(10):
        df = DefectFinder(f"{DATA_DIR}/ShearConvergenceMapData_smoothing_{seed}.npz")
        df.kappa_bar = smoothing(df.kappa_bar_unsmoothed, sigma=sigma_rad, lmax=256*3)
        df.gamm1_bar, df.gamm2_bar = glass.shear_from_convergence(df.kappa_bar, lmax=256*3)
        df.shear_amp = np.sqrt(df.gamm1_bar**2 + df.gamm2_bar**2)
        df.phi_bar = 0.5 * np.arctan2(df.gamm2_bar, df.gamm1_bar)

        cl_kappa = hp.anafast(df.kappa_bar, lmax=256*3)
        gq_sim_.append(charge_correlation_from_cl_kappa(cl_kappa, theta_cf_rad))

        defect_npz = f"{DATA_DIR}/defects_data_charge_sigma{sigma:.6f}_{seed}.npz"
        defect_prefix = f"{DATA_DIR}/defects_data_charge_sigma{sigma:.6f}_{seed}"
        df.find_defects(savepath=defect_npz, cluster=True)
        df.load_defects(defect_npz)
        df.save_defects(savepath=defect_prefix)

        pos_path = f"{defect_prefix}_charge_+0.5.fits"
        neg_path = f"{defect_prefix}_charge_-0.5.fits"
        if os.path.exists(pos_path) and os.path.exists(neg_path):
            theta_data, gq_data, _, _ = df.charge_correlation(
                {0.5: pos_path, -0.5: neg_path},
                min_sep=theta_cf_arcmin.min(), max_sep=theta_cf_arcmin.max(), n_bins=len(theta_cf_arcmin),
                random_multiplier=100, random_seed=seed
            )
            gq_finder_.append(interp1d(theta_data, gq_data, bounds_error=False, fill_value=np.nan)(theta_cf_arcmin))

    gq_sim_ = np.array(gq_sim_)
    gq_finder_ = np.array(gq_finder_) if len(gq_finder_) > 0 else np.full((1, len(theta_cf_arcmin)), np.nan)

    charge_corr_sim_list_smoothing.append(np.nanmean(gq_sim_, axis=0))
    charge_corr_sim_std_list_smoothing.append(np.nanstd(gq_sim_, axis=0))
    charge_corr_finder_list_smoothing.append(np.nanmean(gq_finder_, axis=0))
    charge_corr_finder_std_list_smoothing.append(np.nanstd(gq_finder_, axis=0))

charge_corr_theory_list_smoothing = np.array(charge_corr_theory_list_smoothing)
charge_corr_sim_list_smoothing = np.array(charge_corr_sim_list_smoothing)
charge_corr_sim_std_list_smoothing = np.array(charge_corr_sim_std_list_smoothing)
charge_corr_finder_list_smoothing = np.array(charge_corr_finder_list_smoothing)
charge_corr_finder_std_list_smoothing = np.array(charge_corr_finder_std_list_smoothing)

np.savez(
    f"{DATA_DIR}/charge_correlation_vs_smoothing.npz",
    smoothing_sizes_arcmin=smoothing_sizes_arcmin,
    theta_cf_arcmin=theta_cf_arcmin,
    charge_corr_theory_list_smoothing=charge_corr_theory_list_smoothing,
    charge_corr_sim_list_smoothing=charge_corr_sim_list_smoothing,
    charge_corr_sim_std_list_smoothing=charge_corr_sim_std_list_smoothing,
    charge_corr_finder_list_smoothing=charge_corr_finder_list_smoothing,
    charge_corr_finder_std_list_smoothing=charge_corr_finder_std_list_smoothing,
)

# read results
data = np.load(f"{DATA_DIR}/charge_correlation_vs_smoothing.npz")
smoothing_sizes_arcmin = data['smoothing_sizes_arcmin']
theta_cf_arcmin = data['theta_cf_arcmin']
charge_corr_theory_list_smoothing = data['charge_corr_theory_list_smoothing']
charge_corr_sim_list_smoothing = data['charge_corr_sim_list_smoothing']
charge_corr_sim_std_list_smoothing = data['charge_corr_sim_std_list_smoothing']
charge_corr_finder_list_smoothing = data['charge_corr_finder_list_smoothing']
charge_corr_finder_std_list_smoothing = data['charge_corr_finder_std_list_smoothing']

fig, axes = plt.subplots(
    2, 2, figsize=(4, 4), sharex=True, sharey=True,
    gridspec_kw={'hspace': 0.0, 'wspace': 0.0}
)

target_smoothing = np.array([1.0, 4.0, 8.0, 12.0])
smoothing_sizes_loaded = smoothing_sizes_arcmin / pixel_size

for i_plot, sval in enumerate(target_smoothing):
    ax = axes.ravel()[i_plot]
    i_sigma = int(np.argmin(np.abs(smoothing_sizes_loaded - sval)))
    matched_smoothing = smoothing_sizes_loaded[i_sigma]

    # If the requested smoothing is missing in the saved data, mark the panel.
    if np.abs(matched_smoothing - sval) > 0.25:
        ax.text(0.5, 0.5, 'Not available', transform=ax.transAxes, ha='center', va='center')
        ax.set_title(rf'$\theta_{{\rm G}}={sval:.0f}\,\theta_{{\rm pix}}$')
        ax.set_xscale('log')
        ax.set_xlim(1e1, 1e3)
        continue

    theory = charge_corr_theory_list_smoothing[i_sigma]
    sim = charge_corr_sim_list_smoothing[i_sigma]
    sim_std = charge_corr_sim_std_list_smoothing[i_sigma]
    finder = charge_corr_finder_list_smoothing[i_sigma]
    finder_std = charge_corr_finder_std_list_smoothing[i_sigma]

    ax.plot(theta_cf_arcmin, theory, ls='--', color='k', lw=1.2, label='Theory')
    ax.plot(theta_cf_arcmin, sim, color=color_map(0), lw=1.2, label='Expected')
    ax.fill_between(
        theta_cf_arcmin,
        sim - sim_std/np.sqrt(10),
        sim + sim_std/np.sqrt(10),
        color=color_map(0), alpha=0.22
    )
    ax.errorbar(
        theta_cf_arcmin,
        finder,
        yerr=finder_std/np.sqrt(10),
        color=color_map(3), lw=1.0, ms=2.8, fmt='o', capsize=1.8, label='Measured'
    )
    ax.axhline(0, color='gray', linestyle='-.', lw=0.8)
    ax.set_xscale('log')
    ax.set_xlim(1e1, 1e3)
    ax.set_xticks([10, 100])
    # ax.set_title(rf'$\theta_{{\rm G}}={sval:.0f}\,\theta_{{\rm pix}}$')
    ax.text(0.9, 0.1, rf'$\theta_{{\rm G}}={sval:.0f}\,\theta_{{\rm pix}}$', transform=ax.transAxes, ha='right', va='bottom')

axes[0, 0].set_ylabel(r'$g_Q(\theta)$')
axes[1, 0].set_ylabel(r'$g_Q(\theta)$')
axes[1, 0].set_xlabel(r'$\theta$ [arcmin]')
axes[1, 1].set_xlabel(r'$\theta$ [arcmin]')
handles, labels = ax.get_legend_handles_labels()
fig.legend(handles, labels, loc='upper center', ncol=3, bbox_to_anchor=(0.5, 0.98))
plt.savefig("outfiles/charge_correlation_vs_smoothing.png", dpi=300)

