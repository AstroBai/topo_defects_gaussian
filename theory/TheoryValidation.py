import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from theory.WLDefects import DefectDensity
from GenMapData import GenMap
from DefectFinder import DefectFinder
import numpy as np
import pyccl as ccl
from tqdm import tqdm

Oms = np.array([0.1, 0.2, 0.3, 0.4, 0.5])
sigma8s = np.array([0.4, 0.6, 0.8, 1.0, 1.2])

density_sim_list_Om = []
density_sim_std_list_Om = []
density_finder_list_Om = []
density_finder_std_list_Om = []

DATA_DIR = "/cephfs/jbai/DefectValidation"
for Om in tqdm(Oms):
    cosmo_params = {
        'Oc': Om - 0.05,
        'Ob': 0.05,
        'h': 0.7,
        'sigma8': 0.8
    }
    density_sim_ = []
    density_finder_ = []
    for seed in range(10):
        os.makedirs(DATA_DIR, exist_ok=True)
        # kappa_bar_unsmoothed, kappa_bar, gamm1_bar, gamm2_bar, phi_bar = GenMap(cosmo_params, nside=256, lmax=64, Plot=False, Savepath=f"outfiles/shear_field_patch_hires_Om{Om}.png", random_seed=seed)
        # np.savez(f"{DATA_DIR}/ShearConvergenceMapData_Om{Om}_{seed}.npz",
        #         kappa_bar_unsmoothed=kappa_bar_unsmoothed,
        #         kappa_bar=kappa_bar,
        #         gamm1_bar=gamm1_bar,
        #         gamm2_bar=gamm2_bar,
        #         phi_bar=phi_bar)
        df = DefectFinder(f"{DATA_DIR}/ShearConvergenceMapData_Om{Om}_{seed}.npz")
        # df.find_defects(savepath=f"{DATA_DIR}/defects_data_Om{Om}_{seed}.npz", cluster=True)
        df.load_defects(f"{DATA_DIR}/defects_data_Om{Om}_{seed}.npz")
        df.save_defects(savepath=f"{DATA_DIR}/defects_data_Om{Om}_{seed}")
        
        density_sim = df.number_estimation()
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

for sigma8 in tqdm(sigma8s):
    cosmo_params = {
        'Oc': 0.25,
        'Ob': 0.05,
        'h': 0.7,
        'sigma8': sigma8
    }
    density_sim_ = []
    density_finder_ = []
    for seed in range(10):
        os.makedirs(DATA_DIR, exist_ok=True)
        # kappa_bar_unsmoothed, kappa_bar, gamm1_bar, gamm2_bar, phi_bar = GenMap(cosmo_params, nside=256, lmax=64, Plot=False, Savepath=f"outfiles/shear_field_patch_hires_sigma8{sigma8}.png", random_seed=seed)
        # np.savez(f"{DATA_DIR}/ShearConvergenceMapData_sigma8{sigma8}_{seed}.npz",
        #         kappa_bar_unsmoothed=kappa_bar_unsmoothed,
        #         kappa_bar=kappa_bar,
        #         gamm1_bar=gamm1_bar,
        #         gamm2_bar=gamm2_bar,
        #         phi_bar=phi_bar)
        df = DefectFinder(f"{DATA_DIR}/ShearConvergenceMapData_sigma8{sigma8}_{seed}.npz")
        # df.find_defects(savepath=f"{DATA_DIR}/defects_data_sigma8{sigma8}_{seed}.npz", cluster=True)
        df.load_defects(f"{DATA_DIR}/defects_data_sigma8{sigma8}_{seed}.npz")
        df.save_defects(savepath=f"{DATA_DIR}/defects_data_sigma8{sigma8}_{seed}")
        
        density_sim = df.number_estimation()
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
ax1.fill_between(Oms, np.array(density_sim_list_Om) - np.array(density_sim_std_list_Om), np.array(density_sim_list_Om) + np.array(density_sim_std_list_Om), alpha=0.3, label=r'Expected ($\Omega_{\rm m}$)', color=color_map(0))
ax1.errorbar(Oms, density_finder_list_Om, yerr=density_finder_std_list_Om, label=r'Measured ($\Omega_{\rm m}$)', fmt='o', markersize=3, color=color_map(0))
ax1.set_xlabel(r'$\Omega_{\rm m}$')
ax1.set_ylabel('Defect Density')
ax1.set_xticks(Oms)

# Plot sigma8 on top axis
ax2 = ax1.twiny()
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