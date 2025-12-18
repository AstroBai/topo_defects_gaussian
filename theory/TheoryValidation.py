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

density_sim_list = []
density_sim_std_list = []
density_finder_list = []
density_finder_std_list = []

for Om in tqdm(Oms):
    density_sim_ = []
    density_finder_ = []
    for seed in range(10):
        # kappa_bar_unsmoothed, kappa_bar, gamm1_bar, gamm2_bar, phi_bar = GenMap(cosmo_params, nside=256, lmax=64, Plot=False, Savepath=f"outfiles/shear_field_patch_hires_Om{Om}.png", random_seed=seed)
        # np.savez(f"outfiles/ShearConvergenceMapData_Om{Om}_{seed}.npz",
        #         kappa_bar_unsmoothed=kappa_bar_unsmoothed,
        #         kappa_bar=kappa_bar,
        #         gamm1_bar=gamm1_bar,
        #         gamm2_bar=gamm2_bar,
        #         phi_bar=phi_bar)
        df = DefectFinder(f"outfiles/ShearConvergenceMapData_Om{Om}_{seed}.npz")
        df.find_defects(savepath=f"outfiles/defects_data_Om{Om}_{seed}.npz", cluster=True)
        df.load_defects(f"outfiles/defects_data_Om{Om}_{seed}.npz")
        df.save_defects(savepath=f"outfiles/defects_data_Om{Om}_{seed}")
        
        density_sim = df.number_estimation()
        density_finder, pos, neg = df.number()
        print(pos, neg)
        density_sim_.append(density_sim)
        density_finder_.append(density_finder)

    density_sim_list.append(np.mean(density_sim_))
    density_finder_list.append(np.mean(density_finder_))
    density_sim_std_list.append(np.std(density_sim_))
    density_finder_std_list.append(np.std(density_finder_))


import matplotlib.pyplot as plt
import matplotlib as mpl
mpl.rcParams['xtick.direction'] = 'in'
mpl.rcParams['ytick.direction'] = 'in'

plt.figure(figsize=(4, 3))
plt.fill_between(Oms, np.array(density_sim_list) - np.array(density_sim_std_list), np.array(density_sim_list) + np.array(density_sim_std_list), alpha=0.3, label='Expected')
plt.errorbar(Oms, density_finder_list, yerr=density_finder_std_list, label='Measured', fmt='o', markersize=3, color='r')
plt.xlabel(r'$\Omega_m$')
plt.ylabel('Defect Density')
plt.legend()
plt.tight_layout()
plt.savefig("outfiles/defect_density_vs_Om.png", dpi=300)