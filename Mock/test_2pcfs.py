import numpy as np  
import os
import sys
from tqdm import tqdm
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from DefectFinder import DefectFinder
realizations = 1000
DATA_DIR = '/cephfs/jbai/DefectMockGaussian'
pn_list = []
for i in tqdm(range(realizations)):
    sigmas = [np.radians(30/60)]#[np.radians(15/60), np.radians(30/60), np.radians(90/60)]
    for sigma in sigmas:
        s = int(np.degrees(sigma)*60)
        df = DefectFinder(f"{DATA_DIR}/Maps/realisation_{i}.npz")
        df.load_defects(f"{DATA_DIR}/Defects/realisation_{i}_s{s}_defects.npz")
        
        # if s == 15:
        #     savepath = f"{DATA_DIR}/Defects/realisation_{i}_s{s}_defects"
        #     theta, xi_pp, varxi_pp, covmat_pp = df.correlation_function(charge1_path=f"{savepath}_charge_+0.5.fits",
        #                                                                 charge2_path=f"{savepath}_charge_+0.5.fits",
        #                                                                 min_sep=15., max_sep=600, n_bins=30)
        # elif s == 30:
        #     savepath = f"{DATA_DIR}/Defects/realisation_{i}_s{s}_defects"
        #     theta, xi_pp, varxi_pp, covmat_pp = df.correlation_function(charge1_path=f"{savepath}_charge_+0.5.fits",
        #                                                                 charge2_path=f"{savepath}_charge_+0.5.fits",
        #                                                                 min_sep=30., max_sep=600, n_bins=30)
        # elif s == 90:
        #     savepath = f"{DATA_DIR}/Defects/realisation_{i}_s{s}_defects"
        #     theta, xi_pp, varxi_pp, covmat_pp = df.correlation_function(charge1_path=f"{savepath}_charge_+0.5.fits",
        #                                                                 charge2_path=f"{savepath}_charge_+0.5.fits",
        #                                                                 min_sep=90., max_sep=600, n_bins=30)
        # os.makedirs(f"{DATA_DIR}/PosPos", exist_ok=True)
        # np.savez(f"{DATA_DIR}/PosPos/realisation_{i}_s{s}_pos_pos_corr.npz",
        #          theta=theta,
        #          xi_pp=xi_pp,
        #          varxi_pp=varxi_pp,
        #          covmat_pp=covmat_pp)
        
        # if s == 15:
        #     savepath = f"{DATA_DIR}/Defects/realisation_{i}_s{s}_defects"
        #     theta, xi_nn, varxi_nn, covmat_nn = df.correlation_function(charge1_path=f"{savepath}_charge_-0.5.fits",
        #                                                                 charge2_path=f"{savepath}_charge_-0.5.fits",
        #                                                                 min_sep=15., max_sep=600, n_bins=30)
        # elif s == 30:
        #     savepath = f"{DATA_DIR}/Defects/realisation_{i}_s{s}_defects"
        #     theta, xi_nn, varxi_nn, covmat_nn = df.correlation_function(charge1_path=f"{savepath}_charge_-0.5.fits",
        #                                                                 charge2_path=f"{savepath}_charge_-0.5.fits",
        #                                                                 min_sep=30., max_sep=600, n_bins=30)
        # elif s == 90:
        #     savepath = f"{DATA_DIR}/Defects/realisation_{i}_s{s}_defects"
        #     theta, xi_nn, varxi_nn, covmat_nn = df.correlation_function(charge1_path=f"{savepath}_charge_-0.5.fits",
        #                                                                 charge2_path=f"{savepath}_charge_-0.5.fits",
        #                                                                 min_sep=90., max_sep=600, n_bins=30)
        # os.makedirs(f"{DATA_DIR}/NegNeg", exist_ok=True)
        # np.savez(f"{DATA_DIR}/NegNeg/realisation_{i}_s{s}_neg_neg_corr.npz",
        #          theta=theta,
        #          xi_nn=xi_nn,
        #          varxi_nn=varxi_nn,
        #          covmat_nn=covmat_nn)
        
        if s == 15:
            savepath = f"{DATA_DIR}/Defects/realisation_{i}_s{s}_defects"
            theta, xi_pn, varxi_pn, covmat_pn = df.correlation_function(charge1_path=f"{savepath}_charge_+0.5.fits",
                                                                        charge2_path=f"{savepath}_charge_-0.5.fits",
                                                                        min_sep=15., max_sep=600, n_bins=30)
        elif s == 30:
            savepath = f"{DATA_DIR}/Defects/realisation_{i}_s{s}_defects"
            theta, xi_pn, varxi_pn, covmat_pn = df.correlation_function(charge1_path=f"{savepath}_charge_+0.5.fits",
                                                                        charge2_path=f"{savepath}_charge_-0.5.fits",
                                                                        min_sep=30., max_sep=600, n_bins=10)
        elif s == 90:
            savepath = f"{DATA_DIR}/Defects/realisation_{i}_s{s}_defects"
            theta, xi_pn, varxi_pn, covmat_pn = df.correlation_function(charge1_path=f"{savepath}_charge_+0.5.fits",
                                                                        charge2_path=f"{savepath}_charge_-0.5.fits",
                                                                        min_sep=90., max_sep=600, n_bins=30)
        pn_list.append(xi_pn)

pn_list = np.array(pn_list)
np.savez(f"./test_2pcfs.npz",
         theta=theta,
         pn_list=pn_list)

import matplotlib.pyplot as plt
mean_xi_pn = np.mean(pn_list, axis=0)
std_xi_pn = np.std(pn_list, axis=0)
plt.errorbar(theta, mean_xi_pn, yerr=std_xi_pn, fmt='o-')
plt.xlabel("Angular Separation (arcmin)")
plt.ylabel("Defect Cross-Correlation ξ_{+-}(θ)")
plt.title("Defect Cross-Correlation Function")
plt.grid()
plt.savefig("./test_2pcfs.png", dpi=300, bbox_inches='tight')