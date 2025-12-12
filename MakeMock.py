from pyDOE import lhs
import numpy as np
from DefectFinder import DefectFinder
from GenMapData import GenMap
from tqdm import tqdm
import os
from healpy.sphtfunc import smoothing
import gc
import glass
DATA_DIR = '/cephfs/jbai/DefectMockGaussian'
params = {
    'Om': 0.3,
    'S8': 0.8,
}
Ob = 0.049
h = 0.67
w = -1.0

res = 512

for i in tqdm(range(500), desc="Generating mock data"):
    # dump memories
    gc.collect()
    param_dict = {
        'Oc': params['Om'] - Ob,
        'Ob': Ob,
        'h': h,
        'S8': params['S8'],
        'w': w
    }
    # kappa_bar_unsmoothed, kappa_bar, gamm1_bar, gamm2_bar, phi_bar = GenMap(param_dict, nside=res, lmax=res*3, random_type='gaussian', random_seed=i)
    # # create path if not exists
    # os.makedirs(f"{DATA_DIR}/Maps", exist_ok=True)
    # np.savez(f"{DATA_DIR}/Maps/realisation_{i}.npz",
    #          kappa_bar_unsmoothed=kappa_bar_unsmoothed,
    #          kappa_bar=kappa_bar,
    #          gamm1_bar=gamm1_bar,
    #          gamm2_bar=gamm2_bar,
    #          phi_bar=phi_bar)
    

    # Find defects and saves
    os.makedirs(f"{DATA_DIR}/Defects", exist_ok=True)
    df = DefectFinder(f"{DATA_DIR}/Maps/realisation_{i}.npz")
    sigmas = [np.radians(15/60), np.radians(30/60), np.radians(90/60)]
    for sigma in sigmas:
        s = int(np.degrees(sigma)*60)
        df.kappa_bar = smoothing(df.kappa_bar_unsmoothed, sigma=sigma, lmax=res*3)
        # df.gamm1_bar, df.gamm2_bar = glass.shear_from_convergence(df.kappa_bar, lmax=res*3)
        # df.phi_bar = 0.5 * np.arctan2(df.gamm2_bar, df.gamm1_bar)
        # df.find_defects(savepath=f"{DATA_DIR}/Defects/realisation_{i}_s{s}_defects.npz", cluster=True)
        df.load_defects(defect_datapath=f"{DATA_DIR}/Defects/realisation_{i}_s{s}_defects.npz")
        df.save_defects(savepath=f"{DATA_DIR}/Defects/realisation_{i}_s{s}_defects")
        # Save convergence PDF
        if s == 15:
            bin_edges = np.array([-0.01, -0.005, 0.0, 0.005, 0.01])
        elif s == 30:
            bin_edges = np.array([-0.008, -0.004, 0.0, 0.004, 0.008])
        elif s == 90:
            bin_edges = np.array([-0.005, -0.0025, 0.0, 0.0025, 0.005])
        hist, bin_edges = np.histogram(df.kappa_bar, bins=bin_edges, density=True)
        os.makedirs(f"{DATA_DIR}/PDFs", exist_ok=True)
        np.savez(f"{DATA_DIR}/PDFs/realisation_{i}_s{s}_kappa_pdf.npz",
                 hist=hist,
                 bin_edges=bin_edges)

        # save defect PDFs
        if s == 15:
            bins = np.array([-0.01, -0.005, 0.005, 0.01])
        elif s == 30:
            bins = np.array([-0.008, -0.004, 0.004, 0.008])
        elif s == 90:
            bins = np.array([-0.005, -0.0025, 0.0025, 0.005])
        kappa_pos = df.convergence_dist(charge=0.5)
        kappa_pos_cum = np.array([np.sum(kappa_pos < b) for b in bins]) / 4 / np.pi

        os.makedirs(f"{DATA_DIR}/DefectPDFs", exist_ok=True)
        np.savez(f"{DATA_DIR}/DefectPDFs/realisation_{i}_s{s}_defect_kappa_pos_pdf.npz",
                hist=kappa_pos_cum,
                bin_edges=bins)

        kappa_neg = df.convergence_dist(charge=-0.5)
        kappa_neg_cum = np.array([np.sum(kappa_neg < b) for b in bins]) / 4 / np.pi

        os.makedirs(f"{DATA_DIR}/DefectPDFs", exist_ok=True)
        np.savez(f"{DATA_DIR}/DefectPDFs/realisation_{i}_s{s}_defect_kappa_neg_pdf.npz",
                hist=kappa_neg_cum,
                bin_edges=bins)
        
        if s == 15:
            theta, xi_kk, varxi_kk, covmat_kk = df.convergence_correlation(min_sep=15., max_sep=600, n_bins=10, smoothed=True)
        elif s == 30:
            theta, xi_kk, varxi_kk, covmat_kk = df.convergence_correlation(min_sep=30., max_sep=600, n_bins=10, smoothed=True)
        elif s == 90:
            theta, xi_kk, varxi_kk, covmat_kk = df.convergence_correlation(min_sep=90., max_sep=600, n_bins=10, smoothed=True)
        os.makedirs(f"{DATA_DIR}/KKs", exist_ok=True)
        np.savez(f"{DATA_DIR}/KKs/realisation_{i}_s{s}_kappa_kappa_corr.npz",
                 theta=theta,
                 xi_kk=xi_kk,
                 varxi_kk=varxi_kk,
                 covmat_kk=covmat_kk)
