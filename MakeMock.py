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

for i in tqdm(range(1000), desc="Generating mock data"):
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
        df.gamm1_bar, df.gamm2_bar = glass.shear_from_convergence(df.kappa_bar, lmax=res*3)
        df.phi_bar = 0.5 * np.arctan2(df.gamm2_bar, df.gamm1_bar)
        # df.find_defects(savepath=f"{DATA_DIR}/Defects/realisation_{i}_s{s}_defects.npz", cluster=True)
        df.load_defects(defect_datapath=f"{DATA_DIR}/Defects/realisation_{i}_s{s}_defects.npz")
        df.save_defects(savepath=f"{DATA_DIR}/Defects/realisation_{i}_s{s}_defects")
        # # Save convergence PDF
        # if s == 15:
        #     bin_edges = np.linspace(-0.01, 0.01, 11)
        # elif s == 30:
        #     bin_edges = np.linspace(-0.008, 0.008, 11)
        # elif s == 90:
        #     bin_edges = np.linspace(-0.005, 0.005, 11)
        # hist, bin_edges = np.histogram(df.kappa_bar, bins=bin_edges, density=True)
        # os.makedirs(f"{DATA_DIR}/PDFs", exist_ok=True)
        # np.savez(f"{DATA_DIR}/PDFs/realisation_{i}_s{s}_kappa_pdf.npz",
        #          hist=hist,
        #          bin_edges=bin_edges)

        # # save defect PDFs
        # if s == 15:
        #     bins = np.linspace(-0.01, 0.01, 10)
        # elif s == 30:
        #     bins = np.linspace(-0.008, 0.008, 10)
        # elif s == 90:
        #     bins = np.linspace(-0.005, 0.005, 10)
        # kappa_pos = df.convergence_dist(charge=0.5)
        # kappa_pos_cum = np.array([np.sum(kappa_pos < b) for b in bins]) / 4 / np.pi

        # os.makedirs(f"{DATA_DIR}/DefectPDFs", exist_ok=True)
        # np.savez(f"{DATA_DIR}/DefectPDFs/realisation_{i}_s{s}_defect_kappa_pos_pdf.npz",
        #         hist=kappa_pos_cum,
        #         bin_edges=bins)

        # kappa_neg = df.convergence_dist(charge=-0.5)
        # kappa_neg_cum = np.array([np.sum(kappa_neg < b) for b in bins]) / 4 / np.pi

        # os.makedirs(f"{DATA_DIR}/DefectPDFs", exist_ok=True)
        # np.savez(f"{DATA_DIR}/DefectPDFs/realisation_{i}_s{s}_defect_kappa_neg_pdf.npz",
        #         hist=kappa_neg_cum,
        #         bin_edges=bins)
        
        # if s == 15:
        #     theta, xi_kk, varxi_kk, covmat_kk = df.convergence_correlation(min_sep=15., max_sep=600, n_bins=30, smoothed=True)
        # elif s == 30:
        #     theta, xi_kk, varxi_kk, covmat_kk = df.convergence_correlation(min_sep=30., max_sep=600, n_bins=30, smoothed=True)
        # elif s == 90:
        #     theta, xi_kk, varxi_kk, covmat_kk = df.convergence_correlation(min_sep=90., max_sep=600, n_bins=30, smoothed=True)
        # os.makedirs(f"{DATA_DIR}/KKs", exist_ok=True)
        # np.savez(f"{DATA_DIR}/KKs/realisation_{i}_s{s}_kappa_kappa_corr.npz",
        #          theta=theta,
        #          xi_kk=xi_kk,
        #          varxi_kk=varxi_kk,
        #          covmat_kk=covmat_kk)

        if s == 15:
            savepath = f"{DATA_DIR}/Defects/realisation_{i}_s{s}_defects"
            theta, xi_pp, varxi_pp, covmat_pp = df.correlation_function(charge1_path=f"{savepath}_charge_+0.5.fits",
                                                                        charge2_path=f"{savepath}_charge_+0.5.fits",
                                                                        min_sep=15., max_sep=100, n_bins=10)
        elif s == 30:
            savepath = f"{DATA_DIR}/Defects/realisation_{i}_s{s}_defects"
            theta, xi_pp, varxi_pp, covmat_pp = df.correlation_function(charge1_path=f"{savepath}_charge_+0.5.fits",
                                                                        charge2_path=f"{savepath}_charge_+0.5.fits",
                                                                        min_sep=30., max_sep=200, n_bins=10)
        elif s == 90:
            savepath = f"{DATA_DIR}/Defects/realisation_{i}_s{s}_defects"
            theta, xi_pp, varxi_pp, covmat_pp = df.correlation_function(charge1_path=f"{savepath}_charge_+0.5.fits",
                                                                        charge2_path=f"{savepath}_charge_+0.5.fits",
                                                                        min_sep=90., max_sep=600, n_bins=10)
        os.makedirs(f"{DATA_DIR}/PosPos", exist_ok=True)
        np.savez(f"{DATA_DIR}/PosPos/realisation_{i}_s{s}_pos_pos_corr.npz",
                 theta=theta,
                 xi_pp=xi_pp,
                 varxi_pp=varxi_pp,
                 covmat_pp=covmat_pp)
        
        if s == 15:
            savepath = f"{DATA_DIR}/Defects/realisation_{i}_s{s}_defects"
            theta, xi_nn, varxi_nn, covmat_nn = df.correlation_function(charge1_path=f"{savepath}_charge_-0.5.fits",
                                                                        charge2_path=f"{savepath}_charge_-0.5.fits",
                                                                        min_sep=15., max_sep=100, n_bins=10)
        elif s == 30:
            savepath = f"{DATA_DIR}/Defects/realisation_{i}_s{s}_defects"
            theta, xi_nn, varxi_nn, covmat_nn = df.correlation_function(charge1_path=f"{savepath}_charge_-0.5.fits",
                                                                        charge2_path=f"{savepath}_charge_-0.5.fits",
                                                                        min_sep=30., max_sep=200, n_bins=10)
        elif s == 90:
            savepath = f"{DATA_DIR}/Defects/realisation_{i}_s{s}_defects"
            theta, xi_nn, varxi_nn, covmat_nn = df.correlation_function(charge1_path=f"{savepath}_charge_-0.5.fits",
                                                                        charge2_path=f"{savepath}_charge_-0.5.fits",
                                                                        min_sep=90., max_sep=600, n_bins=10)
        os.makedirs(f"{DATA_DIR}/NegNeg", exist_ok=True)
        np.savez(f"{DATA_DIR}/NegNeg/realisation_{i}_s{s}_neg_neg_corr.npz",
                 theta=theta,
                 xi_nn=xi_nn,
                 varxi_nn=varxi_nn,
                 covmat_nn=covmat_nn)
        
        if s == 15:
            savepath = f"{DATA_DIR}/Defects/realisation_{i}_s{s}_defects"
            theta, xi_pn, varxi_pn, covmat_pn = df.correlation_function(charge1_path=f"{savepath}_charge_+0.5.fits",
                                                                        charge2_path=f"{savepath}_charge_-0.5.fits",
                                                                        min_sep=15., max_sep=100, n_bins=10)
        elif s == 30:
            savepath = f"{DATA_DIR}/Defects/realisation_{i}_s{s}_defects"
            theta, xi_pn, varxi_pn, covmat_pn = df.correlation_function(charge1_path=f"{savepath}_charge_+0.5.fits",
                                                                        charge2_path=f"{savepath}_charge_-0.5.fits",
                                                                        min_sep=30., max_sep=200, n_bins=10)
        elif s == 90:
            savepath = f"{DATA_DIR}/Defects/realisation_{i}_s{s}_defects"
            theta, xi_pn, varxi_pn, covmat_pn = df.correlation_function(charge1_path=f"{savepath}_charge_+0.5.fits",
                                                                        charge2_path=f"{savepath}_charge_-0.5.fits",
                                                                        min_sep=90., max_sep=600, n_bins=10)
        os.makedirs(f"{DATA_DIR}/PosNeg", exist_ok=True)
        np.savez(f"{DATA_DIR}/PosNeg/realisation_{i}_s{s}_pos_neg_corr.npz",
                 theta=theta,
                 xi_pn=xi_pn,
                 varxi_pn=varxi_pn,
                 covmat_pn=covmat_pn)
        
        # if s == 15:
        #     theta, xi_pk, varxi_pk, covmat_pk = df.defect_convergence_correlation(charge=0.5,
        #                                                                 min_sep=15., max_sep=1000, n_bins=30)
        # elif s == 30:
        #     theta, xi_pk, varxi_pk, covmat_pk = df.defect_convergence_correlation(charge=0.5,
        #                                                                 min_sep=30., max_sep=1000, n_bins=30)
        # elif s == 90:
        #     theta, xi_pk, varxi_pk, covmat_pk = df.defect_convergence_correlation(charge=0.5,
        #                                                                 min_sep=90., max_sep=1000, n_bins=30)
        # os.makedirs(f"{DATA_DIR}/PKs", exist_ok=True)
        # np.savez(f"{DATA_DIR}/PKs/realisation_{i}_s{s}_pos_kappa_corr.npz",
        #          theta=theta,
        #          xi_pk=xi_pk,
        #          varxi_pk=varxi_pk,
        #          covmat_pk=covmat_pk)
        
        # if s == 15:
        #     theta, xi_nk, varxi_nk, covmat_nk = df.defect_convergence_correlation(charge=-0.5,
        #                                                                 min_sep=15., max_sep=1000, n_bins=30)
        # elif s == 30:
        #     theta, xi_nk, varxi_nk, covmat_nk = df.defect_convergence_correlation(charge=-0.5,
        #                                                                 min_sep=30., max_sep=1000, n_bins=30)
        # elif s == 90:
        #     theta, xi_nk, varxi_nk, covmat_nk = df.defect_convergence_correlation(charge=-0.5,
        #                                                                 min_sep=90., max_sep=1000, n_bins=30)
        # os.makedirs(f"{DATA_DIR}/NKs", exist_ok=True)
        # np.savez(f"{DATA_DIR}/NKs/realisation_{i}_s{s}_neg_kappa_corr.npz",
        #          theta=theta,
        #          xi_nk=xi_nk,
        #          varxi_nk=varxi_nk,
        #          covmat_nk=covmat_nk)

        # pos-peak, pos-minima, neg-peak, neg-minima

        # savepath = f"{DATA_DIR}/Defects/realisation_{i}_s{s}"
        # df.find_peaks(type='peaks', savepath=savepath)
        # df.find_peaks(type='minima', savepath=savepath)

        # if s == 15:
        #     min_sep, max_sep, n_bins = 15., 600., 30
        # elif s == 30:
        #     min_sep, max_sep, n_bins = 30., 600., 30
        # elif s == 90:
        #     min_sep, max_sep, n_bins = 90., 600., 30
            
        # defect_path_base = f"{DATA_DIR}/Defects/realisation_{i}_s{s}_defects"

        # # Pos-Peak
        # theta, xi, varxi, covmat = df.defect_peak_correlation(charge_path=f"{defect_path_base}_charge_+0.5.fits", 
        #                                                       peak_path=f"{savepath}_peaks.fits", 
        #                                                       min_sep=min_sep, max_sep=max_sep, n_bins=n_bins)
        # os.makedirs(f"{DATA_DIR}/PosPeak", exist_ok=True)
        # np.savez(f"{DATA_DIR}/PosPeak/realisation_{i}_s{s}_pos_peak_corr.npz", theta=theta, xi=xi, varxi=varxi, covmat=covmat)
        
        # # Pos-Minima
        # theta, xi, varxi, covmat = df.defect_peak_correlation(charge_path=f"{defect_path_base}_charge_+0.5.fits",
        #                                                       peak_path=f"{savepath}_minima.fits",
        #                                                       min_sep=min_sep, max_sep=max_sep, n_bins=n_bins)
        # os.makedirs(f"{DATA_DIR}/PosMin", exist_ok=True)
        # np.savez(f"{DATA_DIR}/PosMin/realisation_{i}_s{s}_pos_min_corr.npz", theta=theta, xi=xi, varxi=varxi, covmat=covmat)
        
        # # Neg-Peak
        # theta, xi, varxi, covmat = df.defect_peak_correlation(charge_path=f"{defect_path_base}_charge_-0.5.fits",
        #                                                       peak_path=f"{savepath}_peaks.fits",
        #                                                       min_sep=min_sep, max_sep=max_sep, n_bins=n_bins)
        # os.makedirs(f"{DATA_DIR}/NegPeak", exist_ok=True)
        # np.savez(f"{DATA_DIR}/NegPeak/realisation_{i}_s{s}_neg_peak_corr.npz", theta=theta, xi=xi, varxi=varxi, covmat=covmat)
        
        # # Neg-Minima
        # theta, xi, varxi, covmat = df.defect_peak_correlation(charge_path=f"{defect_path_base}_charge_-0.5.fits",
        #                                                       peak_path=f"{savepath}_minima.fits",
        #                                                       min_sep=min_sep, max_sep=max_sep, n_bins=n_bins)
        # os.makedirs(f"{DATA_DIR}/NegMin", exist_ok=True)
        # np.savez(f"{DATA_DIR}/NegMin/realisation_{i}_s{s}_neg_min_corr.npz", theta=theta, xi=xi, varxi=varxi, covmat=covmat)