import numpy as np
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from GenMapData import GenMap
from DefectFinder import DefectFinder
from scipy.interpolate import interp1d
from tqdm import tqdm
from healpy.sphtfunc import smoothing
import glass
DATA_DIR = "/home/jbai/cephfs_copy/cephfs/jbai/DefectFisherGaussian"
fid = {'Om': 0.3,
       'sigma8': 0.8}
Ob = 0.049
h = 0.67
w = -1.0
# surrounding cosmology params, e.g. Om +/- delta_Om
delta_params = {'Om': 0.1,
                'sigma8': 0.1}

params = []
for key in fid.keys():
    p_minus = fid.copy()
    p_plus = fid.copy()
    p_minus[key] -= delta_params[key]
    p_plus[key] += delta_params[key]
    params.append(p_minus)
    params.append(p_plus)

params = np.array(params)
print(params)
res = 512
n_seeds = 100  # Number of random seeds per parameter point

for i, param in enumerate(params):
    param_dict = {
        'Oc': param['Om'] - Ob,
        'Ob': Ob,
        'h': h,
        'sigma8': param['sigma8'],
        'w': w
    }
    
    # Initialize arrays to store results from all seeds for each smoothing scale
    all_kappa_pdf_s15 = []
    all_kappa_pdf_s30 = []
    all_kappa_pdf_s90 = []
    
    all_kappa_pos_cum_s15 = []
    all_kappa_pos_cum_s30 = []
    all_kappa_pos_cum_s90 = []
    
    all_kappa_neg_cum_s15 = []
    all_kappa_neg_cum_s30 = []
    all_kappa_neg_cum_s90 = []
    
    all_xi_kk_s15 = []
    all_xi_kk_s30 = []
    all_xi_kk_s90 = []
    
    all_varxi_kk_s15 = []
    all_varxi_kk_s30 = []
    all_varxi_kk_s90 = []
    
    all_covmat_kk_s15 = []
    all_covmat_kk_s30 = []
    all_covmat_kk_s90 = []

    all_xi_pp_s15 = []
    all_xi_pp_s30 = []
    all_xi_pp_s90 = []  

    all_varxi_pp_s15 = []
    all_varxi_pp_s30 = []
    all_varxi_pp_s90 = []

    all_covmat_pp_s15 = []
    all_covmat_pp_s30 = []
    all_covmat_pp_s90 = []

    all_xi_nn_s15 = []
    all_xi_nn_s30 = []
    all_xi_nn_s90 = []

    all_varxi_nn_s15 = []
    all_varxi_nn_s30 = []
    all_varxi_nn_s90 = []

    all_covmat_nn_s15 = []
    all_covmat_nn_s30 = []
    all_covmat_nn_s90 = []

    all_xi_pn_s15 = []
    all_xi_pn_s30 = []
    all_xi_pn_s90 = []

    all_varxi_pn_s15 = []
    all_varxi_pn_s30 = []
    all_varxi_pn_s90 = []

    all_covmat_pn_s15 = []
    all_covmat_pn_s30 = []
    all_covmat_pn_s90 = []

    all_xi_pk_s15 = []
    all_xi_pk_s30 = []
    all_xi_pk_s90 = []

    all_varxi_pk_s15 = []
    all_varxi_pk_s30 = []
    all_varxi_pk_s90 = []

    all_covmat_pk_s15 = []
    all_covmat_pk_s30 = []
    all_covmat_pk_s90 = []

    all_xi_nk_s15 = []
    all_xi_nk_s30 = []
    all_xi_nk_s90 = []  

    all_varxi_nk_s15 = []
    all_varxi_nk_s30 = []
    all_varxi_nk_s90 = []

    all_covmat_nk_s15 = []
    all_covmat_nk_s30 = []
    all_covmat_nk_s90 = []

    # Initialize new lists for Peak/Minima correlations
    all_xi_pos_peak_s15, all_varxi_pos_peak_s15, all_covmat_pos_peak_s15 = [], [], []
    all_xi_pos_peak_s30, all_varxi_pos_peak_s30, all_covmat_pos_peak_s30 = [], [], []
    all_xi_pos_peak_s90, all_varxi_pos_peak_s90, all_covmat_pos_peak_s90 = [], [], []

    all_xi_pos_min_s15, all_varxi_pos_min_s15, all_covmat_pos_min_s15 = [], [], []
    all_xi_pos_min_s30, all_varxi_pos_min_s30, all_covmat_pos_min_s30 = [], [], []
    all_xi_pos_min_s90, all_varxi_pos_min_s90, all_covmat_pos_min_s90 = [], [], []

    all_xi_neg_peak_s15, all_varxi_neg_peak_s15, all_covmat_neg_peak_s15 = [], [], []
    all_xi_neg_peak_s30, all_varxi_neg_peak_s30, all_covmat_neg_peak_s30 = [], [], []
    all_xi_neg_peak_s90, all_varxi_neg_peak_s90, all_covmat_neg_peak_s90 = [], [], []

    all_xi_neg_min_s15, all_varxi_neg_min_s15, all_covmat_neg_min_s15 = [], [], []
    all_xi_neg_min_s30, all_varxi_neg_min_s30, all_covmat_neg_min_s30 = [], [], []
    all_xi_neg_min_s90, all_varxi_neg_min_s90, all_covmat_neg_min_s90 = [], [], []
    
    all_nn_combined_s15 = []
    all_nn_combined_s30 = []
    all_nn_combined_s90 = []
    
    for seed in tqdm(range(n_seeds), desc=f"Seeds for param set {i}"):
        kappa_bar_unsmoothed, kappa_bar, gamm1_bar, gamm2_bar, phi_bar = GenMap(param_dict, nside=res, lmax=res*3, random_type='gaussian', random_seed=seed * 1000 + i)
        os.makedirs(f"{DATA_DIR}/TrainMaps", exist_ok=True)
        np.savez(f"{DATA_DIR}/TrainMaps/train_map_{i}_seed_{seed}.npz",
                kappa_bar_unsmoothed=kappa_bar_unsmoothed,
                kappa_bar=kappa_bar,
                gamm1_bar=gamm1_bar,
                gamm2_bar=gamm2_bar,
                phi_bar=phi_bar)
        # Find defects
        df = DefectFinder(data_path=f"{DATA_DIR}/TrainMaps/train_map_{i}_seed_{seed}.npz")
        sigmas = [np.radians(15/60), np.radians(30/60), np.radians(90/60)]
        os.makedirs(f"{DATA_DIR}/TrainDefects", exist_ok=True)
        
        for sigma in sigmas:            
            s = int(np.degrees(sigma)*60)
            df.kappa_bar = smoothing(df.kappa_bar_unsmoothed, sigma=sigma, lmax=res*3)
            df.gamm1_bar, df.gamm2_bar = glass.shear_from_convergence(df.kappa_bar, lmax=res*3)
            df.phi_bar = 0.5 * np.arctan2(df.gamm2_bar, df.gamm1_bar)
            df.find_defects(savepath=f"{DATA_DIR}/TrainDefects/realisation_{i}_s{s}_seed_{seed}_defects.npz", cluster=True)
            df.load_defects(defect_datapath=f"{DATA_DIR}/TrainDefects/realisation_{i}_s{s}_seed_{seed}_defects.npz")
            df.save_defects(savepath=f"{DATA_DIR}/TrainDefects/realisation_{i}_s{s}_seed_{seed}_defects")

            # Save convergence PDF
            if s == 15:
                bin_edges = np.linspace(-0.01, 0.01, 11)
            elif s == 30:
                bin_edges = np.linspace(-0.008, 0.008, 11)
            elif s == 90:
                bin_edges = np.linspace(-0.005, 0.005, 11)
            hist, bin_edges = np.histogram(df.kappa_bar, bins=bin_edges, density=True)
            
            # save defect PDFs
            if s == 15:
                bins = np.linspace(-0.01, 0.01, 10)
            elif s == 30:
                bins = np.linspace(-0.008, 0.008, 10)
            elif s == 90:
                bins = np.linspace(-0.005, 0.005, 10)   
            kappa_pos = df.convergence_dist(charge=0.5)
            kappa_pos_cum = np.array([np.sum(kappa_pos < b) for b in bins]) / 4 / np.pi

            kappa_neg = df.convergence_dist(charge=-0.5)
            kappa_neg_cum = np.array([np.sum(kappa_neg < b) for b in bins]) / 4 / np.pi

            if s == 15:
                theta, xi_kk, varxi_kk, covmat_kk = df.convergence_correlation(min_sep=15., max_sep=600, n_bins=30, smoothed=True)
            elif s == 30:
                theta, xi_kk, varxi_kk, covmat_kk = df.convergence_correlation(min_sep=30., max_sep=600, n_bins=30, smoothed=True)
            elif s == 90:
                theta, xi_kk, varxi_kk, covmat_kk = df.convergence_correlation(min_sep=90., max_sep=600, n_bins=30, smoothed=True)

            if s == 15:
                savepath=f"{DATA_DIR}/TrainDefects/realisation_{i}_s{s}_seed_{seed}_defects"
                theta, xi_pp, varxi_pp, covmat_pp = df.correlation_function(charge1_path=f"{savepath}_charge_+0.5.fits",
                                                                            charge2_path=f"{savepath}_charge_+0.5.fits",
                                                                            min_sep=15., max_sep=100, n_bins=10)
            elif s == 30:
                savepath=f"{DATA_DIR}/TrainDefects/realisation_{i}_s{s}_seed_{seed}_defects"
                theta, xi_pp, varxi_pp, covmat_pp = df.correlation_function(charge1_path=f"{savepath}_charge_+0.5.fits",
                                                                            charge2_path=f"{savepath}_charge_+0.5.fits",
                                                                            min_sep=30., max_sep=200, n_bins=10)
            elif s == 90:
                savepath=f"{DATA_DIR}/TrainDefects/realisation_{i}_s{s}_seed_{seed}_defects"
                theta, xi_pp, varxi_pp, covmat_pp = df.correlation_function(charge1_path=f"{savepath}_charge_+0.5.fits",
                                                                            charge2_path=f"{savepath}_charge_+0.5.fits",
                                                                            min_sep=90., max_sep=600, n_bins=10)
            
            if s == 15:
                savepath=f"{DATA_DIR}/TrainDefects/realisation_{i}_s{s}_seed_{seed}_defects"
                theta, xi_nn, varxi_nn, covmat_nn = df.correlation_function(charge1_path=f"{savepath}_charge_-0.5.fits",
                                                                            charge2_path=f"{savepath}_charge_-0.5.fits",
                                                                            min_sep=15., max_sep=100, n_bins=10)
            elif s == 30:
                savepath=f"{DATA_DIR}/TrainDefects/realisation_{i}_s{s}_seed_{seed}_defects"
                theta, xi_nn, varxi_nn, covmat_nn = df.correlation_function(charge1_path=f"{savepath}_charge_-0.5.fits",
                                                                            charge2_path=f"{savepath}_charge_-0.5.fits",
                                                                            min_sep=30., max_sep=200, n_bins=10)
            elif s == 90:
                savepath=f"{DATA_DIR}/TrainDefects/realisation_{i}_s{s}_seed_{seed}_defects"
                theta, xi_nn, varxi_nn, covmat_nn = df.correlation_function(charge1_path=f"{savepath}_charge_-0.5.fits",
                                                                            charge2_path=f"{savepath}_charge_-0.5.fits",
                                                                            min_sep=90., max_sep=600, n_bins=10)

            
            if s == 15:
                savepath=f"{DATA_DIR}/TrainDefects/realisation_{i}_s{s}_seed_{seed}_defects"
                theta, xi_pn, varxi_pn, covmat_pn = df.correlation_function(charge1_path=f"{savepath}_charge_+0.5.fits",
                                                                            charge2_path=f"{savepath}_charge_-0.5.fits",
                                                                            min_sep=15., max_sep=100, n_bins=10)
            elif s == 30:
                savepath=f"{DATA_DIR}/TrainDefects/realisation_{i}_s{s}_seed_{seed}_defects"
                theta, xi_pn, varxi_pn, covmat_pn = df.correlation_function(charge1_path=f"{savepath}_charge_+0.5.fits",
                                                                            charge2_path=f"{savepath}_charge_-0.5.fits",
                                                                            min_sep=30., max_sep=200, n_bins=10)
            elif s == 90:
                savepath=f"{DATA_DIR}/TrainDefects/realisation_{i}_s{s}_seed_{seed}_defects"
                theta, xi_pn, varxi_pn, covmat_pn = df.correlation_function(charge1_path=f"{savepath}_charge_+0.5.fits",
                                                                            charge2_path=f"{savepath}_charge_-0.5.fits",
                                                                            min_sep=90., max_sep=600, n_bins=10)

            
            if s == 15:
                theta, xi_pk, varxi_pk, covmat_pk = df.defect_convergence_correlation(charge=0.5,
                                                                            min_sep=15., max_sep=1000, n_bins=30)
            elif s == 30:
                theta, xi_pk, varxi_pk, covmat_pk = df.defect_convergence_correlation(charge=0.5,
                                                                            min_sep=30., max_sep=1000, n_bins=30)
            elif s == 90:
                theta, xi_pk, varxi_pk, covmat_pk = df.defect_convergence_correlation(charge=0.5,
                                                                            min_sep=90., max_sep=1000, n_bins=30)
            
            if s == 15:
                theta, xi_nk, varxi_nk, covmat_nk = df.defect_convergence_correlation(charge=-0.5,
                                                                            min_sep=15., max_sep=1000, n_bins=30)
            elif s == 30:
                theta, xi_nk, varxi_nk, covmat_nk = df.defect_convergence_correlation(charge=-0.5,
                                                                            min_sep=30., max_sep=1000, n_bins=30)
            elif s == 90:
                theta, xi_nk, varxi_nk, covmat_nk = df.defect_convergence_correlation(charge=-0.5,
                                                                            min_sep=90., max_sep=1000, n_bins=30)
                

            # pos-peak, pos-minima, neg-peak, neg-minima

            savepath = f"{DATA_DIR}/TrainDefects/realisation_{i}_s{s}_seed_{seed}"
            df.find_peaks(type='peaks', savepath=savepath)
            df.find_peaks(type='minima', savepath=savepath)

            if s == 15:
                min_sep, max_sep, n_bins = 15., 600., 30
            elif s == 30:
                min_sep, max_sep, n_bins = 30., 600., 30
            elif s == 90:
                min_sep, max_sep, n_bins = 90., 600., 30
                
            defect_path_base = f"{DATA_DIR}/TrainDefects/realisation_{i}_s{s}_seed_{seed}_defects"

            # Pos-Peak
            theta, xi_pos_peak, varxi_pos_peak, covmat_pos_peak = df.defect_peak_correlation(charge_path=f"{defect_path_base}_charge_+0.5.fits", 
                                                                peak_path=f"{savepath}_peaks.fits", 
                                                                min_sep=min_sep, max_sep=max_sep, n_bins=n_bins)
            
            # Pos-Minima
            theta, xi_pos_min, varxi_pos_min, covmat_pos_min = df.defect_peak_correlation(charge_path=f"{defect_path_base}_charge_+0.5.fits",
                                                                peak_path=f"{savepath}_minima.fits",
                                                                min_sep=min_sep, max_sep=max_sep, n_bins=n_bins)
            
            # Neg-Peak
            theta, xi_neg_peak, varxi_neg_peak, covmat_neg_peak = df.defect_peak_correlation(charge_path=f"{defect_path_base}_charge_-0.5.fits",
                                                                peak_path=f"{savepath}_peaks.fits",
                                                                min_sep=min_sep, max_sep=max_sep, n_bins=n_bins)
            
            # Neg-Minima
            theta, xi_neg_min, varxi_neg_min, covmat_neg_min = df.defect_peak_correlation(charge_path=f"{defect_path_base}_charge_-0.5.fits",
                                                                peak_path=f"{savepath}_minima.fits",
                                                                min_sep=min_sep, max_sep=max_sep, n_bins=n_bins)

            # NNs calculation
            nn_pos_same1 = df.nearest_distance(charge=0.5, k=1)
            nn_neg_same1 = df.nearest_distance(charge=-0.5, k=1)
            nn_pos_opp1 = df.nearest_opposite_distance(charge=0.5, k=1)
            nn_neg_opp1 = df.nearest_opposite_distance(charge=-0.5, k=1)
            nn_pos_same2 = df.nearest_distance(charge=0.5, k=2)
            nn_neg_same2 = df.nearest_distance(charge=-0.5, k=2)
            nn_pos_opp2 = df.nearest_opposite_distance(charge=0.5, k=2)
            nn_neg_opp2 = df.nearest_opposite_distance(charge=-0.5, k=2)
            nn_pos_same4 = df.nearest_distance(charge=0.5, k=4)
            nn_neg_same4 = df.nearest_distance(charge=-0.5, k=4)
            nn_pos_opp4 = df.nearest_opposite_distance(charge=0.5, k=4)
            nn_neg_opp4 = df.nearest_opposite_distance(charge=-0.5, k=4)
            nn_pos_same8 = df.nearest_distance(charge=0.5, k=8)
            nn_neg_same8 = df.nearest_distance(charge=-0.5, k=8)
            nn_pos_opp8 = df.nearest_opposite_distance(charge=0.5, k=8)
            nn_neg_opp8 = df.nearest_opposite_distance(charge=-0.5, k=8)
            
            if s == 15:
                bin_edges1 = np.linspace(30, 100, 10)
                bin_edges2 = np.linspace(30, 200, 10)
                bin_edges4 = np.linspace(30, 300, 10)
                bin_edges8 = np.linspace(30, 400, 10)
            elif s == 30:
                bin_edges1 = np.linspace(30, 300, 10)
                bin_edges2 = np.linspace(30, 450, 10)
                bin_edges4 = np.linspace(30, 600, 10)
                bin_edges8 = np.linspace(30, 750, 10)
            elif s == 90:
                bin_edges1 = np.linspace(30, 750, 10)
                bin_edges2 = np.linspace(30, 1125, 10)
                bin_edges4 = np.linspace(30, 1500, 10)
                bin_edges8 = np.linspace(30, 1875, 10)
            
            # cumulative number counts
            nn_pos_same1_cum = np.array([np.sum(nn_pos_same1 < b) for b in bin_edges1]) / 4 / np.pi
            nn_neg_same1_cum = np.array([np.sum(nn_neg_same1 < b) for b in bin_edges1]) / 4 / np.pi
            nn_pos_opp1_cum = np.array([np.sum(nn_pos_opp1 < b) for b in bin_edges1]) / 4 / np.pi
            nn_neg_opp1_cum = np.array([np.sum(nn_neg_opp1 < b) for b in bin_edges1]) / 4 / np.pi
            nn_pos_same2_cum = np.array([np.sum(nn_pos_same2 < b) for b in bin_edges2]) / 4 / np.pi
            nn_neg_same2_cum = np.array([np.sum(nn_neg_same2 < b) for b in bin_edges2]) / 4 / np.pi
            nn_pos_opp2_cum = np.array([np.sum(nn_pos_opp2 < b) for b in bin_edges2]) / 4 / np.pi
            nn_neg_opp2_cum = np.array([np.sum(nn_neg_opp2 < b) for b in bin_edges2]) / 4 / np.pi
            nn_pos_same4_cum = np.array([np.sum(nn_pos_same4 < b) for b in bin_edges4]) / 4 / np.pi
            nn_neg_same4_cum = np.array([np.sum(nn_neg_same4 < b) for b in bin_edges4]) / 4 / np.pi
            nn_pos_opp4_cum = np.array([np.sum(nn_pos_opp4 < b) for b in bin_edges4]) / 4 / np.pi
            nn_neg_opp4_cum = np.array([np.sum(nn_neg_opp4 < b) for b in bin_edges4]) / 4 / np.pi
            nn_pos_same8_cum = np.array([np.sum(nn_pos_same8 < b) for b in bin_edges8]) / 4 / np.pi
            nn_neg_same8_cum = np.array([np.sum(nn_neg_same8 < b) for b in bin_edges8]) / 4 / np.pi
            nn_pos_opp8_cum = np.array([np.sum(nn_pos_opp8 < b) for b in bin_edges8]) / 4 / np.pi
            nn_neg_opp8_cum = np.array([np.sum(nn_neg_opp8 < b) for b in bin_edges8]) / 4 / np.pi
            
            # combine all nn data for this seed
            nn_combined = np.hstack([
                    nn_pos_same1_cum, nn_neg_same1_cum, nn_pos_opp1_cum, nn_neg_opp1_cum,
                    nn_pos_same2_cum, nn_neg_same2_cum, nn_pos_opp2_cum, nn_neg_opp2_cum,
                    nn_pos_same4_cum, nn_neg_same4_cum, nn_pos_opp4_cum, nn_neg_opp4_cum,
                    nn_pos_same8_cum, nn_neg_same8_cum, nn_pos_opp8_cum, nn_neg_opp8_cum
            ])
            
            # Save each individual realization
            bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
            os.makedirs(f"{DATA_DIR}/TrainPDFs", exist_ok=True)
            np.savez(f"{DATA_DIR}/TrainPDFs/train_{i}_s{s}_seed_{seed}_kappa_pdf.npz",
                    hist=hist,
                    bin_centers=bin_centers)

            os.makedirs(f"{DATA_DIR}/TrainDPDFs", exist_ok=True)
            np.savez(f"{DATA_DIR}/TrainDPDFs/train_{i}_s{s}_seed_{seed}_defect_kappa_pos_pdf.npz",
                    hist=kappa_pos_cum,
                    bin_centers=bins)

            np.savez(f"{DATA_DIR}/TrainDPDFs/train_{i}_s{s}_seed_{seed}_defect_kappa_neg_pdf.npz",
                    hist=kappa_neg_cum,
                    bin_centers=bins)

            os.makedirs(f"{DATA_DIR}/TrainKKs", exist_ok=True)
            np.savez(f"{DATA_DIR}/TrainKKs/train_{i}_s{s}_seed_{seed}_kappa_kappa_corr.npz",
                    theta=theta,
                    xi_kk=xi_kk,
                    varxi_kk=varxi_kk,
                    covmat_kk=covmat_kk)
            
            os.makedirs(f"{DATA_DIR}/TrainPosPos", exist_ok=True)
            np.savez(f"{DATA_DIR}/TrainPosPos/train_{i}_s{s}_seed_{seed}_pos_pos_corr.npz",
                    theta=theta,
                    xi_pp=xi_pp,
                    varxi_pp=varxi_pp,
                    covmat_pp=covmat_pp)
            os.makedirs(f"{DATA_DIR}/TrainNegNeg", exist_ok=True)
            np.savez(f"{DATA_DIR}/TrainNegNeg/train_{i}_s{s}_seed_{seed}_neg_neg_corr.npz",
                    theta=theta,    
                    xi_nn=xi_nn,
                    varxi_nn=varxi_nn,
                    covmat_nn=covmat_nn)
            os.makedirs(f"{DATA_DIR}/TrainPosNeg", exist_ok=True)
            np.savez(f"{DATA_DIR}/TrainPosNeg/train_{i}_s{s}_seed_{seed}_pos_neg_corr.npz",
                    theta=theta,
                    xi_pn=xi_pn,
                    varxi_pn=varxi_pn,
                    covmat_pn=covmat_pn)
            os.makedirs(f"{DATA_DIR}/TrainPKs", exist_ok=True)
            np.savez(f"{DATA_DIR}/TrainPKs/train_{i}_s{s}_seed_{seed}_pos_kappa_corr.npz",
                    theta=theta,
                    xi_pk=xi_pk,
                    varxi_pk=varxi_pk,
                    covmat_pk=covmat_pk)
            os.makedirs(f"{DATA_DIR}/TrainNKs", exist_ok=True)
            np.savez(f"{DATA_DIR}/TrainNKs/train_{i}_s{s}_seed_{seed}_neg_kappa_corr.npz",
                    theta=theta,
                    xi_nk=xi_nk,
                    varxi_nk=varxi_nk,
                    covmat_nk=covmat_nk)
            

            os.makedirs(f"{DATA_DIR}/TrainNNs", exist_ok=True)
            np.savez(f"{DATA_DIR}/TrainNNs/train_{i}_s{s}_seed_{seed}_NNs_combined.npz",
                    nn_combined=nn_combined)
            
            # Append to corresponding smoothing scale lists
            if s == 15:
                all_kappa_pdf_s15.append(hist)
                all_kappa_pos_cum_s15.append(kappa_pos_cum)
                all_kappa_neg_cum_s15.append(kappa_neg_cum)
                all_xi_kk_s15.append(xi_kk)
                all_varxi_kk_s15.append(varxi_kk)
                all_covmat_kk_s15.append(covmat_kk)
                all_xi_pp_s15.append(xi_pp)
                all_varxi_pp_s15.append(varxi_pp)
                all_covmat_pp_s15.append(covmat_pp)
                all_xi_nn_s15.append(xi_nn)
                all_varxi_nn_s15.append(varxi_nn)
                all_covmat_nn_s15.append(covmat_nn)
                all_xi_pn_s15.append(xi_pn)
                all_varxi_pn_s15.append(varxi_pn)
                all_covmat_pn_s15.append(covmat_pn)
                all_xi_pk_s15.append(xi_pk)
                all_varxi_pk_s15.append(varxi_pk)
                all_covmat_pk_s15.append(covmat_pk)
                all_xi_nk_s15.append(xi_nk)
                all_varxi_nk_s15.append(varxi_nk)
                all_covmat_nk_s15.append(covmat_nk) 

                all_xi_pos_peak_s15.append(xi_pos_peak)
                all_varxi_pos_peak_s15.append(varxi_pos_peak)
                all_covmat_pos_peak_s15.append(covmat_pos_peak)
                all_xi_pos_min_s15.append(xi_pos_min)
                all_varxi_pos_min_s15.append(varxi_pos_min)
                all_covmat_pos_min_s15.append(covmat_pos_min)
                all_xi_neg_peak_s15.append(xi_neg_peak)
                all_varxi_neg_peak_s15.append(varxi_neg_peak)
                all_covmat_neg_peak_s15.append(covmat_neg_peak)
                all_xi_neg_min_s15.append(xi_neg_min)
                all_varxi_neg_min_s15.append(varxi_neg_min)
                all_covmat_neg_min_s15.append(covmat_neg_min)

                all_nn_combined_s15.append(nn_combined)
            elif s == 30:
                all_kappa_pdf_s30.append(hist)
                all_kappa_pos_cum_s30.append(kappa_pos_cum)
                all_kappa_neg_cum_s30.append(kappa_neg_cum)
                all_xi_kk_s30.append(xi_kk)
                all_varxi_kk_s30.append(varxi_kk)
                all_covmat_kk_s30.append(covmat_kk)
                all_xi_pp_s30.append(xi_pp)
                all_varxi_pp_s30.append(varxi_pp)
                all_covmat_pp_s30.append(covmat_pp)
                all_xi_nn_s30.append(xi_nn)
                all_varxi_nn_s30.append(varxi_nn)
                all_covmat_nn_s30.append(covmat_nn)
                all_xi_pn_s30.append(xi_pn)
                all_varxi_pn_s30.append(varxi_pn)
                all_covmat_pn_s30.append(covmat_pn)
                all_xi_pk_s30.append(xi_pk)
                all_varxi_pk_s30.append(varxi_pk)
                all_covmat_pk_s30.append(covmat_pk)
                all_xi_nk_s30.append(xi_nk)
                all_varxi_nk_s30.append(varxi_nk)
                all_covmat_nk_s30.append(covmat_nk)
                
                all_xi_pos_peak_s30.append(xi_pos_peak)
                all_varxi_pos_peak_s30.append(varxi_pos_peak)
                all_covmat_pos_peak_s30.append(covmat_pos_peak)
                all_xi_pos_min_s30.append(xi_pos_min)
                all_varxi_pos_min_s30.append(varxi_pos_min)
                all_covmat_pos_min_s30.append(covmat_pos_min)
                all_xi_neg_peak_s30.append(xi_neg_peak)
                all_varxi_neg_peak_s30.append(varxi_neg_peak)
                all_covmat_neg_peak_s30.append(covmat_neg_peak)
                all_xi_neg_min_s30.append(xi_neg_min)
                all_varxi_neg_min_s30.append(varxi_neg_min)
                all_covmat_neg_min_s30.append(covmat_neg_min)

                all_nn_combined_s30.append(nn_combined)
            elif s == 90:
                all_kappa_pdf_s90.append(hist)
                all_kappa_pos_cum_s90.append(kappa_pos_cum)
                all_kappa_neg_cum_s90.append(kappa_neg_cum)
                all_xi_kk_s90.append(xi_kk)
                all_varxi_kk_s90.append(varxi_kk)
                all_covmat_kk_s90.append(covmat_kk)
                all_xi_pp_s90.append(xi_pp)
                all_varxi_pp_s90.append(varxi_pp)
                all_covmat_pp_s90.append(covmat_pp)
                all_xi_nn_s90.append(xi_nn)
                all_varxi_nn_s90.append(varxi_nn)
                all_covmat_nn_s90.append(covmat_nn)
                all_xi_pn_s90.append(xi_pn)
                all_varxi_pn_s90.append(varxi_pn)
                all_covmat_pn_s90.append(covmat_pn)
                all_xi_pk_s90.append(xi_pk)
                all_varxi_pk_s90.append(varxi_pk)
                all_covmat_pk_s90.append(covmat_pk)
                all_xi_nk_s90.append(xi_nk)
                all_varxi_nk_s90.append(varxi_nk)
                all_covmat_nk_s90.append(covmat_nk)

                all_xi_pos_peak_s90.append(xi_pos_peak)
                all_varxi_pos_peak_s90.append(varxi_pos_peak)
                all_covmat_pos_peak_s90.append(covmat_pos_peak)
                all_xi_pos_min_s90.append(xi_pos_min)
                all_varxi_pos_min_s90.append(varxi_pos_min)
                all_covmat_pos_min_s90.append(covmat_pos_min)
                all_xi_neg_peak_s90.append(xi_neg_peak)
                all_varxi_neg_peak_s90.append(varxi_neg_peak)
                all_covmat_neg_peak_s90.append(covmat_neg_peak)
                all_xi_neg_min_s90.append(xi_neg_min)
                all_varxi_neg_min_s90.append(varxi_neg_min)
                all_covmat_neg_min_s90.append(covmat_neg_min)

                all_nn_combined_s90.append(nn_combined)
    
    # Calculate averages over all seeds for each smoothing scale
    bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
    
    for s, (
        kappa_pdf_list,
        kappa_pos_list,
        kappa_neg_list,
        xi_kk_list,
        varxi_kk_list,
        covmat_kk_list,
        nn_list,
        xi_pp_list,
        varxi_pp_list,
        covmat_pp_list,
        xi_nn_list,
        varxi_nn_list,
        covmat_nn_list,
        xi_pn_list,
        varxi_pn_list,
        covmat_pn_list,
        xi_pk_list,
        varxi_pk_list,
        covmat_pk_list,
        xi_nk_list,
        varxi_nk_list,
        covmat_nk_list,
        xi_pos_peak_list, varxi_pos_peak_list, covmat_pos_peak_list,
        xi_pos_min_list, varxi_pos_min_list, covmat_pos_min_list,
        xi_neg_peak_list, varxi_neg_peak_list, covmat_neg_peak_list,
        xi_neg_min_list, varxi_neg_min_list, covmat_neg_min_list
    ) in [
        (15, (
            all_kappa_pdf_s15, all_kappa_pos_cum_s15, all_kappa_neg_cum_s15,
            all_xi_kk_s15, all_varxi_kk_s15, all_covmat_kk_s15, all_nn_combined_s15,
            all_xi_pp_s15, all_varxi_pp_s15, all_covmat_pp_s15,
            all_xi_nn_s15, all_varxi_nn_s15, all_covmat_nn_s15,
            all_xi_pn_s15, all_varxi_pn_s15, all_covmat_pn_s15,
            all_xi_pk_s15, all_varxi_pk_s15, all_covmat_pk_s15,
            all_xi_nk_s15, all_varxi_nk_s15, all_covmat_nk_s15,
            all_xi_pos_peak_s15, all_varxi_pos_peak_s15, all_covmat_pos_peak_s15,
            all_xi_pos_min_s15, all_varxi_pos_min_s15, all_covmat_pos_min_s15,
            all_xi_neg_peak_s15, all_varxi_neg_peak_s15, all_covmat_neg_peak_s15,
            all_xi_neg_min_s15, all_varxi_neg_min_s15, all_covmat_neg_min_s15
        )),
        (30, (
            all_kappa_pdf_s30, all_kappa_pos_cum_s30, all_kappa_neg_cum_s30,
            all_xi_kk_s30, all_varxi_kk_s30, all_covmat_kk_s30, all_nn_combined_s30,
            all_xi_pp_s30, all_varxi_pp_s30, all_covmat_pp_s30,
            all_xi_nn_s30, all_varxi_nn_s30, all_covmat_nn_s30,
            all_xi_pn_s30, all_varxi_pn_s30, all_covmat_pn_s30,
            all_xi_pk_s30, all_varxi_pk_s30, all_covmat_pk_s30,
            all_xi_nk_s30, all_varxi_nk_s30, all_covmat_nk_s30,
            all_xi_pos_peak_s30, all_varxi_pos_peak_s30, all_covmat_pos_peak_s30,
            all_xi_pos_min_s30, all_varxi_pos_min_s30, all_covmat_pos_min_s30,
            all_xi_neg_peak_s30, all_varxi_neg_peak_s30, all_covmat_neg_peak_s30,
            all_xi_neg_min_s30, all_varxi_neg_min_s30, all_covmat_neg_min_s30
        )),
        (90, (
            all_kappa_pdf_s90, all_kappa_pos_cum_s90, all_kappa_neg_cum_s90,
            all_xi_kk_s90, all_varxi_kk_s90, all_covmat_kk_s90, all_nn_combined_s90,
            all_xi_pp_s90, all_varxi_pp_s90, all_covmat_pp_s90,
            all_xi_nn_s90, all_varxi_nn_s90, all_covmat_nn_s90,
            all_xi_pn_s90, all_varxi_pn_s90, all_covmat_pn_s90,
            all_xi_pk_s90, all_varxi_pk_s90, all_covmat_pk_s90,
            all_xi_nk_s90, all_varxi_nk_s90, all_covmat_nk_s90,
            all_xi_pos_peak_s90, all_varxi_pos_peak_s90, all_covmat_pos_peak_s90,
            all_xi_pos_min_s90, all_varxi_pos_min_s90, all_covmat_pos_min_s90,
            all_xi_neg_peak_s90, all_varxi_neg_peak_s90, all_covmat_neg_peak_s90,
            all_xi_neg_min_s90, all_varxi_neg_min_s90, all_covmat_neg_min_s90
        )),
    ]:
        kappa_pdf_avg = np.mean(kappa_pdf_list, axis=0)
        kappa_pos_cum_avg = np.mean(kappa_pos_list, axis=0)
        kappa_neg_cum_avg = np.mean(kappa_neg_list, axis=0)
        xi_kk_avg = np.mean(xi_kk_list, axis=0)
        varxi_kk_avg = np.mean(varxi_kk_list, axis=0)
        covmat_kk_avg = np.mean(covmat_kk_list, axis=0)
        nn_combined_avg = np.mean(nn_list, axis=0)
        xi_pp_avg = np.mean(xi_pp_list, axis=0)
        varxi_pp_avg = np.mean(varxi_pp_list, axis=0)
        covmat_pp_avg = np.mean(covmat_pp_list, axis=0)
        xi_nn_avg = np.mean(xi_nn_list, axis=0)
        varxi_nn_avg = np.mean(varxi_nn_list, axis=0)
        covmat_nn_avg = np.mean(covmat_nn_list, axis=0)
        xi_pn_avg = np.mean(xi_pn_list, axis=0)
        varxi_pn_avg = np.mean(varxi_pn_list, axis=0)
        covmat_pn_avg = np.mean(covmat_pn_list, axis=0)
        xi_pk_avg = np.mean(xi_pk_list, axis=0)
        varxi_pk_avg = np.mean(varxi_pk_list, axis=0)
        covmat_pk_avg = np.mean(covmat_pk_list, axis=0)
        xi_nk_avg = np.mean(xi_nk_list, axis=0)
        varxi_nk_avg = np.mean(varxi_nk_list, axis=0)
        covmat_nk_avg = np.mean(covmat_nk_list, axis=0)

        # Average Peak/Minima results
        xi_pos_peak_avg = np.mean(xi_pos_peak_list, axis=0)
        varxi_pos_peak_avg = np.mean(varxi_pos_peak_list, axis=0)
        covmat_pos_peak_avg = np.mean(covmat_pos_peak_list, axis=0)
        
        xi_pos_min_avg = np.mean(xi_pos_min_list, axis=0)
        varxi_pos_min_avg = np.mean(varxi_pos_min_list, axis=0)
        covmat_pos_min_avg = np.mean(covmat_pos_min_list, axis=0)
        
        xi_neg_peak_avg = np.mean(xi_neg_peak_list, axis=0)
        varxi_neg_peak_avg = np.mean(varxi_neg_peak_list, axis=0)
        covmat_neg_peak_avg = np.mean(covmat_neg_peak_list, axis=0)
        
        xi_neg_min_avg = np.mean(xi_neg_min_list, axis=0)
        varxi_neg_min_avg = np.mean(varxi_neg_min_list, axis=0)
        covmat_neg_min_avg = np.mean(covmat_neg_min_list, axis=0)

        # Save averaged results   
        os.makedirs(f"{DATA_DIR}/TrainPDFs", exist_ok=True)
        np.savez(f"{DATA_DIR}/TrainPDFs/train_{i}_s{s}_kappa_pdf.npz",
                hist=kappa_pdf_avg,
                bin_centers=bin_centers)

        os.makedirs(f"{DATA_DIR}/TrainDPDFs", exist_ok=True)
        np.savez(f"{DATA_DIR}/TrainDPDFs/train_{i}_s{s}_defect_kappa_pos_pdf.npz",
                hist=kappa_pos_cum_avg,
                bin_centers=bins)

        np.savez(f"{DATA_DIR}/TrainDPDFs/train_{i}_s{s}_defect_kappa_neg_pdf.npz",
                hist=kappa_neg_cum_avg,
                bin_centers=bins)

        os.makedirs(f"{DATA_DIR}/TrainKKs", exist_ok=True)
        np.savez(f"{DATA_DIR}/TrainKKs/train_{i}_s{s}_kappa_kappa_corr.npz",
                theta=theta,
                xi_kk=xi_kk_avg,
                varxi_kk=varxi_kk_avg,
                covmat_kk=covmat_kk_avg)

        os.makedirs(f"{DATA_DIR}/TrainNNs", exist_ok=True)
        np.savez(f"{DATA_DIR}/TrainNNs/train_{i}_s{s}_NNs_combined.npz",
                nn_combined=nn_combined_avg)

        os.makedirs(f"{DATA_DIR}/TrainPosPos", exist_ok=True)
        np.savez(f"{DATA_DIR}/TrainPosPos/train_{i}_s{s}_pos_pos_corr.npz",
            theta=theta,
            xi_pp=xi_pp_avg,
            varxi_pp=varxi_pp_avg,
            covmat_pp=covmat_pp_avg)

        os.makedirs(f"{DATA_DIR}/TrainNegNeg", exist_ok=True)
        np.savez(f"{DATA_DIR}/TrainNegNeg/train_{i}_s{s}_neg_neg_corr.npz",
            theta=theta,
            xi_nn=xi_nn_avg,
            varxi_nn=varxi_nn_avg,
            covmat_nn=covmat_nn_avg)

        os.makedirs(f"{DATA_DIR}/TrainPosNeg", exist_ok=True)
        np.savez(f"{DATA_DIR}/TrainPosNeg/train_{i}_s{s}_pos_neg_corr.npz",
            theta=theta,
            xi_pn=xi_pn_avg,
            varxi_pn=varxi_pn_avg,
            covmat_pn=covmat_pn_avg)

        os.makedirs(f"{DATA_DIR}/TrainPKs", exist_ok=True)
        np.savez(f"{DATA_DIR}/TrainPKs/train_{i}_s{s}_pos_kappa_corr.npz",
            theta=theta,
            xi_pk=xi_pk_avg,
            varxi_pk=varxi_pk_avg,
            covmat_pk=covmat_pk_avg)

        os.makedirs(f"{DATA_DIR}/TrainNKs", exist_ok=True)
        np.savez(f"{DATA_DIR}/TrainNKs/train_{i}_s{s}_neg_kappa_corr.npz",
            theta=theta,
            xi_nk=xi_nk_avg,
            varxi_nk=varxi_nk_avg,
            covmat_nk=covmat_nk_avg)

        os.makedirs(f"{DATA_DIR}/TrainPosPeak", exist_ok=True)
        np.savez(f"{DATA_DIR}/TrainPosPeak/train_{i}_s{s}_pos_peak_corr.npz", theta=theta, xi=xi_pos_peak_avg, varxi=varxi_pos_peak_avg, covmat=covmat_pos_peak_avg)

        os.makedirs(f"{DATA_DIR}/TrainPosMin", exist_ok=True)
        np.savez(f"{DATA_DIR}/TrainPosMin/train_{i}_s{s}_pos_min_corr.npz", theta=theta, xi=xi_pos_min_avg, varxi=varxi_pos_min_avg, covmat=covmat_pos_min_avg)
        
        os.makedirs(f"{DATA_DIR}/TrainNegPeak", exist_ok=True)
        np.savez(f"{DATA_DIR}/TrainNegPeak/train_{i}_s{s}_neg_peak_corr.npz", theta=theta, xi=xi_neg_peak_avg, varxi=varxi_neg_peak_avg, covmat=covmat_neg_peak_avg)

        os.makedirs(f"{DATA_DIR}/TrainNegMin", exist_ok=True)
        np.savez(f"{DATA_DIR}/TrainNegMin/train_{i}_s{s}_neg_min_corr.npz", theta=theta, xi=xi_neg_min_avg, varxi=varxi_neg_min_avg, covmat=covmat_neg_min_avg)