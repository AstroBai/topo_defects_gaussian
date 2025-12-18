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
DATA_DIR = "/cephfs/jbai/DefectFisherGaussian"
fid = {'Om': 0.3,
       'sigma8': 0.8}
Ob = 0.049
h = 0.67
w = -1.0
# surrounding cosmology params, e.g. Om +/- delta_Om
delta_params = {'Om': 0.02,
                'sigma8': 0.02}

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
n_seeds = 60  # Number of random seeds per parameter point

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
                bin_edges = np.array([-0.01, -0.005, 0.0, 0.005, 0.01])
            elif s == 30:
                bin_edges = np.array([-0.008, -0.004, 0.0, 0.004, 0.008])
            elif s == 90:
                bin_edges = np.array([-0.005, -0.0025, 0.0, 0.0025, 0.005])
            hist, bin_edges = np.histogram(df.kappa_bar, bins=bin_edges, density=True)
            
            # save defect PDFs
            if s == 15:
                bins = np.array([-0.01, -0.005, 0.005, 0.01])
            elif s == 30:
                bins = np.array([-0.008, -0.004, 0.004, 0.008])
            elif s == 90:
                bins = np.array([-0.005, -0.0025, 0.0025, 0.005])   
            kappa_pos = df.convergence_dist(charge=0.5)
            kappa_pos_cum = np.array([np.sum(kappa_pos < b) for b in bins]) / 4 / np.pi

            kappa_neg = df.convergence_dist(charge=-0.5)
            kappa_neg_cum = np.array([np.sum(kappa_neg < b) for b in bins]) / 4 / np.pi

            if s == 15:
                theta, xi_kk, varxi_kk, covmat_kk = df.convergence_correlation(min_sep=15., max_sep=600, n_bins=10, smoothed=True)
            elif s == 30:
                theta, xi_kk, varxi_kk, covmat_kk = df.convergence_correlation(min_sep=30., max_sep=600, n_bins=10, smoothed=True)
            elif s == 90:
                theta, xi_kk, varxi_kk, covmat_kk = df.convergence_correlation(min_sep=90., max_sep=600, n_bins=10, smoothed=True)

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
                bin_edges1 = np.linspace(30, 100, 5)
                bin_edges2 = np.linspace(30, 200, 5)
                bin_edges4 = np.linspace(30, 300, 5)
                bin_edges8 = np.linspace(30, 400, 5)
            elif s == 30:
                bin_edges1 = np.linspace(30, 300, 5)
                bin_edges2 = np.linspace(30, 450, 5)
                bin_edges4 = np.linspace(30, 600, 5)
                bin_edges8 = np.linspace(30, 750, 5)
            elif s == 90:
                bin_edges1 = np.linspace(30, 750, 5)
                bin_edges2 = np.linspace(30, 1125, 5)
                bin_edges4 = np.linspace(30, 1500, 5)
                bin_edges8 = np.linspace(30, 1875, 5)
            
            # cumulative number counts
            nn_pos_same1_hist, _ = np.histogram(nn_pos_same1, bins=bin_edges1, density=False)
            nn_neg_same1_hist, _ = np.histogram(nn_neg_same1, bins=bin_edges1, density=False)
            nn_pos_opp1_hist, _ = np.histogram(nn_pos_opp1, bins=bin_edges1, density=False)
            nn_neg_opp1_hist, _ = np.histogram(nn_neg_opp1, bins=bin_edges1, density=False)
            nn_pos_same2_hist, _ = np.histogram(nn_pos_same2, bins=bin_edges2, density=False)    
            nn_neg_same2_hist, _ = np.histogram(nn_neg_same2, bins=bin_edges2, density=False)
            nn_pos_opp2_hist, _ = np.histogram(nn_pos_opp2, bins=bin_edges2, density=False)
            nn_neg_opp2_hist, _ = np.histogram(nn_neg_opp2, bins=bin_edges2, density=False)
            nn_pos_same4_hist, _ = np.histogram(nn_pos_same4, bins=bin_edges4, density=False)
            nn_neg_same4_hist, _ = np.histogram(nn_neg_same4, bins=bin_edges4, density=False)
            nn_pos_opp4_hist, _ = np.histogram(nn_pos_opp4, bins=bin_edges4, density=False)
            nn_neg_opp4_hist, _ = np.histogram(nn_neg_opp4, bins=bin_edges4, density=False)
            nn_pos_same8_hist, _ = np.histogram(nn_pos_same8, bins=bin_edges8, density=False)
            nn_neg_same8_hist, _ = np.histogram(nn_neg_same8, bins=bin_edges8, density=False)
            nn_pos_opp8_hist, _ = np.histogram(nn_pos_opp8, bins=bin_edges8, density=False)
            nn_neg_opp8_hist, _ = np.histogram(nn_neg_opp8, bins=bin_edges8, density=False)
            
            # make cumulative
            nn_pos_same1_cum = np.cumsum(nn_pos_same1_hist) / 4 / np.pi
            nn_neg_same1_cum = np.cumsum(nn_neg_same1_hist) / 4 / np.pi
            nn_pos_opp1_cum = np.cumsum(nn_pos_opp1_hist) / 4 / np.pi
            nn_neg_opp1_cum = np.cumsum(nn_neg_opp1_hist) / 4 / np.pi
            nn_pos_same2_cum = np.cumsum(nn_pos_same2_hist) / 4 / np.pi
            nn_neg_same2_cum = np.cumsum(nn_neg_same2_hist) / 4 / np.pi
            nn_pos_opp2_cum = np.cumsum(nn_pos_opp2_hist) / 4 / np.pi
            nn_neg_opp2_cum = np.cumsum(nn_neg_opp2_hist) / 4 / np.pi
            nn_pos_same4_cum = np.cumsum(nn_pos_same4_hist) / 4 / np.pi
            nn_neg_same4_cum = np.cumsum(nn_neg_same4_hist) / 4 / np.pi
            nn_pos_opp4_cum = np.cumsum(nn_pos_opp4_hist) / 4 / np.pi
            nn_neg_opp4_cum = np.cumsum(nn_neg_opp4_hist) / 4 / np.pi
            nn_pos_same8_cum = np.cumsum(nn_pos_same8_hist) / 4 / np.pi
            nn_neg_same8_cum = np.cumsum(nn_neg_same8_hist) / 4 / np.pi
            nn_pos_opp8_cum = np.cumsum(nn_pos_opp8_hist) / 4 / np.pi
            nn_neg_opp8_cum = np.cumsum(nn_neg_opp8_hist) / 4 / np.pi
            
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
                all_nn_combined_s15.append(nn_combined)
            elif s == 30:
                all_kappa_pdf_s30.append(hist)
                all_kappa_pos_cum_s30.append(kappa_pos_cum)
                all_kappa_neg_cum_s30.append(kappa_neg_cum)
                all_xi_kk_s30.append(xi_kk)
                all_varxi_kk_s30.append(varxi_kk)
                all_covmat_kk_s30.append(covmat_kk)
                all_nn_combined_s30.append(nn_combined)
            elif s == 90:
                all_kappa_pdf_s90.append(hist)
                all_kappa_pos_cum_s90.append(kappa_pos_cum)
                all_kappa_neg_cum_s90.append(kappa_neg_cum)
                all_xi_kk_s90.append(xi_kk)
                all_varxi_kk_s90.append(varxi_kk)
                all_covmat_kk_s90.append(covmat_kk)
                all_nn_combined_s90.append(nn_combined)
    
    # Calculate averages over all seeds for each smoothing scale
    bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
    
    for s, (kappa_pdf_list, kappa_pos_list, kappa_neg_list, xi_kk_list, varxi_kk_list, covmat_kk_list, nn_list) in [
        (15, (all_kappa_pdf_s15, all_kappa_pos_cum_s15, all_kappa_neg_cum_s15, all_xi_kk_s15, all_varxi_kk_s15, all_covmat_kk_s15, all_nn_combined_s15)),
        (30, (all_kappa_pdf_s30, all_kappa_pos_cum_s30, all_kappa_neg_cum_s30, all_xi_kk_s30, all_varxi_kk_s30, all_covmat_kk_s30, all_nn_combined_s30)),
        (90, (all_kappa_pdf_s90, all_kappa_pos_cum_s90, all_kappa_neg_cum_s90, all_xi_kk_s90, all_varxi_kk_s90, all_covmat_kk_s90, all_nn_combined_s90))
    ]:
        kappa_pdf_avg = np.mean(kappa_pdf_list, axis=0)
        kappa_pos_cum_avg = np.mean(kappa_pos_list, axis=0)
        kappa_neg_cum_avg = np.mean(kappa_neg_list, axis=0)
        xi_kk_avg = np.mean(xi_kk_list, axis=0)
        varxi_kk_avg = np.mean(varxi_kk_list, axis=0)
        covmat_kk_avg = np.mean(covmat_kk_list, axis=0)
        nn_combined_avg = np.mean(nn_list, axis=0)

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