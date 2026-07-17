import numpy as np

DATA_DIR = "/home/jbai/cephfs_copy/cephfs/jbai/DefectFisherGaussian"
fid = {'Om': 0.3,
       'sigma8': 0.8}
Ob = 0.049
h = 0.67
w = -1.0
# surrounding cosmology params, e.g. Om +/- delta_Om
delta_params = {'Om': 0.1,
                'sigma8': 0.1}
n_seeds = 100

params = []
for key in fid.keys():
    p_minus = fid.copy()
    p_plus = fid.copy()
    p_minus[key] -= delta_params[key]
    p_plus[key] += delta_params[key]
    params.append(p_minus)
    params.append(p_plus)

params = np.array(params)

pdfs_s15 = []
pdfs_s30 = []
pdfs_s90 = []

dpdfs_pos_s15 = []
dpdfs_pos_s30 = []
dpdfs_pos_s90 = []

dpdfs_neg_s15 = []
dpdfs_neg_s30 = []
dpdfs_neg_s90 = []

# dk_corrs_pos = []
# dk_corrs_neg = []
kk_corrs_s15 = []
kk_corrs_s30 = []
kk_corrs_s90 = []

nn_combined_s15 = []
nn_combined_s30 = []
nn_combined_s90 = []
for seed in range(n_seeds):
    for i in range(4):
        data = np.load(f"{DATA_DIR}/TrainPDFs/train_{i}_s15_seed_{seed}_kappa_pdf.npz")
        pdfs_s15.append((data['hist']))
        data = np.load(f"{DATA_DIR}/TrainPDFs/train_{i}_s30_seed_{seed}_kappa_pdf.npz")
        pdfs_s30.append((data['hist']))
        data = np.load(f"{DATA_DIR}/TrainPDFs/train_{i}_s90_seed_{seed}_kappa_pdf.npz")
        pdfs_s90.append((data['hist']))
        data_pos = np.load(f"{DATA_DIR}/TrainDPDFs/train_{i}_s15_seed_{seed}_defect_kappa_pos_pdf.npz")
        dpdfs_pos_s15.append((data_pos['hist']))
        data_pos = np.load(f"{DATA_DIR}/TrainDPDFs/train_{i}_s30_seed_{seed}_defect_kappa_pos_pdf.npz")
        dpdfs_pos_s30.append((data_pos['hist']))
        data_pos = np.load(f"{DATA_DIR}/TrainDPDFs/train_{i}_s90_seed_{seed}_defect_kappa_pos_pdf.npz")
        dpdfs_pos_s90.append((data_pos['hist']))

        data_neg = np.load(f"{DATA_DIR}/TrainDPDFs/train_{i}_s15_seed_{seed}_defect_kappa_neg_pdf.npz")
        dpdfs_neg_s15.append((data_neg['hist']))
        data_neg = np.load(f"{DATA_DIR}/TrainDPDFs/train_{i}_s30_seed_{seed}_defect_kappa_neg_pdf.npz")
        dpdfs_neg_s30.append((data_neg['hist']))
        data_neg = np.load(f"{DATA_DIR}/TrainDPDFs/train_{i}_s90_seed_{seed}_defect_kappa_neg_pdf.npz")
        dpdfs_neg_s90.append((data_neg['hist']))
        data_kk = np.load(f"{DATA_DIR}/TrainKKs/train_{i}_s15_seed_{seed}_kappa_kappa_corr.npz")
        kk_corrs_s15.append((data_kk['xi_kk']))
        data_kk = np.load(f"{DATA_DIR}/TrainKKs/train_{i}_s30_seed_{seed}_kappa_kappa_corr.npz")
        kk_corrs_s30.append((data_kk['xi_kk']))
        data_kk = np.load(f"{DATA_DIR}/TrainKKs/train_{i}_s90_seed_{seed}_kappa_kappa_corr.npz")
        kk_corrs_s90.append((data_kk['xi_kk']))
        data_nn_combined = np.load(f"{DATA_DIR}/TrainNNs/train_{i}_s15_seed_{seed}_NNs_combined.npz")
        nn_combined_s15.append((data_nn_combined['nn_combined']))
        data_nn_combined = np.load(f"{DATA_DIR}/TrainNNs/train_{i}_s30_seed_{seed}_NNs_combined.npz")
        nn_combined_s30.append((data_nn_combined['nn_combined']))
        data_nn_combined = np.load(f"{DATA_DIR}/TrainNNs/train_{i}_s90_seed_{seed}_NNs_combined.npz")
        nn_combined_s90.append((data_nn_combined['nn_combined']))

# reshape, num_samples x num_params*hist_bins
pdfs_s15 = np.array(pdfs_s15)
pdfs_s30 = np.array(pdfs_s30)
pdfs_s90 = np.array(pdfs_s90)
dpdfs_pos_s15 = np.array(dpdfs_pos_s15)
dpdfs_pos_s30 = np.array(dpdfs_pos_s30)
dpdfs_pos_s90 = np.array(dpdfs_pos_s90)
dpdfs_neg_s15 = np.array(dpdfs_neg_s15)
dpdfs_neg_s30 = np.array(dpdfs_neg_s30)
dpdfs_neg_s90 = np.array(dpdfs_neg_s90)
kk_corrs_s15 = np.array(kk_corrs_s15)
kk_corrs_s30 = np.array(kk_corrs_s30)
kk_corrs_s90 = np.array(kk_corrs_s90)
nn_combined_s15 = np.array(nn_combined_s15)
nn_combined_s30 = np.array(nn_combined_s30)
nn_combined_s90 = np.array(nn_combined_s90)

print(f"pdfs_s15 shape: {pdfs_s15.shape}")
print(f"nn_combined_s15 shape: {nn_combined_s15.shape}")

pdfs_s15 = pdfs_s15.reshape(n_seeds, 4, -1)
pdfs_s30 = pdfs_s30.reshape(n_seeds, 4, -1)
pdfs_s90 = pdfs_s90.reshape(n_seeds, 4, -1)
dpdfs_pos_s15 = dpdfs_pos_s15.reshape(n_seeds, 4, -1)
dpdfs_pos_s30 = dpdfs_pos_s30.reshape(n_seeds, 4, -1)
dpdfs_pos_s90 = dpdfs_pos_s90.reshape(n_seeds, 4, -1)
dpdfs_neg_s15 = dpdfs_neg_s15.reshape(n_seeds, 4, -1)
dpdfs_neg_s30 = dpdfs_neg_s30.reshape(n_seeds, 4, -1)
dpdfs_neg_s90 = dpdfs_neg_s90.reshape(n_seeds, 4, -1)
kk_corrs_s15 = kk_corrs_s15.reshape(n_seeds, 4, -1)
kk_corrs_s30 = kk_corrs_s30.reshape(n_seeds, 4, -1)
kk_corrs_s90 = kk_corrs_s90.reshape(n_seeds, 4, -1)
nn_combined_s15 = nn_combined_s15.reshape(n_seeds, 4, -1)
nn_combined_s30 = nn_combined_s30.reshape(n_seeds, 4, -1)
nn_combined_s90 = nn_combined_s90.reshape(n_seeds, 4, -1)
print(f"After reshape - pdfs_s15: {pdfs_s15.shape}")
print(f"After reshape - nn_combined_s15: {nn_combined_s15.shape}")

# compute mean over seeds
pdfs_s15_mean = np.mean(pdfs_s15, axis=0)
pdfs_s30_mean = np.mean(pdfs_s30, axis=0)
pdfs_s90_mean = np.mean(pdfs_s90, axis=0)
dpdfs_pos_s15_mean = np.mean(dpdfs_pos_s15, axis=0)
dpdfs_pos_s30_mean = np.mean(dpdfs_pos_s30, axis=0)
dpdfs_pos_s90_mean = np.mean(dpdfs_pos_s90, axis=0)
dpdfs_neg_s15_mean = np.mean(dpdfs_neg_s15, axis=0)
dpdfs_neg_s30_mean = np.mean(dpdfs_neg_s30, axis=0)
dpdfs_neg_s90_mean = np.mean(dpdfs_neg_s90, axis=0)
kk_corrs_s15_mean = np.mean(kk_corrs_s15, axis=0)
kk_corrs_s30_mean = np.mean(kk_corrs_s30, axis=0)
kk_corrs_s90_mean = np.mean(kk_corrs_s90, axis=0)
nn_combined_s15_mean = np.mean(nn_combined_s15, axis=0)
nn_combined_s30_mean = np.mean(nn_combined_s30, axis=0)
nn_combined_s90_mean = np.mean(nn_combined_s90, axis=0)

# load fiducial data
MOCK_DIR = '../Mock'
pdfs_fid_s15 = np.load(f'{MOCK_DIR}/PDF_mean_s15.npy')
cov_pdfs_fid_s15 = np.load(f'{MOCK_DIR}/CovPDF_s15.npy')
pdfs_fid_s30 = np.load(f'{MOCK_DIR}/PDF_mean_s30.npy')
cov_pdfs_fid_s30 = np.load(f'{MOCK_DIR}/CovPDF_s30.npy')
pdfs_fid_s90 = np.load(f'{MOCK_DIR}/PDF_mean_s90.npy')
cov_pdfs_fid_s90 = np.load(f'{MOCK_DIR}/CovPDF_s90.npy')

dpdfs_pos_fid_s15 = np.load(f'{MOCK_DIR}/DPDF_pos_mean_s15.npy')
cov_dpdfs_pos_s15 = np.load(f'{MOCK_DIR}/CovDPDF_pos_s15.npy')
dpdfs_neg_fid_s15 = np.load(f'{MOCK_DIR}/DPDF_neg_mean_s15.npy')
cov_dpdfs_neg_s15 = np.load(f'{MOCK_DIR}/CovDPDF_neg_s15.npy')
dpdfs_pos_fid_s30 = np.load(f'{MOCK_DIR}/DPDF_pos_mean_s30.npy')
cov_dpdfs_pos_s30 = np.load(f'{MOCK_DIR}/CovDPDF_pos_s30.npy')
dpdfs_neg_fid_s30 = np.load(f'{MOCK_DIR}/DPDF_neg_mean_s30.npy')
cov_dpdfs_neg_s30 = np.load(f'{MOCK_DIR}/CovDPDF_neg_s30.npy')
dpdfs_pos_fid_s90 = np.load(f'{MOCK_DIR}/DPDF_pos_mean_s90.npy')
cov_dpdfs_pos_s90 = np.load(f'{MOCK_DIR}/CovDPDF_pos_s90.npy')
dpdfs_neg_fid_s90 = np.load(f'{MOCK_DIR}/DPDF_neg_mean_s90.npy')
cov_dpdfs_neg_s90 = np.load(f'{MOCK_DIR}/CovDPDF_neg_s90.npy')

# Load NNs data - use .npz instead of .npy
nns_fid_s15 = np.load(f'{MOCK_DIR}/NNs_mean_s15.npy', allow_pickle=True).item()
nn_pos_same1_s15 = nns_fid_s15['nn_pos_same1_mean']
nn_neg_same1_s15 = nns_fid_s15['nn_neg_same1_mean']
nn_pos_opp1_s15 = nns_fid_s15['nn_pos_opp1_mean']
nn_neg_opp1_s15 = nns_fid_s15['nn_neg_opp1_mean']
nn_pos_same2_s15 = nns_fid_s15['nn_pos_same2_mean']
nn_neg_same2_s15 = nns_fid_s15['nn_neg_same2_mean']
nn_pos_opp2_s15 = nns_fid_s15['nn_pos_opp2_mean']
nn_neg_opp2_s15 = nns_fid_s15['nn_neg_opp2_mean']
nn_pos_same4_s15 = nns_fid_s15['nn_pos_same4_mean']
nn_neg_same4_s15 = nns_fid_s15['nn_neg_same4_mean']
nn_pos_opp4_s15 = nns_fid_s15['nn_pos_opp4_mean']
nn_neg_opp4_s15 = nns_fid_s15['nn_neg_opp4_mean']
nn_pos_same8_s15 = nns_fid_s15['nn_pos_same8_mean']
nn_neg_same8_s15 = nns_fid_s15['nn_neg_same8_mean']
nn_pos_opp8_s15 = nns_fid_s15['nn_pos_opp8_mean']
nn_neg_opp8_s15 = nns_fid_s15['nn_neg_opp8_mean']
cov_nns_fid_s15 = np.load(f'{MOCK_DIR}/CovNNs_s15.npy', allow_pickle=True).item()
cov_nn_pos_same1_s15 = cov_nns_fid_s15['cov_nn_pos_same1']
cov_nn_neg_same1_s15 = cov_nns_fid_s15['cov_nn_neg_same1']
cov_nn_pos_opp1_s15 = cov_nns_fid_s15['cov_nn_pos_opp1']
cov_nn_neg_opp1_s15 = cov_nns_fid_s15['cov_nn_neg_opp1']
cov_nn_pos_same2_s15 = cov_nns_fid_s15['cov_nn_pos_same2']
cov_nn_neg_same2_s15 = cov_nns_fid_s15['cov_nn_neg_same2']
cov_nn_pos_opp2_s15 = cov_nns_fid_s15['cov_nn_pos_opp2']
cov_nn_neg_opp2_s15 = cov_nns_fid_s15['cov_nn_neg_opp2']
cov_nn_pos_same4_s15 = cov_nns_fid_s15['cov_nn_pos_same4']
cov_nn_neg_same4_s15 = cov_nns_fid_s15['cov_nn_neg_same4']
cov_nn_pos_opp4_s15 = cov_nns_fid_s15['cov_nn_pos_opp4']
cov_nn_neg_opp4_s15 = cov_nns_fid_s15['cov_nn_neg_opp4']
cov_nn_pos_same8_s15 = cov_nns_fid_s15['cov_nn_pos_same8']
cov_nn_neg_same8_s15 = cov_nns_fid_s15['cov_nn_neg_same8']
cov_nn_pos_opp8_s15 = cov_nns_fid_s15['cov_nn_pos_opp8']
cov_nn_neg_opp8_s15 = cov_nns_fid_s15['cov_nn_neg_opp8']

nns_fid_s30 = np.load(f'{MOCK_DIR}/NNs_mean_s30.npy', allow_pickle=True).item()
nn_pos_same1_s30 = nns_fid_s30['nn_pos_same1_mean']
nn_neg_same1_s30 = nns_fid_s30['nn_neg_same1_mean']
nn_pos_opp1_s30 = nns_fid_s30['nn_pos_opp1_mean']
nn_neg_opp1_s30 = nns_fid_s30['nn_neg_opp1_mean']
nn_pos_same2_s30 = nns_fid_s30['nn_pos_same2_mean']
nn_neg_same2_s30 = nns_fid_s30['nn_neg_same2_mean']
nn_pos_opp2_s30 = nns_fid_s30['nn_pos_opp2_mean']
nn_neg_opp2_s30 = nns_fid_s30['nn_neg_opp2_mean']
nn_pos_same4_s30 = nns_fid_s30['nn_pos_same4_mean']
nn_neg_same4_s30 = nns_fid_s30['nn_neg_same4_mean']
nn_pos_opp4_s30 = nns_fid_s30['nn_pos_opp4_mean']
nn_neg_opp4_s30 = nns_fid_s30['nn_neg_opp4_mean']
nn_pos_same8_s30 = nns_fid_s30['nn_pos_same8_mean']
nn_neg_same8_s30 = nns_fid_s30['nn_neg_same8_mean']
nn_pos_opp8_s30 = nns_fid_s30['nn_pos_opp8_mean']
nn_neg_opp8_s30 = nns_fid_s30['nn_neg_opp8_mean']
cov_nns_fid_s30 = np.load(f'{MOCK_DIR}/CovNNs_s30.npy', allow_pickle=True).item()
cov_nn_pos_same1_s30 = cov_nns_fid_s30['cov_nn_pos_same1']
cov_nn_neg_same1_s30 = cov_nns_fid_s30['cov_nn_neg_same1']
cov_nn_pos_opp1_s30 = cov_nns_fid_s30['cov_nn_pos_opp1']
cov_nn_neg_opp1_s30 = cov_nns_fid_s30['cov_nn_neg_opp1']
cov_nn_pos_same2_s30 = cov_nns_fid_s30['cov_nn_pos_same2']
cov_nn_neg_same2_s30 = cov_nns_fid_s30['cov_nn_neg_same2']
cov_nn_pos_opp2_s30 = cov_nns_fid_s30['cov_nn_pos_opp2']
cov_nn_neg_opp2_s30 = cov_nns_fid_s30['cov_nn_neg_opp2']
cov_nn_pos_same4_s30 = cov_nns_fid_s30['cov_nn_pos_same4']
cov_nn_neg_same4_s30 = cov_nns_fid_s30['cov_nn_neg_same4']
cov_nn_pos_opp4_s30 = cov_nns_fid_s30['cov_nn_pos_opp4']
cov_nn_neg_opp4_s30 = cov_nns_fid_s30['cov_nn_neg_opp4']
cov_nn_pos_same8_s30 = cov_nns_fid_s30['cov_nn_pos_same8']
cov_nn_neg_same8_s30 = cov_nns_fid_s30['cov_nn_neg_same8']
cov_nn_pos_opp8_s30 = cov_nns_fid_s30['cov_nn_pos_opp8']
cov_nn_neg_opp8_s30 = cov_nns_fid_s30['cov_nn_neg_opp8']

nns_fid_s90 = np.load(f'{MOCK_DIR}/NNs_mean_s90.npy', allow_pickle=True).item()
nn_pos_same1_s90 = nns_fid_s90['nn_pos_same1_mean']
nn_neg_same1_s90 = nns_fid_s90['nn_neg_same1_mean']
nn_pos_opp1_s90 = nns_fid_s90['nn_pos_opp1_mean']
nn_neg_opp1_s90 = nns_fid_s90['nn_neg_opp1_mean']
nn_pos_same2_s90 = nns_fid_s90['nn_pos_same2_mean']
nn_neg_same2_s90 = nns_fid_s90['nn_neg_same2_mean']
nn_pos_opp2_s90 = nns_fid_s90['nn_pos_opp2_mean']
nn_neg_opp2_s90 = nns_fid_s90['nn_neg_opp2_mean']
nn_pos_same4_s90 = nns_fid_s90['nn_pos_same4_mean']
nn_neg_same4_s90 = nns_fid_s90['nn_neg_same4_mean']
nn_pos_opp4_s90 = nns_fid_s90['nn_pos_opp4_mean']
nn_neg_opp4_s90 = nns_fid_s90['nn_neg_opp4_mean']
nn_pos_same8_s90 = nns_fid_s90['nn_pos_same8_mean']
nn_neg_same8_s90 = nns_fid_s90['nn_neg_same8_mean']
nn_pos_opp8_s90 = nns_fid_s90['nn_pos_opp8_mean']
nn_neg_opp8_s90 = nns_fid_s90['nn_neg_opp8_mean']
cov_nns_fid_s90 = np.load(f'{MOCK_DIR}/CovNNs_s90.npy', allow_pickle=True).item()
cov_nn_pos_same1_s90 = cov_nns_fid_s90['cov_nn_pos_same1']
cov_nn_neg_same1_s90 = cov_nns_fid_s90['cov_nn_neg_same1']
cov_nn_pos_opp1_s90 = cov_nns_fid_s90['cov_nn_pos_opp1']
cov_nn_neg_opp1_s90 = cov_nns_fid_s90['cov_nn_neg_opp1']
cov_nn_pos_same2_s90 = cov_nns_fid_s90['cov_nn_pos_same2']
cov_nn_neg_same2_s90 = cov_nns_fid_s90['cov_nn_neg_same2']
cov_nn_pos_opp2_s90 = cov_nns_fid_s90['cov_nn_pos_opp2']
cov_nn_neg_opp2_s90 = cov_nns_fid_s90['cov_nn_neg_opp2']
cov_nn_pos_same4_s90 = cov_nns_fid_s90['cov_nn_pos_same4']
cov_nn_neg_same4_s90 = cov_nns_fid_s90['cov_nn_neg_same4']
cov_nn_pos_opp4_s90 = cov_nns_fid_s90['cov_nn_pos_opp4']
cov_nn_neg_opp4_s90 = cov_nns_fid_s90['cov_nn_neg_opp4']
cov_nn_pos_same8_s90 = cov_nns_fid_s90['cov_nn_pos_same8']
cov_nn_neg_same8_s90 = cov_nns_fid_s90['cov_nn_neg_same8']
cov_nn_pos_opp8_s90 = cov_nns_fid_s90['cov_nn_pos_opp8']
cov_nn_neg_opp8_s90 = cov_nns_fid_s90['cov_nn_neg_opp8']

kk_fid_s15 = np.load(f'{MOCK_DIR}/KK_mean_s15.npy')
cov_kk_fid_s15 = np.load(f'{MOCK_DIR}/CovKK_s15.npy')
kk_fid_s30 = np.load(f'{MOCK_DIR}/KK_mean_s30.npy')
cov_kk_fid_s30 = np.load(f'{MOCK_DIR}/CovKK_s30.npy')
kk_fid_s90 = np.load(f'{MOCK_DIR}/KK_mean_s90.npy')
cov_kk_fid_s90 = np.load(f'{MOCK_DIR}/CovKK_s90.npy')

pp_fid_s15 = np.load(f'{MOCK_DIR}/PosPos_mean_s15.npy')
cov_pp_fid_s15 = np.load(f'{MOCK_DIR}/CovPosPos_s15.npy')
nn_fid_s15 = np.load(f'{MOCK_DIR}/NegNeg_mean_s15.npy')
cov_nn_fid_s15 = np.load(f'{MOCK_DIR}/CovNegNeg_s15.npy')
pn_fid_s15 = np.load(f'{MOCK_DIR}/PosNeg_mean_s15.npy')
cov_pn_fid_s15 = np.load(f'{MOCK_DIR}/CovPosNeg_s15.npy')
pp_fid_s30 = np.load(f'{MOCK_DIR}/PosPos_mean_s30.npy')
cov_pp_fid_s30 = np.load(f'{MOCK_DIR}/CovPosPos_s30.npy')
nn_fid_s30 = np.load(f'{MOCK_DIR}/NegNeg_mean_s30.npy')
cov_nn_fid_s30 = np.load(f'{MOCK_DIR}/CovNegNeg_s30.npy')
pn_fid_s30 = np.load(f'{MOCK_DIR}/PosNeg_mean_s30.npy')
cov_pn_fid_s30 = np.load(f'{MOCK_DIR}/CovPosNeg_s30.npy')
pp_fid_s90 = np.load(f'{MOCK_DIR}/PosPos_mean_s90.npy')
cov_pp_fid_s90 = np.load(f'{MOCK_DIR}/CovPosPos_s90.npy')
nn_fid_s90 = np.load(f'{MOCK_DIR}/NegNeg_mean_s90.npy')
cov_nn_fid_s90 = np.load(f'{MOCK_DIR}/CovNegNeg_s90.npy')
pn_fid_s90 = np.load(f'{MOCK_DIR}/PosNeg_mean_s90.npy')
cov_pn_fid_s90 = np.load(f'{MOCK_DIR}/CovPosNeg_s90.npy')


# plot derivs
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.ticker import FixedLocator, FixedFormatter
mpl.rcParams['xtick.direction'] = 'in'
mpl.rcParams['ytick.direction'] = 'in'
mpl.rcParams['font.size'] = 8


def get_err(cov):
    return np.sqrt(np.diag(cov)) / np.sqrt(n_seeds)

def get_cumsum_err(cov):
    n = cov.shape[0]
    T = np.tril(np.ones((n, n)))
    cov_cdf = T @ cov @ T.T
    return np.sqrt(np.diag(cov_cdf)) / np.sqrt(n_seeds)


# compare cdf+ cdf-
color_map = plt.get_cmap('tab10')
plt.figure(figsize=(4, 3))
bin_edges = np.linspace(-0.008, 0.008, 11)
bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

# Errors
err_pos = get_err(cov_dpdfs_pos_s30)
err_neg = get_err(cov_dpdfs_neg_s30)
err_pdf_cdf = get_cumsum_err(cov_pdfs_fid_s30)
print(get_err(cov_dpdfs_pos_s15))

norm_pos = dpdfs_pos_fid_s30[-1]
norm_neg = dpdfs_neg_fid_s30[-1]
cdf_pdf = np.cumsum(pdfs_fid_s30)
norm_pdf = cdf_pdf[-1]

# removed interpolation
plt.plot(bin_centers, dpdfs_pos_fid_s30/norm_pos, label=r'$C^{\rm P}(\kappa)$', c=color_map(0))
plt.fill_between(bin_centers, (dpdfs_pos_fid_s30 - err_pos)/norm_pos, (dpdfs_pos_fid_s30 + err_pos)/norm_pos, color=color_map(0), alpha=0.3)

plt.plot(bin_centers, dpdfs_neg_fid_s30/norm_neg, label=r'$C^{\rm N}(\kappa)$', c=color_map(1), linestyle='-.')
# plt.fill_between(bin_centers, (dpdfs_neg_fid_s30 - err_neg)/norm_neg, (dpdfs_neg_fid_s30 + err_neg)/norm_neg, color=color_map(1), alpha=0.3)

plt.plot(bin_centers, cdf_pdf/norm_pdf, label=r'$C^\text{All Pixels}(\kappa)$', c='grey', linestyle='--')
plt.fill_between(bin_centers, (cdf_pdf - err_pdf_cdf)/norm_pdf, (cdf_pdf + err_pdf_cdf)/norm_pdf, color='grey', alpha=0.3)

plt.xlabel(r'$\kappa$')
plt.ylabel(r'$\text{CDF} (\kappa) \text{ (Normalised)}$')
plt.title(r'$\theta_G$ = 30 arcmin')
plt.legend(loc='upper left')
plt.text(0.9, 0.1, 'Gaussian', ha='right', va='bottom', transform=plt.gca().transAxes, fontsize=12)
# plt.grid(True, alpha=0.3)
plt.savefig('outfiles/compare_cdf.png', dpi=300, bbox_inches='tight')
plt.close()

# compare cdf no normal, and no pdf
plt.figure(figsize=(4, 3))
bin_edges = np.linspace(-0.005, 0.005, 11)
bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
# removed interpolation
plt.plot(bin_centers, dpdfs_pos_fid_s30, label=r'$C^{\rm P}(\kappa)$', c=color_map(0))
plt.fill_between(bin_centers, dpdfs_pos_fid_s30 - err_pos, dpdfs_pos_fid_s30 + err_pos, color=color_map(0), alpha=0.3)

plt.plot(bin_centers, dpdfs_neg_fid_s30, label=r'$C^{\rm N}(\kappa)$', c=color_map(1), linestyle='-.')
# plt.fill_between(bin_centers, dpdfs_neg_fid_s30 - err_neg, dpdfs_neg_fid_s30 + err_neg, color=color_map(1), alpha=0.3)

# Normalize PDF to match the scale of positive defects CDF
scaling_factor = norm_pos / norm_pdf
plt.plot(bin_centers, cdf_pdf * scaling_factor, label=r'$C^\text{All Pixels}(\kappa)$', c='grey', linestyle='--')
plt.fill_between(bin_centers, (cdf_pdf - err_pdf_cdf) * scaling_factor, (cdf_pdf + err_pdf_cdf) * scaling_factor, color='grey', alpha=0.3)

plt.xlabel(r'$\kappa$')
plt.ylabel(r'$\text{CDF} (\kappa)$')
plt.title(r'$\theta_G$ = 30 arcmin')
plt.legend(loc='upper left')
plt.text(0.9, 0.1, 'Gaussian', ha='right', va='bottom', transform=plt.gca().transAxes, fontsize=12)
# plt.grid(True, alpha=0.3)
plt.savefig('outfiles/compare_cdf_nonormal.png', dpi=300, bbox_inches='tight')
plt.close()
print("Errors / Values for CDF P at s30:")
for i in range(len(bin_centers)):
    print(f"Bin center: {bin_centers[i]:.4f}, Value: {dpdfs_pos_fid_s30[i]:.4f}, Error: {err_pos[i]:.4f}, Error/Value: {err_pos[i]/dpdfs_pos_fid_s30[i]:.6f}")  


# compare 1,2,4,8 NN pos same/opp 
plt.figure(figsize=(4, 3))
bin_edges1 = np.linspace(30, 300, 10)
bin_edges2 = np.linspace(30, 450, 10)
bin_edges4 = np.linspace(30, 600, 10)
bin_edges8 = np.linspace(30, 750, 10)

# removed interpolation
# Errors
e_ps1 = get_err(cov_nn_pos_same1_s30)
e_po1 = get_err(cov_nn_pos_opp1_s30)
e_ps2 = get_err(cov_nn_pos_same2_s30)
e_po2 = get_err(cov_nn_pos_opp2_s30)
e_ps4 = get_err(cov_nn_pos_same4_s30)
e_po4 = get_err(cov_nn_pos_opp4_s30)
e_ps8 = get_err(cov_nn_pos_same8_s30)
e_po8 = get_err(cov_nn_pos_opp8_s30)

plt.plot(bin_edges1, nn_pos_same1_s30, label='1NN pos-pos', c=color_map(0))
plt.fill_between(bin_edges1, nn_pos_same1_s30-e_ps1, nn_pos_same1_s30+e_ps1, color=color_map(0), alpha=0.3)
plt.plot(bin_edges1, nn_pos_opp1_s30, label='1NN pos-neg', c=color_map(0), linestyle='--')
plt.fill_between(bin_edges1, nn_pos_opp1_s30-e_po1, nn_pos_opp1_s30+e_po1, color=color_map(0), alpha=0.3)

plt.plot(bin_edges2[1:], nn_pos_same2_s30[1:], label='2NN pos-pos', c=color_map(1))
plt.fill_between(bin_edges2[1:], (nn_pos_same2_s30-e_ps2)[1:], (nn_pos_same2_s30+e_ps2)[1:], color=color_map(1), alpha=0.3)
plt.plot(bin_edges2[1:], nn_pos_opp2_s30[1:], label='2NN pos-neg', c=color_map(1), linestyle='--')
plt.fill_between(bin_edges2[1:], (nn_pos_opp2_s30-e_po2)[1:], (nn_pos_opp2_s30+e_po2)[1:], color=color_map(1), alpha=0.3)

plt.plot(bin_edges4[2:], nn_pos_same4_s30[2:], label='4NN pos-pos', c=color_map(2))
plt.fill_between(bin_edges4[2:], (nn_pos_same4_s30-e_ps4)[2:], (nn_pos_same4_s30+e_ps4)[2:], color=color_map(2), alpha=0.3)
plt.plot(bin_edges4[2:], nn_pos_opp4_s30[2:], label='4NN pos-neg', c=color_map(2), linestyle='--')
plt.fill_between(bin_edges4[2:], (nn_pos_opp4_s30-e_po4)[2:], (nn_pos_opp4_s30+e_po4)[2:], color=color_map(2), alpha=0.3)

plt.plot(bin_edges8[3:], nn_pos_same8_s30[3:], label='8NN pos-pos', c=color_map(3))
plt.fill_between(bin_edges8[3:], (nn_pos_same8_s30-e_ps8)[3:], (nn_pos_same8_s30+e_ps8)[3:], color=color_map(3), alpha=0.3)
plt.plot(bin_edges8[3:], nn_pos_opp8_s30[3:], label='8NN pos-neg', c=color_map(3), linestyle='--')
plt.fill_between(bin_edges8[3:], (nn_pos_opp8_s30-e_po8)[3:], (nn_pos_opp8_s30+e_po8)[3:], color=color_map(3), alpha=0.3)

plt.xlabel(r'$\theta$ [arcmin]')
plt.ylabel(r'$\text{CDF} (\theta)$')
plt.title(r'$\theta_G$ = 30 arcmin')
plt.legend()
# plt.grid(True, alpha=0.3)
plt.savefig('outfiles/compare_knn_cdf.png', dpi=300, bbox_inches='tight')
plt.close()
# print errors / values
print("Errors / Values for 1NN P-P at s30:")
for i in range(len(bin_edges1)):
    print(f"Bin edge: {bin_edges1[i]:.1f}, Value: {nn_pos_same1_s30[i]:.4f}, Error: {e_ps1[i]:.4f}, Error/Value: {e_ps1[i]/nn_pos_same1_s30[i]:.6f}")


# compare ++, +-, --, -+ for 2NN at s90
# plt.figure(figsize=(4.8, 3))
# bin_edges2 = np.linspace(30, 1125, 10)

# # Create interpolation functions for 2NN types
# interp_2nn_pp = interp1d(bin_edges2, nn_pos_same2_s90, kind='quadratic', fill_value=np.nan)
# interp_2nn_pn = interp1d(bin_edges2, nn_pos_opp2_s90, kind='quadratic', fill_value=np.nan)
# interp_2nn_nn = interp1d(bin_edges2, nn_neg_same2_s90, kind='quadratic', fill_value=np.nan)
# interp_2nn_np = interp1d(bin_edges2, nn_neg_opp2_s90, kind='quadratic', fill_value=np.nan)

# # Create 100 bins for plotting
# bin_edges2_fine = np.linspace(bin_edges2[0], bin_edges2[-1], 100)

# plt.plot(bin_edges2_fine[6:], interp_2nn_pp(bin_edges2_fine[6:]), label='2NN P-P', c=color_map(0))
# plt.plot(bin_edges2_fine[6:], interp_2nn_pn(bin_edges2_fine[6:]), label='2NN P-N', c=color_map(0), linestyle='--')
# plt.plot(bin_edges2_fine[6:], interp_2nn_nn(bin_edges2_fine[6:]), label='2NN N-N', c=color_map(1))
# plt.plot(bin_edges2_fine[6:], interp_2nn_np(bin_edges2_fine[6:]), label='2NN N-P', c=color_map(1), linestyle='--')
# plt.xlabel('Distance [arcmin]')
# plt.ylabel('Cumulative Number Distribution')
# # plt.title('Comparison of 2NN types at 90 arcmin')
# plt.legend(loc='upper left')
# plt.text(0.9, 0.1, 'Full-sky', ha='right', va='bottom', transform=plt.gca().transAxes, fontsize=12)
# # plt.grid(True, alpha=0.3)
# plt.savefig('outfiles/compare_2nn_types_cdf.png', dpi=300)
# plt.close()


# dpdf, pdf cov

# all_and_pdf_cov = np.load(f'{MOCK_DIR}/Defects_PDF_all_scales_cov.npy')
# # take first 20 and last 10
# cdf_cov = all_and_pdf_cov[np.ix_(list(range(20)) + list(range(all_and_pdf_cov.shape[0]-10, all_and_pdf_cov.shape[0])), list(range(20)) + list(range(all_and_pdf_cov.shape[0]-10, all_and_pdf_cov.shape[0])))]

# plt.figure(figsize=(4, 3))
# # normalize covariance to correlation matrix
# d = np.sqrt(np.diag(cdf_cov))
# corr = cdf_cov / np.outer(d, d)
# plt.imshow(corr, cmap='RdBu', vmin=-1, vmax=1)
# plt.colorbar(label='Correlation Coefficient')
# # plt.title('Correlation Matrix (DCDF + PDF)')
# # add lines to separate different parts
# plt.axvline(x=9.5, color='black', linestyle='--', linewidth=1)
# plt.axhline(y=9.5, color='black', linestyle='--', linewidth=1)
# plt.axvline(x=19.5, color='black', linestyle='--', linewidth=1)
# plt.axhline(y=19.5, color='black', linestyle='--', linewidth=1)

# # ticks
# plt.xticks([5, 15, 25], [r'DCDF+', r'DCDF-', r'PDF'])
# plt.yticks([5, 15, 25], [r'DCDF+', r'DCDF-', r'PDF'])
# plt.tick_params(axis='both', which='both', length=0)
# plt.savefig('outfiles/corr_matrix_cdf.png', dpi=300, bbox_inches='tight')
# plt.close()

# # compare pp/nn/pn xi at s15, s30, s90
# scales = ['s15', 's30', 's90']
# pp_fids = [pp_fid_s15, pp_fid_s30, pp_fid_s90]
# cov_pp_fids = [cov_pp_fid_s15, cov_pp_fid_s30, cov_pp_fid_s90]
# nn_fids = [nn_fid_s15, nn_fid_s30, nn_fid_s90]
# cov_nn_fids = [cov_nn_fid_s15, cov_nn_fid_s30, cov_nn_fid_s90]
# pn_fids = [pn_fid_s15, pn_fid_s30, pn_fid_s90]
# cov_pn_fids = [cov_pn_fid_s15, cov_pn_fid_s30, cov_pn_fid_s90]
# bins = [np.linspace(15, 600, 30), np.linspace(30, 600, 30), np.linspace(90, 600, 30)]
# fig, axes = plt.subplots(1, 3, figsize=(9, 3), sharey=True)
# for i in range(3):
#     ax = axes[i]
#     bin_centers = bins[i]
#     # Standard Error of Mean: sigma / sqrt(N)
#     err_pp = np.sqrt(np.diag(cov_pp_fids[i])) / np.sqrt(n_seeds)
#     err_nn = np.sqrt(np.diag(cov_nn_fids[i])) / np.sqrt(n_seeds)
#     err_pn = np.sqrt(np.diag(cov_pn_fids[i])) / np.sqrt(n_seeds)

#     ax.plot(bin_centers, pp_fids[i], label='P-P', color=color_map(0))
#     ax.fill_between(bin_centers, pp_fids[i] - err_pp, pp_fids[i] + err_pp, color=color_map(0), alpha=0.3)
#     ax.plot(bin_centers, nn_fids[i], label='N-N', color=color_map(1))
#     ax.fill_between(bin_centers, nn_fids[i] - err_nn, nn_fids[i] + err_nn, color=color_map(1), alpha=0.3)
#     ax.plot(bin_centers, pn_fids[i], label='P-N', color=color_map(2))
#     ax.fill_between(bin_centers, pn_fids[i] - err_pn, pn_fids[i] + err_pn, color=color_map(2), alpha=0.3)
#     ax.axhline(0, color='grey', linestyle='--', linewidth=1)
#     ax.axhline(-1, color='red', linestyle='-.', linewidth=1)
#     ax.set_xlabel(r'$\theta$ [arcmin]')
#     ax.set_xscale('log')
#     ax.xaxis.set_major_locator(FixedLocator([100, 500]))
#     ax.xaxis.set_major_formatter(FixedFormatter(['100', '500']))
#     ax.xaxis.set_minor_locator(FixedLocator([]))
#     if i == 0:
#         ax.set_ylabel(r'$\xi(\theta)$')
#     if i == 0:
#         ax.set_title(r'$\theta_G$ = 15 arcmin')
#     elif i == 1:
#         ax.set_title(r'$\theta_G$ = 30 arcmin')
#     else:
#         ax.set_title(r'$\theta_G$ = 90 arcmin')
# plt.subplots_adjust(wspace=0)
# # plt.tight_layout()
# handles, labels = axes[0].get_legend_handles_labels()
# fig.legend(handles, labels, loc='upper center', ncol=3, bbox_to_anchor=(0.5, 1.05))
# plt.savefig('outfiles/compare_xi_pp_nn_pn.png', dpi=300, bbox_inches='tight')
# plt.close()

# compare pp/nn/pn xi at s30 only
plt.figure(figsize=(4, 3))
bin_centers = np.linspace(30, 200, 10)

err_pp_s30 = np.sqrt(np.diag(cov_pp_fid_s30)) / np.sqrt(n_seeds)
err_nn_s30 = np.sqrt(np.diag(cov_nn_fid_s30)) / np.sqrt(n_seeds)
err_pn_s30 = np.sqrt(np.diag(cov_pn_fid_s30)) / np.sqrt(n_seeds)

plt.plot(bin_centers, pp_fid_s30, label='pos-pos', color=color_map(0))
plt.fill_between(bin_centers, pp_fid_s30 - err_pp_s30, pp_fid_s30 + err_pp_s30, color=color_map(0), alpha=0.3)
plt.plot(bin_centers, nn_fid_s30, label='neg-neg', color=color_map(1))
plt.fill_between(bin_centers, nn_fid_s30 - err_nn_s30, nn_fid_s30 + err_nn_s30, color=color_map(1), alpha=0.3)
plt.plot(bin_centers, pn_fid_s30, label='pos-neg', color=color_map(2))
plt.fill_between(bin_centers, pn_fid_s30 - err_pn_s30, pn_fid_s30 + err_pn_s30, color=color_map(2), alpha=0.3)
plt.axhline(0, color='grey', linestyle='--', linewidth=1)
# plt.axhline(-1, color='red', linestyle='-.', linewidth=1)
plt.xlabel(r'$\theta$ [arcmin]')
plt.xscale('log')
plt.ylabel(r'$g(\theta)$')
plt.title(r'$\theta_G$ = 30 arcmin')
plt.subplots_adjust(wspace=0)
plt.legend()
plt.savefig('outfiles/compare_xi_pp_nn_pn.png', dpi=300, bbox_inches='tight')
plt.close()

print("Errors / Values for P-N at s30:")
for i in range(len(bin_centers)):
    print(f"Bin center: {bin_centers[i]:.1f}, Value: {pn_fid_s30[i]:.4f}, Error: {err_pn_s30[i]:.4f}, Error/Value: {err_pn_s30[i]/pn_fid_s30[i]:.6f}")
print("Errors / Values for N-N at s30:")
for i in range(len(bin_centers)):   
    print(f"Bin center: {bin_centers[i]:.1f}, Value: {nn_fid_s30[i]:.4f}, Error: {err_nn_s30[i]:.4f}, Error/Value: {err_nn_s30[i]/nn_fid_s30[i]:.6f}")  
print("Errors / Values for P-P at s30:")
for i in range(len(bin_centers)):
    print(f"Bin center: {bin_centers[i]:.1f}, Value: {pp_fid_s30[i]:.4f}, Error: {err_pp_s30[i]:.4f}, Error/Value: {err_pp_s30[i]/pp_fid_s30[i]:.6f}")