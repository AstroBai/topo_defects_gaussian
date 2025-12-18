import numpy as np

DATA_DIR = "/cephfs/jbai/DefectFisherGaussian"
fid = {'Om': 0.3,
       'sigma8': 0.8}
Ob = 0.049
h = 0.67
w = -1.0
# surrounding cosmology params, e.g. Om +/- delta_Om
delta_params = {'Om': 0.02,
                'sigma8': 0.02}
n_seeds = 60

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
print(f"After reshape - nn_combined_s15: {nn_combined_s15.shape}")






# compute derivatives
deriv_pdfs_s15 = []
deriv_pdfs_s30 = []
deriv_pdfs_s90 = []

deriv_dpdfs_pos_s15 = []
deriv_dpdfs_pos_s30 = []
deriv_dpdfs_pos_s90 = []

deriv_dpdfs_neg_s15 = []
deriv_dpdfs_neg_s30 = []
deriv_dpdfs_neg_s90 = []

deriv_dpdfs_combined_s15 = []
deriv_dpdfs_combined_s30 = []
deriv_dpdfs_combined_s90 = []

deriv_nns_combined_s15 = []
deriv_nns_combined_s30 = []
deriv_nns_combined_s90 = []
deriv_dpdf_nn_combined_s15 = []
deriv_dpdf_nn_combined_s30 = []
deriv_dpdf_nn_combined_s90 = []

deriv_kk_corrs_s15 = []
deriv_kk_corrs_s30 = []
deriv_kk_corrs_s90 = []
for i in range(0, len(params), 2):
    # pdf derivatives
    hist_plus = pdfs_s15[:,i+1,:]
    hist_minus = pdfs_s15[:,i,:]
    deriv_pdf = (hist_plus - hist_minus) / (2 * delta_params[list(fid.keys())[i//2]])
    deriv_pdfs_s15.append(deriv_pdf)
    hist_plus = pdfs_s30[:,i+1,:]
    hist_minus = pdfs_s30[:,i,:]
    deriv_pdf = (hist_plus - hist_minus) / (2 * delta_params[list(fid.keys())[i//2]])
    deriv_pdfs_s30.append(deriv_pdf)
    hist_plus = pdfs_s90[:,i+1,:]
    hist_minus = pdfs_s90[:,i,:]
    deriv_pdf = (hist_plus - hist_minus) / (2 * delta_params[list(fid.keys())[i//2]])
    deriv_pdfs_s90.append(deriv_pdf)


    # defect pdf derivatives
    hist_pos_plus = dpdfs_pos_s15[:,i+1,:]
    hist_pos_minus = dpdfs_pos_s15[:,i,:]
    deriv_dpdf_pos = (hist_pos_plus - hist_pos_minus) / (2 * delta_params[list(fid.keys())[i//2]])
    deriv_dpdfs_pos_s15.append(deriv_dpdf_pos)
    hist_pos_plus = dpdfs_pos_s30[:,i+1,:]
    hist_pos_minus = dpdfs_pos_s30[:,i,:]
    deriv_dpdf_pos = (hist_pos_plus - hist_pos_minus) / (2 * delta_params[list(fid.keys())[i//2]])
    deriv_dpdfs_pos_s30.append(deriv_dpdf_pos)
    hist_pos_plus = dpdfs_pos_s90[:,i+1,:]
    hist_pos_minus = dpdfs_pos_s90[:,i,:]
    deriv_dpdf_pos = (hist_pos_plus - hist_pos_minus) / (2 * delta_params[list(fid.keys())[i//2]])
    deriv_dpdfs_pos_s90.append(deriv_dpdf_pos)

    hist_neg_plus = dpdfs_neg_s15[:,i+1,:]
    hist_neg_minus = dpdfs_neg_s15[:,i,:]
    deriv_dpdf_neg = (hist_neg_plus - hist_neg_minus) / (2 * delta_params[list(fid.keys())[i//2]])
    deriv_dpdfs_neg_s15.append(deriv_dpdf_neg)
    hist_neg_plus = dpdfs_neg_s30[:,i+1,:]
    hist_neg_minus = dpdfs_neg_s30[:,i,:]
    deriv_dpdf_neg = (hist_neg_plus - hist_neg_minus) / (2 * delta_params[list(fid.keys())[i//2]])
    deriv_dpdfs_neg_s30.append(deriv_dpdf_neg)
    hist_neg_plus = dpdfs_neg_s90[:,i+1,:]
    hist_neg_minus = dpdfs_neg_s90[:,i,:]
    deriv_dpdf_neg = (hist_neg_plus - hist_neg_minus) / (2 * delta_params[list(fid.keys())[i//2]])
    deriv_dpdfs_neg_s90.append(deriv_dpdf_neg)

    # kk correlation derivatives
    xi_kk_plus = kk_corrs_s15[:,i+1,:]
    xi_kk_minus = kk_corrs_s15[:,i,:]
    deriv_kk_corr = (xi_kk_plus - xi_kk_minus) / (2 * delta_params[list(fid.keys())[i//2]])
    deriv_kk_corrs_s15.append(deriv_kk_corr)
    xi_kk_plus = kk_corrs_s30[:,i+1,:]
    xi_kk_minus = kk_corrs_s30[:,i,:]
    deriv_kk_corr = (xi_kk_plus - xi_kk_minus) / (2 * delta_params[list(fid.keys())[i//2]])
    deriv_kk_corrs_s30.append(deriv_kk_corr)
    xi_kk_plus = kk_corrs_s90[:,i+1,:]
    xi_kk_minus = kk_corrs_s90[:,i,:]
    deriv_kk_corr = (xi_kk_plus - xi_kk_minus) / (2 * delta_params[list(fid.keys())[i//2]])
    deriv_kk_corrs_s90.append(deriv_kk_corr)
    
    nn_combined_plus = nn_combined_s15[:,i+1,:]
    nn_combined_minus = nn_combined_s15[:,i,:]
    deriv_nn_combined = (nn_combined_plus - nn_combined_minus) / (2 * delta_params[list(fid.keys())[i//2]]) 
    deriv_nns_combined_s15.append(deriv_nn_combined)
    nn_combined_plus = nn_combined_s30[:,i+1,:]
    nn_combined_minus = nn_combined_s30[:,i,:]
    deriv_nn_combined = (nn_combined_plus - nn_combined_minus) / (2 * delta_params[list(fid.keys())[i//2]]) 
    deriv_nns_combined_s30.append(deriv_nn_combined)
    nn_combined_plus = nn_combined_s90[:,i+1,:]
    nn_combined_minus = nn_combined_s90[:,i,:]
    deriv_nn_combined = (nn_combined_plus - nn_combined_minus) / (2 * delta_params[list(fid.keys())[i//2]]) 
    deriv_nns_combined_s90.append(deriv_nn_combined)




deriv_pdfs_s15 = np.array(deriv_pdfs_s15)
deriv_pdfs_s30 = np.array(deriv_pdfs_s30)
deriv_pdfs_s90 = np.array(deriv_pdfs_s90)

deriv_dpdfs_pos_s15 = np.array(deriv_dpdfs_pos_s15)
deriv_dpdfs_pos_s30 = np.array(deriv_dpdfs_pos_s30)
deriv_dpdfs_pos_s90 = np.array(deriv_dpdfs_pos_s90)

deriv_dpdfs_neg_s15 = np.array(deriv_dpdfs_neg_s15)
deriv_dpdfs_neg_s30 = np.array(deriv_dpdfs_neg_s30)
deriv_dpdfs_neg_s90 = np.array(deriv_dpdfs_neg_s90)
deriv_dpdfs_combined_s15 = np.hstack((deriv_dpdfs_pos_s15, deriv_dpdfs_neg_s15))
deriv_dpdfs_combined_s30 = np.hstack((deriv_dpdfs_pos_s30, deriv_dpdfs_neg_s30))
deriv_dpdfs_combined_s90 = np.hstack((deriv_dpdfs_pos_s90, deriv_dpdfs_neg_s90))
deriv_dpdfs_combined_all_scales = np.hstack((deriv_dpdfs_combined_s15, deriv_dpdfs_combined_s30, deriv_dpdfs_combined_s90))

deriv_kk_corrs_s15 = np.array(deriv_kk_corrs_s15)
deriv_kk_corrs_s30 = np.array(deriv_kk_corrs_s30)
deriv_kk_corrs_s90 = np.array(deriv_kk_corrs_s90)

deriv_nns_combined_s15 = np.array(deriv_nns_combined_s15)
deriv_nns_combined_s30 = np.array(deriv_nns_combined_s30)
deriv_nns_combined_s90 = np.array(deriv_nns_combined_s90)


# create mean derivs and their covs 30 as one group
print(deriv_pdfs_s15.shape)
# deriv_pdfs_s15 shape: (2, 30, 5) - (params, seeds, bins)
# Reshape to (30, 2*5) for covariance calculation
deriv_pdfs_s15_reshaped = deriv_pdfs_s15.transpose(1, 0, 2).reshape(30, -1)
deriv_pdfs_s15_mean = np.mean(deriv_pdfs_s15_reshaped, axis=0)
deriv_pdfs_s15_cov = np.cov(deriv_pdfs_s15_reshaped, rowvar=False)

deriv_pdfs_s30_reshaped = deriv_pdfs_s30.transpose(1, 0, 2).reshape(30, -1)
deriv_pdfs_s30_mean = np.mean(deriv_pdfs_s30_reshaped, axis=0)
deriv_pdfs_s30_cov = np.cov(deriv_pdfs_s30_reshaped, rowvar=False)

deriv_pdfs_s90_reshaped = deriv_pdfs_s90.transpose(1, 0, 2).reshape(30, -1)
deriv_pdfs_s90_mean = np.mean(deriv_pdfs_s90_reshaped, axis=0)
deriv_pdfs_s90_cov = np.cov(deriv_pdfs_s90_reshaped, rowvar=False)

deriv_dpdfs_combined_s15_reshaped = deriv_dpdfs_combined_s15.transpose(1, 0, 2).reshape(30, -1)
deriv_dpdfs_combined_s15_mean = np.mean(deriv_dpdfs_combined_s15_reshaped, axis=0)
deriv_dpdfs_combined_s15_cov = np.cov(deriv_dpdfs_combined_s15_reshaped, rowvar=False)

deriv_dpdfs_combined_s30_reshaped = deriv_dpdfs_combined_s30.transpose(1, 0, 2).reshape(30, -1)
deriv_dpdfs_combined_s30_mean = np.mean(deriv_dpdfs_combined_s30_reshaped, axis=0)
deriv_dpdfs_combined_s30_cov = np.cov(deriv_dpdfs_combined_s30_reshaped, rowvar=False)

deriv_dpdfs_combined_s90_reshaped = deriv_dpdfs_combined_s90.transpose(1, 0, 2).reshape(30, -1)
deriv_dpdfs_combined_s90_mean = np.mean(deriv_dpdfs_combined_s90_reshaped, axis=0)
deriv_dpdfs_combined_s90_cov = np.cov(deriv_dpdfs_combined_s90_reshaped, rowvar=False)

deriv_dpdfs_combined_all_scales_reshaped = deriv_dpdfs_combined_all_scales.transpose(1, 0, 2).reshape(30, -1)
deriv_dpdfs_combined_all_scales_mean = np.mean(deriv_dpdfs_combined_all_scales_reshaped, axis=0)
deriv_dpdfs_combined_all_scales_cov = np.cov(deriv_dpdfs_combined_all_scales_reshaped, rowvar=False)

deriv_kk_corrs_s15_reshaped = deriv_kk_corrs_s15.transpose(1, 0, 2).reshape(30, -1)
deriv_kk_corrs_s15_mean = np.mean(deriv_kk_corrs_s15_reshaped, axis=0)
deriv_kk_corrs_s15_cov = np.cov(deriv_kk_corrs_s15_reshaped, rowvar=False)

deriv_kk_corrs_s30_reshaped = deriv_kk_corrs_s30.transpose(1, 0, 2).reshape(30, -1)
deriv_kk_corrs_s30_mean = np.mean(deriv_kk_corrs_s30_reshaped, axis=0)
deriv_kk_corrs_s30_cov = np.cov(deriv_kk_corrs_s30_reshaped, rowvar=False)

deriv_kk_corrs_s90_reshaped = deriv_kk_corrs_s90.transpose(1, 0, 2).reshape(30, -1)
deriv_kk_corrs_s90_mean = np.mean(deriv_kk_corrs_s90_reshaped, axis=0)
deriv_kk_corrs_s90_cov = np.cov(deriv_kk_corrs_s90_reshaped, rowvar=False)

deriv_nns_combined_s15_reshaped = deriv_nns_combined_s15.transpose(1, 0, 2).reshape(30, -1)
deriv_nns_combined_s15_mean = np.mean(deriv_nns_combined_s15_reshaped, axis=0)
deriv_nns_combined_s15_cov = np.cov(deriv_nns_combined_s15_reshaped, rowvar=False)

deriv_nns_combined_s30_reshaped = deriv_nns_combined_s30.transpose(1, 0, 2).reshape(30, -1)
deriv_nns_combined_s30_mean = np.mean(deriv_nns_combined_s30_reshaped, axis=0)
deriv_nns_combined_s30_cov = np.cov(deriv_nns_combined_s30_reshaped, rowvar=False)

deriv_nns_combined_s90_reshaped = deriv_nns_combined_s90.transpose(1, 0, 2).reshape(30, -1)
deriv_nns_combined_s90_mean = np.mean(deriv_nns_combined_s90_reshaped, axis=0)
deriv_nns_combined_s90_cov = np.cov(deriv_nns_combined_s90_reshaped, rowvar=False)



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

# plot derivs
import matplotlib.pyplot as plt
import matplotlib as mpl
mpl.rcParams['xtick.direction'] = 'in'
mpl.rcParams['ytick.direction'] = 'in'

N_fid = 500
N_deriv = n_seeds

# pdfs, 1x3, share y axis
fig, axes = plt.subplots(1, 3, figsize=(6, 2), sharey=True)

# Subplot 1: s15
ax = axes[0]
bin_edges = np.array([-0.01, -0.005, 0.0, 0.005, 0.01])
bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
print(f"Bin centers for s15: {bin_centers}")
print(deriv_pdfs_s15_mean)
ax.plot(bin_centers, pdfs_fid_s15, c='black')
ax.errorbar(bin_centers, pdfs_fid_s15, yerr=np.sqrt(np.diag(cov_pdfs_fid_s15))/np.sqrt(N_fid), fmt='o', label='Fiducial', c='black')
ax.errorbar(bin_centers, deriv_pdfs_s15_mean[0:4], yerr=np.sqrt(np.diag(deriv_pdfs_s15_cov[0:4, 0:4]))/np.sqrt(N_deriv), fmt='o', label=r'Derv. w.r.t $\Omega_m$')
ax.errorbar(bin_centers, deriv_pdfs_s15_mean[4:8], yerr=np.sqrt(np.diag(deriv_pdfs_s15_cov[4:8, 4:8]))/np.sqrt(N_deriv), fmt='o', label=r'Derv. w.r.t $\sigma_8$')
ax.fill_between(bin_centers, deriv_pdfs_s15_mean[0:4] - np.sqrt(np.diag(deriv_pdfs_s15_cov[0:4, 0:4]))/np.sqrt(N_deriv), deriv_pdfs_s15_mean[0:4] + np.sqrt(np.diag(deriv_pdfs_s15_cov[0:4, 0:4]))/np.sqrt(N_deriv), alpha=0.3)
ax.fill_between(bin_centers, deriv_pdfs_s15_mean[4:8] - np.sqrt(np.diag(deriv_pdfs_s15_cov[4:8, 4:8]))/np.sqrt(N_deriv), deriv_pdfs_s15_mean[4:8] + np.sqrt(np.diag(deriv_pdfs_s15_cov[4:8, 4:8]))/np.sqrt(N_deriv), alpha=0.3)
ax.set_title(r'$\theta_G$ = 15 arcmin')
ax.set_xlabel(r'$\kappa$')
ax.set_ylabel('Probability Density')
ax.set_xticks([-0.005, 0.005])
ax.grid(True, alpha=0.3)


# Subplot 2: s30
ax = axes[1]
bin_edges = np.array([-0.008, -0.004, 0.0, 0.004, 0.008])
bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
ax.plot(bin_centers, pdfs_fid_s30, c='black')
ax.errorbar(bin_centers, pdfs_fid_s30, yerr=np.sqrt(np.diag(cov_pdfs_fid_s30))/np.sqrt(N_fid), fmt='o', label='Fiducial', c='black')
ax.errorbar(bin_centers, deriv_pdfs_s30_mean[0:4], yerr=np.sqrt(np.diag(deriv_pdfs_s30_cov[0:4, 0:4]))/np.sqrt(N_deriv), fmt='o', label=r'Derv. w.r.t $\Omega_m$')
ax.errorbar(bin_centers, deriv_pdfs_s30_mean[4:8], yerr=np.sqrt(np.diag(deriv_pdfs_s30_cov[4:8, 4:8]))/np.sqrt(N_deriv), fmt='o', label=r'Derv. w.r.t $\sigma_8$')
ax.fill_between(bin_centers, deriv_pdfs_s30_mean[0:4] - np.sqrt(np.diag(deriv_pdfs_s30_cov[0:4, 0:4]))/np.sqrt(N_deriv), deriv_pdfs_s30_mean[0:4] + np.sqrt(np.diag(deriv_pdfs_s30_cov[0:4, 0:4]))/np.sqrt(N_deriv), alpha=0.3)
ax.fill_between(bin_centers, deriv_pdfs_s30_mean[4:8] - np.sqrt(np.diag(deriv_pdfs_s30_cov[4:8, 4:8]))/np.sqrt(N_deriv), deriv_pdfs_s30_mean[4:8] + np.sqrt(np.diag(deriv_pdfs_s30_cov[4:8, 4:8]))/np.sqrt(N_deriv), alpha=0.3)
ax.set_title(r'$\theta_G$ = 30 arcmin')
ax.set_xlabel(r'$\kappa$')
ax.set_xticks([-0.004, 0.004])
ax.grid(True, alpha=0.3)


# Subplot 3: s90
ax = axes[2]
bin_edges = np.array([-0.005, -0.0025, 0.0, 0.0025, 0.005])
bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
ax.plot(bin_centers, pdfs_fid_s90, c='black')
ax.errorbar(bin_centers, pdfs_fid_s90, yerr=np.sqrt(np.diag(cov_pdfs_fid_s90))/np.sqrt(N_fid), fmt='o', label='Fiducial', c='black')
ax.errorbar(bin_centers, deriv_pdfs_s90_mean[0:4], yerr=np.sqrt(np.diag(deriv_pdfs_s90_cov[0:4, 0:4]))/np.sqrt(N_deriv), fmt='o', label=r'Derv. w.r.t $\Omega_m$')
ax.errorbar(bin_centers, deriv_pdfs_s90_mean[4:8], yerr=np.sqrt(np.diag(deriv_pdfs_s90_cov[4:8, 4:8]))/np.sqrt(N_deriv), fmt='o', label=r'Derv. w.r.t $\sigma_8$')
ax.fill_between(bin_centers, deriv_pdfs_s90_mean[0:4] - np.sqrt(np.diag(deriv_pdfs_s90_cov[0:4, 0:4]))/np.sqrt(N_deriv), deriv_pdfs_s90_mean[0:4] + np.sqrt(np.diag(deriv_pdfs_s90_cov[0:4, 0:4]))/np.sqrt(N_deriv), alpha=0.3)
ax.fill_between(bin_centers, deriv_pdfs_s90_mean[4:8] - np.sqrt(np.diag(deriv_pdfs_s90_cov[4:8, 4:8]))/np.sqrt(N_deriv), deriv_pdfs_s90_mean[4:8] + np.sqrt(np.diag(deriv_pdfs_s90_cov[4:8, 4:8]))/np.sqrt(N_deriv), alpha=0.3)
ax.set_title(r'$\theta_G$ = 90 arcmin')
ax.set_xlabel(r'$\kappa$')
ax.set_xticks([-0.0025, 0.0025])
ax.grid(True, alpha=0.3)


plt.legend(loc=(1.05, 0.25))
plt.subplots_adjust(wspace=0)
plt.savefig('outfiles/derivs_pdf.png', dpi=300, bbox_inches='tight')
plt.close()

# dpdfs
fig, axes = plt.subplots(2, 3, figsize=(6, 3), sharey='col') # 2x3 because pos and neg
# Subplot 1: s15 pos
ax = axes[0,0]
bin_edges = np.array([-0.01, -0.005, 0.0, 0.005, 0.01])
bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
ax.plot(bin_centers, dpdfs_pos_fid_s15, c='black')
ax.errorbar(bin_centers, dpdfs_pos_fid_s15, yerr=np.sqrt(np.diag(cov_dpdfs_pos_s15))/np.sqrt(N_fid), fmt='o', label='Fiducial', c='black')
ax.errorbar(bin_centers, deriv_dpdfs_combined_s15_mean[0:4], yerr=np.sqrt(np.diag(deriv_dpdfs_combined_s15_cov[0:4, 0:4]))/np.sqrt(N_deriv), fmt='o', label=r'Derv. w.r.t $\Omega_m$')
ax.errorbar(bin_centers, deriv_dpdfs_combined_s15_mean[8:12], yerr=np.sqrt(np.diag(deriv_dpdfs_combined_s15_cov[8:12, 8:12]))/np.sqrt(N_deriv), fmt='o', label=r'Derv. w.r.t $\sigma_8$')
ax.fill_between(bin_centers, deriv_dpdfs_combined_s15_mean[0:4] - np.sqrt(np.diag(deriv_dpdfs_combined_s15_cov[0:4, 0:4]))/np.sqrt(N_deriv), deriv_dpdfs_combined_s15_mean[0:4] + np.sqrt(np.diag(deriv_dpdfs_combined_s15_cov[0:4, 0:4]))/np.sqrt(N_deriv), alpha=0.3)
ax.fill_between(bin_centers, deriv_dpdfs_combined_s15_mean[8:12] - np.sqrt(np.diag(deriv_dpdfs_combined_s15_cov[8:12, 8:12]))/np.sqrt(N_deriv), deriv_dpdfs_combined_s15_mean[8:12] + np.sqrt(np.diag(deriv_dpdfs_combined_s15_cov[8:12, 8:12]))/np.sqrt(N_deriv), alpha=0.3)
ax.set_title(r'$\theta_G$ = 15 arcmin')
ax.set_xlabel(r'$\kappa$')
ax.set_ylabel(r'$C^p(\kappa)$')
ax.set_xticks([-0.005, 0.005])
ax.set_yticks([0, 500, 1000])
ax.grid(True, alpha=0.3)

# Subplot 2: s30 pos
ax = axes[0,1]
bin_edges = np.array([-0.008, -0.004, 0.0, 0.004, 0.008])
bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
ax.plot(bin_centers, dpdfs_pos_fid_s30, c='black')
ax.errorbar(bin_centers, dpdfs_pos_fid_s30, yerr=np.sqrt(np.diag(cov_dpdfs_pos_s30))/np.sqrt(N_fid), fmt='o', label='Fiducial', c='black')   
ax.errorbar(bin_centers, deriv_dpdfs_combined_s30_mean[0:4], yerr=np.sqrt(np.diag(deriv_dpdfs_combined_s30_cov[0:4, 0:4]))/np.sqrt(N_deriv), fmt='o', label=r'Derv. w.r.t $\Omega_m$')
ax.errorbar(bin_centers, deriv_dpdfs_combined_s30_mean[8:12], yerr=np.sqrt(np.diag(deriv_dpdfs_combined_s30_cov[8:12, 8:12]))/np.sqrt(N_deriv), fmt='o', label=r'Derv. w.r.t $\sigma_8$')
ax.fill_between(bin_centers, deriv_dpdfs_combined_s30_mean[0:4] - np.sqrt(np.diag(deriv_dpdfs_combined_s30_cov[0:4, 0:4]))/np.sqrt(N_deriv), deriv_dpdfs_combined_s30_mean[0:4] + np.sqrt(np.diag(deriv_dpdfs_combined_s30_cov[0:4, 0:4]))/np.sqrt(N_deriv), alpha=0.3)
ax.fill_between(bin_centers, deriv_dpdfs_combined_s30_mean[8:12] - np.sqrt(np.diag(deriv_dpdfs_combined_s30_cov[8:12, 8:12]))/np.sqrt(N_deriv), deriv_dpdfs_combined_s30_mean[8:12] + np.sqrt(np.diag(deriv_dpdfs_combined_s30_cov[8:12, 8:12]))/np.sqrt(N_deriv), alpha=0.3)
ax.set_title(r'$\theta_G$ = 30 arcmin')
ax.set_xlabel(r'$\kappa$')
ax.set_xticks([-0.004, 0.004])
ax.set_yticks([0, 150, 300])
ax.grid(True, alpha=0.3)

# Subplot 3: s90 pos
ax = axes[0,2]
bin_edges = np.array([-0.005, -0.0025, 0.0, 0.0025, 0.005])
bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
ax.plot(bin_centers, dpdfs_pos_fid_s90, c='black')
ax.errorbar(bin_centers, dpdfs_pos_fid_s90, yerr=np.sqrt(np.diag(cov_dpdfs_pos_s90))/np.sqrt(N_fid), fmt='o', label='Fiducial', c='black')   
ax.errorbar(bin_centers, deriv_dpdfs_combined_s90_mean[0:4], yerr=np.sqrt(np.diag(deriv_dpdfs_combined_s90_cov[0:4, 0:4]))/np.sqrt(N_deriv), fmt='o', label=r'Derv. w.r.t $\Omega_m$')
ax.errorbar(bin_centers, deriv_dpdfs_combined_s90_mean[8:12], yerr=np.sqrt(np.diag(deriv_dpdfs_combined_s90_cov[8:12, 8:12]))/np.sqrt(N_deriv), fmt='o', label=r'Derv. w.r.t $\sigma_8$')
ax.fill_between(bin_centers, deriv_dpdfs_combined_s90_mean[0:4] - np.sqrt(np.diag(deriv_dpdfs_combined_s90_cov[0:4, 0:4]))/np.sqrt(N_deriv), deriv_dpdfs_combined_s90_mean[0:4] + np.sqrt(np.diag(deriv_dpdfs_combined_s90_cov[0:4, 0:4]))/np.sqrt(N_deriv), alpha=0.3)
ax.fill_between(bin_centers, deriv_dpdfs_combined_s90_mean[8:12] - np.sqrt(np.diag(deriv_dpdfs_combined_s90_cov[8:12, 8:12]))/np.sqrt(N_deriv), deriv_dpdfs_combined_s90_mean[8:12] + np.sqrt(np.diag(deriv_dpdfs_combined_s90_cov[8:12, 8:12]))/np.sqrt(N_deriv), alpha=0.3)
ax.set_title(r'$\theta_G$ = 90 arcmin')
ax.set_xlabel(r'$\kappa$')
ax.set_xticks([-0.0025, 0.0, 0.0025])
ax.grid(True, alpha=0.3)

# Subplot 4: s15 neg
ax = axes[1,0]
bin_edges = np.array([-0.01, -0.005, 0.0, 0.005, 0.01])
bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
ax.plot(bin_centers, dpdfs_neg_fid_s15, c='black')
ax.errorbar(bin_centers, dpdfs_neg_fid_s15, yerr=np.sqrt(np.diag(cov_dpdfs_neg_s15))/np.sqrt(N_fid), fmt='o', label='Fiducial', c='black')   
ax.errorbar(bin_centers, deriv_dpdfs_combined_s15_mean[4:8], yerr=np.sqrt(np.diag(deriv_dpdfs_combined_s15_cov[4:8, 4:8]))/np.sqrt(N_deriv), fmt='o', label=r'Derv. w.r.t $\Omega_m$')
ax.errorbar(bin_centers, deriv_dpdfs_combined_s15_mean[12:16], yerr=np.sqrt(np.diag(deriv_dpdfs_combined_s15_cov[12:16, 12:16]))/np.sqrt(N_deriv), fmt='o', label=r'Derv. w.r.t $\sigma_8$')
ax.fill_between(bin_centers, deriv_dpdfs_combined_s15_mean[4:8] - np.sqrt(np.diag(deriv_dpdfs_combined_s15_cov[4:8, 4:8]))/np.sqrt(N_deriv), deriv_dpdfs_combined_s15_mean[4:8] + np.sqrt(np.diag(deriv_dpdfs_combined_s15_cov[4:8, 4:8]))/np.sqrt(N_deriv), alpha=0.3)
ax.fill_between(bin_centers, deriv_dpdfs_combined_s15_mean[12:16] - np.sqrt(np.diag(deriv_dpdfs_combined_s15_cov[12:16, 12:16]))/np.sqrt(N_deriv), deriv_dpdfs_combined_s15_mean[12:16] + np.sqrt(np.diag(deriv_dpdfs_combined_s15_cov[12:16, 12:16]))/np.sqrt(N_deriv), alpha=0.3)
ax.set_xlabel(r'$\kappa$')
ax.set_ylabel(r'$C^n(\kappa)$')
ax.set_xticks([-0.005, 0.005])
ax.grid(True, alpha=0.3)

# Subplot 5: s30 neg
ax = axes[1,1]
bin_edges = np.array([-0.008, -0.004, 0.0, 0.004, 0.008])
bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
ax.plot(bin_centers, dpdfs_neg_fid_s30, c='black')
ax.errorbar(bin_centers, dpdfs_neg_fid_s30, yerr=np.sqrt(np.diag(cov_dpdfs_neg_s30))/np.sqrt(N_fid), fmt='o', label='Fiducial', c='black')   
ax.errorbar(bin_centers, deriv_dpdfs_combined_s30_mean[4:8], yerr=np.sqrt(np.diag(deriv_dpdfs_combined_s30_cov[4:8, 4:8]))/np.sqrt(N_deriv), fmt='o', label=r'Derv. w.r.t $\Omega_m$')
ax.errorbar(bin_centers, deriv_dpdfs_combined_s30_mean[12:16], yerr=np.sqrt(np.diag(deriv_dpdfs_combined_s30_cov[12:16, 12:16]))/np.sqrt(N_deriv), fmt='o', label=r'Derv. w.r.t $\sigma_8$')
ax.fill_between(bin_centers, deriv_dpdfs_combined_s30_mean[4:8] - np.sqrt(np.diag(deriv_dpdfs_combined_s30_cov[4:8, 4:8]))/np.sqrt(N_deriv), deriv_dpdfs_combined_s30_mean[4:8] + np.sqrt(np.diag(deriv_dpdfs_combined_s30_cov[4:8, 4:8]))/np.sqrt(N_deriv), alpha=0.3)
ax.fill_between(bin_centers, deriv_dpdfs_combined_s30_mean[12:16] - np.sqrt(np.diag(deriv_dpdfs_combined_s30_cov[12:16, 12:16]))/np.sqrt(N_deriv), deriv_dpdfs_combined_s30_mean[12:16] + np.sqrt(np.diag(deriv_dpdfs_combined_s30_cov[12:16, 12:16]))/np.sqrt(N_deriv), alpha=0.3)
ax.set_xlabel(r'$\kappa$')
ax.set_xticks([-0.004, 0.004])
ax.grid(True, alpha=0.3)

# Subplot 6: s90 neg
ax = axes[1,2]
bin_edges = np.array([-0.005, -0.0025, 0.0, 0.0025, 0.005])
bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
ax.plot(bin_centers, dpdfs_neg_fid_s90, c='black')
ax.errorbar(bin_centers, dpdfs_neg_fid_s90, yerr=np.sqrt(np.diag(cov_dpdfs_neg_s90))/np.sqrt(N_fid), fmt='o', label='Fiducial', c='black')   
ax.errorbar(bin_centers, deriv_dpdfs_combined_s90_mean[4:8], yerr=np.sqrt(np.diag(deriv_dpdfs_combined_s90_cov[4:8, 4:8]))/np.sqrt(N_deriv), fmt='o', label=r'Derv. w.r.t $\Omega_m$')
ax.errorbar(bin_centers, deriv_dpdfs_combined_s90_mean[12:16], yerr=np.sqrt(np.diag(deriv_dpdfs_combined_s90_cov[12:16, 12:16]))/np.sqrt(N_deriv), fmt='o', label=r'Derv. w.r.t $\sigma_8$')
ax.fill_between(bin_centers, deriv_dpdfs_combined_s90_mean[4:8] - np.sqrt(np.diag(deriv_dpdfs_combined_s90_cov[4:8, 4:8]))/np.sqrt(N_deriv), deriv_dpdfs_combined_s90_mean[4:8] + np.sqrt(np.diag(deriv_dpdfs_combined_s90_cov[4:8, 4:8]))/np.sqrt(N_deriv), alpha=0.3)
ax.fill_between(bin_centers, deriv_dpdfs_combined_s90_mean[12:16] - np.sqrt(np.diag(deriv_dpdfs_combined_s90_cov[12:16, 12:16]))/np.sqrt(N_deriv), deriv_dpdfs_combined_s90_mean[12:16] + np.sqrt(np.diag(deriv_dpdfs_combined_s90_cov[12:16, 12:16]))/np.sqrt(N_deriv), alpha=0.3)
ax.set_xlabel(r'$\kappa$')
ax.set_xticks([-0.0025, 0.0025])
ax.grid(True, alpha=0.3)
        
plt.legend(loc=(1.05, 0.75))
plt.subplots_adjust(hspace=0, wspace=0)
plt.savefig('outfiles/derivs_dpdf.png', dpi=300, bbox_inches='tight')
plt.close()

# nns only plot 2NN , pos same and pos opp, for all smoothing scales, so 2x3
fig, axes = plt.subplots(2, 3, figsize=(6, 3), sharex='col') 
# Row 1: 2NN pos same
# s15
ax = axes[0,0]
bin_centers = np.linspace(30, 200, 5)[1:]
ax.plot(bin_centers, nn_pos_same2_s15, c='black')
ax.errorbar(bin_centers, nn_pos_same2_s15, yerr=np.sqrt(np.diag(cov_nn_pos_same2_s15))/np.sqrt(N_fid), fmt='o', c='black')   
ax.errorbar(bin_centers, deriv_nns_combined_s15_mean[16:20], yerr=np.sqrt(np.diag(deriv_nns_combined_s15_cov[16:20, 16:20]))/np.sqrt(N_deriv), fmt='o', label=r'Derv. w.r.t $\Omega_m$')
ax.errorbar(bin_centers, deriv_nns_combined_s15_mean[80:84], yerr=np.sqrt(np.diag(deriv_nns_combined_s15_cov[80:84, 80:84]))/np.sqrt(N_deriv), fmt='o', label=r'Derv. w.r.t $\sigma_8$')
ax.fill_between(bin_centers, deriv_nns_combined_s15_mean[16:20] - np.sqrt(np.diag(deriv_nns_combined_s15_cov[16:20, 16:20]))/np.sqrt(N_deriv), deriv_nns_combined_s15_mean[16:20] + np.sqrt(np.diag(deriv_nns_combined_s15_cov[16:20, 16:20]))/np.sqrt(N_deriv), alpha=0.3)
ax.fill_between(bin_centers, deriv_nns_combined_s15_mean[80:84] - np.sqrt(np.diag(deriv_nns_combined_s15_cov[80:84, 80:84]))/np.sqrt(N_deriv), deriv_nns_combined_s15_mean[80:84] + np.sqrt(np.diag(deriv_nns_combined_s15_cov[80:84, 80:84]))/np.sqrt(N_deriv), alpha=0.3)
ax.set_title(r'$\theta_G$ = 15 arcmin')
ax.set_ylabel(r'$C^{pp}_{\rm 2nd}(\theta)$')
ax.grid(True, alpha=0.3)

# s30
ax = axes[0,1]
bin_centers = np.linspace(30, 450, 5)[1:]
ax.plot(bin_centers, nn_pos_same2_s30, c='black')
ax.errorbar(bin_centers, nn_pos_same2_s30, yerr=np.sqrt(np.diag(cov_nn_pos_same2_s30))/np.sqrt(N_fid), fmt='o', c='black')   
ax.errorbar(bin_centers, deriv_nns_combined_s30_mean[16:20], yerr=np.sqrt(np.diag(deriv_nns_combined_s30_cov[16:20, 16:20]))/np.sqrt(N_deriv), fmt='o')
ax.errorbar(bin_centers, deriv_nns_combined_s30_mean[80:84], yerr=np.sqrt(np.diag(deriv_nns_combined_s30_cov[80:84, 80:84]))/np.sqrt(N_deriv), fmt='o')
ax.fill_between(bin_centers, deriv_nns_combined_s30_mean[16:20] - np.sqrt(np.diag(deriv_nns_combined_s30_cov[16:20, 16:20]))/np.sqrt(N_deriv), deriv_nns_combined_s30_mean[16:20] + np.sqrt(np.diag(deriv_nns_combined_s30_cov[16:20, 16:20]))/np.sqrt(N_deriv), alpha=0.3)
ax.fill_between(bin_centers, deriv_nns_combined_s30_mean[80:84] - np.sqrt(np.diag(deriv_nns_combined_s30_cov[80:84, 80:84]))/np.sqrt(N_deriv), deriv_nns_combined_s30_mean[80:84] + np.sqrt(np.diag(deriv_nns_combined_s30_cov[80:84, 80:84]))/np.sqrt(N_deriv), alpha=0.3)
ax.set_title(r'$\theta_G$ = 30 arcmin')
ax.grid(True, alpha=0.3)

# s90
ax = axes[0,2]
bin_centers = np.linspace(30, 1125, 5)[1:]
ax.plot(bin_centers, nn_pos_same2_s90, c='black')
ax.errorbar(bin_centers, nn_pos_same2_s90, yerr=np.sqrt(np.diag(cov_nn_pos_same2_s90))/np.sqrt(N_fid), fmt='o', c='black')   
ax.errorbar(bin_centers, deriv_nns_combined_s90_mean[16:20], yerr=np.sqrt(np.diag(deriv_nns_combined_s90_cov[16:20, 16:20]))/np.sqrt(N_deriv), fmt='o')
ax.errorbar(bin_centers, deriv_nns_combined_s90_mean[80:84], yerr=np.sqrt(np.diag(deriv_nns_combined_s90_cov[80:84, 80:84]))/np.sqrt(N_deriv), fmt='o')
ax.fill_between(bin_centers, deriv_nns_combined_s90_mean[16:20] - np.sqrt(np.diag(deriv_nns_combined_s90_cov[16:20, 16:20]))/np.sqrt(N_deriv), deriv_nns_combined_s90_mean[16:20] + np.sqrt(np.diag(deriv_nns_combined_s90_cov[16:20, 16:20]))/np.sqrt(N_deriv), alpha=0.3)
ax.fill_between(bin_centers, deriv_nns_combined_s90_mean[80:84] - np.sqrt(np.diag(deriv_nns_combined_s90_cov[80:84, 80:84]))/np.sqrt(N_deriv), deriv_nns_combined_s90_mean[80:84] + np.sqrt(np.diag(deriv_nns_combined_s90_cov[80:84, 80:84]))/np.sqrt(N_deriv), alpha=0.3)
ax.set_title(r'$\theta_G$ = 90 arcmin')
ax.grid(True, alpha=0.3)

# Row 2: 2NN pos opp
# s15
ax = axes[1,0]
bin_centers = np.linspace(30, 200, 5)[1:]
ax.plot(bin_centers, nn_pos_opp2_s15, c='black')
ax.errorbar(bin_centers, nn_pos_opp2_s15, yerr=np.sqrt(np.diag(cov_nn_pos_opp2_s15))/np.sqrt(N_fid), fmt='o', c='black')   
ax.errorbar(bin_centers, deriv_nns_combined_s15_mean[24:28], yerr=np.sqrt(np.diag(deriv_nns_combined_s15_cov[24:28, 24:28]))/np.sqrt(N_deriv), fmt='o')
ax.errorbar(bin_centers, deriv_nns_combined_s15_mean[88:92], yerr=np.sqrt(np.diag(deriv_nns_combined_s15_cov[88:92, 88:92]))/np.sqrt(N_deriv), fmt='o')
ax.fill_between(bin_centers, deriv_nns_combined_s15_mean[24:28] - np.sqrt(np.diag(deriv_nns_combined_s15_cov[24:28, 24:28]))/np.sqrt(N_deriv), deriv_nns_combined_s15_mean[24:28] + np.sqrt(np.diag(deriv_nns_combined_s15_cov[24:28, 24:28]))/np.sqrt(N_deriv), alpha=0.3)
ax.fill_between(bin_centers, deriv_nns_combined_s15_mean[88:92] - np.sqrt(np.diag(deriv_nns_combined_s15_cov[88:92, 88:92]))/np.sqrt(N_deriv), deriv_nns_combined_s15_mean[88:92] + np.sqrt(np.diag(deriv_nns_combined_s15_cov[88:92, 88:92]))/np.sqrt(N_deriv), alpha=0.3)
ax.set_ylabel(r'$C^{pn}_{\rm 2nd}(\theta)$')
ax.set_xticks([100, 150])
ax.grid(True, alpha=0.3)

# s30
ax = axes[1,1]
bin_centers = np.linspace(30, 450, 5)[1:]
ax.plot(bin_centers, nn_pos_opp2_s30, c='black')
ax.errorbar(bin_centers, nn_pos_opp2_s30, yerr=np.sqrt(np.diag(cov_nn_pos_opp2_s30))/np.sqrt(N_fid), fmt='o', c='black')   
ax.errorbar(bin_centers, deriv_nns_combined_s30_mean[24:28], yerr=np.sqrt(np.diag(deriv_nns_combined_s30_cov[24:28, 24:28]))/np.sqrt(N_deriv), fmt='o')
ax.errorbar(bin_centers, deriv_nns_combined_s30_mean[88:92], yerr=np.sqrt(np.diag(deriv_nns_combined_s30_cov[88:92, 88:92]))/np.sqrt(N_deriv), fmt='o')
ax.fill_between(bin_centers, deriv_nns_combined_s30_mean[24:28] - np.sqrt(np.diag(deriv_nns_combined_s30_cov[24:28, 24:28]))/np.sqrt(N_deriv), deriv_nns_combined_s30_mean[24:28] + np.sqrt(np.diag(deriv_nns_combined_s30_cov[24:28, 24:28]))/np.sqrt(N_deriv), alpha=0.3)
ax.fill_between(bin_centers, deriv_nns_combined_s30_mean[88:92] - np.sqrt(np.diag(deriv_nns_combined_s30_cov[88:92, 88:92]))/np.sqrt(N_deriv), deriv_nns_combined_s30_mean[88:92] + np.sqrt(np.diag(deriv_nns_combined_s30_cov[88:92, 88:92]))/np.sqrt(N_deriv), alpha=0.3)
ax.set_xticks([150, 350])
ax.grid(True, alpha=0.3)

# s90
ax = axes[1,2]
bin_centers = np.linspace(30, 1125, 5)[1:]
ax.plot(bin_centers, nn_pos_opp2_s90, c='black')
ax.errorbar(bin_centers, nn_pos_opp2_s90, yerr=np.sqrt(np.diag(cov_nn_pos_opp2_s90))/np.sqrt(N_fid), fmt='o', c='black', label='Fiducial')   
ax.errorbar(bin_centers, deriv_nns_combined_s90_mean[24:28], yerr=np.sqrt(np.diag(deriv_nns_combined_s90_cov[24:28, 24:28]))/np.sqrt(N_deriv), fmt='o', label=r'Derv. w.r.t $\Omega_m$')
ax.errorbar(bin_centers, deriv_nns_combined_s90_mean[88:92], yerr=np.sqrt(np.diag(deriv_nns_combined_s90_cov[88:92, 88:92]))/np.sqrt(N_deriv), fmt='o', label=r'Derv. w.r.t $\sigma_8$')
ax.fill_between(bin_centers, deriv_nns_combined_s90_mean[24:28] - np.sqrt(np.diag(deriv_nns_combined_s90_cov[24:28, 24:28]))/np.sqrt(N_deriv), deriv_nns_combined_s90_mean[24:28] + np.sqrt(np.diag(deriv_nns_combined_s90_cov[24:28, 24:28]))/np.sqrt(N_deriv), alpha=0.3)
ax.fill_between(bin_centers, deriv_nns_combined_s90_mean[88:92] - np.sqrt(np.diag(deriv_nns_combined_s90_cov[88:92, 88:92]))/np.sqrt(N_deriv), deriv_nns_combined_s90_mean[88:92] + np.sqrt(np.diag(deriv_nns_combined_s90_cov[88:92, 88:92]))/np.sqrt(N_deriv), alpha=0.3)
ax.set_xticks([600, 1000])
ax.grid(True, alpha=0.3)

plt.legend(loc='upper left', bbox_to_anchor=(1.05, 1.25))
plt.subplots_adjust(hspace=0, wspace=0)
plt.savefig('outfiles/derivs_nns.png', dpi=300, bbox_inches='tight')
plt.close()