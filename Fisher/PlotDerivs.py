import numpy as np

DATA_DIR = "/cephfs/jbai/DefectFisherGaussian"
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
print(params)
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

pp_corrs_s15 = []
pp_corrs_s30 = []
pp_corrs_s90 = []

nn_corrs_s15 = []
nn_corrs_s30 = []
nn_corrs_s90 = []

pn_corrs_s15 = []
pn_corrs_s30 = []
pn_corrs_s90 = []

pk_corrs_s15 = []
pk_corrs_s30 = []
pk_corrs_s90 = []

nk_corrs_s15 = []
nk_corrs_s30 = []
nk_corrs_s90 = []

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

        data_pp = np.load(f"{DATA_DIR}/TrainPosPos/train_{i}_s15_seed_{seed}_pos_pos_corr.npz")
        pp_corrs_s15.append((data_pp['xi_pp']))
        data_pp = np.load(f"{DATA_DIR}/TrainPosPos/train_{i}_s30_seed_{seed}_pos_pos_corr.npz")
        pp_corrs_s30.append((data_pp['xi_pp']))
        data_pp = np.load(f"{DATA_DIR}/TrainPosPos/train_{i}_s90_seed_{seed}_pos_pos_corr.npz")
        pp_corrs_s90.append((data_pp['xi_pp']))

        data_nn = np.load(f"{DATA_DIR}/TrainNegNeg/train_{i}_s15_seed_{seed}_neg_neg_corr.npz")
        nn_corrs_s15.append((data_nn['xi_nn']))
        data_nn = np.load(f"{DATA_DIR}/TrainNegNeg/train_{i}_s30_seed_{seed}_neg_neg_corr.npz")
        nn_corrs_s30.append((data_nn['xi_nn']))
        data_nn = np.load(f"{DATA_DIR}/TrainNegNeg/train_{i}_s90_seed_{seed}_neg_neg_corr.npz")
        nn_corrs_s90.append((data_nn['xi_nn']))

        data_pn = np.load(f"{DATA_DIR}/TrainPosNeg/train_{i}_s15_seed_{seed}_pos_neg_corr.npz")
        pn_corrs_s15.append((data_pn['xi_pn']))
        data_pn = np.load(f"{DATA_DIR}/TrainPosNeg/train_{i}_s30_seed_{seed}_pos_neg_corr.npz")
        pn_corrs_s30.append((data_pn['xi_pn']))
        data_pn = np.load(f"{DATA_DIR}/TrainPosNeg/train_{i}_s90_seed_{seed}_pos_neg_corr.npz")
        pn_corrs_s90.append((data_pn['xi_pn']))

        data_pk = np.load(f"{DATA_DIR}/TrainPKs/train_{i}_s15_seed_{seed}_pos_kappa_corr.npz")
        pk_corrs_s15.append((data_pk['xi_pk']))
        data_pk = np.load(f"{DATA_DIR}/TrainPKs/train_{i}_s30_seed_{seed}_pos_kappa_corr.npz")
        pk_corrs_s30.append((data_pk['xi_pk']))
        data_pk = np.load(f"{DATA_DIR}/TrainPKs/train_{i}_s90_seed_{seed}_pos_kappa_corr.npz")
        pk_corrs_s90.append((data_pk['xi_pk']))

        data_nk = np.load(f"{DATA_DIR}/TrainNKs/train_{i}_s15_seed_{seed}_neg_kappa_corr.npz")
        nk_corrs_s15.append((data_nk['xi_nk']))
        data_nk = np.load(f"{DATA_DIR}/TrainNKs/train_{i}_s30_seed_{seed}_neg_kappa_corr.npz")
        nk_corrs_s30.append((data_nk['xi_nk']))
        data_nk = np.load(f"{DATA_DIR}/TrainNKs/train_{i}_s90_seed_{seed}_neg_kappa_corr.npz")
        nk_corrs_s90.append((data_nk['xi_nk']))
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
pp_corrs_s15 = np.array(pp_corrs_s15)
pp_corrs_s30 = np.array(pp_corrs_s30)
pp_corrs_s90 = np.array(pp_corrs_s90)
nn_corrs_s15 = np.array(nn_corrs_s15)
nn_corrs_s30 = np.array(nn_corrs_s30)
nn_corrs_s90 = np.array(nn_corrs_s90)
pn_corrs_s15 = np.array(pn_corrs_s15)
pn_corrs_s30 = np.array(pn_corrs_s30)
pn_corrs_s90 = np.array(pn_corrs_s90)
pk_corrs_s15 = np.array(pk_corrs_s15)
pk_corrs_s30 = np.array(pk_corrs_s30)
pk_corrs_s90 = np.array(pk_corrs_s90)
nk_corrs_s15 = np.array(nk_corrs_s15)
nk_corrs_s30 = np.array(nk_corrs_s30)
nk_corrs_s90 = np.array(nk_corrs_s90)
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
pp_corrs_s15 = pp_corrs_s15.reshape(n_seeds, 4, -1)
pp_corrs_s30 = pp_corrs_s30.reshape(n_seeds, 4, -1)
pp_corrs_s90 = pp_corrs_s90.reshape(n_seeds, 4, -1)
nn_corrs_s15 = nn_corrs_s15.reshape(n_seeds, 4, -1)
nn_corrs_s30 = nn_corrs_s30.reshape(n_seeds, 4, -1)
nn_corrs_s90 = nn_corrs_s90.reshape(n_seeds, 4, -1)
pn_corrs_s15 = pn_corrs_s15.reshape(n_seeds, 4, -1)
pn_corrs_s30 = pn_corrs_s30.reshape(n_seeds, 4, -1)
pn_corrs_s90 = pn_corrs_s90.reshape(n_seeds, 4, -1)
pk_corrs_s15 = pk_corrs_s15.reshape(n_seeds, 4, -1)
pk_corrs_s30 = pk_corrs_s30.reshape(n_seeds, 4, -1)
pk_corrs_s90 = pk_corrs_s90.reshape(n_seeds, 4, -1)
nk_corrs_s15 = nk_corrs_s15.reshape(n_seeds, 4, -1)
nk_corrs_s30 = nk_corrs_s30.reshape(n_seeds, 4, -1)
nk_corrs_s90 = nk_corrs_s90.reshape(n_seeds, 4, -1)
nn_combined_s15 = nn_combined_s15.reshape(n_seeds, 4, -1)
nn_combined_s30 = nn_combined_s30.reshape(n_seeds, 4, -1)
nn_combined_s90 = nn_combined_s90.reshape(n_seeds, 4, -1)
print(f"After reshape - pdfs_s15: {pdfs_s15.shape}")
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
deriv_pp_corrs_s15 = []
deriv_pp_corrs_s30 = []
deriv_pp_corrs_s90 = []
deriv_nn_corrs_s15 = []
deriv_nn_corrs_s30 = []
deriv_nn_corrs_s90 = []
deriv_pn_corrs_s15 = []
deriv_pn_corrs_s30 = []
deriv_pn_corrs_s90 = []
deriv_pk_corrs_s15 = []
deriv_pk_corrs_s30 = []
deriv_pk_corrs_s90 = []
deriv_nk_corrs_s15 = []
deriv_nk_corrs_s30 = []
deriv_nk_corrs_s90 = []
for i in range(0, len(params), 2):
    print(i)
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

    xi_pp_plus = pp_corrs_s15[:,i+1,:]
    xi_pp_minus = pp_corrs_s15[:,i,:]
    deriv_pp_corr = (xi_pp_plus - xi_pp_minus) / (2 * delta_params[list(fid.keys())[i//2]])
    deriv_pp_corrs_s15.append(deriv_pp_corr)
    xi_pp_plus = pp_corrs_s30[:,i+1,:]
    xi_pp_minus = pp_corrs_s30[:,i,:]
    deriv_pp_corr = (xi_pp_plus - xi_pp_minus) / (2 * delta_params[list(fid.keys())[i//2]])
    deriv_pp_corrs_s30.append(deriv_pp_corr)
    xi_pp_plus = pp_corrs_s90[:,i+1,:]
    xi_pp_minus = pp_corrs_s90[:,i,:]
    deriv_pp_corr = (xi_pp_plus - xi_pp_minus) / (2 * delta_params[list(fid.keys())[i//2]])
    deriv_pp_corrs_s90.append(deriv_pp_corr)

    xi_nn_plus = nn_corrs_s15[:,i+1,:]
    xi_nn_minus = nn_corrs_s15[:,i,:]
    deriv_nn_corr = (xi_nn_plus - xi_nn_minus) / (2 * delta_params[list(fid.keys())[i//2]])
    deriv_nn_corrs_s15.append(deriv_nn_corr)
    xi_nn_plus = nn_corrs_s30[:,i+1,:]
    xi_nn_minus = nn_corrs_s30[:,i,:]
    deriv_nn_corr = (xi_nn_plus - xi_nn_minus) / (2 * delta_params[list(fid.keys())[i//2]])
    deriv_nn_corrs_s30.append(deriv_nn_corr)
    xi_nn_plus = nn_corrs_s90[:,i+1,:]
    xi_nn_minus = nn_corrs_s90[:,i,:]
    deriv_nn_corr = (xi_nn_plus - xi_nn_minus) / (2 * delta_params[list(fid.keys())[i//2]])
    deriv_nn_corrs_s90.append(deriv_nn_corr)

    xi_pn_plus = pn_corrs_s15[:,i+1,:]
    xi_pn_minus = pn_corrs_s15[:,i,:]
    deriv_pn_corr = (xi_pn_plus - xi_pn_minus) / (2 * delta_params[list(fid.keys())[i//2]])
    deriv_pn_corrs_s15.append(deriv_pn_corr)
    xi_pn_plus = pn_corrs_s30[:,i+1,:]
    xi_pn_minus = pn_corrs_s30[:,i,:]
    deriv_pn_corr = (xi_pn_plus - xi_pn_minus) / (2 * delta_params[list(fid.keys())[i//2]])
    deriv_pn_corrs_s30.append(deriv_pn_corr)
    xi_pn_plus = pn_corrs_s90[:,i+1,:]
    xi_pn_minus = pn_corrs_s90[:,i,:]
    deriv_pn_corr = (xi_pn_plus - xi_pn_minus) / (2 * delta_params[list(fid.keys())[i//2]])
    deriv_pn_corrs_s90.append(deriv_pn_corr)

    xi_pk_plus = pk_corrs_s15[:,i+1,:]
    xi_pk_minus = pk_corrs_s15[:,i,:]
    deriv_pk_corr = (xi_pk_plus - xi_pk_minus) / (2 * delta_params[list(fid.keys())[i//2]])
    deriv_pk_corrs_s15.append(deriv_pk_corr)
    xi_pk_plus = pk_corrs_s30[:,i+1,:]
    xi_pk_minus = pk_corrs_s30[:,i,:]
    deriv_pk_corr = (xi_pk_plus - xi_pk_minus) / (2 * delta_params[list(fid.keys())[i//2]])
    deriv_pk_corrs_s30.append(deriv_pk_corr)
    xi_pk_plus = pk_corrs_s90[:,i+1,:]
    xi_pk_minus = pk_corrs_s90[:,i,:]
    deriv_pk_corr = (xi_pk_plus - xi_pk_minus) / (2 * delta_params[list(fid.keys())[i//2]])
    deriv_pk_corrs_s90.append(deriv_pk_corr)

    xi_nk_plus = nk_corrs_s15[:,i+1,:]
    xi_nk_minus = nk_corrs_s15[:,i,:]
    deriv_nk_corr = (xi_nk_plus - xi_nk_minus) / (2 * delta_params[list(fid.keys())[i//2]])
    deriv_nk_corrs_s15.append(deriv_nk_corr)
    xi_nk_plus = nk_corrs_s30[:,i+1,:]
    xi_nk_minus = nk_corrs_s30[:,i,:]
    deriv_nk_corr = (xi_nk_plus - xi_nk_minus) / (2 * delta_params[list(fid.keys())[i//2]])
    deriv_nk_corrs_s30.append(deriv_nk_corr)
    xi_nk_plus = nk_corrs_s90[:,i+1,:]
    xi_nk_minus = nk_corrs_s90[:,i,:]
    deriv_nk_corr = (xi_nk_plus - xi_nk_minus) / (2 * delta_params[list(fid.keys())[i//2]])
    deriv_nk_corrs_s90.append(deriv_nk_corr)
    
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
deriv_pp_corrs_s15 = np.array(deriv_pp_corrs_s15)
deriv_pp_corrs_s30 = np.array(deriv_pp_corrs_s30)
deriv_pp_corrs_s90 = np.array(deriv_pp_corrs_s90)
deriv_nn_corrs_s15 = np.array(deriv_nn_corrs_s15)
deriv_nn_corrs_s30 = np.array(deriv_nn_corrs_s30)
deriv_nn_corrs_s90 = np.array(deriv_nn_corrs_s90)
deriv_pn_corrs_s15 = np.array(deriv_pn_corrs_s15)
deriv_pn_corrs_s30 = np.array(deriv_pn_corrs_s30)
deriv_pn_corrs_s90 = np.array(deriv_pn_corrs_s90)
deriv_pk_corrs_s15 = np.array(deriv_pk_corrs_s15)
deriv_pk_corrs_s30 = np.array(deriv_pk_corrs_s30)
deriv_pk_corrs_s90 = np.array(deriv_pk_corrs_s90)
deriv_nk_corrs_s15 = np.array(deriv_nk_corrs_s15)
deriv_nk_corrs_s30 = np.array(deriv_nk_corrs_s30)
deriv_nk_corrs_s90 = np.array(deriv_nk_corrs_s90)

deriv_nns_combined_s15 = np.array(deriv_nns_combined_s15)
deriv_nns_combined_s30 = np.array(deriv_nns_combined_s30)
deriv_nns_combined_s90 = np.array(deriv_nns_combined_s90)


# create mean derivs and their covs 30 as one group
print(deriv_pdfs_s15.shape)
# deriv_pdfs_s15 shape: (2, 30, 5) - (params, seeds, bins)
# Reshape to (30, 2*5) for covariance calculation
deriv_pdfs_s15_reshaped = deriv_pdfs_s15.transpose(1, 0, 2).reshape(100, -1)
deriv_pdfs_s15_mean = np.mean(deriv_pdfs_s15_reshaped, axis=0)
deriv_pdfs_s15_cov = np.cov(deriv_pdfs_s15_reshaped, rowvar=False)

deriv_pdfs_s30_reshaped = deriv_pdfs_s30.transpose(1, 0, 2).reshape(100, -1)
deriv_pdfs_s30_mean = np.mean(deriv_pdfs_s30_reshaped, axis=0)
deriv_pdfs_s30_cov = np.cov(deriv_pdfs_s30_reshaped, rowvar=False)

deriv_pdfs_s90_reshaped = deriv_pdfs_s90.transpose(1, 0, 2).reshape(100, -1)
deriv_pdfs_s90_mean = np.mean(deriv_pdfs_s90_reshaped, axis=0)
deriv_pdfs_s90_cov = np.cov(deriv_pdfs_s90_reshaped, rowvar=False)

deriv_dpdfs_pos_s15_reshaped = deriv_dpdfs_pos_s15.transpose(1, 0, 2).reshape(100, -1)
deriv_dpdfs_pos_s15_mean = np.mean(deriv_dpdfs_pos_s15_reshaped, axis=0)
deriv_dpdfs_pos_s15_cov = np.cov(deriv_dpdfs_pos_s15_reshaped, rowvar=False)

deriv_dpdfs_pos_s30_reshaped = deriv_dpdfs_pos_s30.transpose(1, 0, 2).reshape(100, -1)
deriv_dpdfs_pos_s30_mean = np.mean(deriv_dpdfs_pos_s30_reshaped, axis=0)
deriv_dpdfs_pos_s30_cov = np.cov(deriv_dpdfs_pos_s30_reshaped, rowvar=False)

deriv_dpdfs_pos_s90_reshaped = deriv_dpdfs_pos_s90.transpose(1, 0, 2).reshape(100, -1)
deriv_dpdfs_pos_s90_mean = np.mean(deriv_dpdfs_pos_s90_reshaped, axis=0)
deriv_dpdfs_pos_s90_cov = np.cov(deriv_dpdfs_pos_s90_reshaped, rowvar=False)

deriv_dpdfs_neg_s15_reshaped = deriv_dpdfs_neg_s15.transpose(1, 0, 2).reshape(100, -1)
deriv_dpdfs_neg_s15_mean = np.mean(deriv_dpdfs_neg_s15_reshaped, axis=0)
deriv_dpdfs_neg_s15_cov = np.cov(deriv_dpdfs_neg_s15_reshaped, rowvar=False)

deriv_dpdfs_neg_s30_reshaped = deriv_dpdfs_neg_s30.transpose(1, 0, 2).reshape(100, -1)
deriv_dpdfs_neg_s30_mean = np.mean(deriv_dpdfs_neg_s30_reshaped, axis=0)
deriv_dpdfs_neg_s30_cov = np.cov(deriv_dpdfs_neg_s30_reshaped, rowvar=False)

deriv_dpdfs_neg_s90_reshaped = deriv_dpdfs_neg_s90.transpose(1, 0, 2).reshape(100, -1)
deriv_dpdfs_neg_s90_mean = np.mean(deriv_dpdfs_neg_s90_reshaped, axis=0)
deriv_dpdfs_neg_s90_cov = np.cov(deriv_dpdfs_neg_s90_reshaped, rowvar=False)

deriv_dpdfs_combined_s15_reshaped = deriv_dpdfs_combined_s15.transpose(1, 0, 2).reshape(100, -1)
deriv_dpdfs_combined_s15_mean = np.mean(deriv_dpdfs_combined_s15_reshaped, axis=0)
deriv_dpdfs_combined_s15_cov = np.cov(deriv_dpdfs_combined_s15_reshaped, rowvar=False)

deriv_dpdfs_combined_s30_reshaped = deriv_dpdfs_combined_s30.transpose(1, 0, 2).reshape(100, -1)
deriv_dpdfs_combined_s30_mean = np.mean(deriv_dpdfs_combined_s30_reshaped, axis=0)
deriv_dpdfs_combined_s30_cov = np.cov(deriv_dpdfs_combined_s30_reshaped, rowvar=False)

deriv_dpdfs_combined_s90_reshaped = deriv_dpdfs_combined_s90.transpose(1, 0, 2).reshape(100, -1)
deriv_dpdfs_combined_s90_mean = np.mean(deriv_dpdfs_combined_s90_reshaped, axis=0)
deriv_dpdfs_combined_s90_cov = np.cov(deriv_dpdfs_combined_s90_reshaped, rowvar=False)

deriv_dpdfs_combined_all_scales_reshaped = deriv_dpdfs_combined_all_scales.transpose(1, 0, 2).reshape(100, -1)
deriv_dpdfs_combined_all_scales_mean = np.mean(deriv_dpdfs_combined_all_scales_reshaped, axis=0)
deriv_dpdfs_combined_all_scales_cov = np.cov(deriv_dpdfs_combined_all_scales_reshaped, rowvar=False)

deriv_kk_corrs_s15_reshaped = deriv_kk_corrs_s15.transpose(1, 0, 2).reshape(100, -1)
deriv_kk_corrs_s15_mean = np.mean(deriv_kk_corrs_s15_reshaped, axis=0)
deriv_kk_corrs_s15_cov = np.cov(deriv_kk_corrs_s15_reshaped, rowvar=False)

deriv_kk_corrs_s30_reshaped = deriv_kk_corrs_s30.transpose(1, 0, 2).reshape(100, -1)
deriv_kk_corrs_s30_mean = np.mean(deriv_kk_corrs_s30_reshaped, axis=0)
deriv_kk_corrs_s30_cov = np.cov(deriv_kk_corrs_s30_reshaped, rowvar=False)

deriv_kk_corrs_s90_reshaped = deriv_kk_corrs_s90.transpose(1, 0, 2).reshape(100, -1)
deriv_kk_corrs_s90_mean = np.mean(deriv_kk_corrs_s90_reshaped, axis=0)
deriv_kk_corrs_s90_cov = np.cov(deriv_kk_corrs_s90_reshaped, rowvar=False)

deriv_pp_corrs_s15_reshaped = deriv_pp_corrs_s15.transpose(1, 0, 2).reshape(100, -1)
deriv_pp_corrs_s15_mean = np.mean(deriv_pp_corrs_s15_reshaped, axis=0)
deriv_pp_corrs_s15_cov = np.cov(deriv_pp_corrs_s15_reshaped, rowvar=False)

deriv_pp_corrs_s30_reshaped = deriv_pp_corrs_s30.transpose(1, 0, 2).reshape(100, -1)
deriv_pp_corrs_s30_mean = np.mean(deriv_pp_corrs_s30_reshaped, axis=0)
deriv_pp_corrs_s30_cov = np.cov(deriv_pp_corrs_s30_reshaped, rowvar=False)

deriv_pp_corrs_s90_reshaped = deriv_pp_corrs_s90.transpose(1, 0, 2).reshape(100, -1)
deriv_pp_corrs_s90_mean = np.mean(deriv_pp_corrs_s90_reshaped, axis=0)
deriv_pp_corrs_s90_cov = np.cov(deriv_pp_corrs_s90_reshaped, rowvar=False)

deriv_nn_corrs_s15_reshaped = deriv_nn_corrs_s15.transpose(1, 0, 2).reshape(100, -1)
deriv_nn_corrs_s15_mean = np.mean(deriv_nn_corrs_s15_reshaped, axis=0)
deriv_nn_corrs_s15_cov = np.cov(deriv_nn_corrs_s15_reshaped, rowvar=False)

deriv_nn_corrs_s30_reshaped = deriv_nn_corrs_s30.transpose(1, 0, 2).reshape(100, -1)
deriv_nn_corrs_s30_mean = np.mean(deriv_nn_corrs_s30_reshaped, axis=0)
deriv_nn_corrs_s30_cov = np.cov(deriv_nn_corrs_s30_reshaped, rowvar=False)

deriv_nn_corrs_s90_reshaped = deriv_nn_corrs_s90.transpose(1, 0, 2).reshape(100, -1)
deriv_nn_corrs_s90_mean = np.mean(deriv_nn_corrs_s90_reshaped, axis=0)
deriv_nn_corrs_s90_cov = np.cov(deriv_nn_corrs_s90_reshaped, rowvar=False)

deriv_pn_corrs_s15_reshaped = deriv_pn_corrs_s15.transpose(1, 0, 2).reshape(100, -1)
deriv_pn_corrs_s15_mean = np.mean(deriv_pn_corrs_s15_reshaped, axis=0)
deriv_pn_corrs_s15_cov = np.cov(deriv_pn_corrs_s15_reshaped, rowvar=False)

deriv_pn_corrs_s30_reshaped = deriv_pn_corrs_s30.transpose(1, 0, 2).reshape(100, -1)
deriv_pn_corrs_s30_mean = np.mean(deriv_pn_corrs_s30_reshaped, axis=0)
deriv_pn_corrs_s30_cov = np.cov(deriv_pn_corrs_s30_reshaped, rowvar=False)

deriv_pn_corrs_s90_reshaped = deriv_pn_corrs_s90.transpose(1, 0, 2).reshape(100, -1)
deriv_pn_corrs_s90_mean = np.mean(deriv_pn_corrs_s90_reshaped, axis=0)
deriv_pn_corrs_s90_cov = np.cov(deriv_pn_corrs_s90_reshaped, rowvar=False)

deriv_pk_corrs_s15_reshaped = deriv_pk_corrs_s15.transpose(1, 0, 2).reshape(100, -1)
deriv_pk_corrs_s15_mean = np.mean(deriv_pk_corrs_s15_reshaped, axis=0)
deriv_pk_corrs_s15_cov = np.cov(deriv_pk_corrs_s15_reshaped, rowvar=False)

deriv_pk_corrs_s30_reshaped = deriv_pk_corrs_s30.transpose(1, 0, 2).reshape(100, -1)
deriv_pk_corrs_s30_mean = np.mean(deriv_pk_corrs_s30_reshaped, axis=0)
deriv_pk_corrs_s30_cov = np.cov(deriv_pk_corrs_s30_reshaped, rowvar=False)

deriv_pk_corrs_s90_reshaped = deriv_pk_corrs_s90.transpose(1, 0, 2).reshape(100, -1)
deriv_pk_corrs_s90_mean = np.mean(deriv_pk_corrs_s90_reshaped, axis=0)
deriv_pk_corrs_s90_cov = np.cov(deriv_pk_corrs_s90_reshaped, rowvar=False)

deriv_nk_corrs_s15_reshaped = deriv_nk_corrs_s15.transpose(1, 0, 2).reshape(100, -1)
deriv_nk_corrs_s15_mean = np.mean(deriv_nk_corrs_s15_reshaped, axis=0)
deriv_nk_corrs_s15_cov = np.cov(deriv_nk_corrs_s15_reshaped, rowvar=False)

deriv_nk_corrs_s30_reshaped = deriv_nk_corrs_s30.transpose(1, 0, 2).reshape(100, -1)
deriv_nk_corrs_s30_mean = np.mean(deriv_nk_corrs_s30_reshaped, axis=0)
deriv_nk_corrs_s30_cov = np.cov(deriv_nk_corrs_s30_reshaped, rowvar=False)

deriv_nk_corrs_s90_reshaped = deriv_nk_corrs_s90.transpose(1, 0, 2).reshape(100, -1)
deriv_nk_corrs_s90_mean = np.mean(deriv_nk_corrs_s90_reshaped, axis=0)
deriv_nk_corrs_s90_cov = np.cov(deriv_nk_corrs_s90_reshaped, rowvar=False)

deriv_nns_combined_s15_reshaped = deriv_nns_combined_s15.transpose(1, 0, 2).reshape(100, -1)
deriv_nns_combined_s15_mean = np.mean(deriv_nns_combined_s15_reshaped, axis=0)
deriv_nns_combined_s15_cov = np.cov(deriv_nns_combined_s15_reshaped, rowvar=False)

deriv_nns_combined_s30_reshaped = deriv_nns_combined_s30.transpose(1, 0, 2).reshape(100, -1)
deriv_nns_combined_s30_mean = np.mean(deriv_nns_combined_s30_reshaped, axis=0)
deriv_nns_combined_s30_cov = np.cov(deriv_nns_combined_s30_reshaped, rowvar=False)

deriv_nns_combined_s90_reshaped = deriv_nns_combined_s90.transpose(1, 0, 2).reshape(100, -1)
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

pp_fid_s15 = np.load(f'{MOCK_DIR}/PosPos_mean_s15.npy')
cov_pp_fid_s15 = np.load(f'{MOCK_DIR}/CovPosPos_s15.npy')
pp_fid_s30 = np.load(f'{MOCK_DIR}/PosPos_mean_s30.npy')
cov_pp_fid_s30 = np.load(f'{MOCK_DIR}/CovPosPos_s30.npy')
pp_fid_s90 = np.load(f'{MOCK_DIR}/PosPos_mean_s90.npy')
cov_pp_fid_s90 = np.load(f'{MOCK_DIR}/CovPosPos_s90.npy')

nn_corr_fid_s15 = np.load(f'{MOCK_DIR}/NegNeg_mean_s15.npy')
cov_nn_corr_fid_s15 = np.load(f'{MOCK_DIR}/CovNegNeg_s15.npy')
nn_corr_fid_s30 = np.load(f'{MOCK_DIR}/NegNeg_mean_s30.npy')
cov_nn_corr_fid_s30 = np.load(f'{MOCK_DIR}/CovNegNeg_s30.npy')
nn_corr_fid_s90 = np.load(f'{MOCK_DIR}/NegNeg_mean_s90.npy')
cov_nn_corr_fid_s90 = np.load(f'{MOCK_DIR}/CovNegNeg_s90.npy')

pn_fid_s15 = np.load(f'{MOCK_DIR}/PosNeg_mean_s15.npy')
cov_pn_fid_s15 = np.load(f'{MOCK_DIR}/CovPosNeg_s15.npy')
pn_fid_s30 = np.load(f'{MOCK_DIR}/PosNeg_mean_s30.npy')
cov_pn_fid_s30 = np.load(f'{MOCK_DIR}/CovPosNeg_s30.npy')
pn_fid_s90 = np.load(f'{MOCK_DIR}/PosNeg_mean_s90.npy')
cov_pn_fid_s90 = np.load(f'{MOCK_DIR}/CovPosNeg_s90.npy')

pk_fid_s15 = np.load(f'{MOCK_DIR}/PosKappa_mean_s15.npy')
cov_pk_fid_s15 = np.load(f'{MOCK_DIR}/CovPosKappa_s15.npy')
pk_fid_s30 = np.load(f'{MOCK_DIR}/PosKappa_mean_s30.npy')
cov_pk_fid_s30 = np.load(f'{MOCK_DIR}/CovPosKappa_s30.npy')
pk_fid_s90 = np.load(f'{MOCK_DIR}/PosKappa_mean_s90.npy')
cov_pk_fid_s90 = np.load(f'{MOCK_DIR}/CovPosKappa_s90.npy')

nk_fid_s15 = np.load(f'{MOCK_DIR}/NegKappa_mean_s15.npy')
cov_nk_fid_s15 = np.load(f'{MOCK_DIR}/CovNegKappa_s15.npy')
nk_fid_s30 = np.load(f'{MOCK_DIR}/NegKappa_mean_s30.npy')
cov_nk_fid_s30 = np.load(f'{MOCK_DIR}/CovNegKappa_s30.npy')
nk_fid_s90 = np.load(f'{MOCK_DIR}/NegKappa_mean_s90.npy')
cov_nk_fid_s90 = np.load(f'{MOCK_DIR}/CovNegKappa_s90.npy')

# plot derivs
import matplotlib.pyplot as plt
import matplotlib as mpl
mpl.rcParams['xtick.direction'] = 'in'
mpl.rcParams['ytick.direction'] = 'in'
mpl.rcParams['font.size'] = 8

N_fid = 500
N_deriv = n_seeds

# pdfs, 1x3, share y axis
fig, axes = plt.subplots(1, 3, figsize=(6, 2), sharey=True)

# Subplot 1: s15
ax = axes[0]
bin_edges = np.linspace(-0.01, 0.01, 11)
bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
print(f"Bin centers for s15: {bin_centers}")
print(deriv_pdfs_s15_mean)
ax.plot(bin_centers, pdfs_fid_s15, c='black')
ax.errorbar(bin_centers, pdfs_fid_s15, yerr=np.sqrt(np.diag(cov_pdfs_fid_s15)), fmt='|', label='Fiducial', c='black')
ax.errorbar(bin_centers, deriv_pdfs_s15_mean[0:10], yerr=np.sqrt(np.diag(deriv_pdfs_s15_cov[0:10, 0:10])), fmt='|', label=r'Derv. w.r.t $\Omega_m$')
ax.errorbar(bin_centers, deriv_pdfs_s15_mean[10:20], yerr=np.sqrt(np.diag(deriv_pdfs_s15_cov[10:20, 10:20])), fmt='|', label=r'Derv. w.r.t $\sigma_8$')
ax.fill_between(bin_centers, deriv_pdfs_s15_mean[0:10] - np.sqrt(np.diag(deriv_pdfs_s15_cov[0:10, 0:10])), deriv_pdfs_s15_mean[0:10] + np.sqrt(np.diag(deriv_pdfs_s15_cov[0:10, 0:10])), alpha=0.3)
ax.fill_between(bin_centers, deriv_pdfs_s15_mean[10:20] - np.sqrt(np.diag(deriv_pdfs_s15_cov[10:20, 10:20])), deriv_pdfs_s15_mean[10:20] + np.sqrt(np.diag(deriv_pdfs_s15_cov[10:20, 10:20])), alpha=0.3)
ax.set_title(r'$\theta_G$ = 15 arcmin')
ax.set_xlabel(r'$\kappa$')
ax.set_ylabel('Probability Density')
ax.set_xticks([-0.005, 0.005])
# ax.grid(True, alpha=0.3)


# Subplot 2: s30
ax = axes[1]
bin_edges = np.linspace(-0.008, 0.008, 11)
bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
ax.plot(bin_centers, pdfs_fid_s30, c='black')
ax.errorbar(bin_centers, pdfs_fid_s30, yerr=np.sqrt(np.diag(cov_pdfs_fid_s30)), fmt='|', label='Fiducial', c='black')
ax.errorbar(bin_centers, deriv_pdfs_s30_mean[0:10], yerr=np.sqrt(np.diag(deriv_pdfs_s30_cov[0:10, 0:10])), fmt='|', label=r'Derv. w.r.t $\Omega_m$')
ax.errorbar(bin_centers, deriv_pdfs_s30_mean[10:20], yerr=np.sqrt(np.diag(deriv_pdfs_s30_cov[10:20, 10:20])), fmt='|', label=r'Derv. w.r.t $\sigma_8$')
ax.fill_between(bin_centers, deriv_pdfs_s30_mean[0:10] - np.sqrt(np.diag(deriv_pdfs_s30_cov[0:10, 0:10])), deriv_pdfs_s30_mean[0:10] + np.sqrt(np.diag(deriv_pdfs_s30_cov[0:10, 0:10])), alpha=0.3)
ax.fill_between(bin_centers, deriv_pdfs_s30_mean[10:20] - np.sqrt(np.diag(deriv_pdfs_s30_cov[10:20, 10:20])), deriv_pdfs_s30_mean[10:20] + np.sqrt(np.diag(deriv_pdfs_s30_cov[10:20, 10:20])), alpha=0.3)
ax.set_title(r'$\theta_G$ = 30 arcmin')
ax.set_xlabel(r'$\kappa$')
ax.set_xticks([-0.004, 0.004])
# ax.grid(True, alpha=0.3)


# Subplot 3: s90
ax = axes[2]
bin_edges = np.linspace(-0.005, 0.005, 11)
bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
ax.plot(bin_centers, pdfs_fid_s90, c='black')
ax.errorbar(bin_centers, pdfs_fid_s90, yerr=np.sqrt(np.diag(cov_pdfs_fid_s90)), fmt='|', label='Fiducial', c='black')
ax.errorbar(bin_centers, deriv_pdfs_s90_mean[0:10], yerr=np.sqrt(np.diag(deriv_pdfs_s90_cov[0:10, 0:10])), fmt='|', label=r'Derv. w.r.t $\Omega_m$')
ax.errorbar(bin_centers, deriv_pdfs_s90_mean[10:20], yerr=np.sqrt(np.diag(deriv_pdfs_s90_cov[10:20, 10:20])), fmt='|', label=r'Derv. w.r.t $\sigma_8$')
ax.fill_between(bin_centers, deriv_pdfs_s90_mean[0:10] - np.sqrt(np.diag(deriv_pdfs_s90_cov[0:10, 0:10])), deriv_pdfs_s90_mean[0:10] + np.sqrt(np.diag(deriv_pdfs_s90_cov[0:10, 0:10])), alpha=0.3)
ax.fill_between(bin_centers, deriv_pdfs_s90_mean[10:20] - np.sqrt(np.diag(deriv_pdfs_s90_cov[10:20, 10:20])), deriv_pdfs_s90_mean[10:20] + np.sqrt(np.diag(deriv_pdfs_s90_cov[10:20, 10:20])), alpha=0.3)
ax.set_title(r'$\theta_G$ = 90 arcmin')
ax.set_xlabel(r'$\kappa$')
ax.set_xticks([-0.0025, 0.0025])
# ax.grid(True, alpha=0.3)


plt.legend(loc=(1.05, 0.25))
plt.subplots_adjust(wspace=0)
plt.savefig('outfiles/derivs_pdf.png', dpi=300, bbox_inches='tight')
plt.close()

# dpdfs
fig, axes = plt.subplots(2, 3, figsize=(6, 3), sharey='col') # 2x3 because pos and neg
# Subplot 1: s15 pos
ax = axes[0,0]
bin_edges = np.linspace(-0.01, 0.01, 11)
bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
ax.plot(bin_centers, dpdfs_pos_fid_s15, c='black')
ax.errorbar(bin_centers, dpdfs_pos_fid_s15, yerr=np.sqrt(np.diag(cov_dpdfs_pos_s15)), fmt='|', label='Fiducial', c='black')
ax.errorbar(bin_centers, deriv_dpdfs_pos_s15_mean[0:10], yerr=np.sqrt(np.diag(deriv_dpdfs_pos_s15_cov[0:10, 0:10])), fmt='|', label=r'Derv. w.r.t $\Omega_m$')
ax.errorbar(bin_centers, deriv_dpdfs_pos_s15_mean[10:20], yerr=np.sqrt(np.diag(deriv_dpdfs_pos_s15_cov[10:20, 10:20])), fmt='|', label=r'Derv. w.r.t $\sigma_8$')
ax.fill_between(bin_centers, deriv_dpdfs_pos_s15_mean[0:10] - np.sqrt(np.diag(deriv_dpdfs_pos_s15_cov[0:10, 0:10])), deriv_dpdfs_pos_s15_mean[0:10] + np.sqrt(np.diag(deriv_dpdfs_pos_s15_cov[0:10, 0:10])), alpha=0.3)
ax.fill_between(bin_centers, deriv_dpdfs_pos_s15_mean[10:20] - np.sqrt(np.diag(deriv_dpdfs_pos_s15_cov[10:20, 10:20])), deriv_dpdfs_pos_s15_mean[10:20] + np.sqrt(np.diag(deriv_dpdfs_pos_s15_cov[10:20, 10:20])), alpha=0.3)
ax.set_title(r'$\theta_G$ = 15 arcmin')
ax.set_xlabel(r'$\kappa$')
ax.set_ylabel(r'$C^p(\kappa)$')
ax.set_xticks([-0.005, 0.005])
ax.set_yticks([0, 500, 1000])
# ax.grid(True, alpha=0.3)

# Subplot 2: s30 pos
ax = axes[0,1]
bin_edges = np.linspace(-0.008, 0.008, 11)
bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
ax.plot(bin_centers, dpdfs_pos_fid_s30, c='black')
ax.errorbar(bin_centers, dpdfs_pos_fid_s30, yerr=np.sqrt(np.diag(cov_dpdfs_pos_s30)), fmt='|', label='Fiducial', c='black')   
ax.errorbar(bin_centers, deriv_dpdfs_pos_s30_mean[0:10], yerr=np.sqrt(np.diag(deriv_dpdfs_pos_s30_cov[0:10, 0:10])), fmt='|', label=r'Derv. w.r.t $\Omega_m$')
ax.errorbar(bin_centers, deriv_dpdfs_pos_s30_mean[10:20], yerr=np.sqrt(np.diag(deriv_dpdfs_pos_s30_cov[10:20, 10:20])), fmt='|', label=r'Derv. w.r.t $\sigma_8$')
ax.fill_between(bin_centers, deriv_dpdfs_pos_s30_mean[0:10] - np.sqrt(np.diag(deriv_dpdfs_pos_s30_cov[0:10, 0:10])), deriv_dpdfs_pos_s30_mean[0:10] + np.sqrt(np.diag(deriv_dpdfs_pos_s30_cov[0:10, 0:10])), alpha=0.3)
ax.fill_between(bin_centers, deriv_dpdfs_pos_s30_mean[10:20] - np.sqrt(np.diag(deriv_dpdfs_pos_s30_cov[10:20, 10:20])), deriv_dpdfs_pos_s30_mean[10:20] + np.sqrt(np.diag(deriv_dpdfs_pos_s30_cov[10:20, 10:20])), alpha=0.3)
ax.set_title(r'$\theta_G$ = 30 arcmin')
ax.set_xlabel(r'$\kappa$')
ax.set_xticks([-0.004, 0.004])
ax.set_yticks([0, 150, 300])
# ax.grid(True, alpha=0.3)

# Subplot 3: s90 pos
ax = axes[0,2]
bin_edges = np.linspace(-0.005, 0.005, 11)
bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
ax.plot(bin_centers, dpdfs_pos_fid_s90, c='black')
ax.errorbar(bin_centers, dpdfs_pos_fid_s90, yerr=np.sqrt(np.diag(cov_dpdfs_pos_s90)), fmt='|', label='Fiducial', c='black')   
ax.errorbar(bin_centers, deriv_dpdfs_pos_s90_mean[0:10], yerr=np.sqrt(np.diag(deriv_dpdfs_pos_s90_cov[0:10, 0:10])), fmt='|', label=r'Derv. w.r.t $\Omega_m$')
ax.errorbar(bin_centers, deriv_dpdfs_pos_s90_mean[10:20], yerr=np.sqrt(np.diag(deriv_dpdfs_pos_s90_cov[10:20, 10:20])), fmt='|', label=r'Derv. w.r.t $\sigma_8$')
ax.fill_between(bin_centers, deriv_dpdfs_pos_s90_mean[0:10] - np.sqrt(np.diag(deriv_dpdfs_pos_s90_cov[0:10, 0:10])), deriv_dpdfs_pos_s90_mean[0:10] + np.sqrt(np.diag(deriv_dpdfs_pos_s90_cov[0:10, 0:10])), alpha=0.3)
ax.fill_between(bin_centers, deriv_dpdfs_pos_s90_mean[10:20] - np.sqrt(np.diag(deriv_dpdfs_pos_s90_cov[10:20, 10:20])), deriv_dpdfs_pos_s90_mean[10:20] + np.sqrt(np.diag(deriv_dpdfs_pos_s90_cov[10:20, 10:20])), alpha=0.3)
ax.set_title(r'$\theta_G$ = 90 arcmin')
ax.set_xlabel(r'$\kappa$')
ax.set_yticks([0, 25, 50])
ax.set_xticks([-0.0025, 0.0, 0.0025])
# ax.grid(True, alpha=0.3)

# Subplot 4: s15 neg
ax = axes[1,0]
bin_edges = np.linspace(-0.01, 0.01, 11)
bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
ax.plot(bin_centers, dpdfs_neg_fid_s15, c='black')
ax.errorbar(bin_centers, dpdfs_neg_fid_s15, yerr=np.sqrt(np.diag(cov_dpdfs_neg_s15)), fmt='|', label='Fiducial', c='black')   
ax.errorbar(bin_centers, deriv_dpdfs_neg_s15_mean[0:10], yerr=np.sqrt(np.diag(deriv_dpdfs_neg_s15_cov[0:10, 0:10])), fmt='|', label=r'Derv. w.r.t $\Omega_m$')
ax.errorbar(bin_centers, deriv_dpdfs_neg_s15_mean[10:20], yerr=np.sqrt(np.diag(deriv_dpdfs_neg_s15_cov[10:20, 10:20])), fmt='|', label=r'Derv. w.r.t $\sigma_8$')
ax.fill_between(bin_centers, deriv_dpdfs_neg_s15_mean[0:10] - np.sqrt(np.diag(deriv_dpdfs_neg_s15_cov[0:10, 0:10])), deriv_dpdfs_neg_s15_mean[0:10] + np.sqrt(np.diag(deriv_dpdfs_neg_s15_cov[0:10, 0:10])), alpha=0.3)
ax.fill_between(bin_centers, deriv_dpdfs_neg_s15_mean[10:20] - np.sqrt(np.diag(deriv_dpdfs_neg_s15_cov[10:20, 10:20])), deriv_dpdfs_neg_s15_mean[10:20] + np.sqrt(np.diag(deriv_dpdfs_neg_s15_cov[10:20, 10:20])), alpha=0.3)
ax.set_xlabel(r'$\kappa$')
ax.set_ylabel(r'$C^n(\kappa)$')
ax.set_xticks([-0.005, 0.005])
ax.set_yticks([0, 500, 1000])
# ax.grid(True, alpha=0.3)

# Subplot 5: s30 neg
ax = axes[1,1]
bin_edges = np.linspace(-0.008, 0.008, 11)
bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
ax.plot(bin_centers, dpdfs_neg_fid_s30, c='black')
ax.errorbar(bin_centers, dpdfs_neg_fid_s30, yerr=np.sqrt(np.diag(cov_dpdfs_neg_s30)), fmt='|', label='Fiducial', c='black')   
ax.errorbar(bin_centers, deriv_dpdfs_neg_s30_mean[0:10], yerr=np.sqrt(np.diag(deriv_dpdfs_neg_s30_cov[0:10, 0:10])), fmt='|', label=r'Derv. w.r.t $\Omega_m$')
ax.errorbar(bin_centers, deriv_dpdfs_neg_s30_mean[10:20], yerr=np.sqrt(np.diag(deriv_dpdfs_neg_s30_cov[10:20, 10:20])), fmt='|', label=r'Derv. w.r.t $\sigma_8$')
ax.fill_between(bin_centers, deriv_dpdfs_neg_s30_mean[0:10] - np.sqrt(np.diag(deriv_dpdfs_neg_s30_cov[0:10, 0:10])), deriv_dpdfs_neg_s30_mean[0:10] + np.sqrt(np.diag(deriv_dpdfs_neg_s30_cov[0:10, 0:10])), alpha=0.3)
ax.fill_between(bin_centers, deriv_dpdfs_neg_s30_mean[10:20] - np.sqrt(np.diag(deriv_dpdfs_neg_s30_cov[10:20, 10:20])), deriv_dpdfs_neg_s30_mean[10:20] + np.sqrt(np.diag(deriv_dpdfs_neg_s30_cov[10:20, 10:20])), alpha=0.3)
ax.set_xlabel(r'$\kappa$')
ax.set_xticks([-0.004, 0.004])
ax.set_yticks([0, 150, 300])
# ax.grid(True, alpha=0.3)

# Subplot 6: s90 neg
ax = axes[1,2]
bin_edges = np.linspace(-0.005, 0.005, 11)
bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
ax.plot(bin_centers, dpdfs_neg_fid_s90, c='black')
ax.errorbar(bin_centers, dpdfs_neg_fid_s90, yerr=np.sqrt(np.diag(cov_dpdfs_neg_s90)), fmt='|', label='Fiducial', c='black')   
ax.errorbar(bin_centers, deriv_dpdfs_neg_s90_mean[0:10], yerr=np.sqrt(np.diag(deriv_dpdfs_neg_s90_cov[0:10, 0:10])), fmt='|', label=r'Derv. w.r.t $\Omega_m$')
ax.errorbar(bin_centers, deriv_dpdfs_neg_s90_mean[10:20], yerr=np.sqrt(np.diag(deriv_dpdfs_neg_s90_cov[10:20, 10:20])), fmt='|', label=r'Derv. w.r.t $\sigma_8$')
ax.fill_between(bin_centers, deriv_dpdfs_neg_s90_mean[0:10] - np.sqrt(np.diag(deriv_dpdfs_neg_s90_cov[0:10, 0:10])), deriv_dpdfs_neg_s90_mean[0:10] + np.sqrt(np.diag(deriv_dpdfs_neg_s90_cov[0:10, 0:10])), alpha=0.3)
ax.fill_between(bin_centers, deriv_dpdfs_neg_s90_mean[10:20] - np.sqrt(np.diag(deriv_dpdfs_neg_s90_cov[10:20, 10:20])), deriv_dpdfs_neg_s90_mean[10:20] + np.sqrt(np.diag(deriv_dpdfs_neg_s90_cov[10:20, 10:20])), alpha=0.3)
ax.set_xlabel(r'$\kappa$')
ax.set_yticks([0, 25, 50])
ax.set_xticks([-0.0025, 0.0025])
# ax.grid(True, alpha=0.3)
        
plt.legend(loc=(1.05, 0.75))
plt.subplots_adjust(hspace=0, wspace=0)
plt.savefig('outfiles/derivs_dpdf.png', dpi=300, bbox_inches='tight')
plt.close()

# nns only plot 2NN , pos same and pos opp, for all smoothing scales, so 2x3
fig, axes = plt.subplots(2, 3, figsize=(6, 3), sharex='col') 
# Row 1: 2NN pos same
# s15
ax = axes[0,0]
bin_centers = np.linspace(30, 200, 10)
ax.plot(bin_centers, nn_pos_same2_s15, c='black')
ax.errorbar(bin_centers, nn_pos_same2_s15, yerr=np.sqrt(np.diag(cov_nn_pos_same2_s15)), fmt='|', c='black')   
ax.errorbar(bin_centers, deriv_nns_combined_s15_mean[40:50], yerr=np.sqrt(np.diag(deriv_nns_combined_s15_cov[40:50, 40:50])), fmt='|', label=r'Derv. w.r.t $\Omega_m$')
ax.errorbar(bin_centers, deriv_nns_combined_s15_mean[200:210], yerr=np.sqrt(np.diag(deriv_nns_combined_s15_cov[200:210, 200:210])), fmt='|', label=r'Derv. w.r.t $\sigma_8$')
ax.fill_between(bin_centers, deriv_nns_combined_s15_mean[40:50] - np.sqrt(np.diag(deriv_nns_combined_s15_cov[40:50, 40:50])), deriv_nns_combined_s15_mean[40:50] + np.sqrt(np.diag(deriv_nns_combined_s15_cov[40:50, 40:50])), alpha=0.3)
ax.fill_between(bin_centers, deriv_nns_combined_s15_mean[200:210] - np.sqrt(np.diag(deriv_nns_combined_s15_cov[200:210, 200:210])), deriv_nns_combined_s15_mean[200:210] + np.sqrt(np.diag(deriv_nns_combined_s15_cov[200:210, 200:210])), alpha=0.3)
ax.set_title(r'$\theta_G$ = 15 arcmin')
ax.set_ylabel(r'$C^{pp}_{\rm 2NN}(\theta)$')
# ax.grid(True, alpha=0.3)

# s30
ax = axes[0,1]
bin_centers = np.linspace(30, 450, 10)
ax.plot(bin_centers, nn_pos_same2_s30, c='black')
ax.errorbar(bin_centers, nn_pos_same2_s30, yerr=np.sqrt(np.diag(cov_nn_pos_same2_s30)), fmt='|', c='black')   
ax.errorbar(bin_centers, deriv_nns_combined_s30_mean[40:50], yerr=np.sqrt(np.diag(deriv_nns_combined_s30_cov[40:50, 40:50])), fmt='|')
ax.errorbar(bin_centers, deriv_nns_combined_s30_mean[200:210], yerr=np.sqrt(np.diag(deriv_nns_combined_s30_cov[200:210, 200:210])), fmt='|')
ax.fill_between(bin_centers, deriv_nns_combined_s30_mean[40:50] - np.sqrt(np.diag(deriv_nns_combined_s30_cov[40:50, 40:50])), deriv_nns_combined_s30_mean[40:50] + np.sqrt(np.diag(deriv_nns_combined_s30_cov[40:50, 40:50])), alpha=0.3)
ax.fill_between(bin_centers, deriv_nns_combined_s30_mean[200:210] - np.sqrt(np.diag(deriv_nns_combined_s30_cov[200:210, 200:210])), deriv_nns_combined_s30_mean[200:210] + np.sqrt(np.diag(deriv_nns_combined_s30_cov[200:210, 200:210])), alpha=0.3)
ax.set_title(r'$\theta_G$ = 30 arcmin')
# ax.grid(True, alpha=0.3)

# s90
ax = axes[0,2]
bin_centers = np.linspace(30, 1125, 10)
ax.plot(bin_centers, nn_pos_same2_s90, c='black')
ax.errorbar(bin_centers, nn_pos_same2_s90, yerr=np.sqrt(np.diag(cov_nn_pos_same2_s90)), fmt='|', c='black')   
ax.errorbar(bin_centers, deriv_nns_combined_s90_mean[40:50], yerr=np.sqrt(np.diag(deriv_nns_combined_s90_cov[40:50, 40:50])), fmt='|')
ax.errorbar(bin_centers, deriv_nns_combined_s90_mean[200:210], yerr=np.sqrt(np.diag(deriv_nns_combined_s90_cov[200:210, 200:210])), fmt='|')
ax.fill_between(bin_centers, deriv_nns_combined_s90_mean[40:50] - np.sqrt(np.diag(deriv_nns_combined_s90_cov[40:50, 40:50])), deriv_nns_combined_s90_mean[40:50] + np.sqrt(np.diag(deriv_nns_combined_s90_cov[40:50, 40:50])), alpha=0.3)
ax.fill_between(bin_centers, deriv_nns_combined_s90_mean[200:210] - np.sqrt(np.diag(deriv_nns_combined_s90_cov[200:210, 200:210])), deriv_nns_combined_s90_mean[200:210] + np.sqrt(np.diag(deriv_nns_combined_s90_cov[200:210, 200:210])), alpha=0.3)
ax.set_title(r'$\theta_G$ = 90 arcmin')
ax.set_yticks([0, 25, 50])
# ax.grid(True, alpha=0.3)

# Row 2: 2NN pos opp
# s15
ax = axes[1,0]
bin_centers = np.linspace(30, 200, 10)
ax.plot(bin_centers, nn_pos_opp2_s15, c='black')
ax.errorbar(bin_centers, nn_pos_opp2_s15, yerr=np.sqrt(np.diag(cov_nn_pos_opp2_s15)), fmt='|', c='black')   
ax.errorbar(bin_centers, deriv_nns_combined_s15_mean[60:70], yerr=np.sqrt(np.diag(deriv_nns_combined_s15_cov[60:70, 60:70])), fmt='|')
ax.errorbar(bin_centers, deriv_nns_combined_s15_mean[220:230], yerr=np.sqrt(np.diag(deriv_nns_combined_s15_cov[220:230, 220:230])), fmt='|')
ax.fill_between(bin_centers, deriv_nns_combined_s15_mean[60:70] - np.sqrt(np.diag(deriv_nns_combined_s15_cov[60:70, 60:70])), deriv_nns_combined_s15_mean[60:70] + np.sqrt(np.diag(deriv_nns_combined_s15_cov[60:70, 60:70])), alpha=0.3)
ax.fill_between(bin_centers, deriv_nns_combined_s15_mean[220:230] - np.sqrt(np.diag(deriv_nns_combined_s15_cov[220:230, 220:230])), deriv_nns_combined_s15_mean[220:230] + np.sqrt(np.diag(deriv_nns_combined_s15_cov[220:230, 220:230])), alpha=0.3)
ax.set_ylabel(r'$C^{pn}_{\rm 2NN}(\theta)$')
ax.set_xticks([100, 150])
ax.set_xlabel(r'$\theta$ (arcmin)')
# ax.grid(True, alpha=0.3)

# s30
ax = axes[1,1]
bin_centers = np.linspace(30, 450, 10)
ax.plot(bin_centers, nn_pos_opp2_s30, c='black')
ax.errorbar(bin_centers, nn_pos_opp2_s30, yerr=np.sqrt(np.diag(cov_nn_pos_opp2_s30)), fmt='|', c='black')   
ax.errorbar(bin_centers, deriv_nns_combined_s30_mean[60:70], yerr=np.sqrt(np.diag(deriv_nns_combined_s30_cov[60:70, 60:70])), fmt='|')
ax.errorbar(bin_centers, deriv_nns_combined_s30_mean[220:230], yerr=np.sqrt(np.diag(deriv_nns_combined_s30_cov[220:230, 220:230])), fmt='|')
ax.fill_between(bin_centers, deriv_nns_combined_s30_mean[60:70] - np.sqrt(np.diag(deriv_nns_combined_s30_cov[60:70, 60:70])), deriv_nns_combined_s30_mean[60:70] + np.sqrt(np.diag(deriv_nns_combined_s30_cov[60:70, 60:70])), alpha=0.3)
ax.fill_between(bin_centers, deriv_nns_combined_s30_mean[220:230] - np.sqrt(np.diag(deriv_nns_combined_s30_cov[220:230, 220:230])), deriv_nns_combined_s30_mean[220:230] + np.sqrt(np.diag(deriv_nns_combined_s30_cov[220:230, 220:230])), alpha=0.3)
ax.set_xticks([150, 350])
ax.set_xlabel(r'$\theta$ (arcmin)')
# ax.grid(True, alpha=0.3)

# s90
ax = axes[1,2]
bin_centers = np.linspace(30, 1125, 10)
ax.plot(bin_centers, nn_pos_opp2_s90, c='black')
ax.errorbar(bin_centers, nn_pos_opp2_s90, yerr=np.sqrt(np.diag(cov_nn_pos_opp2_s90)), fmt='|', c='black', label='Fiducial')   
ax.errorbar(bin_centers, deriv_nns_combined_s90_mean[60:70], yerr=np.sqrt(np.diag(deriv_nns_combined_s90_cov[60:70, 60:70])), fmt='|', label=r'Derv. w.r.t $\Omega_m$')
ax.errorbar(bin_centers, deriv_nns_combined_s90_mean[220:230], yerr=np.sqrt(np.diag(deriv_nns_combined_s90_cov[220:230, 220:230])), fmt='|', label=r'Derv. w.r.t $\sigma_8$')
ax.fill_between(bin_centers, deriv_nns_combined_s90_mean[60:70] - np.sqrt(np.diag(deriv_nns_combined_s90_cov[60:70, 60:70])), deriv_nns_combined_s90_mean[60:70] + np.sqrt(np.diag(deriv_nns_combined_s90_cov[60:70, 60:70])), alpha=0.3)
ax.fill_between(bin_centers, deriv_nns_combined_s90_mean[220:230] - np.sqrt(np.diag(deriv_nns_combined_s90_cov[220:230, 220:230])), deriv_nns_combined_s90_mean[220:230] + np.sqrt(np.diag(deriv_nns_combined_s90_cov[220:230, 220:230])), alpha=0.3)
ax.set_xticks([600, 1000])
ax.set_xlabel(r'$\theta$ (arcmin)')
# ax.grid(True, alpha=0.3)

plt.legend(loc='upper left', bbox_to_anchor=(1.05, 1.25))
plt.subplots_adjust(hspace=0, wspace=0)
plt.savefig('outfiles/derivs_nns.png', dpi=300, bbox_inches='tight')
plt.close()


# plot xi_pp, xi_nn, xi_pn derivatives, all smoothing scales, 3x3
fig, axes = plt.subplots(3, 3, figsize=(12, 6), sharex='col', sharey='row')
# Row 1: xi_pp
# s15
ax = axes[0,0]
bin_centers = np.logspace(np.log10(15), np.log10(600), 30)
ax.plot(bin_centers, pp_fid_s15, c='black')
ax.errorbar(bin_centers, pp_fid_s15, yerr=np.sqrt(np.diag(cov_pp_fid_s15)), fmt='|', c='black')   
ax.errorbar(bin_centers, deriv_pp_corrs_s15_mean[0:30], yerr=np.sqrt(np.diag(deriv_pp_corrs_s15_cov[0:30, 0:30])), fmt='|', label=r'Derv. w.r.t $\Omega_m$')
ax.errorbar(bin_centers, deriv_pp_corrs_s15_mean[30:60], yerr=np.sqrt(np.diag(deriv_pp_corrs_s15_cov[30:60, 30:60])), fmt='|', label=r'Derv. w.r.t $\sigma_8$')
ax.fill_between(bin_centers, deriv_pp_corrs_s15_mean[0:30] - np.sqrt(np.diag(deriv_pp_corrs_s15_cov[0:30, 0:30])), deriv_pp_corrs_s15_mean[0:30] + np.sqrt(np.diag(deriv_pp_corrs_s15_cov[0:30, 0:30])), alpha=0.3)
ax.fill_between(bin_centers, deriv_pp_corrs_s15_mean[30:60] - np.sqrt(np.diag(deriv_pp_corrs_s15_cov[30:60, 30:60])), deriv_pp_corrs_s15_mean[30:60] + np.sqrt(np.diag(deriv_pp_corrs_s15_cov[30:60, 30:60])), alpha=0.3)
ax.set_title(r'$\theta_G$ = 15 arcmin')
ax.set_ylabel(r'$\xi^{pp}(\theta)$')
ax.set_xscale('log')
ax.set_xlim(15, 600)
# ax.grid(True, alpha=0.3)


# Subplot 2: s30
ax = axes[0,1]
bin_centers = np.logspace(np.log10(30), np.log10(600), 30)
ax.plot(bin_centers, pp_fid_s30, c='black')
ax.errorbar(bin_centers, pp_fid_s30, yerr=np.sqrt(np.diag(cov_pp_fid_s30)), fmt='|', c='black')
ax.errorbar(bin_centers, deriv_pp_corrs_s30_mean[0:30], yerr=np.sqrt(np.diag(deriv_pp_corrs_s30_cov[0:30, 0:30])), fmt='|')
ax.errorbar(bin_centers, deriv_pp_corrs_s30_mean[30:60], yerr=np.sqrt(np.diag(deriv_pp_corrs_s30_cov[30:60, 30:60])), fmt='|')
ax.fill_between(bin_centers, deriv_pp_corrs_s30_mean[0:30] - np.sqrt(np.diag(deriv_pp_corrs_s30_cov[0:30, 0:30])), deriv_pp_corrs_s30_mean[0:30] + np.sqrt(np.diag(deriv_pp_corrs_s30_cov[0:30, 0:30])), alpha=0.3)
ax.fill_between(bin_centers, deriv_pp_corrs_s30_mean[30:60] - np.sqrt(np.diag(deriv_pp_corrs_s30_cov[30:60, 30:60])), deriv_pp_corrs_s30_mean[30:60] + np.sqrt(np.diag(deriv_pp_corrs_s30_cov[30:60, 30:60])), alpha=0.3)
ax.set_title(r'$\theta_G$ = 30 arcmin')
ax.set_xscale('log')
ax.set_xlim(30, 600)

# s90
ax = axes[0,2]
bin_centers = np.logspace(np.log10(90), np.log10(600), 30)
ax.plot(bin_centers, pp_fid_s90, c='black')
ax.errorbar(bin_centers, pp_fid_s90, yerr=np.sqrt(np.diag(cov_pp_fid_s90)), fmt='|', c='black')
ax.errorbar(bin_centers, deriv_pp_corrs_s90_mean[0:30], yerr=np.sqrt(np.diag(deriv_pp_corrs_s90_cov[0:30, 0:30])), fmt='|')
ax.errorbar(bin_centers, deriv_pp_corrs_s90_mean[30:60], yerr=np.sqrt(np.diag(deriv_pp_corrs_s90_cov[30:60, 30:60])), fmt='|')
ax.fill_between(bin_centers, deriv_pp_corrs_s90_mean[0:30] - np.sqrt(np.diag(deriv_pp_corrs_s90_cov[0:30, 0:30])), deriv_pp_corrs_s90_mean[0:30] + np.sqrt(np.diag(deriv_pp_corrs_s90_cov[0:30, 0:30])), alpha=0.3)
ax.fill_between(bin_centers, deriv_pp_corrs_s90_mean[30:60] - np.sqrt(np.diag(deriv_pp_corrs_s90_cov[30:60, 30:60])), deriv_pp_corrs_s90_mean[30:60] + np.sqrt(np.diag(deriv_pp_corrs_s90_cov[30:60, 30:60])), alpha=0.3)
ax.set_title(r'$\theta_G$ = 90 arcmin')
ax.set_xscale('log')
ax.set_xlim(90, 600)

# Row 2: xi_nn
ax = axes[1,0]
bin_centers = np.logspace(np.log10(15), np.log10(600), 30)
ax.plot(bin_centers, nn_corr_fid_s15, c='black')
ax.errorbar(bin_centers, nn_corr_fid_s15, yerr=np.sqrt(np.diag(cov_nn_corr_fid_s15)), fmt='|', c='black')
ax.errorbar(bin_centers, deriv_nn_corrs_s15_mean[0:30], yerr=np.sqrt(np.diag(deriv_nn_corrs_s15_cov[0:30, 0:30])), fmt='|')
ax.errorbar(bin_centers, deriv_nn_corrs_s15_mean[30:60], yerr=np.sqrt(np.diag(deriv_nn_corrs_s15_cov[30:60, 30:60])), fmt='|')
ax.fill_between(bin_centers, deriv_nn_corrs_s15_mean[0:30] - np.sqrt(np.diag(deriv_nn_corrs_s15_cov[0:30, 0:30])), deriv_nn_corrs_s15_mean[0:30] + np.sqrt(np.diag(deriv_nn_corrs_s15_cov[0:30, 0:30])), alpha=0.3)
ax.fill_between(bin_centers, deriv_nn_corrs_s15_mean[30:60] - np.sqrt(np.diag(deriv_nn_corrs_s15_cov[30:60, 30:60])), deriv_nn_corrs_s15_mean[30:60] + np.sqrt(np.diag(deriv_nn_corrs_s15_cov[30:60, 30:60])), alpha=0.3)
ax.set_ylabel(r'$\xi^{nn}(\theta)$')
ax.set_xscale('log')

ax = axes[1,1]
bin_centers = np.logspace(np.log10(30), np.log10(600), 30)
ax.plot(bin_centers, nn_corr_fid_s30, c='black')
ax.errorbar(bin_centers, nn_corr_fid_s30, yerr=np.sqrt(np.diag(cov_nn_corr_fid_s30)), fmt='|', c='black')
ax.errorbar(bin_centers, deriv_nn_corrs_s30_mean[0:30], yerr=np.sqrt(np.diag(deriv_nn_corrs_s30_cov[0:30, 0:30])), fmt='|')
ax.errorbar(bin_centers, deriv_nn_corrs_s30_mean[30:60], yerr=np.sqrt(np.diag(deriv_nn_corrs_s30_cov[30:60, 30:60])), fmt='|')
ax.fill_between(bin_centers, deriv_nn_corrs_s30_mean[0:30] - np.sqrt(np.diag(deriv_nn_corrs_s30_cov[0:30, 0:30])), deriv_nn_corrs_s30_mean[0:30] + np.sqrt(np.diag(deriv_nn_corrs_s30_cov[0:30, 0:30])), alpha=0.3)
ax.fill_between(bin_centers, deriv_nn_corrs_s30_mean[30:60] - np.sqrt(np.diag(deriv_nn_corrs_s30_cov[30:60, 30:60])), deriv_nn_corrs_s30_mean[30:60] + np.sqrt(np.diag(deriv_nn_corrs_s30_cov[30:60, 30:60])), alpha=0.3)
ax.set_xscale('log')

ax = axes[1,2]
bin_centers = np.logspace(np.log10(90), np.log10(600), 30)
ax.plot(bin_centers, nn_corr_fid_s90, c='black')
ax.errorbar(bin_centers, nn_corr_fid_s90, yerr=np.sqrt(np.diag(cov_nn_corr_fid_s90)), fmt='|', c='black')
ax.errorbar(bin_centers, deriv_nn_corrs_s90_mean[0:30], yerr=np.sqrt(np.diag(deriv_nn_corrs_s90_cov[0:30, 0:30])), fmt='|')
ax.errorbar(bin_centers, deriv_nn_corrs_s90_mean[30:60], yerr=np.sqrt(np.diag(deriv_nn_corrs_s90_cov[30:60, 30:60])), fmt='|')
ax.fill_between(bin_centers, deriv_nn_corrs_s90_mean[0:30] - np.sqrt(np.diag(deriv_nn_corrs_s90_cov[0:30, 0:30])), deriv_nn_corrs_s90_mean[0:30] + np.sqrt(np.diag(deriv_nn_corrs_s90_cov[0:30, 0:30])), alpha=0.3)
ax.fill_between(bin_centers, deriv_nn_corrs_s90_mean[30:60] - np.sqrt(np.diag(deriv_nn_corrs_s90_cov[30:60, 30:60])), deriv_nn_corrs_s90_mean[30:60] + np.sqrt(np.diag(deriv_nn_corrs_s90_cov[30:60, 30:60])), alpha=0.3)
ax.set_xscale('log')

# Row 3: xi_pn
ax = axes[2,0]
bin_centers = np.logspace(np.log10(15), np.log10(600), 30)
ax.plot(bin_centers, pn_fid_s15, c='black')
ax.errorbar(bin_centers, pn_fid_s15, yerr=np.sqrt(np.diag(cov_pn_fid_s15)), fmt='|', c='black')
ax.errorbar(bin_centers, deriv_pn_corrs_s15_mean[0:30], yerr=np.sqrt(np.diag(deriv_pn_corrs_s15_cov[0:30, 0:30])), fmt='|')
ax.errorbar(bin_centers, deriv_pn_corrs_s15_mean[30:60], yerr=np.sqrt(np.diag(deriv_pn_corrs_s15_cov[30:60, 30:60])), fmt='|')
ax.fill_between(bin_centers, deriv_pn_corrs_s15_mean[0:30] - np.sqrt(np.diag(deriv_pn_corrs_s15_cov[0:30, 0:30])), deriv_pn_corrs_s15_mean[0:30] + np.sqrt(np.diag(deriv_pn_corrs_s15_cov[0:30, 0:30])), alpha=0.3)
ax.fill_between(bin_centers, deriv_pn_corrs_s15_mean[30:60] - np.sqrt(np.diag(deriv_pn_corrs_s15_cov[30:60, 30:60])), deriv_pn_corrs_s15_mean[30:60] + np.sqrt(np.diag(deriv_pn_corrs_s15_cov[30:60, 30:60])), alpha=0.3)
ax.set_ylabel(r'$\xi^{pn}(\theta)$')
ax.set_xscale('log')
ax.set_xlabel(r'$\theta$ (arcmin)')

ax = axes[2,1]
bin_centers = np.logspace(np.log10(30), np.log10(600), 30)
ax.plot(bin_centers, pn_fid_s30, c='black')
ax.errorbar(bin_centers, pn_fid_s30, yerr=np.sqrt(np.diag(cov_pn_fid_s30)), fmt='|', c='black')
ax.errorbar(bin_centers, deriv_pn_corrs_s30_mean[0:30], yerr=np.sqrt(np.diag(deriv_pn_corrs_s30_cov[0:30, 0:30])), fmt='|')
ax.errorbar(bin_centers, deriv_pn_corrs_s30_mean[30:60], yerr=np.sqrt(np.diag(deriv_pn_corrs_s30_cov[30:60, 30:60])), fmt='|')
ax.fill_between(bin_centers, deriv_pn_corrs_s30_mean[0:30] - np.sqrt(np.diag(deriv_pn_corrs_s30_cov[0:30, 0:30])), deriv_pn_corrs_s30_mean[0:30] + np.sqrt(np.diag(deriv_pn_corrs_s30_cov[0:30, 0:30])), alpha=0.3)
ax.fill_between(bin_centers, deriv_pn_corrs_s30_mean[30:60] - np.sqrt(np.diag(deriv_pn_corrs_s30_cov[30:60, 30:60])), deriv_pn_corrs_s30_mean[30:60] + np.sqrt(np.diag(deriv_pn_corrs_s30_cov[30:60, 30:60])), alpha=0.3)
ax.set_xscale('log')
ax.set_xlabel(r'$\theta$ (arcmin)')

ax = axes[2,2]
bin_centers = np.logspace(np.log10(90), np.log10(600), 30)
ax.plot(bin_centers, pn_fid_s90, c='black')
ax.errorbar(bin_centers, pn_fid_s90, yerr=np.sqrt(np.diag(cov_pn_fid_s90)), fmt='|', c='black')
ax.errorbar(bin_centers, deriv_pn_corrs_s90_mean[0:30], yerr=np.sqrt(np.diag(deriv_pn_corrs_s90_cov[0:30, 0:30])), fmt='|')
ax.errorbar(bin_centers, deriv_pn_corrs_s90_mean[30:60], yerr=np.sqrt(np.diag(deriv_pn_corrs_s90_cov[30:60, 30:60])), fmt='|')
ax.fill_between(bin_centers, deriv_pn_corrs_s90_mean[0:30] - np.sqrt(np.diag(deriv_pn_corrs_s90_cov[0:30, 0:30])), deriv_pn_corrs_s90_mean[0:30] + np.sqrt(np.diag(deriv_pn_corrs_s90_cov[0:30, 0:30])), alpha=0.3)
ax.fill_between(bin_centers, deriv_pn_corrs_s90_mean[30:60] - np.sqrt(np.diag(deriv_pn_corrs_s90_cov[30:60, 30:60])), deriv_pn_corrs_s90_mean[30:60] + np.sqrt(np.diag(deriv_pn_corrs_s90_cov[30:60, 30:60])), alpha=0.3)
ax.set_xscale('log')
ax.set_xlabel(r'$\theta$ (arcmin)')

for col in range(3):
    axes[0,col].tick_params(labelbottom=False)
    axes[1,col].tick_params(labelbottom=False)

handles, labels = axes[0,0].get_legend_handles_labels()
fig.legend(handles, labels, loc='upper center', ncol=3, bbox_to_anchor=(0.5, 1.02))
plt.subplots_adjust(hspace=0, wspace=0)
plt.savefig('outfiles/derivs_corrs.png', dpi=300, bbox_inches='tight')
plt.close()

# compare cdf+ cdf-
from scipy.interpolate import interp1d
color_map = plt.get_cmap('tab10')
plt.figure(figsize=(4, 3))
bin_edges = np.linspace(-0.005, 0.005, 11)
bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

# Create interpolation functions
interp_pos = interp1d(bin_centers, dpdfs_pos_fid_s90/dpdfs_pos_fid_s90[-1], kind='quadratic', fill_value=np.nan)
interp_neg = interp1d(bin_centers, dpdfs_neg_fid_s90/dpdfs_neg_fid_s90[-1], kind='quadratic', fill_value=np.nan)
interp_cdf = interp1d(bin_centers, np.cumsum(pdfs_fid_s90)/np.cumsum(pdfs_fid_s90)[-1], kind='quadratic', fill_value=np.nan)

# Create 100 bins for plotting
bin_centers_fine = np.linspace(bin_centers[0], bin_centers[-1], 100)

plt.plot(bin_centers_fine, interp_pos(bin_centers_fine), label='DCDF +', c=color_map(0))
plt.plot(bin_centers_fine, interp_neg(bin_centers_fine), label='DCDF -', c=color_map(1), linestyle='-.')
plt.plot(bin_centers_fine, interp_cdf(bin_centers_fine), label='CDF All Pixels', c='grey', linestyle='--')

plt.xlabel(r'$\kappa$')
plt.ylabel('Normalised Cumulative Distribution')
# plt.title('Comparison of CDFs')
plt.legend(loc='upper left')
plt.text(0.9, 0.1, 'Gaussian', ha='right', va='bottom', transform=plt.gca().transAxes, fontsize=12)
# plt.grid(True, alpha=0.3)
plt.savefig('outfiles/compare_cdf.png', dpi=300, bbox_inches='tight')
plt.close()

# compare 1,2,4,8 NN pos same/opp 
plt.figure(figsize=(4, 3))
bin_edges1 = np.linspace(30, 750, 10)
bin_edges2 = np.linspace(30, 1125, 10)
bin_edges4 = np.linspace(30, 1500, 10)
bin_edges8 = np.linspace(30, 1875, 10)

# Create interpolation functions
interp_1nn_same = interp1d(bin_edges1, nn_pos_same1_s90, kind='quadratic', fill_value=np.nan)
interp_1nn_opp = interp1d(bin_edges1, nn_pos_opp1_s90, kind='quadratic', fill_value=np.nan)
interp_2nn_same = interp1d(bin_edges2, nn_pos_same2_s90, kind='quadratic', fill_value=np.nan)
interp_2nn_opp = interp1d(bin_edges2, nn_pos_opp2_s90, kind='quadratic', fill_value=np.nan)
interp_4nn_same = interp1d(bin_edges4, nn_pos_same4_s90, kind='quadratic', fill_value=np.nan)
interp_4nn_opp = interp1d(bin_edges4, nn_pos_opp4_s90, kind='quadratic', fill_value=np.nan)
interp_8nn_same = interp1d(bin_edges8, nn_pos_same8_s90, kind='quadratic', fill_value=np.nan)
interp_8nn_opp = interp1d(bin_edges8, nn_pos_opp8_s90, kind='quadratic', fill_value=np.nan)

# Create 100 bins for plotting
bin_edges1_fine = np.linspace(bin_edges1[0], bin_edges1[-1], 100)
bin_edges2_fine = np.linspace(bin_edges2[0], bin_edges2[-1], 100)
bin_edges4_fine = np.linspace(bin_edges4[0], bin_edges4[-1], 100)
bin_edges8_fine = np.linspace(bin_edges8[0], bin_edges8[-1], 100)

plt.plot(bin_edges1_fine[5:], interp_1nn_same(bin_edges1_fine[5:]), label='1NN ++', c=color_map(0))
plt.plot(bin_edges1_fine[5:], interp_1nn_opp(bin_edges1_fine[5:]), label='1NN +-', c=color_map(0), linestyle='--')
plt.plot(bin_edges2_fine[10:], interp_2nn_same(bin_edges2_fine[10:]), label='2NN ++', c=color_map(1))
plt.plot(bin_edges2_fine[10:], interp_2nn_opp(bin_edges2_fine[10:]), label='2NN +-', c=color_map(1), linestyle='--')
plt.plot(bin_edges4_fine[20:], interp_4nn_same(bin_edges4_fine[20:]), label='4NN ++', c=color_map(2))
plt.plot(bin_edges4_fine[20:], interp_4nn_opp(bin_edges4_fine[20:]), label='4NN +-', c=color_map(2), linestyle='--')
plt.plot(bin_edges8_fine[30:], interp_8nn_same(bin_edges8_fine[30:]), label='8NN ++', c=color_map(3))
plt.plot(bin_edges8_fine[30:], interp_8nn_opp(bin_edges8_fine[30:]), label='8NN +-', c=color_map(3), linestyle='--')
plt.xlabel('Distance (arcmin)')
plt.ylabel('Cumulative Number Distribution')
# plt.title('Comparison of kNNs')
plt.legend()
# plt.grid(True, alpha=0.3)
plt.savefig('outfiles/compare_knn_cdf.png', dpi=300, bbox_inches='tight')
plt.close()

# compare ++, +-, --, -+ for 2NN at s90
plt.figure(figsize=(4.8, 3))
bin_edges2 = np.linspace(30, 1125, 10)

# Create interpolation functions for 2NN types
interp_2nn_pp = interp1d(bin_edges2, nn_pos_same2_s90, kind='quadratic', fill_value=np.nan)
interp_2nn_pn = interp1d(bin_edges2, nn_pos_opp2_s90, kind='quadratic', fill_value=np.nan)
interp_2nn_nn = interp1d(bin_edges2, nn_neg_same2_s90, kind='quadratic', fill_value=np.nan)
interp_2nn_np = interp1d(bin_edges2, nn_neg_opp2_s90, kind='quadratic', fill_value=np.nan)

# Create 100 bins for plotting
bin_edges2_fine = np.linspace(bin_edges2[0], bin_edges2[-1], 100)

plt.plot(bin_edges2_fine[6:], interp_2nn_pp(bin_edges2_fine[6:]), label='2NN ++', c=color_map(0))
plt.plot(bin_edges2_fine[6:], interp_2nn_pn(bin_edges2_fine[6:]), label='2NN +-', c=color_map(0), linestyle='--')
plt.plot(bin_edges2_fine[6:], interp_2nn_nn(bin_edges2_fine[6:]), label='2NN --', c=color_map(1))
plt.plot(bin_edges2_fine[6:], interp_2nn_np(bin_edges2_fine[6:]), label='2NN -+', c=color_map(1), linestyle='--')
plt.xlabel('Distance (arcmin)')
plt.ylabel('Cumulative Number Distribution')
# plt.title('Comparison of 2NN types at 90 arcmin')
plt.legend(loc='upper left')
plt.text(0.9, 0.1, 'Full-sky', ha='right', va='bottom', transform=plt.gca().transAxes, fontsize=12)
# plt.grid(True, alpha=0.3)
plt.savefig('outfiles/compare_2nn_types_cdf.png', dpi=300)
plt.close()


# dpdf, pdf cov

all_and_pdf_cov = np.load(f'{MOCK_DIR}/Defects_PDF_all_scales_cov.npy')
# take first 20 and last 10
cdf_cov = all_and_pdf_cov[np.ix_(list(range(20)) + list(range(all_and_pdf_cov.shape[0]-10, all_and_pdf_cov.shape[0])), list(range(20)) + list(range(all_and_pdf_cov.shape[0]-10, all_and_pdf_cov.shape[0])))]

plt.figure(figsize=(4, 3))
# normalize covariance to correlation matrix
d = np.sqrt(np.diag(cdf_cov))
corr = cdf_cov / np.outer(d, d)
plt.imshow(corr, cmap='RdBu', vmin=-1, vmax=1)
plt.colorbar(label='Correlation Coefficient')
# plt.title('Correlation Matrix (DCDF + PDF)')
# add lines to separate different parts
plt.axvline(x=9.5, color='black', linestyle='--', linewidth=1)
plt.axhline(y=9.5, color='black', linestyle='--', linewidth=1)
plt.axvline(x=19.5, color='black', linestyle='--', linewidth=1)
plt.axhline(y=19.5, color='black', linestyle='--', linewidth=1)

# ticks
plt.xticks([5, 15, 25], [r'DCDF+', r'DCDF-', r'PDF'])
plt.yticks([5, 15, 25], [r'DCDF+', r'DCDF-', r'PDF'])
plt.tick_params(axis='both', which='both', length=0)
plt.savefig('outfiles/corr_matrix_cdf.png', dpi=300, bbox_inches='tight')
plt.close()


