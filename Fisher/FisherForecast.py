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
for i in range(4):
    data = np.load(f"{DATA_DIR}/TrainPDFs/train_{i}_s15_kappa_pdf.npz")
    print("PDF S15:", data['hist'])
    pdfs_s15.append((data['hist'], data['bin_centers']))
    data = np.load(f"{DATA_DIR}/TrainPDFs/train_{i}_s30_kappa_pdf.npz")
    print("PDF S30:", data['hist'])
    pdfs_s30.append((data['hist'], data['bin_centers']))
    data = np.load(f"{DATA_DIR}/TrainPDFs/train_{i}_s90_kappa_pdf.npz")
    print("PDF S90:", data['hist'])
    pdfs_s90.append((data['hist'], data['bin_centers']))

    data_pos = np.load(f"{DATA_DIR}/TrainDPDFs/train_{i}_s15_defect_kappa_pos_pdf.npz")
    dpdfs_pos_s15.append((data_pos['hist'], data_pos['bin_centers']))
    print("DPDF POS S15:", data_pos['hist'])
    data_pos = np.load(f"{DATA_DIR}/TrainDPDFs/train_{i}_s30_defect_kappa_pos_pdf.npz")
    dpdfs_pos_s30.append((data_pos['hist'], data_pos['bin_centers']))
    print("DPDF POS S30:", data_pos['hist'])
    data_pos = np.load(f"{DATA_DIR}/TrainDPDFs/train_{i}_s90_defect_kappa_pos_pdf.npz")
    dpdfs_pos_s90.append((data_pos['hist'], data_pos['bin_centers']))
    print("DPDF POS S90:", data_pos['hist'])

    data_neg = np.load(f"{DATA_DIR}/TrainDPDFs/train_{i}_s15_defect_kappa_neg_pdf.npz")
    dpdfs_neg_s15.append((data_neg['hist'], data_neg['bin_centers']))
    data_neg = np.load(f"{DATA_DIR}/TrainDPDFs/train_{i}_s30_defect_kappa_neg_pdf.npz")
    dpdfs_neg_s30.append((data_neg['hist'], data_neg['bin_centers']))
    data_neg = np.load(f"{DATA_DIR}/TrainDPDFs/train_{i}_s90_defect_kappa_neg_pdf.npz")
    dpdfs_neg_s90.append((data_neg['hist'], data_neg['bin_centers']))

    data_kk = np.load(f"{DATA_DIR}/TrainKKs/train_{i}_s15_kappa_kappa_corr.npz")
    kk_corrs_s15.append((data_kk['theta'], data_kk['xi_kk']))
    data_kk = np.load(f"{DATA_DIR}/TrainKKs/train_{i}_s30_kappa_kappa_corr.npz")
    kk_corrs_s30.append((data_kk['theta'], data_kk['xi_kk']))
    data_kk = np.load(f"{DATA_DIR}/TrainKKs/train_{i}_s90_kappa_kappa_corr.npz")
    kk_corrs_s90.append((data_kk['theta'], data_kk['xi_kk']))

    data_pp = np.load(f"{DATA_DIR}/TrainPosPos/train_{i}_s15_pos_pos_corr.npz")
    pp_corrs_s15.append((data_pp['theta'], data_pp['xi_pp']))
    data_pp = np.load(f"{DATA_DIR}/TrainPosPos/train_{i}_s30_pos_pos_corr.npz")
    pp_corrs_s30.append((data_pp['theta'], data_pp['xi_pp']))
    data_pp = np.load(f"{DATA_DIR}/TrainPosPos/train_{i}_s90_pos_pos_corr.npz")
    pp_corrs_s90.append((data_pp['theta'], data_pp['xi_pp']))

    data_nn = np.load(f"{DATA_DIR}/TrainNegNeg/train_{i}_s15_neg_neg_corr.npz")
    nn_corrs_s15.append((data_nn['theta'], data_nn['xi_nn']))
    data_nn = np.load(f"{DATA_DIR}/TrainNegNeg/train_{i}_s30_neg_neg_corr.npz")
    nn_corrs_s30.append((data_nn['theta'], data_nn['xi_nn']))
    data_nn = np.load(f"{DATA_DIR}/TrainNegNeg/train_{i}_s90_neg_neg_corr.npz")
    nn_corrs_s90.append((data_nn['theta'], data_nn['xi_nn']))

    data_pn = np.load(f"{DATA_DIR}/TrainPosNeg/train_{i}_s15_pos_neg_corr.npz")
    pn_corrs_s15.append((data_pn['theta'], data_pn['xi_pn']))
    data_pn = np.load(f"{DATA_DIR}/TrainPosNeg/train_{i}_s30_pos_neg_corr.npz")
    pn_corrs_s30.append((data_pn['theta'], data_pn['xi_pn']))
    data_pn = np.load(f"{DATA_DIR}/TrainPosNeg/train_{i}_s90_pos_neg_corr.npz")
    pn_corrs_s90.append((data_pn['theta'], data_pn['xi_pn']))

    data_pk = np.load(f"{DATA_DIR}/TrainPKs/train_{i}_s15_pos_kappa_corr.npz")
    pk_corrs_s15.append((data_pk['theta'], data_pk['xi_pk']))
    data_pk = np.load(f"{DATA_DIR}/TrainPKs/train_{i}_s30_pos_kappa_corr.npz")
    pk_corrs_s30.append((data_pk['theta'], data_pk['xi_pk']))
    data_pk = np.load(f"{DATA_DIR}/TrainPKs/train_{i}_s90_pos_kappa_corr.npz")
    pk_corrs_s90.append((data_pk['theta'], data_pk['xi_pk']))

    data_nk = np.load(f"{DATA_DIR}/TrainNKs/train_{i}_s15_neg_kappa_corr.npz")
    nk_corrs_s15.append((data_nk['theta'], data_nk['xi_nk']))
    data_nk = np.load(f"{DATA_DIR}/TrainNKs/train_{i}_s30_neg_kappa_corr.npz")
    nk_corrs_s30.append((data_nk['theta'], data_nk['xi_nk']))
    data_nk = np.load(f"{DATA_DIR}/TrainNKs/train_{i}_s90_neg_kappa_corr.npz")
    nk_corrs_s90.append((data_nk['theta'], data_nk['xi_nk']))

    

    data_nn_combined = np.load(f"{DATA_DIR}/TrainNNs/train_{i}_s15_NNs_combined.npz")
    print("NN COMBINED S15:", data_nn_combined['nn_combined'])
    nn_combined_s15.append((data_nn_combined['nn_combined']))
    data_nn_combined = np.load(f"{DATA_DIR}/TrainNNs/train_{i}_s30_NNs_combined.npz")
    print("NN COMBINED S30:", data_nn_combined['nn_combined']) 
    nn_combined_s30.append((data_nn_combined['nn_combined']))
    data_nn_combined = np.load(f"{DATA_DIR}/TrainNNs/train_{i}_s90_NNs_combined.npz")
    nn_combined_s90.append((data_nn_combined['nn_combined']))
    print("NN COMBINED S90:", data_nn_combined['nn_combined'])


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
    # pdf derivatives
    hist_plus, bin_centers_plus = pdfs_s15[i+1]
    hist_minus, bin_centers_minus = pdfs_s15[i]
    deriv_pdf = (hist_plus - hist_minus) / (2 * delta_params[list(fid.keys())[i//2]])
    deriv_pdfs_s15.append(deriv_pdf)
    hist_plus, bin_centers_plus = pdfs_s30[i+1]
    hist_minus, bin_centers_minus = pdfs_s30[i]
    deriv_pdf = (hist_plus - hist_minus) / (2 * delta_params[list(fid.keys())[i//2]])
    deriv_pdfs_s30.append(deriv_pdf)
    hist_plus, bin_centers_plus = pdfs_s90[i+1]
    hist_minus, bin_centers_minus = pdfs_s90[i]
    deriv_pdf = (hist_plus - hist_minus) / (2 * delta_params[list(fid.keys())[i//2]])
    deriv_pdfs_s90.append(deriv_pdf)


    # defect pdf derivatives
    hist_pos_plus, bin_centers_pos_plus = dpdfs_pos_s15[i+1]
    hist_pos_minus, bin_centers_pos_minus = dpdfs_pos_s15[i]
    deriv_dpdf_pos = (hist_pos_plus - hist_pos_minus) / (2 * delta_params[list(fid.keys())[i//2]])
    deriv_dpdfs_pos_s15.append(deriv_dpdf_pos)
    hist_pos_plus, bin_centers_pos_plus = dpdfs_pos_s30[i+1]
    hist_pos_minus, bin_centers_pos_minus = dpdfs_pos_s30[i]
    deriv_dpdf_pos = (hist_pos_plus - hist_pos_minus) / (2 * delta_params[list(fid.keys())[i//2]])
    deriv_dpdfs_pos_s30.append(deriv_dpdf_pos)
    hist_pos_plus, bin_centers_pos_plus = dpdfs_pos_s90[i+1]
    hist_pos_minus, bin_centers_pos_minus = dpdfs_pos_s90[i]
    deriv_dpdf_pos = (hist_pos_plus - hist_pos_minus) / (2 * delta_params[list(fid.keys())[i//2]])
    deriv_dpdfs_pos_s90.append(deriv_dpdf_pos)

    hist_neg_plus, bin_centers_neg_plus = dpdfs_neg_s15[i+1]
    hist_neg_minus, bin_centers_neg_minus = dpdfs_neg_s15[i]
    deriv_dpdf_neg = (hist_neg_plus - hist_neg_minus) / (2 * delta_params[list(fid.keys())[i//2]])
    deriv_dpdfs_neg_s15.append(deriv_dpdf_neg)
    hist_neg_plus, bin_centers_neg_plus = dpdfs_neg_s30[i+1]
    hist_neg_minus, bin_centers_neg_minus = dpdfs_neg_s30[i]
    deriv_dpdf_neg = (hist_neg_plus - hist_neg_minus) / (2 * delta_params[list(fid.keys())[i//2]])
    deriv_dpdfs_neg_s30.append(deriv_dpdf_neg)
    hist_neg_plus, bin_centers_neg_plus = dpdfs_neg_s90[i+1]
    hist_neg_minus, bin_centers_neg_minus = dpdfs_neg_s90[i]
    deriv_dpdf_neg = (hist_neg_plus - hist_neg_minus) / (2 * delta_params[list(fid.keys())[i//2]])
    deriv_dpdfs_neg_s90.append(deriv_dpdf_neg)

    # kk correlation derivatives
    theta_kk_plus, xi_kk_plus = kk_corrs_s15[i+1]
    theta_kk_minus, xi_kk_minus = kk_corrs_s15[i]
    deriv_kk_corr = (xi_kk_plus - xi_kk_minus) / (2 * delta_params[list(fid.keys())[i//2]])
    deriv_kk_corrs_s15.append(deriv_kk_corr)
    theta_kk_plus, xi_kk_plus = kk_corrs_s30[i+1]
    theta_kk_minus, xi_kk_minus = kk_corrs_s30[i]
    deriv_kk_corr = (xi_kk_plus - xi_kk_minus) / (2 * delta_params[list(fid.keys())[i//2]])
    deriv_kk_corrs_s30.append(deriv_kk_corr)
    theta_kk_plus, xi_kk_plus = kk_corrs_s90[i+1]
    theta_kk_minus, xi_kk_minus = kk_corrs_s90[i]
    deriv_kk_corr = (xi_kk_plus - xi_kk_minus) / (2 * delta_params[list(fid.keys())[i//2]])
    deriv_kk_corrs_s90.append(deriv_kk_corr)

    theta_pp_plus, xi_pp_plus = pp_corrs_s15[i+1]
    theta_pp_minus, xi_pp_minus = pp_corrs_s15[i]
    deriv_pp_corr = (xi_pp_plus - xi_pp_minus) / (2 * delta_params[list(fid.keys())[i//2]])
    deriv_pp_corrs_s15.append(deriv_pp_corr)
    theta_pp_plus, xi_pp_plus = pp_corrs_s30[i+1]
    theta_pp_minus, xi_pp_minus = pp_corrs_s30[i]
    deriv_pp_corr = (xi_pp_plus - xi_pp_minus) / (2 * delta_params[list(fid.keys())[i//2]])
    deriv_pp_corrs_s30.append(deriv_pp_corr)
    theta_pp_plus, xi_pp_plus = pp_corrs_s90[i+1]
    theta_pp_minus, xi_pp_minus = pp_corrs_s90[i]
    deriv_pp_corr = (xi_pp_plus - xi_pp_minus) / (2 * delta_params[list(fid.keys())[i//2]])
    deriv_pp_corrs_s90.append(deriv_pp_corr)

    theta_nn_plus, xi_nn_plus = nn_corrs_s15[i+1]
    theta_nn_minus, xi_nn_minus = nn_corrs_s15[i]
    deriv_nn_corr = (xi_nn_plus - xi_nn_minus) / (2 * delta_params[list(fid.keys())[i//2]])
    deriv_nn_corrs_s15.append(deriv_nn_corr)
    theta_nn_plus, xi_nn_plus = nn_corrs_s30[i+1]
    theta_nn_minus, xi_nn_minus = nn_corrs_s30[i]
    deriv_nn_corr = (xi_nn_plus - xi_nn_minus) / (2 * delta_params[list(fid.keys())[i//2]])
    deriv_nn_corrs_s30.append(deriv_nn_corr)
    theta_nn_plus, xi_nn_plus = nn_corrs_s90[i+1]
    theta_nn_minus, xi_nn_minus = nn_corrs_s90[i]
    deriv_nn_corr = (xi_nn_plus - xi_nn_minus) / (2 * delta_params[list(fid.keys())[i//2]])
    deriv_nn_corrs_s90.append(deriv_nn_corr)

    theta_pn_plus, xi_pn_plus = pn_corrs_s15[i+1]
    theta_pn_minus, xi_pn_minus = pn_corrs_s15[i]
    deriv_pn_corr = (xi_pn_plus - xi_pn_minus) / (2 * delta_params[list(fid.keys())[i//2]])
    deriv_pn_corrs_s15.append(deriv_pn_corr)
    theta_pn_plus, xi_pn_plus = pn_corrs_s30[i+1]
    theta_pn_minus, xi_pn_minus = pn_corrs_s30[i]
    deriv_pn_corr = (xi_pn_plus - xi_pn_minus) / (2 * delta_params[list(fid.keys())[i//2]])
    deriv_pn_corrs_s30.append(deriv_pn_corr)
    theta_pn_plus, xi_pn_plus = pn_corrs_s90[i+1]
    theta_pn_minus, xi_pn_minus = pn_corrs_s90[i]
    deriv_pn_corr = (xi_pn_plus - xi_pn_minus) / (2 * delta_params[list(fid.keys())[i//2]])
    deriv_pn_corrs_s90.append(deriv_pn_corr)

    theta_pk_plus, xi_pk_plus = pk_corrs_s15[i+1]
    theta_pk_minus, xi_pk_minus = pk_corrs_s15[i]
    deriv_pk_corr = (xi_pk_plus - xi_pk_minus) / (2 * delta_params[list(fid.keys())[i//2]])
    deriv_pk_corrs_s15.append(deriv_pk_corr)
    theta_pk_plus, xi_pk_plus = pk_corrs_s30[i+1]
    theta_pk_minus, xi_pk_minus = pk_corrs_s30[i]
    deriv_pk_corr = (xi_pk_plus - xi_pk_minus) / (2 * delta_params[list(fid.keys())[i//2]])
    deriv_pk_corrs_s30.append(deriv_pk_corr)
    theta_pk_plus, xi_pk_plus = pk_corrs_s90[i+1]
    theta_pk_minus, xi_pk_minus = pk_corrs_s90[i]
    deriv_pk_corr = (xi_pk_plus - xi_pk_minus) / (2 * delta_params[list(fid.keys())[i//2]])
    deriv_pk_corrs_s90.append(deriv_pk_corr)

    theta_nk_plus, xi_nk_plus = nk_corrs_s15[i+1]
    theta_nk_minus, xi_nk_minus = nk_corrs_s15[i]
    deriv_nk_corr = (xi_nk_plus - xi_nk_minus) / (2 * delta_params[list(fid.keys())[i//2]])
    deriv_nk_corrs_s15.append(deriv_nk_corr)
    theta_nk_plus, xi_nk_plus = nk_corrs_s30[i+1]
    theta_nk_minus, xi_nk_minus = nk_corrs_s30[i]
    deriv_nk_corr = (xi_nk_plus - xi_nk_minus) / (2 * delta_params[list(fid.keys())[i//2]])
    deriv_nk_corrs_s30.append(deriv_nk_corr)
    theta_nk_plus, xi_nk_plus = nk_corrs_s90[i+1]
    theta_nk_minus, xi_nk_minus = nk_corrs_s90[i]
    deriv_nk_corr = (xi_nk_plus - xi_nk_minus) / (2 * delta_params[list(fid.keys())[i//2]])
    deriv_nk_corrs_s90.append(deriv_nk_corr)
    
    nn_combined_plus = nn_combined_s15[i+1]
    nn_combined_minus = nn_combined_s15[i]
    deriv_nn_combined = (nn_combined_plus - nn_combined_minus) / (2 * delta_params[list(fid.keys())[i//2]]) 
    deriv_nns_combined_s15.append(deriv_nn_combined)
    nn_combined_plus = nn_combined_s30[i+1]
    nn_combined_minus = nn_combined_s30[i]
    deriv_nn_combined = (nn_combined_plus - nn_combined_minus) / (2 * delta_params[list(fid.keys())[i//2]]) 
    deriv_nns_combined_s30.append(deriv_nn_combined)
    nn_combined_plus = nn_combined_s90[i+1]
    nn_combined_minus = nn_combined_s90[i]
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
print("DERIV DPDFS COMBINED S90:", deriv_dpdfs_combined_s90)
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
deriv_dcorrs_combined_s15 = np.hstack((deriv_pp_corrs_s15, deriv_nn_corrs_s15, deriv_pn_corrs_s15))
deriv_dcorrs_combined_s30 = np.hstack((deriv_pp_corrs_s30, deriv_nn_corrs_s30, deriv_pn_corrs_s30))
deriv_dcorrs_combined_s90 = np.hstack((deriv_pp_corrs_s90, deriv_nn_corrs_s90, deriv_pn_corrs_s90))
deriv_dcorrs_combined_all_scales = np.hstack((deriv_dcorrs_combined_s15, deriv_dcorrs_combined_s30, deriv_dcorrs_combined_s90))
deriv_nns_combined_s15 = np.array(deriv_nns_combined_s15)
deriv_nns_combined_s30 = np.array(deriv_nns_combined_s30)
deriv_nns_combined_s90 = np.array(deriv_nns_combined_s90)
print("DERIV NNS COMBINED S90:", deriv_nns_combined_s90)
deriv_dpdf_nn_combined_s15 = np.hstack((deriv_dpdfs_combined_s15, deriv_nns_combined_s15))
deriv_dpdf_nn_combined_s30 = np.hstack((deriv_dpdfs_combined_s30, deriv_nns_combined_s30))
deriv_dpdf_nn_combined_s90 = np.hstack((deriv_dpdfs_combined_s90, deriv_nns_combined_s90))
deriv_dpdf_nn_combined_all_scales = np.hstack((deriv_dpdf_nn_combined_s15, deriv_dpdf_nn_combined_s30, deriv_dpdf_nn_combined_s90))
deriv_defects_pdf_combined_all_scales = np.hstack((deriv_dpdf_nn_combined_all_scales, deriv_pdfs_s15, deriv_pdfs_s30, deriv_pdfs_s90))

MOCKS_DATA_DIR = '../Mock'
# fisher matrix computation
n = 1000 # number of realizations
cov_pos_s15 = np.load(f"{MOCKS_DATA_DIR}/CovDPDF_pos_s15.npy")
inv_cov_pos_s15 = np.linalg.inv(cov_pos_s15) * (n-10-2) / (n-1)
cov_pos_s30 = np.load(f"{MOCKS_DATA_DIR}/CovDPDF_pos_s30.npy")
inv_cov_pos_s30 = np.linalg.inv(cov_pos_s30) * (n-10-2) / (n-1)
cov_pos_s90 = np.load(f"{MOCKS_DATA_DIR}/CovDPDF_pos_s90.npy")
inv_cov_pos_s90 = np.linalg.inv(cov_pos_s90) * (n-10-2) / (n-1)

cov_neg_s15 = np.load(f"{MOCKS_DATA_DIR}/CovDPDF_neg_s15.npy")
inv_cov_neg_s15 = np.linalg.inv(cov_neg_s15) * (n-10-2) / (n-1)
cov_neg_s30 = np.load(f"{MOCKS_DATA_DIR}/CovDPDF_neg_s30.npy")
inv_cov_neg_s30 = np.linalg.inv(cov_neg_s30) * (n-10-2) / (n-1)
cov_neg_s90 = np.load(f"{MOCKS_DATA_DIR}/CovDPDF_neg_s90.npy")
inv_cov_neg_s90 = np.linalg.inv(cov_neg_s90) * (n-10-2) / (n-1)

cov_dpdf_all_scales = np.load(f"{MOCKS_DATA_DIR}/DPDF_all_scales_cov.npy")
inv_cov_dpdf_all_scales = np.linalg.inv(cov_dpdf_all_scales) * (n-10*3*2-2) / (n-1)


cov_pdf_s15 = np.load(f"{MOCKS_DATA_DIR}/CovPDF_s15.npy")
inv_cov_pdf_s15 = np.linalg.inv(cov_pdf_s15) * (n-10-2) / (n-1)
cov_pdf_s30 = np.load(f"{MOCKS_DATA_DIR}/CovPDF_s30.npy")
inv_cov_pdf_s30 = np.linalg.inv(cov_pdf_s30) * (n-10-2) / (n-1)
cov_pdf_s90 = np.load(f"{MOCKS_DATA_DIR}/CovPDF_s90.npy")
inv_cov_pdf_s90 = np.linalg.inv(cov_pdf_s90) * (n-10-2) / (n-1)
cov_pdf_all_scales = np.load(f"{MOCKS_DATA_DIR}/PDF_all_scales_cov.npy")
inv_cov_pdf_all_scales = np.linalg.inv(cov_pdf_all_scales) * (n-10*3-2) / (n-1)


cov_dpdf_combined_s15 = np.load(f"{MOCKS_DATA_DIR}/CovDPDF_combined_s15.npy")
inv_cov_dpdf_combined_s15 = np.linalg.inv(cov_dpdf_combined_s15) * (n-10*2-2) / (n-1)
cov_dpdf_combined_s30 = np.load(f"{MOCKS_DATA_DIR}/CovDPDF_combined_s30.npy")
inv_cov_dpdf_combined_s30 = np.linalg.inv(cov_dpdf_combined_s30) * (n-10*2-2) / (n-1)
cov_dpdf_combined_s90 = np.load(f"{MOCKS_DATA_DIR}/CovDPDF_combined_s90.npy")
inv_cov_dpdf_combined_s90 = np.linalg.inv(cov_dpdf_combined_s90) * (n-10*2-2) / (n-1)
cov_dpdf_combined_all_scales = np.load(f"{MOCKS_DATA_DIR}/DPDF_all_scales_cov.npy")
inv_cov_dpdf_combined_all_scales = np.linalg.inv(cov_dpdf_combined_all_scales) * (n-10*2*3-2) / (n-1)

    
cov_kk_s15 = np.load(f"{MOCKS_DATA_DIR}/CovKK_s15.npy")
inv_cov_kk_s15 = np.linalg.inv(cov_kk_s15) * (n-30-2) / (n-1)
cov_kk_s30 = np.load(f"{MOCKS_DATA_DIR}/CovKK_s30.npy")
inv_cov_kk_s30 = np.linalg.inv(cov_kk_s30) * (n-30-2) / (n-1)
cov_kk_s90 = np.load(f"{MOCKS_DATA_DIR}/CovKK_s90.npy")
inv_cov_kk_s90 = np.linalg.inv(cov_kk_s90) * (n-30-2) / (n-1)
cov_kk_all_scales = np.load(f"{MOCKS_DATA_DIR}/CovKK_s15.npy")
inv_cov_kk_all_scales = np.linalg.inv(cov_kk_all_scales) * (n-30-2) / (n-1)

cov_pp_s15 = np.load(f"{MOCKS_DATA_DIR}/CovPosPos_s15.npy")
inv_cov_pp_s15 = np.linalg.inv(cov_pp_s15) * (n-30-2) / (n-1)
cov_pp_s30 = np.load(f"{MOCKS_DATA_DIR}/CovPosPos_s30.npy")
inv_cov_pp_s30 = np.linalg.inv(cov_pp_s30) * (n-30-2) / (n-1)
cov_pp_s90 = np.load(f"{MOCKS_DATA_DIR}/CovPosPos_s90.npy")
inv_cov_pp_s90 = np.linalg.inv(cov_pp_s90) * (n-30-2) / (n-1)

cov_nn_corr_s15 = np.load(f"{MOCKS_DATA_DIR}/CovNegNeg_s15.npy")
inv_cov_nn_corr_s15 = np.linalg.inv(cov_nn_corr_s15) * (n-30-2) / (n-1)
cov_nn_corr_s30 = np.load(f"{MOCKS_DATA_DIR}/CovNegNeg_s30.npy")
inv_cov_nn_corr_s30 = np.linalg.inv(cov_nn_corr_s30) * (n-30-2) / (n-1)
cov_nn_corr_s90 = np.load(f"{MOCKS_DATA_DIR}/CovNegNeg_s90.npy")
inv_cov_nn_corr_s90 = np.linalg.inv(cov_nn_corr_s90) * (n-30-2) / (n-1)

cov_pn_s15 = np.load(f"{MOCKS_DATA_DIR}/CovPosNeg_s15.npy")
inv_cov_pn_s15 = np.linalg.inv(cov_pn_s15) * (n-30-2) / (n-1)
cov_pn_s30 = np.load(f"{MOCKS_DATA_DIR}/CovPosNeg_s30.npy")
inv_cov_pn_s30 = np.linalg.inv(cov_pn_s30) * (n-30-2) / (n-1)
cov_pn_s90 = np.load(f"{MOCKS_DATA_DIR}/CovPosNeg_s90.npy")
inv_cov_pn_s90 = np.linalg.inv(cov_pn_s90) * (n-30-2) / (n-1)

cov_dcorrs_combined_s15 = np.load(f"{MOCKS_DATA_DIR}/CovDCorr_combined_s15.npy")
inv_cov_dcorrs_combined_s15 = np.linalg.inv(cov_dcorrs_combined_s15) * (n-30*3-2) / (n-1)
cov_dcorrs_combined_s30 = np.load(f"{MOCKS_DATA_DIR}/CovDCorr_combined_s30.npy")
inv_cov_dcorrs_combined_s30 = np.linalg.inv(cov_dcorrs_combined_s30) * (n-30*3-2) / (n-1)
cov_dcorrs_combined_s90 = np.load(f"{MOCKS_DATA_DIR}/CovDCorr_combined_s90.npy")
inv_cov_dcorrs_combined_s90 = np.linalg.inv(cov_dcorrs_combined_s90) * (n-30*3-2) / (n-1)
cov_dcorrs_all_scales = np.load(f"{MOCKS_DATA_DIR}/DCorr_all_scales_cov.npy")
inv_cov_dcorrs_all_scales = np.linalg.inv(cov_dcorrs_all_scales) * (n-30*3*3-2) / (n-1)

cov_pk_s15 = np.load(f"{MOCKS_DATA_DIR}/CovPosKappa_s15.npy")
inv_cov_pk_s15 = np.linalg.inv(cov_pk_s15) * (n-30-2) / (n-1)
cov_pk_s30 = np.load(f"{MOCKS_DATA_DIR}/CovPosKappa_s30.npy")
inv_cov_pk_s30 = np.linalg.inv(cov_pk_s30) * (n-30-2) / (n-1)
cov_pk_s90 = np.load(f"{MOCKS_DATA_DIR}/CovPosKappa_s90.npy")
inv_cov_pk_s90 = np.linalg.inv(cov_pk_s90) * (n-30-2) / (n-1)

cov_nk_s15 = np.load(f"{MOCKS_DATA_DIR}/CovNegKappa_s15.npy")
inv_cov_nk_s15 = np.linalg.inv(cov_nk_s15) * (n-30-2) / (n-1)
cov_nk_s30 = np.load(f"{MOCKS_DATA_DIR}/CovNegKappa_s30.npy")
inv_cov_nk_s30 = np.linalg.inv(cov_nk_s30) * (n-30-2) / (n-1)
cov_nk_s90 = np.load(f"{MOCKS_DATA_DIR}/CovNegKappa_s90.npy")
inv_cov_nk_s90 = np.linalg.inv(cov_nk_s90) * (n-30-2) / (n-1)


cov_nn_s15 = np.load(f"{MOCKS_DATA_DIR}/CovNNs_combined_s15.npy")
inv_cov_nn_s15 = np.linalg.inv(cov_nn_s15) * (n-16*10-2) / (n-1)
cov_nn_s30 = np.load(f"{MOCKS_DATA_DIR}/CovNNs_combined_s30.npy")
inv_cov_nn_s30 = np.linalg.inv(cov_nn_s30) * (n-16*10-2) / (n-1)
cov_nn_s90 = np.load(f"{MOCKS_DATA_DIR}/CovNNs_combined_s90.npy")
inv_cov_nn_s90 = np.linalg.inv(cov_nn_s90) * (n-16*10-2) / (n-1)
cov_nn_all_scales = np.load(f"{MOCKS_DATA_DIR}/NN_all_scales_cov.npy")
inv_cov_nn_all_scales = np.linalg.inv(cov_nn_all_scales) * (n-16*10*3-2) / (n-1)
cov_dpdf_nn_combined_s15 = np.load(f"{MOCKS_DATA_DIR}/CovCDF_combined_s15.npy")
inv_cov_dpdf_nn_combined_s15 = np.linalg.inv(cov_dpdf_nn_combined_s15) * (n-16*10-20-2) / (n-1)
cov_dpdf_nn_combined_s30 = np.load(f"{MOCKS_DATA_DIR}/CovCDF_combined_s30.npy")
inv_cov_dpdf_nn_combined_s30 = np.linalg.inv(cov_dpdf_nn_combined_s30) * (n-16*10-20-2) / (n-1)
cov_dpdf_nn_combined_s90 = np.load(f"{MOCKS_DATA_DIR}/CovCDF_combined_s90.npy")
inv_cov_dpdf_nn_combined_s90 = np.linalg.inv(cov_dpdf_nn_combined_s90 ) * (n-16*10-20-2) / (n-1)
cov_dpdf_nn_combined_all_scales = np.load(f"{MOCKS_DATA_DIR}/Defects_all_scales_cov.npy")
inv_cov_dpdf_nn_combined_all_scales = np.linalg.inv(cov_dpdf_nn_combined_all_scales) * (n-16*10*3-20*3-2) / (n-1)

cov_defects_pdf_combined_all_scales = np.load(f"{MOCKS_DATA_DIR}/Defects_PDF_all_scales_cov.npy")
inv_cov_defects_pdf_combined_all_scales = np.linalg.inv(cov_defects_pdf_combined_all_scales) * (n-16*10*3-20*3-10*3-2) / (n-1)

cov_all_all_scales = np.load(f"{MOCKS_DATA_DIR}/All_all_scales_cov.npy")
inv_cov_all_all_scales = np.linalg.inv(cov_all_all_scales) * (n-16*10*3-20*3-10*3-30-2) / (n-1)

# compute fisher matrix
n_params = len(fid)
fisher_matrix_pdfs_s15 = np.zeros((n_params, n_params))
fisher_matrix_pdfs_s30 = np.zeros((n_params, n_params))
fisher_matrix_pdfs_s90 = np.zeros((n_params, n_params))
fisher_matrix_pdfs_all_scales = np.zeros((n_params, n_params))

fisher_matrix_dpdfs_combined_s15 = np.zeros((n_params, n_params))
fisher_matrix_dpdfs_combined_s30 = np.zeros((n_params, n_params))
fisher_matrix_dpdfs_combined_s90 = np.zeros((n_params, n_params))
fisher_matrix_dpdfs_combined_all_scales = np.zeros((n_params, n_params))    

fisher_matrix_kk_s15 = np.zeros((n_params, n_params))
fisher_matrix_kk_s30 = np.zeros((n_params, n_params))
fisher_matrix_kk_s90 = np.zeros((n_params, n_params))
fisher_matrix_kk_all_scales = np.zeros((n_params, n_params))

fisher_matrix_pp_s15 = np.zeros((n_params, n_params))
fisher_matrix_pp_s30 = np.zeros((n_params, n_params))
fisher_matrix_pp_s90 = np.zeros((n_params, n_params))

fisher_matrix_nn_corr_s15 = np.zeros((n_params, n_params))
fisher_matrix_nn_corr_s30 = np.zeros((n_params, n_params))
fisher_matrix_nn_corr_s90 = np.zeros((n_params, n_params))

fisher_matrix_pn_s15 = np.zeros((n_params, n_params))
fisher_matrix_pn_s30 = np.zeros((n_params, n_params))
fisher_matrix_pn_s90 = np.zeros((n_params, n_params))

fisher_matrix_pk_s15 = np.zeros((n_params, n_params))
fisher_matrix_pk_s30 = np.zeros((n_params, n_params))
fisher_matrix_pk_s90 = np.zeros((n_params, n_params))

fisher_matrix_nk_s15 = np.zeros((n_params, n_params))
fisher_matrix_nk_s30 = np.zeros((n_params, n_params))
fisher_matrix_nk_s90 = np.zeros((n_params, n_params))

fisher_matrix_dcorrs_combined_s15 = np.zeros((n_params, n_params))
fisher_matrix_dcorrs_combined_s30 = np.zeros((n_params, n_params))
fisher_matrix_dcorrs_combined_s90 = np.zeros((n_params, n_params))
fisher_matrix_dcorrs_combined_all_scales = np.zeros((n_params, n_params))

fisher_matrix_nns_s15 = np.zeros((n_params, n_params))
fisher_matrix_nns_s30 = np.zeros((n_params, n_params))
fisher_matrix_nns_s90 = np.zeros((n_params, n_params))
fisher_matrix_nns_all_scales = np.zeros((n_params, n_params))

fisher_matrix_dpdf_nn_combined_s15 = np.zeros((n_params, n_params))
fisher_matrix_dpdf_nn_combined_s30 = np.zeros((n_params, n_params))
fisher_matrix_dpdf_nn_combined_s90 = np.zeros((n_params, n_params))
fisher_matrix_dpdf_nn_combined_all_scales = np.zeros((n_params, n_params))

fisher_matrix_defects_pdf_combined_all_scales = np.zeros((n_params, n_params))
fisher_matrix_all_all_scales = np.zeros((n_params, n_params))


for i in range(n_params):
    deriv_pdf_i_s15 = deriv_pdfs_s15[i]
    deriv_pdf_i_s30 = deriv_pdfs_s30[i]
    deriv_pdf_i_s90 = deriv_pdfs_s90[i]
    deriv_pdf_i_all_scales = np.concatenate([deriv_pdf_i_s15, deriv_pdf_i_s30, deriv_pdf_i_s90])

    deriv_dpdfs_combined_i_s15 = deriv_dpdfs_combined_s15[i]
    deriv_dpdfs_combined_i_s30 = deriv_dpdfs_combined_s30[i]
    deriv_dpdfs_combined_i_s90 = deriv_dpdfs_combined_s90[i]
    deriv_dpdfs_combined_i_all_scales = np.concatenate([deriv_dpdfs_combined_i_s15, deriv_dpdfs_combined_i_s30, deriv_dpdfs_combined_i_s90])

    deriv_kk_corrs_i_s15 = deriv_kk_corrs_s15[i]
    deriv_kk_corrs_i_s30 = deriv_kk_corrs_s30[i]
    deriv_kk_corrs_i_s90 = deriv_kk_corrs_s90[i]
    deriv_kk_corrs_i_all_scales = deriv_kk_corrs_s15[i]

    deriv_pp_corrs_i_s15 = deriv_pp_corrs_s15[i]
    deriv_pp_corrs_i_s30 = deriv_pp_corrs_s30[i]
    deriv_pp_corrs_i_s90 = deriv_pp_corrs_s90[i]

    deriv_nn_corrs_i_s15 = deriv_nn_corrs_s15[i]
    deriv_nn_corrs_i_s30 = deriv_nn_corrs_s30[i]
    deriv_nn_corrs_i_s90 = deriv_nn_corrs_s90[i]

    deriv_pn_corrs_i_s15 = deriv_pn_corrs_s15[i]
    deriv_pn_corrs_i_s30 = deriv_pn_corrs_s30[i]
    deriv_pn_corrs_i_s90 = deriv_pn_corrs_s90[i]

    deriv_pk_corrs_i_s15 = deriv_pk_corrs_s15[i]
    deriv_pk_corrs_i_s30 = deriv_pk_corrs_s30[i]
    deriv_pk_corrs_i_s90 = deriv_pk_corrs_s90[i]

    deriv_nk_corrs_i_s15 = deriv_nk_corrs_s15[i]
    deriv_nk_corrs_i_s30 = deriv_nk_corrs_s30[i]
    deriv_nk_corrs_i_s90 = deriv_nk_corrs_s90[i]

    deriv_dcorrs_i_s15 = np.concatenate([deriv_pp_corrs_i_s15, deriv_nn_corrs_i_s15, deriv_pn_corrs_i_s15])
    deriv_dcorrs_i_s30 = np.concatenate([deriv_pp_corrs_i_s30, deriv_nn_corrs_i_s30, deriv_pn_corrs_i_s30])
    deriv_dcorrs_i_s90 = np.concatenate([deriv_pp_corrs_i_s90, deriv_nn_corrs_i_s90, deriv_pn_corrs_i_s90])
    deriv_dcorrs_i_all_scales = np.concatenate([deriv_dcorrs_i_s15, deriv_dcorrs_i_s30, deriv_dcorrs_i_s90])

    deriv_nns_combined_i_s15 = deriv_nns_combined_s15[i] # + 1e-10
    deriv_nns_combined_i_s30 = deriv_nns_combined_s30[i] # + 1e-10
    deriv_nns_combined_i_s90 = deriv_nns_combined_s90[i] # + 1e-10
    deriv_nns_combined_i_all_scales = np.concatenate([deriv_nns_combined_i_s15, deriv_nns_combined_i_s30, deriv_nns_combined_i_s90])

    deriv_dpdf_nn_combined_i_s15 = deriv_dpdf_nn_combined_s15[i] # + 1e-10
    deriv_dpdf_nn_combined_i_s30 = deriv_dpdf_nn_combined_s30[i] # + 1e-10
    deriv_dpdf_nn_combined_i_s90 = deriv_dpdf_nn_combined_s90[i] # + 1e-10
    deriv_dpdf_nn_combined_i_all_scales = np.concatenate([deriv_dpdf_nn_combined_i_s15, deriv_dpdf_nn_combined_i_s30, deriv_dpdf_nn_combined_i_s90])

    deriv_defects_pdf_combined_i_all_scales = np.concatenate([deriv_dpdf_nn_combined_i_all_scales, deriv_pdf_i_all_scales])
    deriv_all_i_all_scales = np.concatenate([deriv_defects_pdf_combined_i_all_scales, deriv_kk_corrs_i_all_scales])
    for j in range(n_params):
        deriv_pdf_j_s15 = deriv_pdfs_s15[j]
        deriv_pdf_j_s30 = deriv_pdfs_s30[j]
        deriv_pdf_j_s90 = deriv_pdfs_s90[j]
        deriv_pdf_j_all_scales = np.concatenate([deriv_pdf_j_s15, deriv_pdf_j_s30, deriv_pdf_j_s90])

        deriv_dpdfs_combined_j_s15 = deriv_dpdfs_combined_s15[j]
        deriv_dpdfs_combined_j_s30 = deriv_dpdfs_combined_s30[j]
        deriv_dpdfs_combined_j_s90 = deriv_dpdfs_combined_s90[j]
        deriv_dpdfs_combined_j_all_scales = np.concatenate([deriv_dpdfs_combined_j_s15, deriv_dpdfs_combined_j_s30, deriv_dpdfs_combined_j_s90])

        # deriv_dk_corr_pos_j = deriv_dk_corrs_pos[j]
        # deriv_dk_corr_neg_j = deriv_dk_corrs_neg[j]
        # deriv_dk_corrs_combined_j = deriv_dk_corrs_combined[j]
        # deriv_defects_combined_j = deriv_defects_combined[j]

        deriv_kk_corrs_j_s15 = deriv_kk_corrs_s15[j]
        deriv_kk_corrs_j_s30 = deriv_kk_corrs_s30[j]
        deriv_kk_corrs_j_s90 = deriv_kk_corrs_s90[j]
        deriv_kk_corrs_j_all_scales = deriv_kk_corrs_s15[j]

        deriv_pp_corrs_j_s15 = deriv_pp_corrs_s15[j]
        deriv_pp_corrs_j_s30 = deriv_pp_corrs_s30[j]
        deriv_pp_corrs_j_s90 = deriv_pp_corrs_s90[j]

        deriv_nn_corrs_j_s15 = deriv_nn_corrs_s15[j]
        deriv_nn_corrs_j_s30 = deriv_nn_corrs_s30[j]
        deriv_nn_corrs_j_s90 = deriv_nn_corrs_s90[j]

        deriv_pn_corrs_j_s15 = deriv_pn_corrs_s15[j]
        deriv_pn_corrs_j_s30 = deriv_pn_corrs_s30[j]
        deriv_pn_corrs_j_s90 = deriv_pn_corrs_s90[j]

        deriv_pk_corrs_j_s15 = deriv_pk_corrs_s15[j]
        deriv_pk_corrs_j_s30 = deriv_pk_corrs_s30[j]
        deriv_pk_corrs_j_s90 = deriv_pk_corrs_s90[j]

        deriv_nk_corrs_j_s15 = deriv_nk_corrs_s15[j]
        deriv_nk_corrs_j_s30 = deriv_nk_corrs_s30[j]
        deriv_nk_corrs_j_s90 = deriv_nk_corrs_s90[j]

        deriv_dcorrs_j_s15 = np.concatenate([deriv_pp_corrs_j_s15, deriv_nn_corrs_j_s15, deriv_pn_corrs_j_s15])
        deriv_dcorrs_j_s30 = np.concatenate([deriv_pp_corrs_j_s30, deriv_nn_corrs_j_s30, deriv_pn_corrs_j_s30])
        deriv_dcorrs_j_s90 = np.concatenate([deriv_pp_corrs_j_s90, deriv_nn_corrs_j_s90, deriv_pn_corrs_j_s90])
        deriv_dcorrs_j_all_scales = np.concatenate([deriv_dcorrs_j_s15, deriv_dcorrs_j_s30, deriv_dcorrs_j_s90])

        deriv_nns_combined_j_s15 = deriv_nns_combined_s15[j] # + 1e-10
        deriv_nns_combined_j_s30 = deriv_nns_combined_s30[j] # + 1e-10
        deriv_nns_combined_j_s90 = deriv_nns_combined_s90[j] # + 1e-10
        deriv_nns_combined_j_all_scales = np.concatenate([deriv_nns_combined_j_s15, deriv_nns_combined_j_s30, deriv_nns_combined_j_s90])

        deriv_dpdf_nn_combined_j_s15 = deriv_dpdf_nn_combined_s15[j] # + 1e-10
        deriv_dpdf_nn_combined_j_s30 = deriv_dpdf_nn_combined_s30[j] # + 1e-10
        deriv_dpdf_nn_combined_j_s90 = deriv_dpdf_nn_combined_s90[j] # + 1e-10
        deriv_dpdf_nn_combined_j_all_scales = np.concatenate([deriv_dpdf_nn_combined_j_s15, deriv_dpdf_nn_combined_j_s30, deriv_dpdf_nn_combined_j_s90])

        deriv_defects_pdf_combined_j_all_scales = np.concatenate([deriv_dpdf_nn_combined_j_all_scales, deriv_pdf_j_all_scales])
        deriv_all_j_all_scales = np.concatenate([deriv_defects_pdf_combined_j_all_scales, deriv_kk_corrs_j_all_scales])

        fisher_matrix_pdfs_s15[i, j] = deriv_pdf_i_s15 @ inv_cov_pdf_s15 @ deriv_pdf_j_s15
        fisher_matrix_pdfs_s30[i, j] = deriv_pdf_i_s30 @ inv_cov_pdf_s30 @ deriv_pdf_j_s30
        fisher_matrix_pdfs_s90[i, j] = deriv_pdf_i_s90 @ inv_cov_pdf_s90 @ deriv_pdf_j_s90
        fisher_matrix_pdfs_all_scales[i, j] = deriv_pdf_i_all_scales @ inv_cov_pdf_all_scales @ deriv_pdf_j_all_scales

        fisher_matrix_dpdfs_combined_s15[i, j] = deriv_dpdfs_combined_i_s15 @ inv_cov_dpdf_combined_s15 @ deriv_dpdfs_combined_j_s15
        fisher_matrix_dpdfs_combined_s30[i, j] = deriv_dpdfs_combined_i_s30 @ inv_cov_dpdf_combined_s30 @ deriv_dpdfs_combined_j_s30
        fisher_matrix_dpdfs_combined_s90[i, j] = deriv_dpdfs_combined_i_s90 @ inv_cov_dpdf_combined_s90 @ deriv_dpdfs_combined_j_s90
        fisher_matrix_dpdfs_combined_all_scales[i, j] = deriv_dpdfs_combined_i_all_scales @ inv_cov_dpdf_combined_all_scales @ deriv_dpdfs_combined_j_all_scales

        fisher_matrix_kk_s15[i, j] = deriv_kk_corrs_i_s15 @ inv_cov_kk_s15 @ deriv_kk_corrs_j_s15
        fisher_matrix_kk_s30[i, j] = deriv_kk_corrs_i_s30 @ inv_cov_kk_s30 @ deriv_kk_corrs_j_s30
        fisher_matrix_kk_s90[i, j] = deriv_kk_corrs_i_s90 @ inv_cov_kk_s90 @ deriv_kk_corrs_j_s90
        fisher_matrix_kk_all_scales[i, j] = deriv_kk_corrs_i_all_scales @ inv_cov_kk_all_scales @ deriv_kk_corrs_j_all_scales

        fisher_matrix_pp_s15[i, j] = deriv_pp_corrs_i_s15 @ inv_cov_pp_s15 @ deriv_pp_corrs_j_s15
        fisher_matrix_pp_s30[i, j] = deriv_pp_corrs_i_s30 @ inv_cov_pp_s30 @ deriv_pp_corrs_j_s30
        fisher_matrix_pp_s90[i, j] = deriv_pp_corrs_i_s90 @ inv_cov_pp_s90 @ deriv_pp_corrs_j_s90

        fisher_matrix_nn_corr_s15[i, j] = deriv_nn_corrs_i_s15 @ inv_cov_nn_corr_s15 @ deriv_nn_corrs_j_s15
        fisher_matrix_nn_corr_s30[i, j] = deriv_nn_corrs_i_s30 @ inv_cov_nn_corr_s30 @ deriv_nn_corrs_j_s30
        fisher_matrix_nn_corr_s90[i, j] = deriv_nn_corrs_i_s90 @ inv_cov_nn_corr_s90 @ deriv_nn_corrs_j_s90

        fisher_matrix_pn_s15[i, j] = deriv_pn_corrs_i_s15 @ inv_cov_pn_s15 @ deriv_pn_corrs_j_s15
        fisher_matrix_pn_s30[i, j] = deriv_pn_corrs_i_s30 @ inv_cov_pn_s30 @ deriv_pn_corrs_j_s30
        fisher_matrix_pn_s90[i, j] = deriv_pn_corrs_i_s90 @ inv_cov_pn_s90 @ deriv_pn_corrs_j_s90

        fisher_matrix_pk_s15[i, j] = deriv_pk_corrs_i_s15 @ inv_cov_pk_s15 @ deriv_pk_corrs_j_s15
        fisher_matrix_pk_s30[i, j] = deriv_pk_corrs_i_s30 @ inv_cov_pk_s30 @ deriv_pk_corrs_j_s30
        fisher_matrix_pk_s90[i, j] = deriv_pk_corrs_i_s90 @ inv_cov_pk_s90 @ deriv_pk_corrs_j_s90

        fisher_matrix_nk_s15[i, j] = deriv_nk_corrs_i_s15 @ inv_cov_nk_s15 @ deriv_nk_corrs_j_s15
        fisher_matrix_nk_s30[i, j] = deriv_nk_corrs_i_s30 @ inv_cov_nk_s30 @ deriv_nk_corrs_j_s30
        fisher_matrix_nk_s90[i, j] = deriv_nk_corrs_i_s90 @ inv_cov_nk_s90 @ deriv_nk_corrs_j_s90

        fisher_matrix_dcorrs_combined_s15[i, j] = deriv_dcorrs_i_s15 @ inv_cov_dcorrs_combined_s15 @ deriv_dcorrs_j_s15
        fisher_matrix_dcorrs_combined_s30[i, j] = deriv_dcorrs_i_s30 @ inv_cov_dcorrs_combined_s30 @ deriv_dcorrs_j_s30
        fisher_matrix_dcorrs_combined_s90[i, j] = deriv_dcorrs_i_s90 @ inv_cov_dcorrs_combined_s90 @ deriv_dcorrs_j_s90
        fisher_matrix_dcorrs_combined_all_scales[i, j] = deriv_dcorrs_i_all_scales @ inv_cov_dcorrs_all_scales @ deriv_dcorrs_j_all_scales


        fisher_matrix_nns_s15[i, j] = deriv_nns_combined_i_s15 @ inv_cov_nn_s15 @ deriv_nns_combined_j_s15
        fisher_matrix_nns_s30[i, j] = deriv_nns_combined_i_s30 @ inv_cov_nn_s30 @ deriv_nns_combined_j_s30
        fisher_matrix_nns_s90[i, j] = deriv_nns_combined_i_s90 @ inv_cov_nn_s90 @ deriv_nns_combined_j_s90
        fisher_matrix_nns_all_scales[i, j] = deriv_nns_combined_i_all_scales @ inv_cov_nn_all_scales @ deriv_nns_combined_j_all_scales

        fisher_matrix_dpdf_nn_combined_s15[i, j] = deriv_dpdf_nn_combined_i_s15 @ inv_cov_dpdf_nn_combined_s15 @ deriv_dpdf_nn_combined_j_s15
        fisher_matrix_dpdf_nn_combined_s30[i, j] = deriv_dpdf_nn_combined_i_s30 @ inv_cov_dpdf_nn_combined_s30 @ deriv_dpdf_nn_combined_j_s30
        fisher_matrix_dpdf_nn_combined_s90[i, j] = deriv_dpdf_nn_combined_i_s90 @ inv_cov_dpdf_nn_combined_s90 @ deriv_dpdf_nn_combined_j_s90
        fisher_matrix_dpdf_nn_combined_all_scales[i, j] = deriv_dpdf_nn_combined_i_all_scales @ inv_cov_dpdf_nn_combined_all_scales @ deriv_dpdf_nn_combined_j_all_scales

        fisher_matrix_defects_pdf_combined_all_scales[i, j] = deriv_defects_pdf_combined_i_all_scales @ inv_cov_defects_pdf_combined_all_scales @ deriv_defects_pdf_combined_j_all_scales

        fisher_matrix_all_all_scales[i, j] = deriv_all_i_all_scales @ inv_cov_all_all_scales @ deriv_all_j_all_scales

Fisher_dpdf_combined_s15 = fisher_matrix_dpdfs_combined_s15
Fisher_dpdf_combined_s30 = fisher_matrix_dpdfs_combined_s30
Fisher_dpdf_combined_s90 = fisher_matrix_dpdfs_combined_s90
Fisher_dpdf_combined_all_scales = fisher_matrix_dpdfs_combined_all_scales


Fisher_pdf_s15 = fisher_matrix_pdfs_s15
Fisher_pdf_s30 = fisher_matrix_pdfs_s30
Fisher_pdf_s90 = fisher_matrix_pdfs_s90
Fisher_pdf_all_scales = fisher_matrix_pdfs_all_scales

Fisher_kk_s15 = fisher_matrix_kk_s15
Fisher_kk_s30 = fisher_matrix_kk_s30
Fisher_kk_s90 = fisher_matrix_kk_s90
Fisher_kk_all_scales = fisher_matrix_kk_all_scales

Fisher_pp_s15 = fisher_matrix_pp_s15
Fisher_pp_s30 = fisher_matrix_pp_s30
Fisher_pp_s90 = fisher_matrix_pp_s90

Fisher_nn_corr_s15 = fisher_matrix_nn_corr_s15
Fisher_nn_corr_s30 = fisher_matrix_nn_corr_s30
Fisher_nn_corr_s90 = fisher_matrix_nn_corr_s90

Fisher_pn_s15 = fisher_matrix_pn_s15
Fisher_pn_s30 = fisher_matrix_pn_s30
Fisher_pn_s90 = fisher_matrix_pn_s90

Fisher_pk_s15 = fisher_matrix_pk_s15
Fisher_pk_s30 = fisher_matrix_pk_s30
Fisher_pk_s90 = fisher_matrix_pk_s90

Fisher_nk_s15 = fisher_matrix_nk_s15
Fisher_nk_s30 = fisher_matrix_nk_s30
Fisher_nk_s90 = fisher_matrix_nk_s90

Fisher_dcorrs_combined_s15 = fisher_matrix_dcorrs_combined_s15
Fisher_dcorrs_combined_s30 = fisher_matrix_dcorrs_combined_s30
Fisher_dcorrs_combined_s90 = fisher_matrix_dcorrs_combined_s90
Fisher_dcorrs_combined_all_scales = fisher_matrix_dcorrs_combined_all_scales

Fisher_nns_s15 = fisher_matrix_nns_s15
Fisher_nns_s30 = fisher_matrix_nns_s30
Fisher_nns_s90 = fisher_matrix_nns_s90
Fisher_nns_all_scales = fisher_matrix_nns_all_scales

Fisher_dpdf_nn_combined_s15 = fisher_matrix_dpdf_nn_combined_s15
Fisher_dpdf_nn_combined_s30 = fisher_matrix_dpdf_nn_combined_s30
Fisher_dpdf_nn_combined_s90 = fisher_matrix_dpdf_nn_combined_s90
Fisher_dpdf_nn_combined_all_scales = fisher_matrix_dpdf_nn_combined_all_scales

Fisher_defects_pdf_combined_all_scales = fisher_matrix_defects_pdf_combined_all_scales
Fisher_all_all_scales = fisher_matrix_all_all_scales
# covs ================
from getdist.gaussian_mixtures import GaussianND
mean = np.array([0.3, 0.8])
names = ['Om', 'sigma8']
labels = ['\\Omega_m', '\\sigma_8']

covariance_dpdf_combined_s15 = np.linalg.inv(Fisher_dpdf_combined_s15)
covariance_dpdf_combined_s30 = np.linalg.inv(Fisher_dpdf_combined_s30)
covariance_dpdf_combined_s90 = np.linalg.inv(Fisher_dpdf_combined_s90)
print("DPDF COMBINED S90 COVARIANCE:", covariance_dpdf_combined_s90)
covariance_dpdf_combined_all_scales = np.linalg.inv(Fisher_dpdf_combined_all_scales)

# 修复：增加空格，保持数学格式一致
dpdf_combined_s15_samples = GaussianND(mean, covariance_dpdf_combined_s15, names=names, labels=labels, label=r'$C^{p+n}(\kappa), \ \theta_G=15$')
dpdf_combined_s30_samples = GaussianND(mean, covariance_dpdf_combined_s30, names=names, labels=labels, label=r'$C^{p+n}(\kappa), \ \theta_G=30$')
dpdf_combined_s90_samples = GaussianND(mean, covariance_dpdf_combined_s90, names=names, labels=labels, label=r'$C^{p+n}(\kappa), \ \theta_G=90$')
dpdf_combined_all_scales_samples = GaussianND(mean, covariance_dpdf_combined_all_scales, names=names, labels=labels, label=r'$C^{p+n}(\kappa)$, all scales')

covariance_pdf_s15 = np.linalg.inv(Fisher_pdf_s15)
covariance_pdf_s30 = np.linalg.inv(Fisher_pdf_s30)
covariance_pdf_s90 = np.linalg.inv(Fisher_pdf_s90)
covariance_pdf_all_scales = np.linalg.inv(Fisher_pdf_all_scales)

# 修复：统一 PDF 为文本格式，theta_G 为数学格式
pdf_s15_samples = GaussianND(mean, covariance_pdf_s15, names=names, labels=labels, label=r'PDF, $\theta_G=15$')
pdf_s30_samples = GaussianND(mean, covariance_pdf_s30, names=names, labels=labels, label=r'PDF, $\theta_G=30$')
pdf_s90_samples = GaussianND(mean, covariance_pdf_s90, names=names, labels=labels, label=r'PDF, $\theta_G=90$')
pdf_all_scales_samples = GaussianND(mean, covariance_pdf_all_scales, names=names, labels=labels, label=r'PDF, all scales')

covariance_pp_s15 = np.linalg.inv(Fisher_pp_s15)
covariance_pp_s30 = np.linalg.inv(Fisher_pp_s30)
covariance_pp_s90 = np.linalg.inv(Fisher_pp_s90)

covariance_nn_s15 = np.linalg.inv(Fisher_nn_corr_s15)
covariance_nn_s30 = np.linalg.inv(Fisher_nn_corr_s30)
covariance_nn_s90 = np.linalg.inv(Fisher_nn_corr_s90)

covariance_pn_s15 = np.linalg.inv(Fisher_pn_s15)
covariance_pn_s30 = np.linalg.inv(Fisher_pn_s30)
covariance_pn_s90 = np.linalg.inv(Fisher_pn_s90)

pp_s15_samples = GaussianND(mean, covariance_pp_s15, names=names, labels=labels, label=r'$\xi^{++}(\theta), \ \theta_G=15$')
pp_s30_samples = GaussianND(mean, covariance_pp_s30, names=names, labels=labels, label=r'$\xi^{++}(\theta), \ \theta_G=30$')
pp_s90_samples = GaussianND(mean, covariance_pp_s90, names=names, labels=labels, label=r'$\xi^{++}(\theta), \ \theta_G=90$')

nn_s15_samples = GaussianND(mean, covariance_nn_s15, names=names, labels=labels, label=r'$\xi^{--}(\theta), \ \theta_G=15$')
nn_s30_samples = GaussianND(mean, covariance_nn_s30, names=names, labels=labels, label=r'$\xi^{--}(\theta), \ \theta_G=30$')
nn_s90_samples = GaussianND(mean, covariance_nn_s90, names=names, labels=labels, label=r'$\xi^{--}(\theta), \ \theta_G=90$')    

pn_s15_samples = GaussianND(mean, covariance_pn_s15, names=names, labels=labels, label=r'$\xi^{+-}(\theta), \ \theta_G=15$')
pn_s30_samples = GaussianND(mean, covariance_pn_s30, names=names, labels=labels, label=r'$\xi^{+-}(\theta), \ \theta_G=30$')
pn_s90_samples = GaussianND(mean, covariance_pn_s90, names=names, labels=labels, label=r'$\xi^{+-}(\theta), \ \theta_G=90$')

dcorr_combined_s15_samples = GaussianND(mean, np.linalg.inv(Fisher_dcorrs_combined_s15), names=names, labels=labels, label=r'Defect 2PCF., $\theta_G=15$')
dcorr_combined_s30_samples = GaussianND(mean, np.linalg.inv(Fisher_dcorrs_combined_s30), names=names, labels=labels, label=r'Defect 2PCF., $\theta_G=30$')
dcorr_combined_s90_samples = GaussianND(mean, np.linalg.inv(Fisher_dcorrs_combined_s90), names=names, labels=labels, label=r'Defect 2PCF., $\theta_G=90$')
dcorr_combined_all_scales_samples = GaussianND(mean, np.linalg.inv(Fisher_dcorrs_combined_all_scales), names=names, labels=labels, label=r'Defect 2PCF., all scales')

covariance_kk_s15 = np.linalg.inv(Fisher_kk_s15)
covariance_kk_s30 = np.linalg.inv(Fisher_kk_s30)
covariance_kk_s90 = np.linalg.inv(Fisher_kk_s90)
covariance_kk_all_scales = np.linalg.inv(Fisher_kk_all_scales)

# 修复：补全缺失的 $ 符号，修复 all scales 的 $ 位置
kk_s15_samples = GaussianND(mean, covariance_kk_s15, names=names, labels=labels, label=r'$\xi^{\kappa\kappa}(\theta), \ \theta_G=15$')
kk_s30_samples = GaussianND(mean, covariance_kk_s30, names=names, labels=labels, label=r'$\xi^{\kappa\kappa}(\theta), \ \theta_G=30$')
kk_s90_samples = GaussianND(mean, covariance_kk_s90, names=names, labels=labels, label=r'$\xi^{\kappa\kappa}(\theta), \ \theta_G=90$')
kk_all_scales_samples = GaussianND(mean, covariance_kk_all_scales, names=names, labels=labels, label=r'$\xi^{\kappa\kappa}(\theta)$, all scales')

covariance_nns_s15 = np.linalg.inv(Fisher_nns_s15)
covariance_nns_s30 = np.linalg.inv(Fisher_nns_s30)
covariance_nns_s90 = np.linalg.inv(Fisher_nns_s90)
print("NNS S90 COVARIANCE:", covariance_nns_s90)
covariance_nns_all_scales = np.linalg.inv(Fisher_nns_all_scales)

# 修复：补全 $，使用 \mathrm{all} 避免斜体，移除上标中不规范的 "-th"
nns_s15_samples = GaussianND(mean, covariance_nns_s15, names=names, labels=labels, label=r'$C^{\mathrm{all}}_{1,2,4,8}(\theta), \ \theta_G=15$')
nns_s30_samples = GaussianND(mean, covariance_nns_s30, names=names, labels=labels, label=r'$C^{\mathrm{all}}_{1,2,4,8}(\theta), \ \theta_G=30$')
nns_s90_samples = GaussianND(mean, covariance_nns_s90, names=names, labels=labels, label=r'$C^{\mathrm{all}}_{1,2,4,8}(\theta), \ \theta_G=90$')
nns_all_scales_samples = GaussianND(mean, covariance_nns_all_scales, names=names, labels=labels, label=r'$C^{\mathrm{all}}_{1,2,4,8}(\theta)$, all scales')

covariance_dpdf_nn_combined_s15 = np.linalg.inv(Fisher_dpdf_nn_combined_s15)
covariance_dpdf_nn_combined_s30 = np.linalg.inv(Fisher_dpdf_nn_combined_s30)
covariance_dpdf_nn_combined_s90 = np.linalg.inv(Fisher_dpdf_nn_combined_s90)
covariance_dpdf_nn_combined_all_scales = np.linalg.inv(Fisher_dpdf_nn_combined_all_scales)

# 修复：将 DCDF + kNN 放在文本模式，theta_G 放在数学模式
dpdf_nn_combined_s15_samples = GaussianND(mean, covariance_dpdf_nn_combined_s15, names=names, labels=labels, label=r'DCDF + kNN, $\theta_G=15$')
dpdf_nn_combined_s30_samples = GaussianND(mean, covariance_dpdf_nn_combined_s30, names=names, labels=labels, label=r'DCDF + kNN, $\theta_G=30$')
dpdf_nn_combined_s90_samples = GaussianND(mean, covariance_dpdf_nn_combined_s90, names=names, labels=labels, label=r'DCDF + kNN, $\theta_G=90$')
dpdf_nn_combined_all_scales_samples = GaussianND(mean, covariance_dpdf_nn_combined_all_scales, names=names, labels=labels, label=r'DCDF + kNN, all scales')

covariance_defects_pdf_combined_all_scales = np.linalg.inv(Fisher_defects_pdf_combined_all_scales)
defects_pdf_combined_all_scales_samples = GaussianND(mean, covariance_defects_pdf_combined_all_scales, names=names, labels=labels, label=r'Defects + PDF, all scales') 

covariance_all_all_scales = np.linalg.inv(Fisher_all_all_scales)
all_all_scales_samples = GaussianND(mean, covariance_all_all_scales, names=names, labels=labels, label=r'All Data, all scales')
print(np.sqrt(np.diag(covariance_all_all_scales)))

import matplotlib.pyplot as plt
from getdist import plots
import matplotlib as mpl
mpl.rcParams['xtick.direction'] = 'in'
mpl.rcParams['ytick.direction'] = 'in'
mpl.rcParams['font.size'] = 8

# use tab10 colors
color_map = plt.get_cmap('tab10')
# pdf s15 to s90, and all scales
g = plots.get_subplot_plotter(width_inch=4)
g.settings.num_plot_contours = 1
g.plot_2d([pdf_s90_samples, pdf_s30_samples, pdf_s15_samples, pdf_all_scales_samples, kk_all_scales_samples],
          param1="Om", param2="sigma8",
                filled=[False, False, False, False, False],
                lims = [0.2, 0.4, 0.7, 0.9],
                colors=[color_map(i) for i in range(3)] + ['red', 'black'],
                lws=2,
                )
g.add_legend([r"PDF, $\theta_G=90$", r"PDF, $\theta_G=30$", r"PDF, $\theta_G=15$", r"PDF, all scales", r'$\xi^{\kappa\kappa}(\theta)$, all scales'], legend_loc='upper right', fontsize=7)
g.export("outfiles/pdf.png", dpi=300)

g = plots.get_subplot_plotter(width_inch=4)
g.settings.num_plot_contours = 1
g.plot_2d([dpdf_combined_s90_samples, dpdf_combined_s30_samples, dpdf_combined_s15_samples, dpdf_combined_all_scales_samples, kk_all_scales_samples],
                param1="Om", param2="sigma8",
                filled=[False, False, False, False, False],
                lims = [0.2, 0.4, 0.7, 0.9],
                colors=[color_map(i) for i in range(3)] + ['red', 'black'],
                lws=2,
                )
g.add_legend([r'$\text{DCDFs}, \ \theta_G=90$', r'$\text{DCDFs}, \ \theta_G=30$', r'$\text{DCDFs}, \ \theta_G=15$', r'DCDFs, all scales', r'$\xi^{\kappa\kappa}(\theta)$, all scales'], legend_loc='upper right', fontsize=7)
g.export("outfiles/dcdf.png", dpi=300)

g = plots.get_subplot_plotter(width_inch=4)
g.settings.num_plot_contours = 1
g.triangle_plot([kk_s15_samples, kk_s30_samples, kk_s90_samples, kk_all_scales_samples],
                filled=False,
                param_limits={"Om": (0.2, 0.4), "sigma8": (0.7, 0.9)},
                contour_colors=[color_map(i) for i in range(4)],
                contour_lws=1.5)
g.export("outfiles/kk.png", dpi=300)

g = plots.get_subplot_plotter(width_inch=4)
g.settings.num_plot_contours = 1
g.plot_2d([nns_s90_samples, nns_s30_samples, nns_s15_samples, nns_all_scales_samples, kk_all_scales_samples],
                param1="Om", param2="sigma8",
                filled=[False, False, False, False, False],
                lims = [0.2, 0.4, 0.7, 0.9],
                colors=[color_map(i) for i in range(3)] + ['red', 'black'],
                lws=2,
                )
g.add_legend([r'$\text{kNNs}, \ \theta_G=90$', r'$\text{kNNs}, \ \theta_G=30$', r'$\text{kNNs}, \ \theta_G=15$', r'$\text{kNNs}$, all scales', r'$\xi^{\kappa\kappa}(\theta)$, all scales'], legend_loc='upper right', fontsize=7)
g.export("outfiles/nns.png", dpi=300)

g = plots.get_subplot_plotter(width_inch=4)
g.settings.num_plot_contours = 1
g.plot_2d([dpdf_nn_combined_s90_samples, dpdf_nn_combined_s30_samples, dpdf_nn_combined_s15_samples, dpdf_nn_combined_all_scales_samples, kk_all_scales_samples],
                param1="Om", param2="sigma8",
                filled=[False, False, False, False, False],
                lims = [0.2, 0.4, 0.7, 0.9],
                colors=[color_map(i) for i in range(3)] + ['red', 'black'],
                lws=2,
                )
g.add_legend([r'$\text{Defects}, \ \theta_G=90$', r'$\text{Defects}, \ \theta_G=30$', r'$\text{Defects}, \ \theta_G=15$', r'$\text{Defects}$, all scales', r'$\xi^{\kappa\kappa}(\theta)$, all scales'], legend_loc='upper right', fontsize=7)
g.export("outfiles/dcdf_nn_combined.png", dpi=300)

g = plots.get_subplot_plotter(width_inch=4)
g.settings.num_plot_contours = 1
g.plot_2d([kk_all_scales_samples, pdf_all_scales_samples, dpdf_nn_combined_all_scales_samples, defects_pdf_combined_all_scales_samples, all_all_scales_samples],
                param1="Om", param2="sigma8",
                filled=[False, False, False, False, False],
                lims = [0.27, 0.33, 0.77, 0.83],
                colors=[color_map(i) for i in range(3)] + ['red', 'black'],
                lws=2,
                # alphas=[0.7, 0.7, 1.0, 1.0],
                ls=['solid', 'solid', 'solid', 'solid', 'dashed'])    
g.add_legend([r'$\xi^{\kappa\kappa}(\theta)$, all scales', r'PDF, all scales', r'Defects, all scales', r'All HOWLS, all scales', r'All HOWLS + $\xi^{\kappa\kappa}(\theta)$, all scales'], legend_loc='upper right', fontsize=7)
g.add_text("Gaussian", 0.1, 0.1)
g.export("outfiles/all_combined.png", dpi=300)

g = plots.get_subplot_plotter(width_inch=4)
g.settings.num_plot_contours = 1
g.plot_2d([dcorr_combined_s90_samples, dcorr_combined_s30_samples, dcorr_combined_s15_samples, dcorr_combined_all_scales_samples, kk_all_scales_samples],
                param1="Om", param2="sigma8",
                filled=[False, False, False, False, False],
                lims = [0.2, 0.4, 0.7, 0.9],
                colors=[color_map(i) for i in range(3)] + ['red', 'black'],
                lws=2,
                )
g.add_legend([r'Defect 2PCF., $\theta_G=90$', r'Defect 2PCF., $\theta_G=30$', r'Defect 2PCF., $\theta_G=15$', r'Defect 2PCF., all scales', r'$\xi^{\kappa\kappa}(\theta)$, all scales'], legend_loc='upper right', fontsize=7)
g.export("outfiles/dcorrs.png", dpi=300)