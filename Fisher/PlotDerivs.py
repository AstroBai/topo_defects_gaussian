import numpy as np
from scipy.interpolate import interp1d

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

# Initialize new lists for Peak/Minima correlations
ppk_corrs_s15, ppk_corrs_s30, ppk_corrs_s90 = [], [], []
pmn_corrs_s15, pmn_corrs_s30, pmn_corrs_s90 = [], [], []
npk_corrs_s15, npk_corrs_s30, npk_corrs_s90 = [], [], []
nmn_corrs_s15, nmn_corrs_s30, nmn_corrs_s90 = [], [], []

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
        
        # Load Defect-Peak/Minima
        data_ppk = np.load(f"{DATA_DIR}/TrainPosPeak/train_{i}_s15_seed_{seed}_pos_peak_corr.npz")
        ppk_corrs_s15.append(data_ppk['xi'])
        data_ppk = np.load(f"{DATA_DIR}/TrainPosPeak/train_{i}_s30_seed_{seed}_pos_peak_corr.npz")
        ppk_corrs_s30.append(data_ppk['xi'])
        data_ppk = np.load(f"{DATA_DIR}/TrainPosPeak/train_{i}_s90_seed_{seed}_pos_peak_corr.npz")
        ppk_corrs_s90.append(data_ppk['xi'])

        data_pmn = np.load(f"{DATA_DIR}/TrainPosMin/train_{i}_s15_seed_{seed}_pos_min_corr.npz")
        pmn_corrs_s15.append(data_pmn['xi'])
        data_pmn = np.load(f"{DATA_DIR}/TrainPosMin/train_{i}_s30_seed_{seed}_pos_min_corr.npz")
        pmn_corrs_s30.append(data_pmn['xi'])
        data_pmn = np.load(f"{DATA_DIR}/TrainPosMin/train_{i}_s90_seed_{seed}_pos_min_corr.npz")
        pmn_corrs_s90.append(data_pmn['xi'])

        data_npk = np.load(f"{DATA_DIR}/TrainNegPeak/train_{i}_s15_seed_{seed}_neg_peak_corr.npz")
        npk_corrs_s15.append(data_npk['xi'])
        data_npk = np.load(f"{DATA_DIR}/TrainNegPeak/train_{i}_s30_seed_{seed}_neg_peak_corr.npz")
        npk_corrs_s30.append(data_npk['xi'])
        data_npk = np.load(f"{DATA_DIR}/TrainNegPeak/train_{i}_s90_seed_{seed}_neg_peak_corr.npz")
        npk_corrs_s90.append(data_npk['xi'])

        data_nmn = np.load(f"{DATA_DIR}/TrainNegMin/train_{i}_s15_seed_{seed}_neg_min_corr.npz")
        nmn_corrs_s15.append(data_nmn['xi'])
        data_nmn = np.load(f"{DATA_DIR}/TrainNegMin/train_{i}_s30_seed_{seed}_neg_min_corr.npz")
        nmn_corrs_s30.append(data_nmn['xi'])
        data_nmn = np.load(f"{DATA_DIR}/TrainNegMin/train_{i}_s90_seed_{seed}_neg_min_corr.npz")
        nmn_corrs_s90.append(data_nmn['xi'])
        
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

ppk_corrs_s15, ppk_corrs_s30, ppk_corrs_s90 = np.array(ppk_corrs_s15), np.array(ppk_corrs_s30), np.array(ppk_corrs_s90)
pmn_corrs_s15, pmn_corrs_s30, pmn_corrs_s90 = np.array(pmn_corrs_s15), np.array(pmn_corrs_s30), np.array(pmn_corrs_s90)
npk_corrs_s15, npk_corrs_s30, npk_corrs_s90 = np.array(npk_corrs_s15), np.array(npk_corrs_s30), np.array(npk_corrs_s90)
nmn_corrs_s15, nmn_corrs_s30, nmn_corrs_s90 = np.array(nmn_corrs_s15), np.array(nmn_corrs_s30), np.array(nmn_corrs_s90)

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

# Reshape DPC arrays
ppk_corrs_s15 = ppk_corrs_s15.reshape(n_seeds, 4, -1)
ppk_corrs_s30 = ppk_corrs_s30.reshape(n_seeds, 4, -1)
ppk_corrs_s90 = ppk_corrs_s90.reshape(n_seeds, 4, -1)
pmn_corrs_s15 = pmn_corrs_s15.reshape(n_seeds, 4, -1)
pmn_corrs_s30 = pmn_corrs_s30.reshape(n_seeds, 4, -1)
pmn_corrs_s90 = pmn_corrs_s90.reshape(n_seeds, 4, -1)
npk_corrs_s15 = npk_corrs_s15.reshape(n_seeds, 4, -1)
npk_corrs_s30 = npk_corrs_s30.reshape(n_seeds, 4, -1)
npk_corrs_s90 = npk_corrs_s90.reshape(n_seeds, 4, -1)
nmn_corrs_s15 = nmn_corrs_s15.reshape(n_seeds, 4, -1)
nmn_corrs_s30 = nmn_corrs_s30.reshape(n_seeds, 4, -1)
nmn_corrs_s90 = nmn_corrs_s90.reshape(n_seeds, 4, -1)

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
deriv_ppk_corrs_s15, deriv_ppk_corrs_s30, deriv_ppk_corrs_s90 = [], [], []
deriv_pmn_corrs_s15, deriv_pmn_corrs_s30, deriv_pmn_corrs_s90 = [], [], []
deriv_npk_corrs_s15, deriv_npk_corrs_s30, deriv_npk_corrs_s90 = [], [], []
deriv_nmn_corrs_s15, deriv_nmn_corrs_s30, deriv_nmn_corrs_s90 = [], [], []
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
    
    # DPC derivatives
    for scale_idx, (corrs_list, deriv_list) in enumerate([
        ([ppk_corrs_s15, ppk_corrs_s30, ppk_corrs_s90], [deriv_ppk_corrs_s15, deriv_ppk_corrs_s30, deriv_ppk_corrs_s90]),
        ([pmn_corrs_s15, pmn_corrs_s30, pmn_corrs_s90], [deriv_pmn_corrs_s15, deriv_pmn_corrs_s30, deriv_pmn_corrs_s90]),
        ([npk_corrs_s15, npk_corrs_s30, npk_corrs_s90], [deriv_npk_corrs_s15, deriv_npk_corrs_s30, deriv_npk_corrs_s90]),
        ([nmn_corrs_s15, nmn_corrs_s30, nmn_corrs_s90], [deriv_nmn_corrs_s15, deriv_nmn_corrs_s30, deriv_nmn_corrs_s90]),
    ]):
        for s_idx in range(3):
           xi_plus = corrs_list[s_idx][:,i+1,:]
           xi_minus = corrs_list[s_idx][:,i,:]
           deriv_val = (xi_plus - xi_minus) / (2 * delta_params[list(fid.keys())[i//2]])
           deriv_list[s_idx].append(deriv_val)
    
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

deriv_ppk_corrs_s15, deriv_ppk_corrs_s30, deriv_ppk_corrs_s90 = np.array(deriv_ppk_corrs_s15), np.array(deriv_ppk_corrs_s30), np.array(deriv_ppk_corrs_s90)
deriv_pmn_corrs_s15, deriv_pmn_corrs_s30, deriv_pmn_corrs_s90 = np.array(deriv_pmn_corrs_s15), np.array(deriv_pmn_corrs_s30), np.array(deriv_pmn_corrs_s90)
deriv_npk_corrs_s15, deriv_npk_corrs_s30, deriv_npk_corrs_s90 = np.array(deriv_npk_corrs_s15), np.array(deriv_npk_corrs_s30), np.array(deriv_npk_corrs_s90)
deriv_nmn_corrs_s15, deriv_nmn_corrs_s30, deriv_nmn_corrs_s90 = np.array(deriv_nmn_corrs_s15), np.array(deriv_nmn_corrs_s30), np.array(deriv_nmn_corrs_s90)

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

# DPC s30 statistics
deriv_ppk_corrs_s30_reshaped = deriv_ppk_corrs_s30.transpose(1, 0, 2).reshape(100, -1)
deriv_ppk_corrs_s30_mean = np.mean(deriv_ppk_corrs_s30_reshaped, axis=0)
deriv_ppk_corrs_s30_cov = np.cov(deriv_ppk_corrs_s30_reshaped, rowvar=False)

deriv_pmn_corrs_s30_reshaped = deriv_pmn_corrs_s30.transpose(1, 0, 2).reshape(100, -1)
deriv_pmn_corrs_s30_mean = np.mean(deriv_pmn_corrs_s30_reshaped, axis=0)
deriv_pmn_corrs_s30_cov = np.cov(deriv_pmn_corrs_s30_reshaped, rowvar=False)

deriv_npk_corrs_s30_reshaped = deriv_npk_corrs_s30.transpose(1, 0, 2).reshape(100, -1)
deriv_npk_corrs_s30_mean = np.mean(deriv_npk_corrs_s30_reshaped, axis=0)
deriv_npk_corrs_s30_cov = np.cov(deriv_npk_corrs_s30_reshaped, rowvar=False)

deriv_nmn_corrs_s30_reshaped = deriv_nmn_corrs_s30.transpose(1, 0, 2).reshape(100, -1)
deriv_nmn_corrs_s30_mean = np.mean(deriv_nmn_corrs_s30_reshaped, axis=0)
deriv_nmn_corrs_s30_cov = np.cov(deriv_nmn_corrs_s30_reshaped, rowvar=False)


# Calculate statistics for NNs combined
deriv_nns_combined_s15_reshaped = deriv_nns_combined_s15.transpose(1, 0, 2).reshape(100, -1)
deriv_nns_combined_s15_mean = np.mean(deriv_nns_combined_s15_reshaped, axis=0)
deriv_nns_combined_s15_cov = np.cov(deriv_nns_combined_s15_reshaped, rowvar=False)

deriv_nns_combined_s30_reshaped = deriv_nns_combined_s30.transpose(1, 0, 2).reshape(100, -1)
deriv_nns_combined_s30_mean = np.mean(deriv_nns_combined_s30_reshaped, axis=0)
deriv_nns_combined_s30_cov = np.cov(deriv_nns_combined_s30_reshaped, rowvar=False)

deriv_nns_combined_s90_reshaped = deriv_nns_combined_s90.transpose(1, 0, 2).reshape(100, -1)
deriv_nns_combined_s90_mean = np.mean(deriv_nns_combined_s90_reshaped, axis=0)
deriv_nns_combined_s90_cov = np.cov(deriv_nns_combined_s90_reshaped, rowvar=False)



# plot derivs
import matplotlib.pyplot as plt
import matplotlib as mpl
mpl.rcParams['xtick.direction'] = 'in'
mpl.rcParams['ytick.direction'] = 'in'
mpl.rcParams['font.size'] = 8

N_fid = 1000
N_deriv = n_seeds

# # pdfs, 1x3, share y axis
# fig, axes = plt.subplots(1, 3, figsize=(6, 2), sharey=True)

# # Subplot 1: s15
# ax = axes[0]
# bin_edges = np.linspace(-0.01, 0.01, 11)
# bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

# # Errors: sigma / sqrt(N)
# fid_err = np.sqrt(np.diag(cov_pdfs_fid_s15)) / np.sqrt(N_fid)
# om_err = np.sqrt(np.diag(deriv_pdfs_s15_cov[0:10, 0:10])) / np.sqrt(n_seeds)
# s8_err = np.sqrt(np.diag(deriv_pdfs_s15_cov[10:20, 10:20])) / np.sqrt(n_seeds)

# ax.plot(bin_centers, pdfs_fid_s15, c='black')
# ax.errorbar(bin_centers, pdfs_fid_s15, yerr=fid_err, fmt='|', label='Fiducial', c='black')
# ax.errorbar(bin_centers, deriv_pdfs_s15_mean[0:10], yerr=om_err, fmt='|', label=r'Derv. w.r.t $\Omega_{\rm m}$')
# ax.errorbar(bin_centers, deriv_pdfs_s15_mean[10:20], yerr=s8_err, fmt='|', label=r'Derv. w.r.t $\sigma_8$')
# ax.fill_between(bin_centers, deriv_pdfs_s15_mean[0:10] - om_err, deriv_pdfs_s15_mean[0:10] + om_err, alpha=0.3)
# ax.fill_between(bin_centers, deriv_pdfs_s15_mean[10:20] - s8_err, deriv_pdfs_s15_mean[10:20] + s8_err, alpha=0.3)
# ax.set_title(r'$\theta_G$ = 15 arcmin')
# ax.set_xlabel(r'$\kappa$')
# ax.set_ylabel('Probability Density')
# ax.set_xticks([-0.005, 0.005])


# # Subplot 2: s30
# ax = axes[1]
# bin_edges = np.linspace(-0.008, 0.008, 11)
# bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

# fid_err = np.sqrt(np.diag(cov_pdfs_fid_s30)) / np.sqrt(N_fid)
# om_err = np.sqrt(np.diag(deriv_pdfs_s30_cov[0:10, 0:10])) / np.sqrt(n_seeds)
# s8_err = np.sqrt(np.diag(deriv_pdfs_s30_cov[10:20, 10:20])) / np.sqrt(n_seeds)

# ax.plot(bin_centers, pdfs_fid_s30, c='black')
# ax.errorbar(bin_centers, pdfs_fid_s30, yerr=fid_err, fmt='|', label='Fiducial', c='black')
# ax.errorbar(bin_centers, deriv_pdfs_s30_mean[0:10], yerr=om_err, fmt='|', label=r'Derv. w.r.t $\Omega_{\rm m}$')
# ax.errorbar(bin_centers, deriv_pdfs_s30_mean[10:20], yerr=s8_err, fmt='|', label=r'Derv. w.r.t $\sigma_8$')
# ax.fill_between(bin_centers, deriv_pdfs_s30_mean[0:10] - om_err, deriv_pdfs_s30_mean[0:10] + om_err, alpha=0.3)
# ax.fill_between(bin_centers, deriv_pdfs_s30_mean[10:20] - s8_err, deriv_pdfs_s30_mean[10:20] + s8_err, alpha=0.3)
# ax.set_title(r'$\theta_G$ = 30 arcmin')
# ax.set_xlabel(r'$\kappa$')
# ax.set_xticks([-0.004, 0.004])


# # Subplot 3: s90
# ax = axes[2]
# bin_edges = np.linspace(-0.005, 0.005, 11)
# bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

# fid_err = np.sqrt(np.diag(cov_pdfs_fid_s90)) / np.sqrt(N_fid)
# om_err = np.sqrt(np.diag(deriv_pdfs_s90_cov[0:10, 0:10])) / np.sqrt(n_seeds)
# s8_err = np.sqrt(np.diag(deriv_pdfs_s90_cov[10:20, 10:20])) / np.sqrt(n_seeds)

# ax.plot(bin_centers, pdfs_fid_s90, c='black')
# ax.errorbar(bin_centers, pdfs_fid_s90, yerr=fid_err, fmt='|', label='Fiducial', c='black')
# ax.errorbar(bin_centers, deriv_pdfs_s90_mean[0:10], yerr=om_err, fmt='|', label=r'Derv. w.r.t $\Omega_{\rm m}$')
# ax.errorbar(bin_centers, deriv_pdfs_s90_mean[10:20], yerr=s8_err, fmt='|', label=r'Derv. w.r.t $\sigma_8$')
# ax.fill_between(bin_centers, deriv_pdfs_s90_mean[0:10] - om_err, deriv_pdfs_s90_mean[0:10] + om_err, alpha=0.3)
# ax.fill_between(bin_centers, deriv_pdfs_s90_mean[10:20] - s8_err, deriv_pdfs_s90_mean[10:20] + s8_err, alpha=0.3)
# ax.set_title(r'$\theta_G$ = 90 arcmin')
# ax.set_xlabel(r'$\kappa$')
# ax.set_xticks([-0.0025, 0.0025])


# plt.legend(loc=(1.05, 0.25))
# plt.subplots_adjust(wspace=0)
# plt.savefig('outfiles/derivs_pdf.png', dpi=300, bbox_inches='tight')
# plt.close()

# # dpdfs
# fig, axes = plt.subplots(2, 3, figsize=(6, 3), sharey='col') # 2x3 because pos and neg
# # Subplot 1: s15 pos
# ax = axes[0,0]
# bin_edges = np.linspace(-0.01, 0.01, 11)
# bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

# fid_err = np.sqrt(np.diag(cov_dpdfs_pos_s15)) / np.sqrt(N_fid)
# om_err = np.sqrt(np.diag(deriv_dpdfs_pos_s15_cov[0:10, 0:10])) / np.sqrt(n_seeds)
# s8_err = np.sqrt(np.diag(deriv_dpdfs_pos_s15_cov[10:20, 10:20])) / np.sqrt(n_seeds)

# ax.plot(bin_centers, dpdfs_pos_fid_s15, c='black')
# ax.errorbar(bin_centers, dpdfs_pos_fid_s15, yerr=fid_err, fmt='|', label='Fiducial', c='black')
# ax.errorbar(bin_centers, deriv_dpdfs_pos_s15_mean[0:10], yerr=om_err, fmt='|', label=r'Derv. w.r.t $\Omega_{\rm m}$')
# ax.errorbar(bin_centers, deriv_dpdfs_pos_s15_mean[10:20], yerr=s8_err, fmt='|', label=r'Derv. w.r.t $\sigma_8$')
# ax.fill_between(bin_centers, deriv_dpdfs_pos_s15_mean[0:10] - om_err, deriv_dpdfs_pos_s15_mean[0:10] + om_err, alpha=0.3)
# ax.fill_between(bin_centers, deriv_dpdfs_pos_s15_mean[10:20] - s8_err, deriv_dpdfs_pos_s15_mean[10:20] + s8_err, alpha=0.3)
# ax.set_title(r'$\theta_G$ = 15 arcmin')
# ax.set_xlabel(r'$\kappa$')
# ax.set_ylabel(r'$C^p(\kappa)$')
# ax.set_xticks([-0.005, 0.005])
# ax.set_yticks([0, 500, 1000])


# # Subplot 2: s30 pos
# ax = axes[0,1]
# bin_edges = np.linspace(-0.008, 0.008, 11)
# bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

# fid_err = np.sqrt(np.diag(cov_dpdfs_pos_s30)) / np.sqrt(N_fid)
# om_err = np.sqrt(np.diag(deriv_dpdfs_pos_s30_cov[0:10, 0:10])) / np.sqrt(n_seeds)
# s8_err = np.sqrt(np.diag(deriv_dpdfs_pos_s30_cov[10:20, 10:20])) / np.sqrt(n_seeds)

# ax.plot(bin_centers, dpdfs_pos_fid_s30, c='black')
# ax.errorbar(bin_centers, dpdfs_pos_fid_s30, yerr=fid_err, fmt='|', label='Fiducial', c='black')   
# ax.errorbar(bin_centers, deriv_dpdfs_pos_s30_mean[0:10], yerr=om_err, fmt='|', label=r'Derv. w.r.t $\Omega_{\rm m}$')
# ax.errorbar(bin_centers, deriv_dpdfs_pos_s30_mean[10:20], yerr=s8_err, fmt='|', label=r'Derv. w.r.t $\sigma_8$')
# ax.fill_between(bin_centers, deriv_dpdfs_pos_s30_mean[0:10] - om_err, deriv_dpdfs_pos_s30_mean[0:10] + om_err, alpha=0.3)
# ax.fill_between(bin_centers, deriv_dpdfs_pos_s30_mean[10:20] - s8_err, deriv_dpdfs_pos_s30_mean[10:20] + s8_err, alpha=0.3)
# ax.set_title(r'$\theta_G$ = 30 arcmin')
# ax.set_xlabel(r'$\kappa$')
# ax.set_xticks([-0.004, 0.004])
# ax.set_yticks([0, 150, 300])


# # Subplot 3: s90 pos
# ax = axes[0,2]
# bin_edges = np.linspace(-0.005, 0.005, 11)
# bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

# fid_err = np.sqrt(np.diag(cov_dpdfs_pos_s90)) / np.sqrt(N_fid)
# om_err = np.sqrt(np.diag(deriv_dpdfs_pos_s90_cov[0:10, 0:10])) / np.sqrt(n_seeds)
# s8_err = np.sqrt(np.diag(deriv_dpdfs_pos_s90_cov[10:20, 10:20])) / np.sqrt(n_seeds)

# ax.plot(bin_centers, dpdfs_pos_fid_s90, c='black')
# ax.errorbar(bin_centers, dpdfs_pos_fid_s90, yerr=fid_err, fmt='|', label='Fiducial', c='black')   
# ax.errorbar(bin_centers, deriv_dpdfs_pos_s90_mean[0:10], yerr=om_err, fmt='|', label=r'Derv. w.r.t $\Omega_{\rm m}$')
# ax.errorbar(bin_centers, deriv_dpdfs_pos_s90_mean[10:20], yerr=s8_err, fmt='|', label=r'Derv. w.r.t $\sigma_8$')
# ax.fill_between(bin_centers, deriv_dpdfs_pos_s90_mean[0:10] - om_err, deriv_dpdfs_pos_s90_mean[0:10] + om_err, alpha=0.3)
# ax.fill_between(bin_centers, deriv_dpdfs_pos_s90_mean[10:20] - s8_err, deriv_dpdfs_pos_s90_mean[10:20] + s8_err, alpha=0.3)
# ax.set_title(r'$\theta_G$ = 90 arcmin')
# ax.set_xlabel(r'$\kappa$')
# ax.set_yticks([0, 25, 50])
# ax.set_xticks([-0.0025, 0.0, 0.0025])


# # Subplot 4: s15 neg
# ax = axes[1,0]
# bin_edges = np.linspace(-0.01, 0.01, 11)
# bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

# fid_err = np.sqrt(np.diag(cov_dpdfs_neg_s15)) / np.sqrt(N_fid)
# om_err = np.sqrt(np.diag(deriv_dpdfs_neg_s15_cov[0:10, 0:10])) / np.sqrt(n_seeds)
# s8_err = np.sqrt(np.diag(deriv_dpdfs_neg_s15_cov[10:20, 10:20])) / np.sqrt(n_seeds)

# ax.plot(bin_centers, dpdfs_neg_fid_s15, c='black')
# ax.errorbar(bin_centers, dpdfs_neg_fid_s15, yerr=fid_err, fmt='|', label='Fiducial', c='black')   
# ax.errorbar(bin_centers, deriv_dpdfs_neg_s15_mean[0:10], yerr=om_err, fmt='|', label=r'Derv. w.r.t $\Omega_{\rm m}$')
# ax.errorbar(bin_centers, deriv_dpdfs_neg_s15_mean[10:20], yerr=s8_err, fmt='|', label=r'Derv. w.r.t $\sigma_8$')
# ax.fill_between(bin_centers, deriv_dpdfs_neg_s15_mean[0:10] - om_err, deriv_dpdfs_neg_s15_mean[0:10] + om_err, alpha=0.3)
# ax.fill_between(bin_centers, deriv_dpdfs_neg_s15_mean[10:20] - s8_err, deriv_dpdfs_neg_s15_mean[10:20] + s8_err, alpha=0.3)
# ax.set_xlabel(r'$\kappa$')
# ax.set_ylabel(r'$C^n(\kappa)$')
# ax.set_xticks([-0.005, 0.005])
# ax.set_yticks([0, 500, 1000])


# # Subplot 5: s30 neg
# ax = axes[1,1]
# bin_edges = np.linspace(-0.008, 0.008, 11)
# bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

# fid_err = np.sqrt(np.diag(cov_dpdfs_neg_s30)) / np.sqrt(N_fid)
# om_err = np.sqrt(np.diag(deriv_dpdfs_neg_s30_cov[0:10, 0:10])) / np.sqrt(n_seeds)
# s8_err = np.sqrt(np.diag(deriv_dpdfs_neg_s30_cov[10:20, 10:20])) / np.sqrt(n_seeds)

# ax.plot(bin_centers, dpdfs_neg_fid_s30, c='black')
# ax.errorbar(bin_centers, dpdfs_neg_fid_s30, yerr=fid_err, fmt='|', label='Fiducial', c='black')   
# ax.errorbar(bin_centers, deriv_dpdfs_neg_s30_mean[0:10], yerr=om_err, fmt='|', label=r'Derv. w.r.t $\Omega_{\rm m}$')
# ax.errorbar(bin_centers, deriv_dpdfs_neg_s30_mean[10:20], yerr=s8_err, fmt='|', label=r'Derv. w.r.t $\sigma_8$')
# ax.fill_between(bin_centers, deriv_dpdfs_neg_s30_mean[0:10] - om_err, deriv_dpdfs_neg_s30_mean[0:10] + om_err, alpha=0.3)
# ax.fill_between(bin_centers, deriv_dpdfs_neg_s30_mean[10:20] - s8_err, deriv_dpdfs_neg_s30_mean[10:20] + s8_err, alpha=0.3)
# ax.set_xlabel(r'$\kappa$')
# ax.set_xticks([-0.004, 0.004])
# ax.set_yticks([0, 150, 300])


# # Subplot 6: s90 neg
# ax = axes[1,2]
# bin_edges = np.linspace(-0.005, 0.005, 11)
# bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

# fid_err = np.sqrt(np.diag(cov_dpdfs_neg_s90)) / np.sqrt(N_fid)
# om_err = np.sqrt(np.diag(deriv_dpdfs_neg_s90_cov[0:10, 0:10])) / np.sqrt(n_seeds)
# s8_err = np.sqrt(np.diag(deriv_dpdfs_neg_s90_cov[10:20, 10:20])) / np.sqrt(n_seeds)

# ax.plot(bin_centers, dpdfs_neg_fid_s90, c='black')
# ax.errorbar(bin_centers, dpdfs_neg_fid_s90, yerr=fid_err, fmt='|', label='Fiducial', c='black')   
# ax.errorbar(bin_centers, deriv_dpdfs_neg_s90_mean[0:10], yerr=om_err, fmt='|', label=r'Derv. w.r.t $\Omega_{\rm m}$')
# ax.errorbar(bin_centers, deriv_dpdfs_neg_s90_mean[10:20], yerr=s8_err, fmt='|', label=r'Derv. w.r.t $\sigma_8$')
# ax.fill_between(bin_centers, deriv_dpdfs_neg_s90_mean[0:10] - om_err, deriv_dpdfs_neg_s90_mean[0:10] + om_err, alpha=0.3)
# ax.fill_between(bin_centers, deriv_dpdfs_neg_s90_mean[10:20] - s8_err, deriv_dpdfs_neg_s90_mean[10:20] + s8_err, alpha=0.3)
# ax.set_xlabel(r'$\kappa$')
# ax.set_yticks([0, 25, 50])
# ax.set_xticks([-0.0025, 0.0025])


# handles, labels = ax.get_legend_handles_labels()
# fig.legend(handles, labels, loc='upper center', ncol=3, bbox_to_anchor=(0.5, 1.05))
# plt.subplots_adjust(hspace=0, wspace=0)
# plt.savefig('outfiles/derivs_dcdf.png', dpi=300, bbox_inches='tight')
# plt.close()

#derivs nns_s30 
# 2pp/2nn Left, 2pn/2np Right
fig, axes = plt.subplots(1, 2, figsize=(5, 2), sharey=True)
bin_centers = np.linspace(30, 450, 10)

# Left Panel: PP and NN
ax = axes[0]

# PP Omega_m
y = deriv_nns_combined_s30_mean[40:50]
y_err = np.sqrt(np.diag(deriv_nns_combined_s30_cov[40:50, 40:50]))
y_mean_err = y_err / np.sqrt(n_seeds)

ax.plot(bin_centers, y, c='C0', ls='-', label=r'$\partial C^{\rm PP}_{\rm 2NN}(\theta) / \partial \Omega_{\rm m}$')
# ax.fill_between(bin_centers, y - y_err, y + y_err, alpha=0.1, color='C0')
ax.fill_between(bin_centers, y - y_mean_err, y + y_mean_err, alpha=0.3, color='C0')

# PP sigma_8
y = deriv_nns_combined_s30_mean[200:210]
y_err = np.sqrt(np.diag(deriv_nns_combined_s30_cov[200:210, 200:210]))
y_mean_err = y_err / np.sqrt(n_seeds)

ax.plot(bin_centers, y, c='C1', ls='-', label=r'$\partial C^{\rm PP}_{\rm 2NN}(\theta) / \partial \sigma_8$')
# ax.fill_between(bin_centers, y - y_err, y + y_err, alpha=0.1, color='C1')
ax.fill_between(bin_centers, y - y_mean_err, y + y_mean_err, alpha=0.3, color='C1')

# NN Omega_m
y = deriv_nns_combined_s30_mean[50:60]
ax.plot(bin_centers, y, c='C0', ls='--', label=r'$\partial C^{\rm NN}_{\rm 2NN}(\theta) / \partial \Omega_{\rm m}$')

# NN sigma_8
y = deriv_nns_combined_s30_mean[210:220]
ax.plot(bin_centers, y, c='C1', ls='--', label=r'$\partial C^{\rm NN}_{\rm 2NN}(\theta) / \partial \sigma_8$')

ax.set_ylabel(r'$\partial {\rm CDF}(\theta)/\partial p$')
ax.set_xlabel(r'$\theta$ [arcmin]')
ax.legend(loc='upper center', ncol=2, bbox_to_anchor=(0.5, 1.3), fontsize=6)
ax.axhline(0, color='gray', linestyle='--', linewidth=0.8)
# Right Panel: PN and NP
ax = axes[1]

# PN Omega_m
y = deriv_nns_combined_s30_mean[60:70]
y_err = np.sqrt(np.diag(deriv_nns_combined_s30_cov[60:70, 60:70]))
y_mean_err = y_err / np.sqrt(n_seeds)

ax.plot(bin_centers, y, c='C0', ls='-', label=r'$\partial C^{\rm PN}_{\rm 2NN}(\theta) / \partial \Omega_{\rm m}$')
# ax.fill_between(bin_centers, y - y_err, y + y_err, alpha=0.1, color='C0')
ax.fill_between(bin_centers, y - y_mean_err, y + y_mean_err, alpha=0.3, color='C0')

# PN sigma_8
y = deriv_nns_combined_s30_mean[220:230]
y_err = np.sqrt(np.diag(deriv_nns_combined_s30_cov[220:230, 220:230]))
y_mean_err = y_err / np.sqrt(n_seeds)

ax.plot(bin_centers, y, c='C1', ls='-', label=r'$\partial C^{\rm PN}_{\rm 2NN}(\theta) / \partial \sigma_8$')
# ax.fill_between(bin_centers, y - y_err, y + y_err, alpha=0.1, color='C1')
ax.fill_between(bin_centers, y - y_mean_err, y + y_mean_err, alpha=0.3, color='C1')

# NP Omega_m
y = deriv_nns_combined_s30_mean[70:80]
ax.plot(bin_centers, y, c='C0', ls='--', label=r'$\partial C^{\rm NP}_{\rm 2NN}(\theta) / \partial \Omega_{\rm m}$')

# NP sigma_8
y = deriv_nns_combined_s30_mean[230:240]
ax.plot(bin_centers, y, c='C1', ls='--', label=r'$\partial C^{\rm NP}_{\rm 2NN}(\theta) / \partial \sigma_8$')
ax.axhline(0, color='gray', linestyle='--', linewidth=0.8)
ax.set_xlabel(r'$\theta$ [arcmin]')
ax.legend(loc='upper center', ncol=2, bbox_to_anchor=(0.5, 1.3), fontsize=6)

plt.subplots_adjust(hspace=0, wspace=0)
plt.savefig('outfiles/derivs_nns_s30.png', dpi=300, bbox_inches='tight')
plt.close()

# # plot xi_pp, xi_nn, xi_pn derivatives, all smoothing scales, 3x3
# fig, axes = plt.subplots(2, 3, figsize=(6, 3), sharex='col', sharey='row')
# # Row 1: xi_pp
# # s15
# ax = axes[0,0]
# bin_centers = np.logspace(np.log10(15), np.log10(600), 10)
# ax.plot(bin_centers, pp_fid_s15, c='black')
# ax.errorbar(bin_centers, pp_fid_s15, yerr=np.sqrt(np.diag(cov_pp_fid_s15)), fmt='|', c='black')   
# ax.errorbar(bin_centers, deriv_pp_corrs_s15_mean[0:10], yerr=np.sqrt(np.diag(deriv_pp_corrs_s15_cov[0:10, 0:10])), fmt='|', label=r'Derv. w.r.t $\Omega_{\rm m}$')
# ax.errorbar(bin_centers, deriv_pp_corrs_s15_mean[10:20], yerr=np.sqrt(np.diag(deriv_pp_corrs_s15_cov[10:20, 10:20])), fmt='|', label=r'Derv. w.r.t $\sigma_8$')
# ax.fill_between(bin_centers, deriv_pp_corrs_s15_mean[0:10] - np.sqrt(np.diag(deriv_pp_corrs_s15_cov[0:10, 0:10])), deriv_pp_corrs_s15_mean[0:10] + np.sqrt(np.diag(deriv_pp_corrs_s15_cov[0:10, 0:10])), alpha=0.3)
# ax.fill_between(bin_centers, deriv_pp_corrs_s15_mean[10:20] - np.sqrt(np.diag(deriv_pp_corrs_s15_cov[10:20, 10:20])), deriv_pp_corrs_s15_mean[10:20] + np.sqrt(np.diag(deriv_pp_corrs_s15_cov[10:20, 10:20])), alpha=0.3)
# ax.set_title(r'$\theta_G$ = 15 arcmin')
# ax.set_ylabel(r'$\xi^{pp}(\theta)$')
# ax.set_xscale('log')
# ax.set_xlim(15, 600)
# # ax.grid(True, alpha=0.3)


# # Subplot 2: s30
# ax = axes[0,1]
# bin_centers = np.logspace(np.log10(30), np.log10(600), 10)
# ax.plot(bin_centers, pp_fid_s30, c='black')
# ax.errorbar(bin_centers, pp_fid_s30, yerr=np.sqrt(np.diag(cov_pp_fid_s30)), fmt='|', c='black')
# ax.errorbar(bin_centers, deriv_pp_corrs_s30_mean[0:10], yerr=np.sqrt(np.diag(deriv_pp_corrs_s30_cov[0:10, 0:10])), fmt='|')
# ax.errorbar(bin_centers, deriv_pp_corrs_s30_mean[10:20], yerr=np.sqrt(np.diag(deriv_pp_corrs_s30_cov[10:20, 10:20])), fmt='|')
# ax.fill_between(bin_centers, deriv_pp_corrs_s30_mean[0:10] - np.sqrt(np.diag(deriv_pp_corrs_s30_cov[0:10, 0:10])), deriv_pp_corrs_s30_mean[0:10] + np.sqrt(np.diag(deriv_pp_corrs_s30_cov[0:10, 0:10])), alpha=0.3)
# ax.fill_between(bin_centers, deriv_pp_corrs_s30_mean[10:20] - np.sqrt(np.diag(deriv_pp_corrs_s30_cov[10:20, 10:20])), deriv_pp_corrs_s30_mean[10:20] + np.sqrt(np.diag(deriv_pp_corrs_s30_cov[10:20, 10:20])), alpha=0.3)
# ax.set_title(r'$\theta_G$ = 30 arcmin')
# ax.set_xscale('log')
# ax.set_xlim(30, 600)

# # s90
# ax = axes[0,2]
# bin_centers = np.logspace(np.log10(90), np.log10(600), 10)
# ax.plot(bin_centers, pp_fid_s90, c='black')
# ax.errorbar(bin_centers, pp_fid_s90, yerr=np.sqrt(np.diag(cov_pp_fid_s90)), fmt='|', c='black')
# ax.errorbar(bin_centers, deriv_pp_corrs_s90_mean[0:10], yerr=np.sqrt(np.diag(deriv_pp_corrs_s90_cov[0:10, 0:10])), fmt='|')
# ax.errorbar(bin_centers, deriv_pp_corrs_s90_mean[10:20], yerr=np.sqrt(np.diag(deriv_pp_corrs_s90_cov[10:20, 10:20])), fmt='|')
# ax.fill_between(bin_centers, deriv_pp_corrs_s90_mean[0:10] - np.sqrt(np.diag(deriv_pp_corrs_s90_cov[0:10, 0:10])), deriv_pp_corrs_s90_mean[0:10] + np.sqrt(np.diag(deriv_pp_corrs_s90_cov[0:10, 0:10])), alpha=0.3)
# ax.fill_between(bin_centers, deriv_pp_corrs_s90_mean[10:20] - np.sqrt(np.diag(deriv_pp_corrs_s90_cov[10:20, 10:20])), deriv_pp_corrs_s90_mean[10:20] + np.sqrt(np.diag(deriv_pp_corrs_s90_cov[10:20, 10:20])), alpha=0.3)
# ax.set_title(r'$\theta_G$ = 90 arcmin')
# ax.set_xscale('log')
# ax.set_xlim(90, 600)

# # Row 3: xi_pn
# ax = axes[1,0]
# bin_centers = np.logspace(np.log10(15), np.log10(600), 10)
# ax.plot(bin_centers, pn_fid_s15, c='black')
# ax.errorbar(bin_centers, pn_fid_s15, yerr=np.sqrt(np.diag(cov_pn_fid_s15)), fmt='|', c='black')
# ax.errorbar(bin_centers, deriv_pn_corrs_s15_mean[0:10], yerr=np.sqrt(np.diag(deriv_pn_corrs_s15_cov[0:10, 0:10])), fmt='|')
# ax.errorbar(bin_centers, deriv_pn_corrs_s15_mean[10:20], yerr=np.sqrt(np.diag(deriv_pn_corrs_s15_cov[10:20, 10:20])), fmt='|')
# ax.fill_between(bin_centers, deriv_pn_corrs_s15_mean[0:10] - np.sqrt(np.diag(deriv_pn_corrs_s15_cov[0:10, 0:10])), deriv_pn_corrs_s15_mean[0:10] + np.sqrt(np.diag(deriv_pn_corrs_s15_cov[0:10, 0:10])), alpha=0.3)
# ax.fill_between(bin_centers, deriv_pn_corrs_s15_mean[10:20] - np.sqrt(np.diag(deriv_pn_corrs_s15_cov[10:20, 10:20])), deriv_pn_corrs_s15_mean[10:20] + np.sqrt(np.diag(deriv_pn_corrs_s15_cov[10:20, 10:20])), alpha=0.3)
# ax.set_ylabel(r'$\xi^{pn}(\theta)$')
# ax.set_xscale('log')
# ax.set_xlabel(r'$\theta$ [arcmin]')

# ax = axes[1,1]
# bin_centers = np.logspace(np.log10(30), np.log10(600), 10)
# ax.plot(bin_centers, pn_fid_s30, c='black')
# ax.errorbar(bin_centers, pn_fid_s30, yerr=np.sqrt(np.diag(cov_pn_fid_s30)), fmt='|', c='black')
# ax.errorbar(bin_centers, deriv_pn_corrs_s30_mean[0:10], yerr=np.sqrt(np.diag(deriv_pn_corrs_s30_cov[0:10, 0:10])), fmt='|')
# ax.errorbar(bin_centers, deriv_pn_corrs_s30_mean[10:20], yerr=np.sqrt(np.diag(deriv_pn_corrs_s30_cov[10:20, 10:20])), fmt='|')
# ax.fill_between(bin_centers, deriv_pn_corrs_s30_mean[0:10] - np.sqrt(np.diag(deriv_pn_corrs_s30_cov[0:10, 0:10])), deriv_pn_corrs_s30_mean[0:10] + np.sqrt(np.diag(deriv_pn_corrs_s30_cov[0:10, 0:10])), alpha=0.3)
# ax.fill_between(bin_centers, deriv_pn_corrs_s30_mean[10:20] - np.sqrt(np.diag(deriv_pn_corrs_s30_cov[10:20, 10:20])), deriv_pn_corrs_s30_mean[10:20] + np.sqrt(np.diag(deriv_pn_corrs_s30_cov[10:20, 10:20])), alpha=0.3)
# ax.set_xscale('log')
# ax.set_xlabel(r'$\theta$ [arcmin]')

# ax = axes[1,2]
# bin_centers = np.logspace(np.log10(90), np.log10(600), 10)
# ax.plot(bin_centers, pn_fid_s90, c='black')
# ax.errorbar(bin_centers, pn_fid_s90, yerr=np.sqrt(np.diag(cov_pn_fid_s90)), fmt='|', c='black')
# ax.errorbar(bin_centers, deriv_pn_corrs_s90_mean[0:10], yerr=np.sqrt(np.diag(deriv_pn_corrs_s90_cov[0:10, 0:10])), fmt='|')
# ax.errorbar(bin_centers, deriv_pn_corrs_s90_mean[10:20], yerr=np.sqrt(np.diag(deriv_pn_corrs_s90_cov[10:20, 10:20])), fmt='|')
# ax.fill_between(bin_centers, deriv_pn_corrs_s90_mean[0:10] - np.sqrt(np.diag(deriv_pn_corrs_s90_cov[0:10, 0:10])), deriv_pn_corrs_s90_mean[0:10] + np.sqrt(np.diag(deriv_pn_corrs_s90_cov[0:10, 0:10])), alpha=0.3)
# ax.fill_between(bin_centers, deriv_pn_corrs_s90_mean[10:20] - np.sqrt(np.diag(deriv_pn_corrs_s90_cov[10:20, 10:20])), deriv_pn_corrs_s90_mean[10:20] + np.sqrt(np.diag(deriv_pn_corrs_s90_cov[10:20, 10:20])), alpha=0.3)
# ax.set_xscale('log')
# ax.set_xlabel(r'$\theta$ [arcmin]')

# for col in range(3):
#     axes[0,col].tick_params(labelbottom=False)
#     axes[1,col].set_xticks([100, 500])
#     axes[1,col].set_xticklabels(['100', '500'])

# handles, labels = axes[0,0].get_legend_handles_labels()
# fig.legend(handles, labels, loc='upper center', ncol=3, bbox_to_anchor=(0.5, 1.05))
# plt.subplots_adjust(hspace=0, wspace=0)
# plt.savefig('outfiles/derivs_corrs.png', dpi=300, bbox_inches='tight')
# plt.close()

# derivs_corrs_s30 only plot xi_pp, xi_nn, xi_pn for s30
fig, axes = plt.subplots(1, 3, figsize=(9, 2))
bin_centers = np.logspace(np.log10(30), np.log10(600), 30)

# xi_pp
ax = axes[0]
y = deriv_pp_corrs_s30_mean[0:30]
y_err = np.sqrt(np.diag(deriv_pp_corrs_s30_cov[0:30, 0:30]))
y_mean_err = y_err / np.sqrt(n_seeds)

ax.plot(bin_centers, y, c='C0', ls='-', label=r'$\partial \xi (\theta) / \partial \Omega_{\rm m}$')
# ax.fill_between(bin_centers, y - y_err, y + y_err, alpha=0.1, color='C0')
ax.fill_between(bin_centers, y - y_mean_err, y + y_mean_err, alpha=0.3, color='C0')

y = deriv_pp_corrs_s30_mean[30:60]
y_err = np.sqrt(np.diag(deriv_pp_corrs_s30_cov[30:60, 30:60])) 
y_mean_err = y_err / np.sqrt(n_seeds)

ax.plot(bin_centers, y, c='C1', ls='-', label=r'$\partial \xi (\theta) / \partial \sigma_8$')
# ax.fill_between(bin_centers, y - y_err, y + y_err, alpha=0.1, color='C1')
ax.fill_between(bin_centers, y - y_mean_err, y + y_mean_err, alpha=0.3, color='C1')
# ax.set_ylim(-1, 1)
ax.set_title(r'$\partial \xi^{\rm PP}(\theta) / \partial p$')
ax.set_xscale('log')
ax.set_xlabel(r'$\theta$ [arcmin]')
ax.set_ylabel(r'$\partial \xi(\theta)/\partial p$')
ax.axhline(0, color='gray', linestyle='--', linewidth=0.8)
# xi_nn
ax = axes[1]
y = deriv_nn_corrs_s30_mean[0:30]
y_err = np.sqrt(np.diag(deriv_nn_corrs_s30_cov[0:30, 0:30]))
y_mean_err = y_err / np.sqrt(n_seeds)

ax.plot(bin_centers, y, c='C0', ls='-')
# ax.fill_between(bin_centers, y - y_err, y + y_err, alpha=0.1, color='C0')
ax.fill_between(bin_centers, y - y_mean_err, y + y_mean_err, alpha=0.3, color='C0')

y = deriv_nn_corrs_s30_mean[30:60]
y_err = np.sqrt(np.diag(deriv_nn_corrs_s30_cov[30:60, 30:60])) 

ax.plot(bin_centers, y, c='C1', ls='-')
# ax.fill_between(bin_centers, y - y_err, y + y_err, alpha=0.1, color='C1')
ax.fill_between(bin_centers, y - y_mean_err, y + y_mean_err, alpha=0.3, color='C1')
# ax.set_ylim(-1, 1)
ax.set_title(r'$\partial \xi^{\rm NN}(\theta) / \partial p$')
ax.set_xscale('log')
ax.set_xlabel(r'$\theta$ [arcmin]')
ax.axhline(0, color='gray', linestyle='--', linewidth=0.8)
# xi_pn
ax = axes[2]
y = deriv_pn_corrs_s30_mean[0:30]
y_err = np.sqrt(np.diag(deriv_pn_corrs_s30_cov[0:30, 0:30]))
y_mean_err = y_err / np.sqrt(n_seeds)

ax.plot(bin_centers, y, c='C0', ls='-')
# ax.fill_between(bin_centers, y - y_err, y + y_err, alpha=0.1, color='C0')
ax.fill_between(bin_centers, y - y_mean_err, y + y_mean_err, alpha=0.3, color='C0')

y = deriv_pn_corrs_s30_mean[30:60]
y_err = np.sqrt(np.diag(deriv_pn_corrs_s30_cov[30:60, 30:60])) 

ax.plot(bin_centers, y, c='C1', ls='-')
# ax.fill_between(bin_centers, y - y_err, y + y_err, alpha=0.1, color='C1')
ax.fill_between(bin_centers, y - y_mean_err, y + y_mean_err, alpha=0.3, color='C1')

ax.set_title(r'$\partial \xi^{\rm PN}(\theta) / \partial p$')
ax.set_xscale('log')
ax.set_xlabel(r'$\theta$ [arcmin]')
ax.axhline(0, color='gray', linestyle='--', linewidth=0.8)
handles, labels = axes[0].get_legend_handles_labels()
fig.legend(handles, labels, loc='upper center', ncol=2, bbox_to_anchor=(0.5, 1.15))

plt.subplots_adjust(wspace=0)
plt.savefig('outfiles/derivs_corrs_s30.png', dpi=300, bbox_inches='tight')
plt.close()

# Plot DPC (Peak/Minima) derivatives for s30
fig, axes = plt.subplots(2, 2, figsize=(6, 4))
bin_centers = np.logspace(np.log10(30), np.log10(600), 30)
titles = [r'Pos-Peak ($\xi^{\rm PPk}$)', r'Pos-Minima ($\xi^{\rm PMn}$)', r'Neg-Peak ($\xi^{\rm NPk}$)', r'Neg-Minima ($\xi^{\rm NMn}$)']
means = [deriv_ppk_corrs_s30_mean, deriv_pmn_corrs_s30_mean, deriv_npk_corrs_s30_mean, deriv_nmn_corrs_s30_mean]
covs = [deriv_ppk_corrs_s30_cov, deriv_pmn_corrs_s30_cov, deriv_npk_corrs_s30_cov, deriv_nmn_corrs_s30_cov]

for i, ax in enumerate(axes.flatten()):
    mean = means[i]
    cov = covs[i]
    
    # Omega_m
    y = mean[0:30]
    y_err = np.sqrt(np.diag(cov[0:30, 0:30]))
    y_mean_err = y_err / np.sqrt(n_seeds)

    ax.plot(bin_centers, y, c='C0', ls='-', label=r'$\partial \xi / \partial \Omega_{\rm m}$')
    # ax.fill_between(bin_centers, y - y_err, y + y_err, alpha=0.1, color='C0')
    ax.fill_between(bin_centers, y - y_mean_err, y + y_mean_err, alpha=0.3, color='C0')

    # sigma_8
    y = mean[30:60]
    y_err = np.sqrt(np.diag(cov[30:60, 30:60]))
    y_mean_err = y_err / np.sqrt(n_seeds)

    ax.plot(bin_centers, y, c='C1', ls='-', label=r'$\partial \xi / \partial \sigma_8$')
    # ax.fill_between(bin_centers, y - y_err, y + y_err, alpha=0.1, color='C1')
    ax.fill_between(bin_centers, y - y_mean_err, y + y_mean_err, alpha=0.3, color='C1')
    
    ax.set_title(titles[i])
    ax.set_xscale('log')
    ax.set_xlim(30, 600)
    ax.axhline(0, color='gray', linestyle='--', linewidth=0.8)
    if i >= 2:
        ax.set_xlabel(r'$\theta$ [arcmin]')

handles, labels = axes[0,0].get_legend_handles_labels()
fig.legend(handles, labels, loc='upper center', ncol=2, bbox_to_anchor=(0.5, 1.05))

plt.subplots_adjust(hspace=0.3, wspace=0.2)
plt.savefig('outfiles/derivs_dpc_s30.png', dpi=300, bbox_inches='tight')
plt.close()