import numpy as np
from tqdm import tqdm
import os
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde, norm

def calculate_kl(data_vec):
    """Calculates KL(P || Q) where P is data, Q is Gaussian approximation."""
    mu, std = np.mean(data_vec), np.std(data_vec)
    if std == 0: return 0.0
    
    # Discretize for KL calc
    hist, bin_edges = np.histogram(data_vec, bins=50, density=True)
    pk = hist * np.diff(bin_edges) # probabilities
    
    # Ensure bin centers
    x = (bin_edges[:-1] + bin_edges[1:]) / 2
    qk = norm.pdf(x, mu, std) * np.diff(bin_edges)
    
    # Avoid log(0) and division by zero
    pk = np.where(pk == 0, 1e-10, pk)
    qk = np.where(qk == 0, 1e-10, qk)
    
    return np.sum(pk * np.log(pk / qk))

class FisherProbe:
    def __init__(self, name, samples, derivatives, n_realizations=1000, indices=None, sigma_thresh=1.0):
        self.name = name
        self.samples = samples
        self.derivatives = derivatives
        self.n_realizations = n_realizations
        self.sigma_thresh = sigma_thresh 

        self.mean = np.mean(samples, axis=0)
        self.cov = np.cov(samples, rowvar=False)

        if indices is not None:
            self.indices = indices
            self.recalculated_mask = False
        else:
            self.indices = self._compute_indices(sigma_thresh)
            self.recalculated_mask = True
        
        self.deriv_filt, self.inv_cov_filt = self._apply_mask()

    def _compute_indices(self, sigma_thresh=1.0):
        fid_vec = self.mean
        cov = self.cov
        # S/N Filter
        sigma = np.sqrt(np.diag(cov))
        mask = np.abs(fid_vec) > sigma_thresh * sigma
        
        # KL Filter
        kl_values = np.zeros(fid_vec.shape[0])
        for i in range(fid_vec.shape[0]):
             if mask[i]:
                 kl_values[i] = calculate_kl(self.samples[:, i])
        
        original_count = mask.sum()
        mask = mask & (kl_values <= 0.1)
        kl_removed_count = original_count - mask.sum()
        if kl_removed_count > 0:
            print(f"{self.name}: removed {kl_removed_count} bins due to KL > 0.1")

        if mask.sum() == 0:
            print(f"WARNING: {self.name}: no bins survive S/N + KL cuts. Using all bins.")
            mask[:] = True
            
        # Correlation filtering
        cov_trim = cov[np.ix_(mask, mask)]
        sigma_trim = np.sqrt(np.diag(cov_trim))
        sigma_trim[sigma_trim == 0] = 1.0
        corr = cov_trim / np.outer(sigma_trim, sigma_trim)
        
        corr_mask = np.ones(corr.shape[0], dtype=bool)
        for i in range(corr.shape[0]):
            if not corr_mask[i]: continue
            for j in range(i + 1, corr.shape[0]):
                if not corr_mask[j]: continue
                if np.abs(corr[i, j]) >= 0.9999:
                    # Remove lower S/N
                    idx_i = np.where(mask)[0][i]
                    idx_j = np.where(mask)[0][j]
                    s_i = sigma[idx_i] if sigma[idx_i] > 0 else 1.0
                    s_j = sigma[idx_j] if sigma[idx_j] > 0 else 1.0
                    sn_i = np.abs(fid_vec[idx_i]) / s_i
                    sn_j = np.abs(fid_vec[idx_j]) / s_j
                    if sn_i <= sn_j:
                        corr_mask[i] = False
                    else:
                        corr_mask[j] = False
        
        original_indices = np.where(mask)[0]
        final_indices = original_indices[corr_mask]
        final_mask = np.zeros_like(mask)
        final_mask[final_indices] = True
        
        print(f"{self.name}: kept {final_mask.sum()}/{len(final_mask)} bins")

        # Condition Number Check (Fix for PDF normalization constraint sum(p)=1)
        final_cov = cov[np.ix_(final_mask, final_mask)]
        if final_cov.shape[0] > 0:
            cond_num = np.linalg.cond(final_cov)
            while cond_num > 1e10 and final_mask.sum() > 1:
                # Find the direction of the smallest eigenvalue (null mode)
                vals, vecs = np.linalg.eigh(final_cov)
                # Smallest eigenmode is the first one (eigh returns sorted)
                null_mode = vecs[:, 0]
                
                # Find the bin with the largest contribution to this null mode
                # We map back to the indices within the current final_cov
                current_indices = np.where(final_mask)[0]
                idx_to_remove_local = np.argmax(np.abs(null_mode))
                idx_to_remove_global = current_indices[idx_to_remove_local]
                
                print(f"WARNING: {self.name} covariance is singular (cond={cond_num:.1e}). Removing bin {idx_to_remove_global} to break degeneracy (likely sum constraint).")
                final_mask[idx_to_remove_global] = False
                
                # Re-evaluate
                final_cov = cov[np.ix_(final_mask, final_mask)]
                if final_cov.shape[0] > 0:
                    cond_num = np.linalg.cond(final_cov)
                else:
                    break

        return final_mask

    def _apply_mask(self):
        new_mean = self.mean[self.indices]
        
        # Derivatives shape handling
        if self.derivatives.shape[0] == len(self.indices): 
             new_deriv = self.derivatives[self.indices, :]
        else: 
             new_deriv = self.derivatives[:, self.indices]
             
        new_cov = self.cov[np.ix_(self.indices, self.indices)]
        
        # Hartlap
        n_bins_kept = new_cov.shape[0]
        hartlap = (self.n_realizations - n_bins_kept - 2) / (self.n_realizations - 1)
        new_inv_cov = np.linalg.inv(new_cov) * hartlap
        
        return new_deriv, new_inv_cov

    @classmethod
    def combine(cls, name, probes):
        samples = np.hstack([p.samples for p in probes])
        derivs = np.hstack([p.derivatives for p in probes])
        
        combined_mask = np.concatenate([p.indices for p in probes])
        
        return cls(name, samples, derivs, probes[0].n_realizations, indices=combined_mask)

# def filter_data_block(name, fid_vec, deriv, cov, n_realizations, samples=None, sigma_thresh=1.0):
#     # First filter: remove bins with low signal-to-noise
#     sigma = np.sqrt(np.diag(cov))
#     mask = np.abs(fid_vec) > sigma_thresh * sigma
    
#     # KL Filter
#     if samples is not None:
#         kl_values = np.zeros(fid_vec.shape[0])
#         # Calculate KL only for S/N passed bins
#         for i in range(fid_vec.shape[0]):
#              if mask[i]:
#                  kl_values[i] = calculate_kl(samples[:, i])
        
#         original_count = mask.sum()
#         mask = mask & (kl_values <= 0.1)
#         kl_removed_count = original_count - mask.sum()
#         if kl_removed_count > 0:
#             print(f"{name}: removed {kl_removed_count} bins due to KL > 0.1")

#     if mask.sum() == 0:
#         print(f"WARNING: {name}: no bins survive S/N + KL cuts. Using all bins.")
#         mask[:] = True

#     # Apply first cut
#     cov_trim = cov[np.ix_(mask, mask)]
    
#     # Second filter: remove bins with 100% correlation
#     sigma_trim = np.sqrt(np.diag(cov_trim))
#     sigma_trim[sigma_trim == 0] = 1.0
    
#     corr = cov_trim / np.outer(sigma_trim, sigma_trim)
    
#     # Find pairs with |correlation| >= 0.9999
#     corr_mask = np.ones(corr.shape[0], dtype=bool)
#     for i in range(corr.shape[0]):
#         if not corr_mask[i]: continue
#         for j in range(i + 1, corr.shape[0]):
#             if not corr_mask[j]: continue
#             if np.abs(corr[i, j]) >= 0.9999:
#                 # Remove the bin with lower S/N
#                 idx_i = np.where(mask)[0][i]
#                 idx_j = np.where(mask)[0][j]
                
#                 s_i = sigma[idx_i] if sigma[idx_i] > 0 else 1.0
#                 s_j = sigma[idx_j] if sigma[idx_j] > 0 else 1.0
                
#                 sn_i = np.abs(fid_vec[idx_i]) / s_i
#                 sn_j = np.abs(fid_vec[idx_j]) / s_j
                
#                 if sn_i <= sn_j:
#                     corr_mask[i] = False
#                 else:
#                     corr_mask[j] = False
    
#     # Final mask update
#     original_indices = np.where(mask)[0]
#     final_indices = original_indices[corr_mask]
#     final_mask = np.zeros_like(mask)
#     final_mask[final_indices] = True
    
#     print(f"{name}: kept {final_mask.sum()}/{len(final_mask)} bins")
    
#     # Apply Final Mask
#     new_mean = fid_vec[final_mask]
    
#     # Handle deriv: if deriv is (N_params, N_bins) or (N_bins, N_params)?
#     # In this script, deriv_* seem to be (N_params, N_bins)
#     # Check shape: deriv_pdfs_s15 shape.
#     if deriv.shape[0] == len(final_mask): 
#         # (N_bins, N_params)
#         new_deriv = deriv[final_mask, :]
#     else: 
#         # (N_params, N_bins)
#         new_deriv = deriv[:, final_mask]
        
#     new_cov = cov[np.ix_(final_mask, final_mask)]
    
#     # Invert Covariance (Hartlap factor)
#     # factor = (n_realizations - n_bins - 2) / (n_realizations - 1)
#     n_bins_kept = new_cov.shape[0]
#     hartlap = (n_realizations - n_bins_kept - 2) / (n_realizations - 1)
#     new_inv_cov = np.linalg.inv(new_cov) * hartlap
    
#     return new_deriv, new_inv_cov

RAW_DATA_DIR = '/home/jbai/cephfs_copy/cephfs/jbai/DefectMockGaussian'
def load_mock_samples(scale_deg, n_realizations=1000):
    s = int(scale_deg) # 15, 30, 90
    print(f"Loading samples for scale s{s}...")
    
    # Initialize containers
    data = {
        'pdf': [],
        'dpdf_pos': [],
        'dpdf_neg': [],
        'kk': [],
        'pp': [],
        'nn': [],
        'pn': [],
        'pk': [],
        'nk': [],
        'nns': [],
        'ppk': [], 
        'pmn': [], 
        'npk': [], 
        'nmn': []
    }
    
    for i in tqdm(range(n_realizations), desc=f"Loading s{s}"):
        try:
            # Load PDF
            d = np.load(f"{RAW_DATA_DIR}/PDFs/realisation_{i}_s{s}_kappa_pdf.npz")
            data['pdf'].append(d['hist'])
            
            # Load Defect PDFs
            d = np.load(f"{RAW_DATA_DIR}/DefectPDFs/realisation_{i}_s{s}_defect_kappa_pos_pdf.npz")
            data['dpdf_pos'].append(d['hist'])
            d = np.load(f"{RAW_DATA_DIR}/DefectPDFs/realisation_{i}_s{s}_defect_kappa_neg_pdf.npz")
            data['dpdf_neg'].append(d['hist'])
            
            # Load Corrs
            d = np.load(f"{RAW_DATA_DIR}/KKs/realisation_{i}_s{s}_kappa_kappa_corr.npz")
            data['kk'].append(d['xi_kk'])
            
            d = np.load(f"{RAW_DATA_DIR}/PosPos/realisation_{i}_s{s}_pos_pos_corr.npz")
            data['pp'].append(d['xi_pp'])
            
            d = np.load(f"{RAW_DATA_DIR}/NegNeg/realisation_{i}_s{s}_neg_neg_corr.npz")
            data['nn'].append(d['xi_nn']) # Check key name
            
            d = np.load(f"{RAW_DATA_DIR}/PosNeg/realisation_{i}_s{s}_pos_neg_corr.npz")
            data['pn'].append(d['xi_pn']) # Check key name
            
            # d = np.load(f"{RAW_DATA_DIR}/PKs/realisation_{i}_s{s}_pos_kappa_corr.npz")
            # data['pk'].append(d['xi_pk'])
            
            # d = np.load(f"{RAW_DATA_DIR}/NKs/realisation_{i}_s{s}_neg_kappa_corr.npz")
            # data['nk'].append(d['xi_nk'])

            # Load NNs Combined
            d = np.load(f"{RAW_DATA_DIR}/NNs/realisation_{i}_s{s}_NNs.npz")
            
            nn_comb = []
            for scale_idx in [1, 2, 4, 8]:
                nn_comb.extend(d[f'nn_pos_same{scale_idx}_cum'])
                nn_comb.extend(d[f'nn_neg_same{scale_idx}_cum'])
                nn_comb.extend(d[f'nn_pos_opp{scale_idx}_cum'])
                nn_comb.extend(d[f'nn_neg_opp{scale_idx}_cum'])
            data['nns'].append(np.array(nn_comb))

            # Load Defect-Peak/Fit 
            d = np.load(f"{RAW_DATA_DIR}/PosPeak/realisation_{i}_s{s}_pos_peak_corr.npz")
            data['ppk'].append(d['xi'])
            d = np.load(f"{RAW_DATA_DIR}/PosMin/realisation_{i}_s{s}_pos_min_corr.npz")
            data['pmn'].append(d['xi'])
            d = np.load(f"{RAW_DATA_DIR}/NegPeak/realisation_{i}_s{s}_neg_peak_corr.npz")
            data['npk'].append(d['xi'])
            d = np.load(f"{RAW_DATA_DIR}/NegMin/realisation_{i}_s{s}_neg_min_corr.npz")
            data['nmn'].append(d['xi'])
            
        except FileNotFoundError:
            pass

    # Convert to arrays
    final_data = {}
    for k, v in data.items():
        if len(v) > 0:
            final_data[k] = np.array(v)
            
    return final_data

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

ppk_corrs_s15, ppk_corrs_s30, ppk_corrs_s90 = [], [], []
pmn_corrs_s15, pmn_corrs_s30, pmn_corrs_s90 = [], [], []
npk_corrs_s15, npk_corrs_s30, npk_corrs_s90 = [], [], []
nmn_corrs_s15, nmn_corrs_s30, nmn_corrs_s90 = [], [], []

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

    # Load Defect-Peak/Minima
    data_ppk = np.load(f"{DATA_DIR}/TrainPosPeak/train_{i}_s15_pos_peak_corr.npz")
    ppk_corrs_s15.append((data_ppk['theta'], data_ppk['xi']))
    data_ppk = np.load(f"{DATA_DIR}/TrainPosPeak/train_{i}_s30_pos_peak_corr.npz")
    ppk_corrs_s30.append((data_ppk['theta'], data_ppk['xi']))
    data_ppk = np.load(f"{DATA_DIR}/TrainPosPeak/train_{i}_s90_pos_peak_corr.npz")
    ppk_corrs_s90.append((data_ppk['theta'], data_ppk['xi']))

    data_pmn = np.load(f"{DATA_DIR}/TrainPosMin/train_{i}_s15_pos_min_corr.npz")
    pmn_corrs_s15.append((data_pmn['theta'], data_pmn['xi']))
    data_pmn = np.load(f"{DATA_DIR}/TrainPosMin/train_{i}_s30_pos_min_corr.npz")
    pmn_corrs_s30.append((data_pmn['theta'], data_pmn['xi']))
    data_pmn = np.load(f"{DATA_DIR}/TrainPosMin/train_{i}_s90_pos_min_corr.npz")
    pmn_corrs_s90.append((data_pmn['theta'], data_pmn['xi']))

    data_npk = np.load(f"{DATA_DIR}/TrainNegPeak/train_{i}_s15_neg_peak_corr.npz")
    npk_corrs_s15.append((data_npk['theta'], data_npk['xi']))
    data_npk = np.load(f"{DATA_DIR}/TrainNegPeak/train_{i}_s30_neg_peak_corr.npz")
    npk_corrs_s30.append((data_npk['theta'], data_npk['xi']))
    data_npk = np.load(f"{DATA_DIR}/TrainNegPeak/train_{i}_s90_neg_peak_corr.npz")
    npk_corrs_s90.append((data_npk['theta'], data_npk['xi']))

    data_nmn = np.load(f"{DATA_DIR}/TrainNegMin/train_{i}_s15_neg_min_corr.npz")
    nmn_corrs_s15.append((data_nmn['theta'], data_nmn['xi']))
    data_nmn = np.load(f"{DATA_DIR}/TrainNegMin/train_{i}_s30_neg_min_corr.npz")
    nmn_corrs_s30.append((data_nmn['theta'], data_nmn['xi']))
    data_nmn = np.load(f"{DATA_DIR}/TrainNegMin/train_{i}_s90_neg_min_corr.npz")
    nmn_corrs_s90.append((data_nmn['theta'], data_nmn['xi']))
    

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
deriv_ppk_corrs_s15, deriv_ppk_corrs_s30, deriv_ppk_corrs_s90 = [], [], []
deriv_pmn_corrs_s15, deriv_pmn_corrs_s30, deriv_pmn_corrs_s90 = [], [], []
deriv_npk_corrs_s15, deriv_npk_corrs_s30, deriv_npk_corrs_s90 = [], [], []
deriv_nmn_corrs_s15, deriv_nmn_corrs_s30, deriv_nmn_corrs_s90 = [], [], []

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
    
    # Peak/Minima derivatives
    for scale_idx, (corrs_list, deriv_list) in enumerate([
        ([ppk_corrs_s15, ppk_corrs_s30, ppk_corrs_s90], [deriv_ppk_corrs_s15, deriv_ppk_corrs_s30, deriv_ppk_corrs_s90]),
        ([pmn_corrs_s15, pmn_corrs_s30, pmn_corrs_s90], [deriv_pmn_corrs_s15, deriv_pmn_corrs_s30, deriv_pmn_corrs_s90]),
        ([npk_corrs_s15, npk_corrs_s30, npk_corrs_s90], [deriv_npk_corrs_s15, deriv_npk_corrs_s30, deriv_npk_corrs_s90]),
        ([nmn_corrs_s15, nmn_corrs_s30, nmn_corrs_s90], [deriv_nmn_corrs_s15, deriv_nmn_corrs_s30, deriv_nmn_corrs_s90]),
    ]):
        for s_idx in range(3):
           theta_plus, xi_plus = corrs_list[s_idx][i+1]
           theta_minus, xi_minus = corrs_list[s_idx][i]
           deriv_val = (xi_plus - xi_minus) / (2 * delta_params[list(fid.keys())[i//2]])
           deriv_list[s_idx].append(deriv_val)

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
deriv_gq_corrs_s15 = (deriv_pp_corrs_s15 + deriv_nn_corrs_s15 - 2.0 * deriv_pn_corrs_s15) / 4.0
deriv_gq_corrs_s30 = (deriv_pp_corrs_s30 + deriv_nn_corrs_s30 - 2.0 * deriv_pn_corrs_s30) / 4.0
deriv_gq_corrs_s90 = (deriv_pp_corrs_s90 + deriv_nn_corrs_s90 - 2.0 * deriv_pn_corrs_s90) / 4.0
deriv_pk_corrs_s15 = np.array(deriv_pk_corrs_s15)
deriv_pk_corrs_s30 = np.array(deriv_pk_corrs_s30)
deriv_pk_corrs_s90 = np.array(deriv_pk_corrs_s90)
deriv_nk_corrs_s15 = np.array(deriv_nk_corrs_s15)
deriv_nk_corrs_s30 = np.array(deriv_nk_corrs_s30)
deriv_nk_corrs_s90 = np.array(deriv_nk_corrs_s90)
deriv_ppk_corrs_s15 = np.array(deriv_ppk_corrs_s15)
deriv_ppk_corrs_s30 = np.array(deriv_ppk_corrs_s30)
deriv_ppk_corrs_s90 = np.array(deriv_ppk_corrs_s90)
deriv_pmn_corrs_s15 = np.array(deriv_pmn_corrs_s15)
deriv_pmn_corrs_s30 = np.array(deriv_pmn_corrs_s30)
deriv_pmn_corrs_s90 = np.array(deriv_pmn_corrs_s90)
deriv_npk_corrs_s15 = np.array(deriv_npk_corrs_s15)
deriv_npk_corrs_s30 = np.array(deriv_npk_corrs_s30)
deriv_npk_corrs_s90 = np.array(deriv_npk_corrs_s90)
deriv_nmn_corrs_s15 = np.array(deriv_nmn_corrs_s15)
deriv_nmn_corrs_s30 = np.array(deriv_nmn_corrs_s30)
deriv_nmn_corrs_s90 = np.array(deriv_nmn_corrs_s90)

deriv_dpc_combined_s15 = np.hstack((deriv_ppk_corrs_s15, deriv_pmn_corrs_s15, deriv_npk_corrs_s15, deriv_nmn_corrs_s15))
deriv_dpc_combined_s30 = np.hstack((deriv_ppk_corrs_s30, deriv_pmn_corrs_s30, deriv_npk_corrs_s30, deriv_nmn_corrs_s30))
deriv_dpc_combined_s90 = np.hstack((deriv_ppk_corrs_s90, deriv_pmn_corrs_s90, deriv_npk_corrs_s90, deriv_nmn_corrs_s90))
deriv_dpc_combined_all_scales = np.hstack((deriv_dpc_combined_s15, deriv_dpc_combined_s30, deriv_dpc_combined_s90))
print("DERIV DPC COMBINED S90:", deriv_dpc_combined_s90)
deriv_dcorrs_combined_s15 = np.hstack((deriv_pp_corrs_s15, deriv_nn_corrs_s15, deriv_pn_corrs_s15))
deriv_dcorrs_combined_s30 = np.hstack((deriv_pp_corrs_s30, deriv_nn_corrs_s30, deriv_pn_corrs_s30))
deriv_dcorrs_combined_s90 = np.hstack((deriv_pp_corrs_s90, deriv_nn_corrs_s90, deriv_pn_corrs_s90))
deriv_dcorrs_combined_all_scales = np.hstack((deriv_dcorrs_combined_s15, deriv_dcorrs_combined_s30, deriv_dcorrs_combined_s90))
print("DERIV DCORRS COMBINED S90:", deriv_dcorrs_combined_s90)
deriv_gq_corrs_all_scales = np.hstack((deriv_gq_corrs_s15, deriv_gq_corrs_s30, deriv_gq_corrs_s90))
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
n = 1000 # number of realizations or close to it
slics_samples = {}
scale_map = {15: 15, 30: 30, 90: 90} 

# 1. Load Samples
def add_gq_samples(samples):
    if all(k in samples for k in ("pp", "nn", "pn")):
        samples["gq"] = (samples["pp"] + samples["nn"] - 2.0 * samples["pn"]) / 4.0
    return samples

samples_s15 = load_mock_samples(15, n_realizations=n)
samples_s30 = load_mock_samples(30, n_realizations=n)
samples_s90 = load_mock_samples(90, n_realizations=n)
samples_s15 = add_gq_samples(samples_s15)
samples_s30 = add_gq_samples(samples_s30)
samples_s90 = add_gq_samples(samples_s90)
samples_map = {15: samples_s15, 30: samples_s30, 90: samples_s90}

# 2. Define Helper for processing a single probe/scale
def process_probe(name, samples_dict, key, deriv, n_sims=1000, combined_samples=None, passed_indices=None):
    if combined_samples is not None:
        s_data = combined_samples
    else:
        s_data = samples_dict[key]
    
    probe = FisherProbe(name, s_data, deriv, n_sims, indices=passed_indices)
    return probe

# Pre-construct Raw All-Scale vectors before overwriting components
deriv_pdfs_all_raw = np.hstack((deriv_pdfs_s15, deriv_pdfs_s30, deriv_pdfs_s90))
deriv_kk_all_raw = np.hstack((deriv_kk_corrs_s15, deriv_kk_corrs_s30, deriv_kk_corrs_s90))

# 3. Process PDF
probe_pdfs_s15 = process_probe("pdf_s15", samples_s15, 'pdf', deriv_pdfs_s15, n)
deriv_pdfs_s15, inv_cov_pdf_s15 = probe_pdfs_s15.deriv_filt, probe_pdfs_s15.inv_cov_filt

probe_pdfs_s30 = process_probe("pdf_s30", samples_s30, 'pdf', deriv_pdfs_s30, n)
deriv_pdfs_s30, inv_cov_pdf_s30 = probe_pdfs_s30.deriv_filt, probe_pdfs_s30.inv_cov_filt

probe_pdfs_s90 = process_probe("pdf_s90", samples_s90, 'pdf', deriv_pdfs_s90, n)
deriv_pdfs_s90, inv_cov_pdf_s90 = probe_pdfs_s90.deriv_filt, probe_pdfs_s90.inv_cov_filt
print("inv_cov_pdf_s90:", inv_cov_pdf_s90)
print("cov_pdf_s90:", np.linalg.det(probe_pdfs_s90.cov))

# 4. Process DPDF (Pos, Neg)
probe_dpdfs_pos_s15 = process_probe("dpdf_pos_s15", samples_s15, 'dpdf_pos', deriv_dpdfs_pos_s15, n)
deriv_dpdfs_pos_s15, inv_cov_pos_s15 = probe_dpdfs_pos_s15.deriv_filt, probe_dpdfs_pos_s15.inv_cov_filt

probe_dpdfs_pos_s30 = process_probe("dpdf_pos_s30", samples_s30, 'dpdf_pos', deriv_dpdfs_pos_s30, n)
deriv_dpdfs_pos_s30, inv_cov_pos_s30 = probe_dpdfs_pos_s30.deriv_filt, probe_dpdfs_pos_s30.inv_cov_filt

probe_dpdfs_pos_s90 = process_probe("dpdf_pos_s90", samples_s90, 'dpdf_pos', deriv_dpdfs_pos_s90, n)
deriv_dpdfs_pos_s90, inv_cov_pos_s90 = probe_dpdfs_pos_s90.deriv_filt, probe_dpdfs_pos_s90.inv_cov_filt

probe_dpdfs_neg_s15 = process_probe("dpdf_neg_s15", samples_s15, 'dpdf_neg', deriv_dpdfs_neg_s15, n)
deriv_dpdfs_neg_s15, inv_cov_neg_s15 = probe_dpdfs_neg_s15.deriv_filt, probe_dpdfs_neg_s15.inv_cov_filt

probe_dpdfs_neg_s30 = process_probe("dpdf_neg_s30", samples_s30, 'dpdf_neg', deriv_dpdfs_neg_s30, n)
deriv_dpdfs_neg_s30, inv_cov_neg_s30 = probe_dpdfs_neg_s30.deriv_filt, probe_dpdfs_neg_s30.inv_cov_filt

probe_dpdfs_neg_s90 = process_probe("dpdf_neg_s90", samples_s90, 'dpdf_neg', deriv_dpdfs_neg_s90, n)
deriv_dpdfs_neg_s90, inv_cov_neg_s90 = probe_dpdfs_neg_s90.deriv_filt, probe_dpdfs_neg_s90.inv_cov_filt

# 5. Process DPDF Combined (Pos + Neg) per scale
probe_dpdfs_combined_s15 = FisherProbe.combine("dpdf_comb_s15", [probe_dpdfs_pos_s15, probe_dpdfs_neg_s15])
deriv_dpdfs_combined_s15, inv_cov_dpdf_combined_s15 = probe_dpdfs_combined_s15.deriv_filt, probe_dpdfs_combined_s15.inv_cov_filt

probe_dpdfs_combined_s30 = FisherProbe.combine("dpdf_comb_s30", [probe_dpdfs_pos_s30, probe_dpdfs_neg_s30])
deriv_dpdfs_combined_s30, inv_cov_dpdf_combined_s30 = probe_dpdfs_combined_s30.deriv_filt, probe_dpdfs_combined_s30.inv_cov_filt

probe_dpdfs_combined_s90 = FisherProbe.combine("dpdf_comb_s90", [probe_dpdfs_pos_s90, probe_dpdfs_neg_s90])
deriv_dpdfs_combined_s90, inv_cov_dpdf_combined_s90 = probe_dpdfs_combined_s90.deriv_filt, probe_dpdfs_combined_s90.inv_cov_filt

probe_dpdfs_combined_all_scales = FisherProbe.combine("dpdf_comb_all", [probe_dpdfs_combined_s15, probe_dpdfs_combined_s30, probe_dpdfs_combined_s90])
deriv_dpdfs_combined_all_scales, inv_cov_dpdf_combined_all_scales = probe_dpdfs_combined_all_scales.deriv_filt, probe_dpdfs_combined_all_scales.inv_cov_filt

# 6. Process KK
probe_kk_s15 = process_probe("kk_s15", samples_s15, 'kk', deriv_kk_corrs_s15, n)
deriv_kk_corrs_s15, inv_cov_kk_s15 = probe_kk_s15.deriv_filt, probe_kk_s15.inv_cov_filt

probe_kk_s30 = process_probe("kk_s30", samples_s30, 'kk', deriv_kk_corrs_s30, n)
deriv_kk_corrs_s30, inv_cov_kk_s30 = probe_kk_s30.deriv_filt, probe_kk_s30.inv_cov_filt

probe_kk_s90 = process_probe("kk_s90", samples_s90, 'kk', deriv_kk_corrs_s90, n)
deriv_kk_corrs_s90, inv_cov_kk_s90 = probe_kk_s90.deriv_filt, probe_kk_s90.inv_cov_filt

# 7. Process Other Correlations (PP, NN, PN, PK, NK)
probe_pp_s15 = process_probe("pp_s15", samples_s15, 'pp', deriv_pp_corrs_s15, n)
deriv_pp_corrs_s15, inv_cov_pp_s15 = probe_pp_s15.deriv_filt, probe_pp_s15.inv_cov_filt

probe_pp_s30 = process_probe("pp_s30", samples_s30, 'pp', deriv_pp_corrs_s30, n)
deriv_pp_corrs_s30, inv_cov_pp_s30 = probe_pp_s30.deriv_filt, probe_pp_s30.inv_cov_filt

probe_pp_s90 = process_probe("pp_s90", samples_s90, 'pp', deriv_pp_corrs_s90, n)
deriv_pp_corrs_s90, inv_cov_pp_s90 = probe_pp_s90.deriv_filt, probe_pp_s90.inv_cov_filt

probe_nn_corr_s15 = process_probe("nn_s15", samples_s15, 'nn', deriv_nn_corrs_s15, n)
deriv_nn_corrs_s15, inv_cov_nn_corr_s15 = probe_nn_corr_s15.deriv_filt, probe_nn_corr_s15.inv_cov_filt

probe_nn_corr_s30 = process_probe("nn_s30", samples_s30, 'nn', deriv_nn_corrs_s30, n)
deriv_nn_corrs_s30, inv_cov_nn_corr_s30 = probe_nn_corr_s30.deriv_filt, probe_nn_corr_s30.inv_cov_filt

probe_nn_corr_s90 = process_probe("nn_s90", samples_s90, 'nn', deriv_nn_corrs_s90, n)
deriv_nn_corrs_s90, inv_cov_nn_corr_s90 = probe_nn_corr_s90.deriv_filt, probe_nn_corr_s90.inv_cov_filt

probe_pn_s15 = process_probe("pn_s15", samples_s15, 'pn', deriv_pn_corrs_s15, n)
deriv_pn_corrs_s15, inv_cov_pn_s15 = probe_pn_s15.deriv_filt, probe_pn_s15.inv_cov_filt

probe_pn_s30 = process_probe("pn_s30", samples_s30, 'pn', deriv_pn_corrs_s30, n)
deriv_pn_corrs_s30, inv_cov_pn_s30 = probe_pn_s30.deriv_filt, probe_pn_s30.inv_cov_filt

probe_pn_s90 = process_probe("pn_s90", samples_s90, 'pn', deriv_pn_corrs_s90, n)
deriv_pn_corrs_s90, inv_cov_pn_s90 = probe_pn_s90.deriv_filt, probe_pn_s90.inv_cov_filt

probe_gq_s15 = process_probe("gq_s15", samples_s15, 'gq', deriv_gq_corrs_s15, n)
deriv_gq_corrs_s15, inv_cov_gq_s15 = probe_gq_s15.deriv_filt, probe_gq_s15.inv_cov_filt

probe_gq_s30 = process_probe("gq_s30", samples_s30, 'gq', deriv_gq_corrs_s30, n)
deriv_gq_corrs_s30, inv_cov_gq_s30 = probe_gq_s30.deriv_filt, probe_gq_s30.inv_cov_filt

probe_gq_s90 = process_probe("gq_s90", samples_s90, 'gq', deriv_gq_corrs_s90, n)
deriv_gq_corrs_s90, inv_cov_gq_s90 = probe_gq_s90.deriv_filt, probe_gq_s90.inv_cov_filt

probe_gq_all_scales = FisherProbe.combine("gq_all", [probe_gq_s15, probe_gq_s30, probe_gq_s90])
deriv_gq_corrs_all_scales, inv_cov_gq_all_scales = probe_gq_all_scales.deriv_filt, probe_gq_all_scales.inv_cov_filt

# probe_pk_s15 = process_probe("pk_s15", samples_s15, 'pk', deriv_pk_corrs_s15, n)
# deriv_pk_corrs_s15, inv_cov_pk_s15 = probe_pk_s15.deriv_filt, probe_pk_s15.inv_cov_filt

# probe_pk_s30 = process_probe("pk_s30", samples_s30, 'pk', deriv_pk_corrs_s30, n)
# deriv_pk_corrs_s30, inv_cov_pk_s30 = probe_pk_s30.deriv_filt, probe_pk_s30.inv_cov_filt

# probe_pk_s90 = process_probe("pk_s90", samples_s90, 'pk', deriv_pk_corrs_s90, n)
# deriv_pk_corrs_s90, inv_cov_pk_s90 = probe_pk_s90.deriv_filt, probe_pk_s90.inv_cov_filt

# probe_nk_s15 = process_probe("nk_s15", samples_s15, 'nk', deriv_nk_corrs_s15, n)
# deriv_nk_corrs_s15, inv_cov_nk_s15 = probe_nk_s15.deriv_filt, probe_nk_s15.inv_cov_filt

# probe_nk_s30 = process_probe("nk_s30", samples_s30, 'nk', deriv_nk_corrs_s30, n)
# deriv_nk_corrs_s30, inv_cov_nk_s30 = probe_nk_s30.deriv_filt, probe_nk_s30.inv_cov_filt

# probe_nk_s90 = process_probe("nk_s90", samples_s90, 'nk', deriv_nk_corrs_s90, n)
# deriv_nk_corrs_s90, inv_cov_nk_s90 = probe_nk_s90.deriv_filt, probe_nk_s90.inv_cov_filt

# # 7b. Process Defect-Peak/Minima (DPC)
# probe_ppk_s15 = process_probe("ppk_s15", samples_s15, 'ppk', deriv_ppk_corrs_s15, n)
# probe_pmn_s15 = process_probe("pmn_s15", samples_s15, 'pmn', deriv_pmn_corrs_s15, n)
# probe_npk_s15 = process_probe("npk_s15", samples_s15, 'npk', deriv_npk_corrs_s15, n)
# probe_nmn_s15 = process_probe("nmn_s15", samples_s15, 'nmn', deriv_nmn_corrs_s15, n)

# probe_ppk_s30 = process_probe("ppk_s30", samples_s30, 'ppk', deriv_ppk_corrs_s30, n)
# probe_pmn_s30 = process_probe("pmn_s30", samples_s30, 'pmn', deriv_pmn_corrs_s30, n)
# probe_npk_s30 = process_probe("npk_s30", samples_s30, 'npk', deriv_npk_corrs_s30, n)
# probe_nmn_s30 = process_probe("nmn_s30", samples_s30, 'nmn', deriv_nmn_corrs_s30, n)

# probe_ppk_s90 = process_probe("ppk_s90", samples_s90, 'ppk', deriv_ppk_corrs_s90, n)
# probe_pmn_s90 = process_probe("pmn_s90", samples_s90, 'pmn', deriv_pmn_corrs_s90, n)
# probe_npk_s90 = process_probe("npk_s90", samples_s90, 'npk', deriv_npk_corrs_s90, n)
# probe_nmn_s90 = process_probe("nmn_s90", samples_s90, 'nmn', deriv_nmn_corrs_s90, n)

# # 7c. Combine DPC (PPK + PMN + NPK + NMN)
# probe_dpc_combined_s15 = FisherProbe.combine("dpc_s15", [probe_ppk_s15, probe_pmn_s15, probe_npk_s15, probe_nmn_s15])
# deriv_dpc_combined_s15, inv_cov_dpc_combined_s15 = probe_dpc_combined_s15.deriv_filt, probe_dpc_combined_s15.inv_cov_filt

# probe_dpc_combined_s30 = FisherProbe.combine("dpc_s30", [probe_ppk_s30, probe_pmn_s30, probe_npk_s30, probe_nmn_s30])
# deriv_dpc_combined_s30, inv_cov_dpc_combined_s30 = probe_dpc_combined_s30.deriv_filt, probe_dpc_combined_s30.inv_cov_filt

# probe_dpc_combined_s90 = FisherProbe.combine("dpc_s90", [probe_ppk_s90, probe_pmn_s90, probe_npk_s90, probe_nmn_s90])
# deriv_dpc_combined_s90, inv_cov_dpc_combined_s90 = probe_dpc_combined_s90.deriv_filt, probe_dpc_combined_s90.inv_cov_filt

# probe_dpc_combined_all_scales = FisherProbe.combine("dpc_all", [probe_dpc_combined_s15, probe_dpc_combined_s30, probe_dpc_combined_s90])
# deriv_dpc_combined_all_scales, inv_cov_dpc_combined_all_scales = probe_dpc_combined_all_scales.deriv_filt, probe_dpc_combined_all_scales.inv_cov_filt

# 8. Combined Correlations (PP + NN + PN)
probe_dcorrs_combined_s15 = FisherProbe.combine("dcorr_s15", [probe_pp_s15, probe_nn_corr_s15, probe_pn_s15])
deriv_dcorrs_combined_s15, inv_cov_dcorrs_combined_s15 = probe_dcorrs_combined_s15.deriv_filt, probe_dcorrs_combined_s15.inv_cov_filt

probe_dcorrs_combined_s30 = FisherProbe.combine("dcorr_s30", [probe_pp_s30, probe_nn_corr_s30, probe_pn_s30])
deriv_dcorrs_combined_s30, inv_cov_dcorrs_combined_s30 = probe_dcorrs_combined_s30.deriv_filt, probe_dcorrs_combined_s30.inv_cov_filt

probe_dcorrs_combined_s90 = FisherProbe.combine("dcorr_s90", [probe_pp_s90, probe_nn_corr_s90, probe_pn_s90])
deriv_dcorrs_combined_s90, inv_cov_dcorrs_combined_s90 = probe_dcorrs_combined_s90.deriv_filt, probe_dcorrs_combined_s90.inv_cov_filt

probe_dcorrs_combined_all_scales = FisherProbe.combine("dcorr_all", [probe_dcorrs_combined_s15, probe_dcorrs_combined_s30, probe_dcorrs_combined_s90])
deriv_dcorrs_combined_all_scales, inv_cov_dcorrs_all_scales = probe_dcorrs_combined_all_scales.deriv_filt, probe_dcorrs_combined_all_scales.inv_cov_filt

# 9. All Scales Blocks

# PDF All
probe_pdfs_all_scales = FisherProbe.combine("pdf_all", [probe_pdfs_s15, probe_pdfs_s30, probe_pdfs_s90])
deriv_pdfs_all_scales, inv_cov_pdf_all_scales = probe_pdfs_all_scales.deriv_filt, probe_pdfs_all_scales.inv_cov_filt

# KK All
probe_kk_all_scales = FisherProbe.combine("kk_all", [probe_kk_s15, probe_kk_s30, probe_kk_s90])
deriv_kk_all_scales, inv_cov_kk_all_scales = probe_kk_all_scales.deriv_filt, probe_kk_all_scales.inv_cov_filt

# NNs Individual (Moved up)
probe_nns_s15 = process_probe("nns_s15", samples_s15, 'nns', deriv_nns_combined_s15, n)
deriv_nns_combined_s15, inv_cov_nn_s15 = probe_nns_s15.deriv_filt, probe_nns_s15.inv_cov_filt

probe_nns_s30 = process_probe("nns_s30", samples_s30, 'nns', deriv_nns_combined_s30, n)
deriv_nns_combined_s30, inv_cov_nn_s30 = probe_nns_s30.deriv_filt, probe_nns_s30.inv_cov_filt

probe_nns_s90 = process_probe("nns_s90", samples_s90, 'nns', deriv_nns_combined_s90, n)
deriv_nns_combined_s90, inv_cov_nn_s90 = probe_nns_s90.deriv_filt, probe_nns_s90.inv_cov_filt

# NNs All
probe_nns_all_scales = FisherProbe.combine("nns_all", [probe_nns_s15, probe_nns_s30, probe_nns_s90])
deriv_nns_combined_all_scales, inv_cov_nn_all_scales = probe_nns_all_scales.deriv_filt, probe_nns_all_scales.inv_cov_filt

# Combined DPDF + NN per scale
probe_dpdf_nn_combined_s15 = FisherProbe.combine("dpdf_nn_s15", [probe_dpdfs_combined_s15, probe_nns_s15])
deriv_dpdf_nn_combined_s15, inv_cov_dpdf_nn_combined_s15 = probe_dpdf_nn_combined_s15.deriv_filt, probe_dpdf_nn_combined_s15.inv_cov_filt

probe_dpdf_nn_combined_s30 = FisherProbe.combine("dpdf_nn_s30", [probe_dpdfs_combined_s30, probe_nns_s30])
deriv_dpdf_nn_combined_s30, inv_cov_dpdf_nn_combined_s30 = probe_dpdf_nn_combined_s30.deriv_filt, probe_dpdf_nn_combined_s30.inv_cov_filt

probe_dpdf_nn_combined_s90 = FisherProbe.combine("dpdf_nn_s90", [probe_dpdfs_combined_s90, probe_nns_s90])
deriv_dpdf_nn_combined_s90, inv_cov_dpdf_nn_combined_s90 = probe_dpdf_nn_combined_s90.deriv_filt, probe_dpdf_nn_combined_s90.inv_cov_filt

# Combined DPDF + NN All
probe_dpdf_nn_combined_all_scales = FisherProbe.combine("dpdf_nn_all", [probe_dpdf_nn_combined_s15, probe_dpdf_nn_combined_s30, probe_dpdf_nn_combined_s90])
deriv_dpdf_nn_combined_all_scales, inv_cov_dpdf_nn_combined_all_scales = probe_dpdf_nn_combined_all_scales.deriv_filt, probe_dpdf_nn_combined_all_scales.inv_cov_filt

# Defects PDF Combined All (DPDF + NN + PDF All)
probe_defects_pdf_combined_all_scales = FisherProbe.combine("defects_pdf_all", [probe_dpdf_nn_combined_all_scales, probe_pdfs_all_scales])
deriv_defects_pdf_combined_all_scales, inv_cov_defects_pdf_combined_all_scales = probe_defects_pdf_combined_all_scales.deriv_filt, probe_defects_pdf_combined_all_scales.inv_cov_filt

# All All (Defects PDF + KK)
probe_all_all_scales = FisherProbe.combine("all_all", [probe_defects_pdf_combined_all_scales, probe_kk_all_scales])
deriv_all_all_scales, inv_cov_all_all_scales = probe_all_all_scales.deriv_filt, probe_all_all_scales.inv_cov_filt


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

fisher_matrix_gq_s15 = np.zeros((n_params, n_params))
fisher_matrix_gq_s30 = np.zeros((n_params, n_params))
fisher_matrix_gq_s90 = np.zeros((n_params, n_params))
fisher_matrix_gq_all_scales = np.zeros((n_params, n_params))

fisher_matrix_pk_s15 = np.zeros((n_params, n_params))
fisher_matrix_pk_s30 = np.zeros((n_params, n_params))
fisher_matrix_pk_s90 = np.zeros((n_params, n_params))

fisher_matrix_nk_s15 = np.zeros((n_params, n_params))
fisher_matrix_nk_s30 = np.zeros((n_params, n_params))
fisher_matrix_nk_s90 = np.zeros((n_params, n_params))

fisher_matrix_dpc_combined_s15 = np.zeros((n_params, n_params))
fisher_matrix_dpc_combined_s30 = np.zeros((n_params, n_params))
fisher_matrix_dpc_combined_s90 = np.zeros((n_params, n_params))
fisher_matrix_dpc_combined_all_scales = np.zeros((n_params, n_params))

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
    deriv_pdf_i_all_scales = deriv_pdfs_all_scales[i]

    deriv_dpdfs_combined_i_s15 = deriv_dpdfs_combined_s15[i]
    deriv_dpdfs_combined_i_s30 = deriv_dpdfs_combined_s30[i]
    deriv_dpdfs_combined_i_s90 = deriv_dpdfs_combined_s90[i]
    deriv_dpdfs_combined_i_all_scales = deriv_dpdfs_combined_all_scales[i]

    deriv_kk_corrs_i_s15 = deriv_kk_corrs_s15[i]
    deriv_kk_corrs_i_s30 = deriv_kk_corrs_s30[i]
    deriv_kk_corrs_i_s90 = deriv_kk_corrs_s90[i]
    deriv_kk_corrs_i_all_scales = deriv_kk_all_scales[i]

    deriv_pp_corrs_i_s15 = deriv_pp_corrs_s15[i]
    deriv_pp_corrs_i_s30 = deriv_pp_corrs_s30[i]
    deriv_pp_corrs_i_s90 = deriv_pp_corrs_s90[i]

    deriv_nn_corrs_i_s15 = deriv_nn_corrs_s15[i]
    deriv_nn_corrs_i_s30 = deriv_nn_corrs_s30[i]
    deriv_nn_corrs_i_s90 = deriv_nn_corrs_s90[i]

    deriv_pn_corrs_i_s15 = deriv_pn_corrs_s15[i]
    deriv_pn_corrs_i_s30 = deriv_pn_corrs_s30[i]
    deriv_pn_corrs_i_s90 = deriv_pn_corrs_s90[i]

    deriv_gq_corrs_i_s15 = deriv_gq_corrs_s15[i]
    deriv_gq_corrs_i_s30 = deriv_gq_corrs_s30[i]
    deriv_gq_corrs_i_s90 = deriv_gq_corrs_s90[i]
    deriv_gq_corrs_i_all_scales = deriv_gq_corrs_all_scales[i]

    deriv_pk_corrs_i_s15 = deriv_pk_corrs_s15[i]
    deriv_pk_corrs_i_s30 = deriv_pk_corrs_s30[i]
    deriv_pk_corrs_i_s90 = deriv_pk_corrs_s90[i]

    deriv_nk_corrs_i_s15 = deriv_nk_corrs_s15[i]
    deriv_nk_corrs_i_s30 = deriv_nk_corrs_s30[i]
    deriv_nk_corrs_i_s90 = deriv_nk_corrs_s90[i]

    deriv_dpc_i_s15 = deriv_dpc_combined_s15[i]
    deriv_dpc_i_s30 = deriv_dpc_combined_s30[i]
    deriv_dpc_i_s90 = deriv_dpc_combined_s90[i]
    deriv_dpc_i_all_scales = deriv_dpc_combined_all_scales[i]

    deriv_dcorrs_i_s15 = deriv_dcorrs_combined_s15[i]
    deriv_dcorrs_i_s30 = deriv_dcorrs_combined_s30[i]
    deriv_dcorrs_i_s90 = deriv_dcorrs_combined_s90[i]
    deriv_dcorrs_i_all_scales = deriv_dcorrs_combined_all_scales[i]

    deriv_nns_combined_i_s15 = deriv_nns_combined_s15[i]
    deriv_nns_combined_i_s30 = deriv_nns_combined_s30[i]
    deriv_nns_combined_i_s90 = deriv_nns_combined_s90[i]
    deriv_nns_combined_i_all_scales = deriv_nns_combined_all_scales[i]

    deriv_dpdf_nn_combined_i_s15 = deriv_dpdf_nn_combined_s15[i]
    deriv_dpdf_nn_combined_i_s30 = deriv_dpdf_nn_combined_s30[i]
    deriv_dpdf_nn_combined_i_s90 = deriv_dpdf_nn_combined_s90[i]
    deriv_dpdf_nn_combined_i_all_scales = deriv_dpdf_nn_combined_all_scales[i]

    deriv_defects_pdf_combined_i_all_scales = deriv_defects_pdf_combined_all_scales[i]
    deriv_all_i_all_scales = deriv_all_all_scales[i]
    for j in range(n_params):
        deriv_pdf_j_s15 = deriv_pdfs_s15[j]
        deriv_pdf_j_s30 = deriv_pdfs_s30[j]
        deriv_pdf_j_s90 = deriv_pdfs_s90[j]
        deriv_pdf_j_all_scales = deriv_pdfs_all_scales[j]

        deriv_dpdfs_combined_j_s15 = deriv_dpdfs_combined_s15[j]
        deriv_dpdfs_combined_j_s30 = deriv_dpdfs_combined_s30[j]
        deriv_dpdfs_combined_j_s90 = deriv_dpdfs_combined_s90[j]
        deriv_dpdfs_combined_j_all_scales = deriv_dpdfs_combined_all_scales[j]

        # deriv_dk_corr_pos_j = deriv_dk_corrs_pos[j]
        # deriv_dk_corr_neg_j = deriv_dk_corrs_neg[j]
        # deriv_dk_corrs_combined_j = deriv_dk_corrs_combined[j]
        # deriv_defects_combined_j = deriv_defects_combined[j]

        deriv_kk_corrs_j_s15 = deriv_kk_corrs_s15[j]
        deriv_kk_corrs_j_s30 = deriv_kk_corrs_s30[j]
        deriv_kk_corrs_j_s90 = deriv_kk_corrs_s90[j]
        deriv_kk_corrs_j_all_scales = deriv_kk_all_scales[j]

        deriv_pp_corrs_j_s15 = deriv_pp_corrs_s15[j]
        deriv_pp_corrs_j_s30 = deriv_pp_corrs_s30[j]
        deriv_pp_corrs_j_s90 = deriv_pp_corrs_s90[j]

        deriv_nn_corrs_j_s15 = deriv_nn_corrs_s15[j]
        deriv_nn_corrs_j_s30 = deriv_nn_corrs_s30[j]
        deriv_nn_corrs_j_s90 = deriv_nn_corrs_s90[j]

        deriv_pn_corrs_j_s15 = deriv_pn_corrs_s15[j]
        deriv_pn_corrs_j_s30 = deriv_pn_corrs_s30[j]
        deriv_pn_corrs_j_s90 = deriv_pn_corrs_s90[j]

        deriv_gq_corrs_j_s15 = deriv_gq_corrs_s15[j]
        deriv_gq_corrs_j_s30 = deriv_gq_corrs_s30[j]
        deriv_gq_corrs_j_s90 = deriv_gq_corrs_s90[j]
        deriv_gq_corrs_j_all_scales = deriv_gq_corrs_all_scales[j]

        deriv_pk_corrs_j_s15 = deriv_pk_corrs_s15[j]
        deriv_pk_corrs_j_s30 = deriv_pk_corrs_s30[j]
        deriv_pk_corrs_j_s90 = deriv_pk_corrs_s90[j]

        deriv_nk_corrs_j_s15 = deriv_nk_corrs_s15[j]
        deriv_nk_corrs_j_s30 = deriv_nk_corrs_s30[j]
        deriv_nk_corrs_j_s90 = deriv_nk_corrs_s90[j]

        deriv_dpc_j_s15 = deriv_dpc_combined_s15[j]
        deriv_dpc_j_s30 = deriv_dpc_combined_s30[j]
        deriv_dpc_j_s90 = deriv_dpc_combined_s90[j]
        deriv_dpc_j_all_scales = deriv_dpc_combined_all_scales[j]

        deriv_dcorrs_j_s15 = deriv_dcorrs_combined_s15[j]
        deriv_dcorrs_j_s30 = deriv_dcorrs_combined_s30[j]
        deriv_dcorrs_j_s90 = deriv_dcorrs_combined_s90[j]
        deriv_dcorrs_j_all_scales = deriv_dcorrs_combined_all_scales[j]

        deriv_nns_combined_j_s15 = deriv_nns_combined_s15[j]
        deriv_nns_combined_j_s30 = deriv_nns_combined_s30[j]
        deriv_nns_combined_j_s90 = deriv_nns_combined_s90[j]
        deriv_nns_combined_j_all_scales = deriv_nns_combined_all_scales[j]

        deriv_dpdf_nn_combined_j_s15 = deriv_dpdf_nn_combined_s15[j]
        deriv_dpdf_nn_combined_j_s30 = deriv_dpdf_nn_combined_s30[j]
        deriv_dpdf_nn_combined_j_s90 = deriv_dpdf_nn_combined_s90[j]
        deriv_dpdf_nn_combined_j_all_scales = deriv_dpdf_nn_combined_all_scales[j]

        deriv_defects_pdf_combined_j_all_scales = deriv_defects_pdf_combined_all_scales[j]
        deriv_all_j_all_scales = deriv_all_all_scales[j]

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

        fisher_matrix_gq_s15[i, j] = deriv_gq_corrs_i_s15 @ inv_cov_gq_s15 @ deriv_gq_corrs_j_s15
        fisher_matrix_gq_s30[i, j] = deriv_gq_corrs_i_s30 @ inv_cov_gq_s30 @ deriv_gq_corrs_j_s30
        fisher_matrix_gq_s90[i, j] = deriv_gq_corrs_i_s90 @ inv_cov_gq_s90 @ deriv_gq_corrs_j_s90
        fisher_matrix_gq_all_scales[i, j] = deriv_gq_corrs_i_all_scales @ inv_cov_gq_all_scales @ deriv_gq_corrs_j_all_scales

        # fisher_matrix_pk_s15[i, j] = deriv_pk_corrs_i_s15 @ inv_cov_pk_s15 @ deriv_pk_corrs_j_s15
        # fisher_matrix_pk_s30[i, j] = deriv_pk_corrs_i_s30 @ inv_cov_pk_s30 @ deriv_pk_corrs_j_s30
        # fisher_matrix_pk_s90[i, j] = deriv_pk_corrs_i_s90 @ inv_cov_pk_s90 @ deriv_pk_corrs_j_s90

        # fisher_matrix_nk_s15[i, j] = deriv_nk_corrs_i_s15 @ inv_cov_nk_s15 @ deriv_nk_corrs_j_s15
        # fisher_matrix_nk_s30[i, j] = deriv_nk_corrs_i_s30 @ inv_cov_nk_s30 @ deriv_nk_corrs_j_s30
        # fisher_matrix_nk_s90[i, j] = deriv_nk_corrs_i_s90 @ inv_cov_nk_s90 @ deriv_nk_corrs_j_s90

        # fisher_matrix_dpc_combined_s15[i, j] = deriv_dpc_i_s15 @ inv_cov_dpc_combined_s15 @ deriv_dpc_j_s15
        # fisher_matrix_dpc_combined_s30[i, j] = deriv_dpc_i_s30 @ inv_cov_dpc_combined_s30 @ deriv_dpc_j_s30
        # fisher_matrix_dpc_combined_s90[i, j] = deriv_dpc_i_s90 @ inv_cov_dpc_combined_s90 @ deriv_dpc_j_s90
        # fisher_matrix_dpc_combined_all_scales[i, j] = deriv_dpc_i_all_scales @ inv_cov_dpc_combined_all_scales @ deriv_dpc_j_all_scales

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
print("PDF S90 FISHER MATRIX:", fisher_matrix_pdfs_s90)
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

Fisher_gq_s15 = fisher_matrix_gq_s15
Fisher_gq_s30 = fisher_matrix_gq_s30
Fisher_gq_s90 = fisher_matrix_gq_s90
Fisher_gq_all_scales = fisher_matrix_gq_all_scales

# Fisher_pk_s15 = fisher_matrix_pk_s15
# Fisher_pk_s30 = fisher_matrix_pk_s30
# Fisher_pk_s90 = fisher_matrix_pk_s90

# Fisher_nk_s15 = fisher_matrix_nk_s15
# Fisher_nk_s30 = fisher_matrix_nk_s30
# Fisher_nk_s90 = fisher_matrix_nk_s90

# Fisher_dpc_combined_s15 = fisher_matrix_dpc_combined_s15
# Fisher_dpc_combined_s30 = fisher_matrix_dpc_combined_s30
# Fisher_dpc_combined_s90 = fisher_matrix_dpc_combined_s90
# Fisher_dpc_combined_all_scales = fisher_matrix_dpc_combined_all_scales

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
labels = ['\\Omega_{\\rm m}', '\\sigma_{\\rm 8}']

covariance_dpdf_combined_s15 = np.linalg.inv(Fisher_dpdf_combined_s15)
covariance_dpdf_combined_s30 = np.linalg.inv(Fisher_dpdf_combined_s30)
covariance_dpdf_combined_s90 = np.linalg.inv(Fisher_dpdf_combined_s90)
print("DPDF COMBINED S90 COVARIANCE:", covariance_dpdf_combined_s90)
covariance_dpdf_combined_all_scales = np.linalg.inv(Fisher_dpdf_combined_all_scales)

# 修复：增加空格，保持数学格式一致
dpdf_combined_s15_samples = GaussianND(mean, covariance_dpdf_combined_s15, names=names, labels=labels, label=r"$C^{p+n}(\kappa), \ \theta_G=15'$")
dpdf_combined_s30_samples = GaussianND(mean, covariance_dpdf_combined_s30, names=names, labels=labels, label=r"$C^{p+n}(\kappa), \ \theta_G=30'$")
dpdf_combined_s90_samples = GaussianND(mean, covariance_dpdf_combined_s90, names=names, labels=labels, label=r"$C^{p+n}(\kappa), \ \theta_G=90'$")
dpdf_combined_all_scales_samples = GaussianND(mean, covariance_dpdf_combined_all_scales, names=names, labels=labels, label=r'$C^{p+n}(\kappa)$, all scales')

covariance_pdf_s15 = np.linalg.inv(Fisher_pdf_s15)
covariance_pdf_s30 = np.linalg.inv(Fisher_pdf_s30)
covariance_pdf_s90 = np.linalg.inv(Fisher_pdf_s90)
print("PDF S90 COVARIANCE:", covariance_pdf_s90)
covariance_pdf_all_scales = np.linalg.inv(Fisher_pdf_all_scales)

# 修复：统一 PDF 为文本格式，theta_G 为数学格式
pdf_s15_samples = GaussianND(mean, covariance_pdf_s15, names=names, labels=labels, label=r"PDF, $\theta_G=15'$")
pdf_s30_samples = GaussianND(mean, covariance_pdf_s30, names=names, labels=labels, label=r"PDF, $\theta_G=30'$")
pdf_s90_samples = GaussianND(mean, covariance_pdf_s90, names=names, labels=labels, label=r"PDF, $\theta_G=90'$")
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

pp_s15_samples = GaussianND(mean, covariance_pp_s15, names=names, labels=labels, label=r"$\xi^{++}(\theta), \ \theta_G=15'$")
pp_s30_samples = GaussianND(mean, covariance_pp_s30, names=names, labels=labels, label=r"$\xi^{++}(\theta), \ \theta_G=30'$")
pp_s90_samples = GaussianND(mean, covariance_pp_s90, names=names, labels=labels, label=r"$\xi^{++}(\theta), \ \theta_G=90'$")

nn_s15_samples = GaussianND(mean, covariance_nn_s15, names=names, labels=labels, label=r"$\xi^{--}(\theta), \ \theta_G=15'$")
nn_s30_samples = GaussianND(mean, covariance_nn_s30, names=names, labels=labels, label=r"$\xi^{--}(\theta), \ \theta_G=30'$")
nn_s90_samples = GaussianND(mean, covariance_nn_s90, names=names, labels=labels, label=r"$\xi^{--}(\theta), \ \theta_G=90'$")    

pn_s15_samples = GaussianND(mean, covariance_pn_s15, names=names, labels=labels, label=r"$\xi^{+-}(\theta), \ \theta_G=15'$")
pn_s30_samples = GaussianND(mean, covariance_pn_s30, names=names, labels=labels, label=r"$\xi^{+-}(\theta), \ \theta_G=30'$")
pn_s90_samples = GaussianND(mean, covariance_pn_s90, names=names, labels=labels, label=r"$\xi^{+-}(\theta), \ \theta_G=90'$")

covariance_gq_s15 = np.linalg.inv(Fisher_gq_s15)
covariance_gq_s30 = np.linalg.inv(Fisher_gq_s30)
covariance_gq_s90 = np.linalg.inv(Fisher_gq_s90)
covariance_gq_all_scales = np.linalg.inv(Fisher_gq_all_scales)

gq_s15_samples = GaussianND(mean, covariance_gq_s15, names=names, labels=labels, label=r"$g_Q(\theta), \ \theta_G=15'$")
gq_s30_samples = GaussianND(mean, covariance_gq_s30, names=names, labels=labels, label=r"$g_Q(\theta), \ \theta_G=30'$")
gq_s90_samples = GaussianND(mean, covariance_gq_s90, names=names, labels=labels, label=r"$g_Q(\theta), \ \theta_G=90'$")
gq_all_scales_samples = GaussianND(mean, covariance_gq_all_scales, names=names, labels=labels, label=r"$g_Q(\theta)$, all scales")

dcorr_combined_s15_samples = GaussianND(mean, np.linalg.inv(Fisher_dcorrs_combined_s15), names=names, labels=labels, label=r"D-2PCF, $\theta_G=15'$")
dcorr_combined_s30_samples = GaussianND(mean, np.linalg.inv(Fisher_dcorrs_combined_s30), names=names, labels=labels, label=r"D-2PCF, $\theta_G=30'$")
dcorr_combined_s90_samples = GaussianND(mean, np.linalg.inv(Fisher_dcorrs_combined_s90), names=names, labels=labels, label=r"D-2PCF, $\theta_G=90'$")
dcorr_combined_all_scales_samples = GaussianND(mean, np.linalg.inv(Fisher_dcorrs_combined_all_scales), names=names, labels=labels, label=r'D-2PCF, all scales')

# DPC (Defect-Peak/Minima) Samples
# covariance_dpc_combined_s15 = np.linalg.inv(Fisher_dpc_combined_s15)
# covariance_dpc_combined_s30 = np.linalg.inv(Fisher_dpc_combined_s30)
# covariance_dpc_combined_s90 = np.linalg.inv(Fisher_dpc_combined_s90)
# covariance_dpc_combined_all_scales = np.linalg.inv(Fisher_dpc_combined_all_scales)

# dpc_combined_s15_samples = GaussianND(mean, covariance_dpc_combined_s15, names=names, labels=labels, label=r"D-Peak/Min, $\theta_G=15'$")
# dpc_combined_s30_samples = GaussianND(mean, covariance_dpc_combined_s30, names=names, labels=labels, label=r"D-Peak/Min, $\theta_G=30'$")
# dpc_combined_s90_samples = GaussianND(mean, covariance_dpc_combined_s90, names=names, labels=labels, label=r"D-Peak/Min, $\theta_G=90'$")
# dpc_combined_all_scales_samples = GaussianND(mean, covariance_dpc_combined_all_scales, names=names, labels=labels, label=r'D-Peak/Min, all scales')

covariance_kk_s15 = np.linalg.inv(Fisher_kk_s15)
covariance_kk_s30 = np.linalg.inv(Fisher_kk_s30)
covariance_kk_s90 = np.linalg.inv(Fisher_kk_s90)
covariance_kk_all_scales = np.linalg.inv(Fisher_kk_all_scales)

# 修复：补全缺失的 $ 符号，修复 all scales 的 $ 位置
kk_s15_samples = GaussianND(mean, covariance_kk_s15, names=names, labels=labels, label=r"$\kappa\text{-2PCF}, \ \theta_G=15'$")
kk_s30_samples = GaussianND(mean, covariance_kk_s30, names=names, labels=labels, label=r"$\kappa\text{-2PCF}, \ \theta_G=30'$")
kk_s90_samples = GaussianND(mean, covariance_kk_s90, names=names, labels=labels, label=r"$\kappa\text{-2PCF}, \ \theta_G=90'$")
kk_all_scales_samples = GaussianND(mean, covariance_kk_all_scales, names=names, labels=labels, label=r'$\kappa\text{-2PCF}$, all scales')

covariance_nns_s15 = np.linalg.inv(Fisher_nns_s15)
covariance_nns_s30 = np.linalg.inv(Fisher_nns_s30)
covariance_nns_s90 = np.linalg.inv(Fisher_nns_s90)
print("NNS S90 COVARIANCE:", covariance_nns_s90)
covariance_nns_all_scales = np.linalg.inv(Fisher_nns_all_scales)


nns_s15_samples = GaussianND(mean, covariance_nns_s15, names=names, labels=labels, label=r"$C^{\mathrm{all}}_{1,2,4,8}(\theta), \ \theta_G=15'$")
nns_s30_samples = GaussianND(mean, covariance_nns_s30, names=names, labels=labels, label=r"$C^{\mathrm{all}}_{1,2,4,8}(\theta), \ \theta_G=30'$")
nns_s90_samples = GaussianND(mean, covariance_nns_s90, names=names, labels=labels, label=r"$C^{\mathrm{all}}_{1,2,4,8}(\theta), \ \theta_G=90'$")
nns_all_scales_samples = GaussianND(mean, covariance_nns_all_scales, names=names, labels=labels, label=r"$C^{\mathrm{all}}_{1,2,4,8}(\theta)$, all scales")

covariance_dpdf_nn_combined_s15 = np.linalg.inv(Fisher_dpdf_nn_combined_s15)
covariance_dpdf_nn_combined_s30 = np.linalg.inv(Fisher_dpdf_nn_combined_s30)
covariance_dpdf_nn_combined_s90 = np.linalg.inv(Fisher_dpdf_nn_combined_s90)
covariance_dpdf_nn_combined_all_scales = np.linalg.inv(Fisher_dpdf_nn_combined_all_scales)

dpdf_nn_combined_s15_samples = GaussianND(mean, covariance_dpdf_nn_combined_s15, names=names, labels=labels, label=r"DCDF + kNN, $\theta_G=15'$")
dpdf_nn_combined_s30_samples = GaussianND(mean, covariance_dpdf_nn_combined_s30, names=names, labels=labels, label=r"DCDF + kNN, $\theta_G=30'$")
dpdf_nn_combined_s90_samples = GaussianND(mean, covariance_dpdf_nn_combined_s90, names=names, labels=labels, label=r"DCDF + kNN, $\theta_G=90'$")
dpdf_nn_combined_all_scales_samples = GaussianND(mean, covariance_dpdf_nn_combined_all_scales, names=names, labels=labels, label=r"DCDF + kNN, all scales")

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
g.add_legend([r"$\kappa$-PDF, $\theta_G=90'$", r"$\kappa$-PDF, $\theta_G=30'$", r"$\kappa$-PDF, $\theta_G=15'$", r"$\kappa$-PDF, all scales", r'$\kappa\text{-2PCF}$, all scales'], legend_loc='upper right', fontsize=7)
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
g.add_legend([r"$\text{D-CDF}, \ \theta_G=90'$", r"$\text{D-CDF}, \ \theta_G=30'$", r"$\text{D-CDF}, \ \theta_G=15'$", r"D-CDF, all scales", r'$\kappa\text{-2PCF}$, all scales'], legend_loc='upper right', fontsize=7)
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
g.add_legend([r"$\text{kNN}, \ \theta_G=90'$", r"$\text{kNN}, \ \theta_G=30'$", r"$\text{kNN}, \ \theta_G=15'$", r"$\text{kNN}$, all scales", r'$\kappa\text{-2PCF}$, all scales'], legend_loc='upper right', fontsize=7)
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
g.add_legend([r"$\text{Defects}, \ \theta_G=90'$", r"$\text{Defects}, \ \theta_G=30'$", r"$\text{Defects}, \ \theta_G=15'$", r"$\text{Defects}$, all scales", r'$\kappa\text{-2PCF}$, all scales'], legend_loc='upper right', fontsize=7)
g.export("outfiles/dcdf_nn_combined.png", dpi=300)

g = plots.get_subplot_plotter(width_inch=4)
g.settings.num_plot_contours = 1
g.plot_2d([gq_s90_samples, gq_s30_samples, gq_s15_samples, gq_all_scales_samples, kk_all_scales_samples],
                param1="Om", param2="sigma8",
                filled=[False, False, False, False, False],
                lims = [0.2, 0.4, 0.7, 0.9],
                colors=[color_map(i) for i in range(3)] + ['red', 'black'],
                lws=2,
                )
g.add_legend([r"$g_Q(\theta), \ \theta_G=90'$", r"$g_Q(\theta), \ \theta_G=30'$", r"$g_Q(\theta), \ \theta_G=15'$", r"$g_Q(\theta)$, all scales", r'$\kappa\text{-2PCF}$, all scales'], legend_loc='upper right', fontsize=7)
g.export("outfiles/gq.png", dpi=300)

g = plots.get_subplot_plotter(width_inch=4)
g.settings.num_plot_contours = 1
g.plot_2d([kk_all_scales_samples, pdf_all_scales_samples, dpdf_nn_combined_all_scales_samples, defects_pdf_combined_all_scales_samples, all_all_scales_samples],
                param1="Om", param2="sigma8",
                filled=[False, False, False, False, False],
                lims = [0.28, 0.32, 0.77, 0.83],
                colors=['black', 'red'] + [color_map(i) for i in range(3)],
                lws=2,
                # alphas=[0.7, 0.7, 1.0, 1.0],
                ls=['solid', 'solid', 'solid', 'solid', 'dashed'])    
g.add_legend([r'$\kappa\text{-2PCF}$, all scales', r'$\kappa\text{-PDF}$, all scales', r'Defects, all scales', r'Defects + $\kappa\text{-PDF}$, all scales', r'Defects + $\kappa\text{-PDF}$ + $\kappa\text{-2PCF}$, all scales'], legend_loc='upper right', fontsize=7)
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
g.add_legend([r"D-2PCF, $\theta_G=90'$", r"D-2PCF, $\theta_G=30'$", r"D-2PCF, $\theta_G=15'$", r"D-2PCF, all scales", r'$\kappa\text{-2PCF}$, all scales'], legend_loc='upper right', fontsize=7)
g.export("outfiles/dcorrs.png", dpi=300)

# g = plots.get_subplot_plotter(width_inch=4)
# g.settings.num_plot_contours = 1
# g.plot_2d([dpc_combined_s90_samples, dpc_combined_s30_samples, dpc_combined_s15_samples, dpc_combined_all_scales_samples, kk_all_scales_samples],
#                 param1="Om", param2="sigma8",
#                 filled=[False, False, False, False, False],
#                 lims = [0.2, 0.4, 0.7, 0.9],
#                 colors=[color_map(i) for i in range(3)] + ['red', 'black'],
#                 lws=2,
#                 )
# g.add_legend([r"D-Peak/Min, $\theta_G=90'$", r"D-Peak/Min, $\theta_G=30'$", r"D-Peak/Min, $\theta_G=15'$", r"D-Peak/Min, all scales", r'$\kappa\text{-2PCF}$, all scales'], legend_loc='upper right', fontsize=7)
# g.export("outfiles/dpc.png", dpi=300)