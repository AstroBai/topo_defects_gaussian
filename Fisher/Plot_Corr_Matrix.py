import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from tqdm import tqdm
from mpl_toolkits.axes_grid1 import make_axes_locatable

# Configuration
RAW_DATA_DIR = '/cephfs/jbai/DefectMockGaussian'
OUT_DIR = "outfiles/CorrMatrices"
os.makedirs(OUT_DIR, exist_ok=True)

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
            
            d = np.load(f"{RAW_DATA_DIR}/PKs/realisation_{i}_s{s}_pos_kappa_corr.npz")
            data['pk'].append(d['xi_pk'])
            
            d = np.load(f"{RAW_DATA_DIR}/NKs/realisation_{i}_s{s}_neg_kappa_corr.npz")
            data['nk'].append(d['xi_nk'])

            # Load Defect-Peak/Fit 
            d = np.load(f"{RAW_DATA_DIR}/PosPeak/realisation_{i}_s{s}_pos_peak_corr.npz")
            data['ppk'].append(d['xi'])
            d = np.load(f"{RAW_DATA_DIR}/PosMin/realisation_{i}_s{s}_pos_min_corr.npz")
            data['pmn'].append(d['xi'])
            d = np.load(f"{RAW_DATA_DIR}/NegPeak/realisation_{i}_s{s}_neg_peak_corr.npz")
            data['npk'].append(d['xi'])
            d = np.load(f"{RAW_DATA_DIR}/NegMin/realisation_{i}_s{s}_neg_min_corr.npz")
            data['nmn'].append(d['xi'])

            # Load NNs Combined
            d = np.load(f"{RAW_DATA_DIR}/NNs/realisation_{i}_s{s}_NNs.npz")
            nn_comb = []
            for scale_idx in [1, 2, 4, 8]:
                nn_comb.extend(d[f'nn_pos_same{scale_idx}_cum'])
                nn_comb.extend(d[f'nn_neg_same{scale_idx}_cum'])
                nn_comb.extend(d[f'nn_pos_opp{scale_idx}_cum'])
                nn_comb.extend(d[f'nn_neg_opp{scale_idx}_cum'])
            data['nns'].append(np.array(nn_comb))
            
        except FileNotFoundError:
            pass
        except Exception as e:
            # print(f"Error reading {i}: {e}")
            pass

    # Convert to arrays
    final_data = {}
    for k, v in data.items():
        if len(v) > 0:
            final_data[k] = np.array(v)
            
    return final_data

def get_cov_mean(samples):
    """Compute covariance and mean from samples (N_realizations, N_bins)."""
    mean = np.mean(samples, axis=0)
    cov = np.cov(samples, rowvar=False)
    return cov, mean

def get_kept_mask(mean_vec, cov_mat, sigma_thresh=2.0): # Defaulting to 2.0 per user preference "previous logic"
    sigma = np.sqrt(np.diag(cov_mat))
    # Avoid div by zero
    sigma[sigma == 0] = 1.0
    
    # 1. S/N Filter
    mask = np.abs(mean_vec) > sigma_thresh * sigma
    
    # 2. Correlation Filter (remove highly correlated bins)
    # Re-compute correlation on the kept bins
    if mask.sum() > 0:
        cov_trim = cov_mat[np.ix_(mask, mask)]
        s_trim = np.sqrt(np.diag(cov_trim))
        corr_trim = cov_trim / np.outer(s_trim, s_trim)
        
        # We need to map trimmed indices back to original indices
        kept_indices = np.where(mask)[0]
        
        # Iterate to remove correlated bins
        remove_mask = np.zeros(len(kept_indices), dtype=bool)
        
        for i in range(len(kept_indices)):
            if remove_mask[i]: continue
            for j in range(i + 1, len(kept_indices)):
                if remove_mask[j]: continue
                
                if np.abs(corr_trim[i, j]) >= 0.9999: # Threshold from FisherForecast.py
                    # Remove lower S/N
                    idx_i = kept_indices[i]
                    idx_j = kept_indices[j]
                    
                    sn_i = np.abs(mean_vec[idx_i]) / sigma[idx_i]
                    sn_j = np.abs(mean_vec[idx_j]) / sigma[idx_j]
                    
                    if sn_i <= sn_j:
                        remove_mask[i] = True
                    else:
                        remove_mask[j] = True
        
        # Apply removal
        final_indices = kept_indices[~remove_mask]
        final_mask = np.zeros_like(mask)
        final_mask[final_indices] = True
        return final_mask
    else:
        return mask

def plot_corr_matrix(name, cov, mask, filename, component_lengths):
    # Plot FULL matrix
    sigma = np.sqrt(np.diag(cov))
    sigma[sigma == 0] = 1.0
    corr = cov / np.outer(sigma, sigma)
    
    plt.figure(figsize=(10, 10))
    ax = plt.gca()
    
    # Plot using Seaborn Heatmap with linear scale
    sns.heatmap(corr, ax=ax, cmap='RdBu_r', vmin=-1, vmax=1, center=0,
                square=True, cbar=False, xticklabels=False, yticklabels=False)

    # Add black border to the heatmap
    for _, spine in ax.spines.items():
        spine.set_visible(True)
        spine.set_linewidth(1.5)
        spine.set_color('black')
    
    # Create matched colorbar (thinner)
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="3%", pad=0.1)
    
    # Add colorbar
    norm = plt.Normalize(vmin=-1, vmax=1)
    sm = plt.cm.ScalarMappable(cmap='RdBu_r', norm=norm)
    sm.set_array([])
    plt.colorbar(sm, cax=cax, label='Correlation Coefficient')
    
    # Ensure limits are correct after axes modifications
    ax.set_xlim(0, corr.shape[1])
    ax.set_ylim(corr.shape[0], 0)
    
    # Draw Hierarchical Structure
    # component_lengths is list of (label, length) in order
    current_idx = 0
    boundaries = []
    labels = []
    label_positions = []
    
    print(f"\n--- Data Component Proportions for {name} ---")
    
    for label, length in component_lengths:
        start = current_idx
        end = current_idx + length
        
        boundaries.append(end)
        label_pos = start + length / 2.0
        labels.append(label)
        label_positions.append(label_pos)
        
        # Calculate ratio
        ratio_kept = np.sum(mask[start:end]) / length * 100
        print(f"{label}: {length} total bins, {np.sum(mask[start:end])} kept ({ratio_kept:.2f}%)")
        
        current_idx = end
        
    # Draw Black Dashed Lines (Separators)
    if len(boundaries) > 0:
        for b in boundaries[:-1]:
            pos = b - 0.5
            ax.axvline(x=pos, color='black', linestyle='--', linewidth=1.0, alpha=0.7)
            ax.axhline(y=pos, color='black', linestyle='--', linewidth=1.0, alpha=0.7)
            
    # Highlight Kept Bins with Red Boxes
    kept_indices = np.where(mask)[0]
    if len(kept_indices) > 0:
        segments = []
        if len(kept_indices) > 0:
            start_seg = kept_indices[0]
            curr_seg = kept_indices[0]
            for idx in kept_indices[1:]:
                if idx == curr_seg + 1:
                    curr_seg = idx
                else:
                    segments.append((start_seg, curr_seg))
                    start_seg = idx
                    curr_seg = idx
            segments.append((start_seg, curr_seg))
        
        for start, end in segments:
            width = end - start + 1
            xy = (start - 0.5, start - 0.5)
            rect = plt.Rectangle(xy, width, width, 
                                 fill=False, edgecolor='red', lw=2.0, linestyle='-', zorder=10)
            ax.add_patch(rect)
    
    # Add custom labels
    ax.set_xticks(label_positions)
    ax.set_xticklabels(labels, rotation=90, ha='center')
    ax.xaxis.tick_top()
    
    ax.set_yticks(label_positions)
    ax.set_yticklabels(labels, va='center')
    
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved {filename}")

if __name__ == "__main__":
    # Define Scales
    scales = [30] #, 15, 30] 
    
    for s in scales:
        data = load_mock_samples(s)
        
        # 1. kNN + D-CDF
        # Structure: D-CDF (pos, neg), kNN (nns)
        # However, dpdf_pos and dpdf_neg should be combined into D-CDF
        
        # Build vector
        # Order: D-CDF (dpdf_pos, dpdf_neg), kNN (nns)
        
        # Check shapes
        # nns is already combined 1,2,4,8
        
        samples_list = []
        lengths = []
        
        # D-CDF
        if 'dpdf_pos' in data and 'dpdf_neg' in data:
            dcdf = np.hstack([data['dpdf_pos'], data['dpdf_neg']])
            samples_list.append(dcdf)
            lengths.append(('D-CDF', dcdf.shape[1]))
            
        # kNN
        if 'nns' in data:
            knn = data['nns']
            samples_list.append(knn)
            lengths.append(('kNN', knn.shape[1]))
            
        if samples_list:
            full_samples = np.hstack(samples_list)
            cov, mean = get_cov_mean(full_samples)
            mask = get_kept_mask(mean, cov)
            
            plot_corr_matrix(f"dcdf_knn_s{s}", cov, mask, f"{OUT_DIR}/corr_dcdf_knn_s{s}.png", lengths)
            
        # 2. All HOWLS + kk
        # Logic from FisherForecast.py: 
        # dpdf_nn_combined (DCDF + kNN) + kk + pdf + ?
        # The user requested "D-CDF, kNN, PDF, kappa-2PCF" in the other script.
        # Let's do that.
        
        samples_list = []
        lengths = []
        
        # D-CDF
        if 'dpdf_pos' in data and 'dpdf_neg' in data:
            dcdf = np.hstack([data['dpdf_pos'], data['dpdf_neg']])
            samples_list.append(dcdf)
            lengths.append(('D-CDF', dcdf.shape[1]))
            
        # kNN
        if 'nns' in data:
            knn = data['nns']
            samples_list.append(knn)
            lengths.append(('kNN', knn.shape[1]))
        
        # PDF
        if 'pdf' in data:
            pdf = data['pdf']
            samples_list.append(pdf)
            lengths.append(('PDF', pdf.shape[1]))
            
        # kappa-2PCF
        if 'kk' in data:
            kk = data['kk']
            samples_list.append(kk)
            lengths.append((r'$\kappa$-2PCF', kk.shape[1]))

        if samples_list:
            full_samples = np.hstack(samples_list)
            cov, mean = get_cov_mean(full_samples)
            mask = get_kept_mask(mean, cov)
            
            plot_corr_matrix(f"all_howls_kk_s{s}", cov, mask, f"{OUT_DIR}/corr_all_howls_kk_s{s}.png", lengths)
