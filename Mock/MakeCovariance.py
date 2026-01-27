# make covariance matrices for mock data
import numpy as np  
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde
import os

def plot_ridgeline(data_matrix, title, filename_suffix, bin_indices=None):
    """
    Plots a vertical ridgeline plot (joyplot) for the given data matrix.
    data_matrix: shape (n_samples, n_bins)
    """
    n_samples, n_bins = data_matrix.shape
    
    # Select bins (max 20 for visibility)
    if bin_indices is None:
        if n_bins > 20:
             bin_indices = np.linspace(0, n_bins-1, 20, dtype=int)
        else:
            bin_indices = np.arange(n_bins)
    
    selected_data = data_matrix[:, bin_indices]
    n_plot_bins = len(bin_indices)
    
    # Swap figsize dimensions for vertical orientation (bins on x-axis)
    # Width grows with number of bins
    fig, ax = plt.subplots(figsize=(max(6, n_plot_bins * 0.5), 6))
    
    overlap = 0.5
    colors = plt.cm.viridis(np.linspace(0, 1, n_plot_bins))

    for i, bin_idx in enumerate(bin_indices):
        x_data = selected_data[:, i]
        # Handle constant data (zero variance)
        if np.var(x_data) == 0:
            continue
            
        kde = gaussian_kde(x_data)
        # Grid along the value axis (now Y-axis)
        y_grid = np.linspace(x_data.min(), x_data.max(), 100)
        # Density becomes width along X-axis
        x_density = kde(y_grid)
        
        if x_density.max() > 0:
            x_density = x_density / x_density.max() * (1 + overlap)
        
        baseline = i
        # Fill horizontally between x=baseline and x=baseline+density
        ax.fill_betweenx(y_grid, baseline, baseline + x_density, color=colors[i], alpha=0.7, zorder=n_plot_bins-i)
        ax.plot(baseline + x_density, y_grid, color='white', lw=1, zorder=n_plot_bins-i)

    # Set labels
    ax.set_xticks(np.arange(len(bin_indices)))
    ax.set_xticklabels([f"Bin {b}" for b in bin_indices], rotation=90)
    ax.set_title(title)
    ax.set_xlabel("Bin Index")
    ax.set_ylabel("Value")
    
    # Adjust spines
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    # Keep left/bottom spines for value reading
    ax.spines['left'].set_visible(True)
    ax.spines['bottom'].set_visible(True)
    
    plt.tight_layout()
    os.makedirs("outfiles/Ridgelines", exist_ok=True)
    plt.savefig(f"outfiles/Ridgelines/{filename_suffix}.png", dpi=300)
    plt.close()
n_realizations = 1000
DATA_DIR = '/cephfs/jbai/DefectMockGaussian'
# def random_noise(data_vector, noise_level=0.001):
#     abs_values = np.abs(data_vector)
#     positive_values = abs_values[abs_values > 0]
#     if len(positive_values) > 0:
#         noise_sigma = noise_level * np.min(positive_values)
#     else:
#         noise_sigma = noise_level
#     noise = np.random.normal(0, noise_sigma, size=data_vector.shape)
#     return data_vector + noise

sigmas = [np.radians(15/60), np.radians(30/60), np.radians(90/60)]
pdf_all_scales = []
dpdf_all_scales = []
nn_all_scales = []
kk_all_scales = []
defects_all_scales = []
dcorr_all_scales = []
for sigma in sigmas:
    s = int(np.degrees(sigma)*60)

    # full pdf
    pdfs = []
    for i in range(n_realizations):
        data = np.load(f"{DATA_DIR}/PDFs/realisation_{i}_s{s}_kappa_pdf.npz")
        hist = data['hist']
        # hist = random_noise(hist)
        bin_centers = 0.5 * (data['bin_edges'][:-1] + data['bin_edges'][1:])
        pdfs.append(hist)
    pdfs = np.array(pdfs)
    pdf_all_scales.append(pdfs)

    pdfs_mean = np.mean(pdfs, axis=0)
    cov_matrix = np.cov(pdfs, rowvar=False)
    np.save(f"PDF_mean_s{s}.npy", pdfs_mean)
    np.save(f"CovPDF_s{s}.npy", cov_matrix)
    # plt.figure(figsize=(8,6))
    # plt.errorbar(bin_centers, pdfs_mean, yerr=np.sqrt(np.diag(cov_matrix)), fmt='o', label='Mean PDF with error bars')
    # plt.xlabel('Kappa')
    # plt.ylabel('Probability Density')
    # plt.title('Mean Kappa PDF with Covariance Error Bars')
    # plt.legend()
    # plt.savefig(f"outfiles/Mean_Kappa_PDF_with_Covariance_Error_Bars_s{s}.png")
    # plt.close()

    # defect pdf +1/2
    pdfs_pos = []
    for i in range(n_realizations):
        data = np.load(f"{DATA_DIR}/DefectPDFs/realisation_{i}_s{s}_defect_kappa_pos_pdf.npz")
        hist = data['hist']
        bin_centers = data['bin_edges']
        pdfs_pos.append(hist)
    pdfs_pos = np.array(pdfs_pos)
    pdfs_pos_mean = np.mean(pdfs_pos, axis=0)
    cov_matrix_pos = np.cov(pdfs_pos, rowvar=False)
    np.save(f"CovDPDF_pos_s{s}.npy", cov_matrix_pos)
    np.save(f"DPDF_pos_mean_s{s}.npy", pdfs_pos_mean)
    # plt.figure(figsize=(8,6))
    # plt.errorbar(bin_centers, pdfs_pos_mean, yerr=np.sqrt(np.diag(cov_matrix_pos)), fmt='o', label='Mean Defect +1/2 PDF with error bars')
    # plt.xlabel('Kappa')
    # plt.ylabel('Probability Density')
    # plt.title('Mean Defect +1/2 Kappa PDF with Covariance Error Bars')
    # plt.legend()
    # plt.savefig(f"outfiles/Mean_Defect_Positive_Half_Kappa_PDF_with_Covariance_Error_Bars_s{s}.png")
    # plt.close()

    # defect pdf -1/2
    pdfs_neg = []
    for i in range(n_realizations):
        data = np.load(f"{DATA_DIR}/DefectPDFs/realisation_{i}_s{s}_defect_kappa_neg_pdf.npz")
        hist = data['hist']
        bin_centers = data['bin_edges']
        pdfs_neg.append(hist)
    pdfs_neg = np.array(pdfs_neg)
    pdfs_neg_mean = np.mean(pdfs_neg, axis=0)
    cov_matrix_neg = np.cov(pdfs_neg, rowvar=False)
    np.save(f"CovDPDF_neg_s{s}.npy", cov_matrix_neg)
    np.save(f"DPDF_neg_mean_s{s}.npy", pdfs_neg_mean)
    # plt.figure(figsize=(8,6))
    # plt.errorbar(bin_centers, pdfs_neg_mean, yerr=np.sqrt(np.diag(cov_matrix_neg)), fmt='o', label='Mean Defect -1/2 PDF with error bars')
    # plt.xlabel('Kappa')
    # plt.ylabel('Probability Density')
    # plt.title('Mean Defect -1/2 Kappa PDF with Covariance Error Bars')
    # plt.legend()
    # plt.savefig(f"outfiles/Mean_Defect_Negative_Half_Kappa_PDF_with_Covariance_Error_Bars_s{s}.png")
    # plt.close()


    # KK

    kk = []
    for i in range(n_realizations):
        data = np.load(f"{DATA_DIR}/KKs/realisation_{i}_s{s}_kappa_kappa_corr.npz")
        theta = data['theta']
        xi_kk = data['xi_kk']
        kk.append(xi_kk)
    kk = np.array(kk)
    kk_all_scales.append(kk)
    kk_mean = np.mean(kk, axis=0)
    np.save(f"KK_mean_s{s}.npy", kk_mean)
    cov_matrix_kk = np.cov(kk, rowvar=False)
    np.save(f"CovKK_s{s}.npy", cov_matrix_kk)
    # plt.figure(figsize=(8,6))
    # plt.errorbar(theta, kk_mean, yerr=np.sqrt(np.diag(cov_matrix_kk)), fmt='o', label='Mean KK Correlation with error bars')
    # plt.xscale('log')
    # plt.xlabel('Theta (arcmin)')
    # plt.ylabel('KK Correlation')
    # plt.title('Mean KK Correlation with Covariance Error Bars')
    # plt.legend()
    # plt.savefig(f"outfiles/Mean_KK_Correlation_with_Covariance_Error_Bars_s{s}.png")
    # plt.close()

    # PosPos
    pos_pos = []
    for i in range(n_realizations):
        data = np.load(f"{DATA_DIR}/PosPos/realisation_{i}_s{s}_pos_pos_corr.npz")
        theta = data['theta']
        xi_pp = data['xi_pp']
        pos_pos.append(xi_pp)
    pos_pos = np.array(pos_pos)
    pos_pos_mean = np.mean(pos_pos, axis=0)
    np.save(f"PosPos_mean_s{s}.npy", pos_pos_mean)
    cov_matrix_pos_pos = np.cov(pos_pos, rowvar=False)
    np.save(f"CovPosPos_s{s}.npy", cov_matrix_pos_pos)
    # plt.figure(figsize=(8,6))
    # plt.errorbar(theta, pos_pos_mean, yerr=np.sqrt(np.diag(cov_matrix_pos_pos)), fmt='o', label='Mean +1/2 +1/2 Defect Correlation with error bars')
    # plt.xscale('log')
    # plt.xlabel('Theta (arcmin)')
    # plt.ylabel('+1/2 +1/2 Defect Correlation')
    # plt.title('Mean +1/2 +1/2 Defect Correlation with Covariance Error Bars')
    # plt.legend()
    # plt.savefig(f"outfiles/Mean_PosPos_Defect_Correlation_with_Covariance_Error_Bars_s{s}.png")
    # plt.close()

    # NegNeg
    neg_neg = []
    for i in range(n_realizations):
        data = np.load(f"{DATA_DIR}/NegNeg/realisation_{i}_s{s}_neg_neg_corr.npz")
        theta = data['theta']
        xi_nn = data['xi_nn']
        neg_neg.append(xi_nn)
    neg_neg = np.array(neg_neg)
    neg_neg_mean = np.mean(neg_neg, axis=0)
    np.save(f"NegNeg_mean_s{s}.npy", neg_neg_mean)
    cov_matrix_neg_neg = np.cov(neg_neg, rowvar=False)
    np.save(f"CovNegNeg_s{s}.npy", cov_matrix_neg_neg)
    # plt.figure(figsize=(8,6))
    # plt.errorbar(theta, neg_neg_mean, yerr=np.sqrt(np.diag(cov_matrix_neg_neg)), fmt='o', label='Mean -1/2 -1/2 Defect Correlation with error bars')
    # plt.xscale('log')
    # plt.xlabel('Theta (arcmin)')
    # plt.ylabel('-1/2 -1/2 Defect Correlation')
    # plt.title('Mean -1/2 -1/2 Defect Correlation with Covariance Error Bars')
    # plt.legend()
    # plt.savefig(f"outfiles/Mean_NegNeg_Defect_Correlation_with_Covariance_Error_Bars_s{s}.png")
    # plt.close()

    # PosNeg
    pos_neg = []
    for i in range(n_realizations):
        data = np.load(f"{DATA_DIR}/PosNeg/realisation_{i}_s{s}_pos_neg_corr.npz")
        theta = data['theta']
        xi_pn = data['xi_pn']
        pos_neg.append(xi_pn)
    pos_neg = np.array(pos_neg)
    pos_neg_mean = np.mean(pos_neg, axis=0)
    np.save(f"PosNeg_mean_s{s}.npy", pos_neg_mean)
    cov_matrix_pos_neg = np.cov(pos_neg, rowvar=False)
    np.save(f"CovPosNeg_s{s}.npy", cov_matrix_pos_neg)
    # plt.figure(figsize=(8,6))
    # plt.errorbar(theta, pos_neg_mean, yerr=np.sqrt(np.diag(cov_matrix_pos_neg)), fmt='o', label='Mean +1/2 -1/2 Defect Correlation with error bars')
    # plt.xscale('log')
    # plt.xlabel('Theta (arcmin)')
    # plt.ylabel('+1/2 -1/2 Defect Correlation')
    # plt.title('Mean +1/2 -1/2 Defect Correlation with Covariance Error Bars')
    # plt.legend()
    # plt.savefig(f"outfiles/Mean_PosNeg_Defect_Correlation_with_Covariance_Error_Bars_s{s}.png")
    # plt.close()

    #Pos kappa
    pk = []
    for i in range(n_realizations):
        data = np.load(f"{DATA_DIR}/PKs/realisation_{i}_s{s}_pos_kappa_corr.npz")
        theta = data['theta']
        xi_pk = data['xi_pk']
        pk.append(xi_pk)
    pk = np.array(pk)
    pk_mean = np.mean(pk, axis=0)
    np.save(f"PosKappa_mean_s{s}.npy", pk_mean)
    cov_matrix_pk = np.cov(pk, rowvar=False)
    np.save(f"CovPosKappa_s{s}.npy", cov_matrix_pk)
    # plt.figure(figsize=(8,6))
    # plt.errorbar(theta, pk_mean, yerr=np.sqrt(np.diag(cov_matrix_pk)), fmt='o', label='Mean +1/2 Defect-Kappa Correlation with error bars')
    # plt.xscale('log')
    # plt.xlabel('Theta (arcmin)')
    # plt.ylabel('+1/2 Defect-Kappa Correlation')
    # plt.title('Mean +1/2 Defect-Kappa Correlation with Covariance Error Bars')
    # plt.legend()
    # plt.savefig(f"outfiles/Mean_PosKappa_Correlation_with_Covariance_Error_Bars_s{s}.png")
    # plt.close()

    #Neg kappa
    nk = []
    for i in range(n_realizations):
        data = np.load(f"{DATA_DIR}/NKs/realisation_{i}_s{s}_neg_kappa_corr.npz")
        theta = data['theta']
        xi_nk = data['xi_nk']
        nk.append(xi_nk)
    nk = np.array(nk)
    nk_mean = np.mean(nk, axis=0)
    np.save(f"NegKappa_mean_s{s}.npy", nk_mean)
    cov_matrix_nk = np.cov(nk, rowvar=False)
    np.save(f"CovNegKappa_s{s}.npy", cov_matrix_nk)
    # plt.figure(figsize=(8,6))
    # plt.errorbar(theta, nk_mean, yerr=np.sqrt(np.diag(cov_matrix_nk)), fmt='o', label='Mean -1/2 Defect-Kappa Correlation with error bars')
    # plt.xscale('log')
    # plt.xlabel('Theta (arcmin)')
    # plt.ylabel('-1/2 Defect-Kappa Correlation')
    # plt.title('Mean -1/2 Defect-Kappa Correlation with Covariance Error Bars')
    # plt.legend()
    # plt.savefig(f"outfiles/Mean_NegKappa_Correlation_with_Covariance_Error_Bars_s{s}.png")
    # plt.close()

    # pos-peak
    pos_peak = []
    for i in range(n_realizations):
        data = np.load(f"{DATA_DIR}/PosPeak/realisation_{i}_s{s}_pos_peak_corr.npz")
        theta = data['theta']
        pos_peak.append(data['xi'])
    pos_peak = np.array(pos_peak)
    pos_peak_mean = np.mean(pos_peak, axis=0)
    np.save(f"PosPeak_mean_s{s}.npy", pos_peak_mean)
    cov_matrix_pos_peak = np.cov(pos_peak, rowvar=False)
    np.save(f"CovPosPeak_s{s}.npy", cov_matrix_pos_peak)
    plt.figure(figsize=(8,6))
    plt.errorbar(theta, pos_peak_mean, yerr=np.sqrt(np.diag(cov_matrix_pos_peak)), fmt='o', label='Mean +1/2 Defect-Peak Correlation')
    plt.xscale('log')
    plt.xlabel('Theta (arcmin)')
    plt.ylabel('+1/2 Defect-Peak Correlation')
    plt.title('Mean +1/2 Defect-Peak Correlation with Covariance Error Bars')
    plt.legend()
    plt.savefig(f"outfiles/Mean_PosPeak_Correlation_with_Covariance_Error_Bars_s{s}.png")
    plt.close()

    # pos-minima
    pos_min = []
    for i in range(n_realizations):
        data = np.load(f"{DATA_DIR}/PosMin/realisation_{i}_s{s}_pos_min_corr.npz")
        theta = data['theta']
        pos_min.append(data['xi'])
    pos_min = np.array(pos_min)
    pos_min_mean = np.mean(pos_min, axis=0)
    np.save(f"PosMin_mean_s{s}.npy", pos_min_mean)
    cov_matrix_pos_min = np.cov(pos_min, rowvar=False)
    np.save(f"CovPosMin_s{s}.npy", cov_matrix_pos_min)
    plt.figure(figsize=(8,6))
    plt.errorbar(theta, pos_min_mean, yerr=np.sqrt(np.diag(cov_matrix_pos_min)), fmt='o', label='Mean +1/2 Defect-Minima Correlation')
    plt.xscale('log')
    plt.xlabel('Theta (arcmin)')
    plt.ylabel('+1/2 Defect-Minima Correlation')
    plt.title('Mean +1/2 Defect-Minima Correlation with Covariance Error Bars')
    plt.legend()
    plt.savefig(f"outfiles/Mean_PosMin_Correlation_with_Covariance_Error_Bars_s{s}.png")
    plt.close()

    # neg-peak
    neg_peak = []
    for i in range(n_realizations):
        data = np.load(f"{DATA_DIR}/NegPeak/realisation_{i}_s{s}_neg_peak_corr.npz")
        theta = data['theta']
        neg_peak.append(data['xi'])
    neg_peak = np.array(neg_peak)
    neg_peak_mean = np.mean(neg_peak, axis=0)
    np.save(f"NegPeak_mean_s{s}.npy", neg_peak_mean)
    cov_matrix_neg_peak = np.cov(neg_peak, rowvar=False)
    np.save(f"CovNegPeak_s{s}.npy", cov_matrix_neg_peak)
    plt.figure(figsize=(8,6))
    plt.errorbar(theta, neg_peak_mean, yerr=np.sqrt(np.diag(cov_matrix_neg_peak)), fmt='o', label='Mean -1/2 Defect-Peak Correlation')
    plt.xscale('log')
    plt.xlabel('Theta (arcmin)')
    plt.ylabel('-1/2 Defect-Peak Correlation')
    plt.title('Mean -1/2 Defect-Peak Correlation with Covariance Error Bars')
    plt.legend()
    plt.savefig(f"outfiles/Mean_NegPeak_Correlation_with_Covariance_Error_Bars_s{s}.png")
    plt.close()

    # neg-minima
    neg_min = []
    for i in range(n_realizations):
        data = np.load(f"{DATA_DIR}/NegMin/realisation_{i}_s{s}_neg_min_corr.npz")
        theta = data['theta']
        neg_min.append(data['xi'])
    neg_min = np.array(neg_min)
    neg_min_mean = np.mean(neg_min, axis=0)
    np.save(f"NegMin_mean_s{s}.npy", neg_min_mean)
    cov_matrix_neg_min = np.cov(neg_min, rowvar=False)
    np.save(f"CovNegMin_s{s}.npy", cov_matrix_neg_min)
    plt.figure(figsize=(8,6))
    plt.errorbar(theta, neg_min_mean, yerr=np.sqrt(np.diag(cov_matrix_neg_min)), fmt='o', label='Mean -1/2 Defect-Minima Correlation')
    plt.xscale('log')
    plt.xlabel('Theta (arcmin)')
    plt.ylabel('-1/2 Defect-Minima Correlation')
    plt.title('Mean -1/2 Defect-Minima Correlation with Covariance Error Bars')
    plt.legend()
    plt.savefig(f"outfiles/Mean_NegMin_Correlation_with_Covariance_Error_Bars_s{s}.png")
    plt.close()


    # NNs
    nn_pos_same1 = []
    nn_neg_same1 = []
    nn_pos_opp1 = []
    nn_neg_opp1 = []
    nn_pos_same2 = []
    nn_neg_same2 = []
    nn_pos_opp2 = []
    nn_neg_opp2 = []
    nn_pos_same4 = []
    nn_neg_same4 = []
    nn_pos_opp4 = []
    nn_neg_opp4 = []
    nn_pos_same8 = []
    nn_neg_same8 = []
    nn_pos_opp8 = []
    nn_neg_opp8 = []
    for i in range(n_realizations):
        data = np.load(f"{DATA_DIR}/NNs/realisation_{i}_s{s}_NNs.npz")
        nn_pos_same1.append(data['nn_pos_same1_cum'])
        nn_neg_same1.append(data['nn_neg_same1_cum'])
        nn_pos_opp1.append(data['nn_pos_opp1_cum'])
        nn_neg_opp1.append(data['nn_neg_opp1_cum'])
        nn_pos_same2.append(data['nn_pos_same2_cum'])
        nn_neg_same2.append(data['nn_neg_same2_cum'])
        nn_pos_opp2.append(data['nn_pos_opp2_cum'])
        nn_neg_opp2.append(data['nn_neg_opp2_cum'])
        nn_pos_same4.append(data['nn_pos_same4_cum'])
        nn_neg_same4.append(data['nn_neg_same4_cum'])
        nn_pos_opp4.append(data['nn_pos_opp4_cum'])
        nn_neg_opp4.append(data['nn_neg_opp4_cum'])
        nn_pos_same8.append(data['nn_pos_same8_cum'])
        nn_neg_same8.append(data['nn_neg_same8_cum'])
        nn_pos_opp8.append(data['nn_pos_opp8_cum'])
        nn_neg_opp8.append(data['nn_neg_opp8_cum'])
    nn_pos_same1 = np.array(nn_pos_same1)
    nn_neg_same1 = np.array(nn_neg_same1)
    nn_pos_opp1 = np.array(nn_pos_opp1)
    nn_neg_opp1 = np.array(nn_neg_opp1)
    nn_pos_same2 = np.array(nn_pos_same2)
    nn_neg_same2 = np.array(nn_neg_same2)
    nn_pos_opp2 = np.array(nn_pos_opp2)
    nn_neg_opp2 = np.array(nn_neg_opp2)
    nn_pos_same4 = np.array(nn_pos_same4)
    nn_neg_same4 = np.array(nn_neg_same4)
    nn_pos_opp4 = np.array(nn_pos_opp4)
    nn_neg_opp4 = np.array(nn_neg_opp4)
    nn_pos_same8 = np.array(nn_pos_same8)
    nn_neg_same8 = np.array(nn_neg_same8)
    nn_pos_opp8 = np.array(nn_pos_opp8)
    nn_neg_opp8 = np.array(nn_neg_opp8)

    np.save(f"NNs_mean_s{s}.npy", {
        'nn_pos_same1_mean': np.mean(nn_pos_same1, axis=0),
        'nn_neg_same1_mean': np.mean(nn_neg_same1, axis=0),
        'nn_pos_opp1_mean': np.mean(nn_pos_opp1, axis=0),
        'nn_neg_opp1_mean': np.mean(nn_neg_opp1, axis=0),
        'nn_pos_same2_mean': np.mean(nn_pos_same2, axis=0),
        'nn_neg_same2_mean': np.mean(nn_neg_same2, axis=0),
        'nn_pos_opp2_mean': np.mean(nn_pos_opp2, axis=0),
        'nn_neg_opp2_mean': np.mean(nn_neg_opp2, axis=0),
        'nn_pos_same4_mean': np.mean(nn_pos_same4, axis=0),
        'nn_neg_same4_mean': np.mean(nn_neg_same4, axis=0),
        'nn_pos_opp4_mean': np.mean(nn_pos_opp4, axis=0),
        'nn_neg_opp4_mean': np.mean(nn_neg_opp4, axis=0),
        'nn_pos_same8_mean': np.mean(nn_pos_same8, axis=0),
        'nn_neg_same8_mean': np.mean(nn_neg_same8, axis=0),
        'nn_pos_opp8_mean': np.mean(nn_pos_opp8, axis=0),
        'nn_neg_opp8_mean': np.mean(nn_neg_opp8, axis=0)
    })
    cov_matrix_nns = {
        'cov_nn_pos_same1': np.cov(nn_pos_same1, rowvar=False),
        'cov_nn_neg_same1': np.cov(nn_neg_same1, rowvar=False),
        'cov_nn_pos_opp1': np.cov(nn_pos_opp1, rowvar=False),
        'cov_nn_neg_opp1': np.cov(nn_neg_opp1, rowvar=False),
        'cov_nn_pos_same2': np.cov(nn_pos_same2, rowvar=False),
        'cov_nn_neg_same2': np.cov(nn_neg_same2, rowvar=False),
        'cov_nn_pos_opp2': np.cov(nn_pos_opp2, rowvar=False),
        'cov_nn_neg_opp2': np.cov(nn_neg_opp2, rowvar=False),
        'cov_nn_pos_same4': np.cov(nn_pos_same4, rowvar=False),
        'cov_nn_neg_same4': np.cov(nn_neg_same4, rowvar=False),
        'cov_nn_pos_opp4': np.cov(nn_pos_opp4, rowvar=False),
        'cov_nn_neg_opp4': np.cov(nn_neg_opp4, rowvar=False),
        'cov_nn_pos_same8': np.cov(nn_pos_same8, rowvar=False),
        'cov_nn_neg_same8': np.cov(nn_neg_same8, rowvar=False),
        'cov_nn_pos_opp8': np.cov(nn_pos_opp8, rowvar=False),
        'cov_nn_neg_opp8': np.cov(nn_neg_opp8, rowvar=False)
    }
    np.save(f"CovNNs_s{s}.npy", cov_matrix_nns)
    # combined covariance
    nn_combined = np.hstack([
        nn_pos_same1, nn_neg_same1, nn_pos_opp1, nn_neg_opp1,
        nn_pos_same2, nn_neg_same2, nn_pos_opp2, nn_neg_opp2,
        nn_pos_same4, nn_neg_same4, nn_pos_opp4, nn_neg_opp4,
        nn_pos_same8, nn_neg_same8, nn_pos_opp8, nn_neg_opp8
    ])
    nn_all_scales.append(nn_combined)
    cov_matrix_nn_combined = np.cov(nn_combined, rowvar=False)
    np.save(f"CovNNs_combined_s{s}.npy", cov_matrix_nn_combined)
    # plt.figure(figsize=(8,6))
    # plt.errorbar(np.arange(nn_pos_same1.shape[1]), np.mean(nn_pos_same1, axis=0), yerr=np.sqrt(np.diag(cov_matrix_nns['cov_nn_pos_same1'])), fmt='o', label='Mean NN Pos Same 1 with error bars')
    # plt.xlabel('Bin Index')
    # plt.ylabel('Cumulative Number Count')
    # plt.title('Mean NN Pos Same 1 with Covariance Error Bars')
    # plt.legend()
    # plt.savefig(f"outfiles/Mean_NN_Pos_Same_1_with_Covariance_Error_Bars_s{s}.png")
    # plt.close()

    # plt.figure(figsize=(8,6))
    # plt.errorbar(np.arange(nn_neg_opp4.shape[1]), np.mean(nn_neg_opp4, axis=0), yerr=np.sqrt(np.diag(cov_matrix_nns['cov_nn_neg_opp4'])), fmt='o', label='Mean NN Neg Opp 4 with error bars')
    # plt.xlabel('Bin Index')
    # plt.ylabel('Cumulative Number Count')
    # plt.title('Mean NN Neg Opp 4 with Covariance Error Bars')
    # plt.legend()
    # plt.savefig(f"outfiles/Mean_NN_Neg_Opp_4_with_Covariance_Error_Bars_s{s}.png")
    # plt.close()

    # plt.figure(figsize=(8,6))   
    # plt.errorbar(np.arange(nn_pos_same8.shape[1]), np.mean(nn_pos_same8, axis=0), yerr=np.sqrt(np.diag(cov_matrix_nns['cov_nn_pos_same8'])), fmt='o', label='Mean NN Pos Same 8 with error bars')
    # plt.xlabel('Bin Index')
    # plt.ylabel('Cumulative Number Count')
    # plt.title('Mean NN Pos Same 8 with Covariance Error Bars')
    # plt.legend()
    # plt.savefig(f"outfiles/Mean_NN_Pos_Same_8_with_Covariance_Error_Bars_s{s}.png")
    # plt.close()

    # Plot Ridgeline Distributions for all quantities
    # quantities = {
    #     'PDF': pdfs, 'PDF_Pos': pdfs_pos, 'PDF_Neg': pdfs_neg,
    #     'KK': kk, 'PosPos': pos_pos, 'NegNeg': neg_neg, 'PosNeg': pos_neg,
    #     'PosKappa': pk, 'NegKappa': nk,
    #     'NN_Pos_Same_1': nn_pos_same1, 'NN_Neg_Same_1': nn_neg_same1,
    #     'NN_Pos_Opp_1': nn_pos_opp1, 'NN_Neg_Opp_1': nn_neg_opp1,
    #     'NN_Pos_Same_2': nn_pos_same2, 'NN_Neg_Same_2': nn_neg_same2,
    #     'NN_Pos_Opp_2': nn_pos_opp2, 'NN_Neg_Opp_2': nn_neg_opp2,
    #     'NN_Pos_Same_4': nn_pos_same4, 'NN_Neg_Same_4': nn_neg_same4,
    #     'NN_Pos_Opp_4': nn_pos_opp4, 'NN_Neg_Opp_4': nn_neg_opp4,
    #     'NN_Pos_Same_8': nn_pos_same8, 'NN_Neg_Same_8': nn_neg_same8,
    #     'NN_Pos_Opp_8': nn_pos_opp8, 'NN_Neg_Opp_8': nn_neg_opp8
    # }
    
    # print(f"Plotting ridgeline distributions for s={s}...")
    # for name, data_mat in quantities.items():
    #     try:
    #         plot_ridgeline(data_mat, f"Real Distribution of {name} (s{s})", f"{name}_ridgeline_s{s}")
    #     except Exception as e:
    #         print(f"Failed to plot ridgeline for {name}: {e}")


    # DPDF combined covariance
    # first connect data vectors head to tail
    dpdf_combined = np.hstack((pdfs_pos, pdfs_neg)) 
    cov_matrix_dpdf_combined = np.cov(dpdf_combined, rowvar=False)
    np.save(f"CovDPDF_combined_s{s}.npy", cov_matrix_dpdf_combined)
    dpdf_all_scales.append(dpdf_combined)

    # dpdf&cdf combined
    cdf_combined = np.hstack((pdfs_pos, pdfs_neg, nn_combined))
    defects_all_scales.append(cdf_combined)
    cov_matrix_cdf_combined = np.cov(cdf_combined, rowvar=False)
    np.save(f"CovCDF_combined_s{s}.npy", cov_matrix_cdf_combined)

    dcorr_combined = np.hstack((pos_pos, neg_neg, pos_neg))
    dcorr_all_scales.append(np.hstack((pos_pos, neg_neg, pos_neg)))
    cov_matrix_dcorr_combined = np.cov(dcorr_combined, rowvar=False)
    np.save(f"CovDCorr_combined_s{s}.npy", cov_matrix_dcorr_combined)


    





pdf_all_scales = np.hstack(pdf_all_scales)
print("PDF all scales shape:", pdf_all_scales.shape)
pdf_all_scales_mean = np.mean(pdf_all_scales, axis=0)
pdf_all_scales_cov = np.cov(pdf_all_scales, rowvar=False)
np.save("PDF_all_scales_mean.npy", pdf_all_scales_mean)
np.save("PDF_all_scales_cov.npy", pdf_all_scales_cov)

dpdf_all_scales = np.hstack(dpdf_all_scales)
print("DPDF all scales shape:", dpdf_all_scales.shape)
dpdf_all_scales_mean = np.mean(dpdf_all_scales, axis=0)
dpdf_all_scales_cov = np.cov(dpdf_all_scales, rowvar=False)
np.save("DPDF_all_scales_mean.npy", dpdf_all_scales_mean)
np.save("DPDF_all_scales_cov.npy", dpdf_all_scales_cov)

nn_all_scales = np.hstack(nn_all_scales)
print("NN all scales shape:", nn_all_scales.shape)
nn_all_scales_mean = np.mean(nn_all_scales, axis=0)
nn_all_scales_cov = np.cov(nn_all_scales, rowvar=False)
np.save("NN_all_scales_mean.npy", nn_all_scales_mean)
np.save("NN_all_scales_cov.npy", nn_all_scales_cov)

kk_all_scales = np.hstack(kk_all_scales)
print("KK all scales shape:", kk_all_scales.shape)
kk_all_scales_mean = np.mean(kk_all_scales, axis=0)
kk_all_scales_cov = np.cov(kk_all_scales, rowvar=False)
np.save("KK_all_scales_mean.npy", kk_all_scales_mean)
np.save("KK_all_scales_cov.npy", kk_all_scales_cov)

dcorr_all_scales = np.hstack(dcorr_all_scales)
print("DCorr all scales shape:", dcorr_all_scales.shape)
dcorr_all_scales_mean = np.mean(dcorr_all_scales, axis=0)
dcorr_all_scales_cov = np.cov(dcorr_all_scales, rowvar=False)
np.save("DCorr_all_scales_mean.npy", dcorr_all_scales_mean)
np.save("DCorr_all_scales_cov.npy", dcorr_all_scales_cov)

defects_all_scales = np.hstack(defects_all_scales)
print("Defects all scales shape:", defects_all_scales.shape)
defects_all_scales_mean = np.mean(defects_all_scales, axis=0)
defects_all_scales_cov = np.cov(defects_all_scales, rowvar=False)
np.save("Defects_all_scales_mean.npy", defects_all_scales_mean)
np.save("Defects_all_scales_cov.npy", defects_all_scales_cov)

defects_pdf_all_scales = np.hstack((defects_all_scales, pdf_all_scales))
print("Defects+PDF all scales shape:", defects_pdf_all_scales.shape)
defects_pdf_all_scales_mean = np.mean(defects_pdf_all_scales, axis=0)
defects_pdf_all_scales_cov = np.cov(defects_pdf_all_scales, rowvar=False)
np.save("Defects_PDF_all_scales_mean.npy", defects_pdf_all_scales_mean)
np.save("Defects_PDF_all_scales_cov.npy", defects_pdf_all_scales_cov)

all_all_scales = np.hstack((defects_pdf_all_scales, kk_all_scales[:,:30]))
print("All all scales shape:", all_all_scales.shape)
all_all_scales_mean = np.mean(all_all_scales, axis=0)
all_all_scales_cov = np.cov(all_all_scales, rowvar=False)
np.save("All_all_scales_mean.npy", all_all_scales_mean)
np.save("All_all_scales_cov.npy", all_all_scales_cov)



