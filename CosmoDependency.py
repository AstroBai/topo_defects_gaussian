from GenMapData import GenMap
from DefectFinder import DefectFinder
import matplotlib.pyplot as plt
import numpy as np

# change S8 only
cosmo1 = {'h': 0.7,
            'Oc': 0.15,
            'Ob': 0.05,
            'S8': 0.8,
            'w': -1.0}

cosmo2 = {'h': 0.7,
            'Oc': 0.25,
            'Ob': 0.05,
            'S8': 0.8,
            'w': -1.0}

cosmo3 = {'h': 0.7,
            'Oc': 0.35,
            'Ob': 0.05,
            'S8': 0.8,
            'w': -1.0}

res = 512
lmax = int(res/4 - 1)
# generate maps for the three cosmologies
# kappa1_unsmoothed, kappa1, gamm11, gamm21, phi1 = GenMap(cosmo1, nside=res, lmax=res*3, random_seed=24, random_type='gaussian', smooth_angle=np.radians(90/60), smooth_type='post')
# kappa2_unsmoothed, kappa2, gamm12, gamm22, phi2 = GenMap(cosmo2, nside=res, lmax=res*3, random_seed=24, random_type='gaussian', smooth_angle=np.radians(90/60), smooth_type='post')
# kappa3_unsmoothed, kappa3, gamm13, gamm23, phi3 = GenMap(cosmo3, nside=res, lmax=res*3, random_seed=24, random_type='gaussian', smooth_angle=np.radians(90/60), smooth_type='post')

# np.savez("outfiles/map_data_cosmo1.npz", kappa_bar_unsmoothed=kappa1_unsmoothed, kappa_bar=kappa1, gamm1_bar=gamm11, gamm2_bar=gamm21, phi_bar=phi1)
# np.savez("outfiles/map_data_cosmo2.npz", kappa_bar_unsmoothed=kappa2_unsmoothed, kappa_bar=kappa2, gamm1_bar=gamm12, gamm2_bar=gamm22, phi_bar=phi2)
# np.savez("outfiles/map_data_cosmo3.npz", kappa_bar_unsmoothed=kappa3_unsmoothed, kappa_bar=kappa3, gamm1_bar=gamm13, gamm2_bar=gamm23, phi_bar=phi3)

# find defects in the three maps
df1 = DefectFinder(data_path="outfiles/map_data_cosmo1.npz")
df2 = DefectFinder(data_path="outfiles/map_data_cosmo2.npz")
df3 = DefectFinder(data_path="outfiles/map_data_cosmo3.npz")

df1.find_defects(savepath="outfiles/defects_cosmo1.npz")
df2.find_defects(savepath="outfiles/defects_cosmo2.npz")
df3.find_defects(savepath="outfiles/defects_cosmo3.npz")

df1.load_defects(defect_datapath="outfiles/defects_cosmo1.npz")
df2.load_defects(defect_datapath="outfiles/defects_cosmo2.npz")
df3.load_defects(defect_datapath="outfiles/defects_cosmo3.npz")

df1.save_defects(savepath="outfiles/defects_cosmo1")
df2.save_defects(savepath="outfiles/defects_cosmo2")
df3.save_defects(savepath="outfiles/defects_cosmo3")

# print numbers
print(df1.number(), df1.number_estimation())
print(df2.number(), df2.number_estimation())
print(df3.number(), df3.number_estimation())

nearest_pos_dist1 = df1.nearest_distance(charge=0.5, k=1)
nearest_pos_dist2 = df2.nearest_distance(charge=0.5, k=1)
nearest_pos_dist3 = df3.nearest_distance(charge=0.5, k=1)
plt.figure(figsize=(8, 6))
plt.hist(nearest_pos_dist1, bins=30, density=False, cumulative=True, histtype='step', label=f'Oc={cosmo1["Oc"]}')
plt.hist(nearest_pos_dist2, bins=30, density=False, cumulative=True, histtype='step', label=f'Oc={cosmo2["Oc"]}')
plt.hist(nearest_pos_dist3, bins=30, density=False, cumulative=True, histtype='step', label=f'Oc={cosmo3["Oc"]}')
# plt.xscale('log')
plt.xlabel('Nearest +1/2 Defect Distance [arcmin]', fontsize=14)
plt.ylabel('Cumulative Number Distribution', fontsize=14)
plt.title('Nearest +1/2 Defect Distance Distribution', fontsize=16)
plt.legend()
plt.grid()
plt.tight_layout()
plt.savefig("outfiles/nearest_pos_distance_comparison.png", dpi=300, bbox_inches='tight')


nearest_neg_dist1 = df1.nearest_distance(charge=-0.5, k=1)
nearest_neg_dist2 = df2.nearest_distance(charge=-0.5, k=1)
nearest_neg_dist3 = df3.nearest_distance(charge=-0.5, k=1)

plt.figure(figsize=(8, 6))
plt.hist(nearest_neg_dist1, bins=30, density=False, cumulative=True, histtype='step', label=f'Oc={cosmo1["Oc"]}')
plt.hist(nearest_neg_dist2, bins=30, density=False, cumulative=True, histtype='step', label=f'Oc={cosmo2["Oc"]}')
plt.hist(nearest_neg_dist3, bins=30, density=False, cumulative=True, histtype='step', label=f'Oc={cosmo3["Oc"]}')
# plt.xscale('log')
plt.xlabel('Nearest -1/2 Defect Distance [arcmin]', fontsize=14)
plt.ylabel('Cumulative Number Distribution', fontsize=14)
plt.title('Nearest -1/2 Defect Distance Distribution', fontsize=16)
plt.legend()
plt.grid()
plt.tight_layout()
plt.savefig("outfiles/nearest_neg_distance_comparison.png", dpi=300, bbox_inches='tight')



nearest_opp_pos_dist1 = df1.nearest_opposite_distance(charge=0.5, k=1)
nearest_opp_pos_dist2 = df2.nearest_opposite_distance(charge=0.5, k=1)
nearest_opp_pos_dist3 = df3.nearest_opposite_distance(charge=0.5, k=1)
plt.figure(figsize=(8, 6))
plt.hist(nearest_opp_pos_dist1, bins=30, density=False, cumulative=True, histtype='step', label=f'Oc={cosmo1["Oc"]}')
plt.hist(nearest_opp_pos_dist2, bins=30, density=False, cumulative=True, histtype='step', label=f'Oc={cosmo2["Oc"]}')
plt.hist(nearest_opp_pos_dist3, bins=30, density=False, cumulative=True, histtype='step', label=f'Oc={cosmo3["Oc"]}')
# plt.xscale('log')
plt.xlabel('Nearest Opposite Charge Defect Distance [arcmin]', fontsize=14)
plt.ylabel('Cumulative Number Distribution', fontsize=14)
plt.title('Nearest Opposite Charge Defect Distance Distribution', fontsize=16)
plt.legend()
plt.grid()
plt.tight_layout()
plt.savefig("outfiles/nearest_opp_pos_distance_comparison.png", dpi=300, bbox_inches='tight')

nearest_opp_neg_dist1 = df1.nearest_opposite_distance(charge=-0.5, k=1)
nearest_opp_neg_dist2 = df2.nearest_opposite_distance(charge=-0.5, k=1)
nearest_opp_neg_dist3 = df3.nearest_opposite_distance(charge=-0.5, k=1)
plt.figure(figsize=(8, 6))
plt.hist(nearest_opp_neg_dist1, bins=30, density=False, cumulative=True, histtype='step', label=f'Oc={cosmo1["Oc"]}')
plt.hist(nearest_opp_neg_dist2, bins=30, density=False, cumulative=True, histtype='step', label=f'Oc={cosmo2["Oc"]}')
plt.hist(nearest_opp_neg_dist3, bins=30, density=False, cumulative=True, histtype='step', label=f'Oc={cosmo3["Oc"]}')
# plt.xscale('log')
plt.xlabel('Nearest Opposite Charge Defect Distance [arcmin]', fontsize=14)
plt.ylabel('Cumulative Number Distribution', fontsize=14)
plt.title('Nearest Opposite Charge Defect Distance Distribution', fontsize=16)
plt.legend()
plt.grid()
plt.tight_layout()
plt.savefig("outfiles/nearest_opp_neg_distance_comparison.png", dpi=300, bbox_inches='tight')

# kappa dist
kappa_pos1 = df1.convergence_dist(charge=0.5)
kappa_pos2 = df2.convergence_dist(charge=0.5)
kappa_pos3 = df3.convergence_dist(charge=0.5)
bins = np.array([-0.005, -0.0025, 0.0025, 0.005])
bins = np.linspace(-0.005, 0.005, 50)

kappa_pos_cum1 = np.array([np.sum(kappa_pos1 < b) for b in bins]) / 4 / np.pi
kappa_pos_cum2 = np.array([np.sum(kappa_pos2 < b) for b in bins]) / 4 / np.pi
kappa_pos_cum3 = np.array([np.sum(kappa_pos3 < b) for b in bins]) / 4 / np.pi

plt.figure(figsize=(8, 6))
plt.plot(bins, kappa_pos_cum1, label=f'Oc={cosmo1["Oc"]}')
plt.plot(bins, kappa_pos_cum2, label=f'Oc={cosmo2["Oc"]}')
plt.plot(bins, kappa_pos_cum3, label=f'Oc={cosmo3["Oc"]}')
plt.xlabel('Convergence κ at +1/2 Defect Locations', fontsize=14)
plt.ylabel('Cumulative Number < κ', fontsize=14)
plt.title('Cumulative Count of +1/2 Defects with κ < threshold', fontsize=16)
plt.legend()
plt.grid()
plt.tight_layout()
plt.savefig("outfiles/kappa_pos_pdf_comparison.png", dpi=300, bbox_inches='tight')

kappa_neg1 = df1.convergence_dist(charge=-0.5)
kappa_neg2 = df2.convergence_dist(charge=-0.5)
kappa_neg3 = df3.convergence_dist(charge=-0.5)
bins = np.array([-0.005, -0.0025, 0.0025, 0.005])
bins = np.linspace(-0.005, 0.005, 50)
kappa_neg_cum1 = np.array([np.sum(kappa_neg1 < b) for b in bins]) / 4 / np.pi
kappa_neg_cum2 = np.array([np.sum(kappa_neg2 < b) for b in bins]) / 4 / np.pi
kappa_neg_cum3 = np.array([np.sum(kappa_neg3 < b) for b in bins]) / 4 / np.pi
# smooth the histogram using a simple moving average
# kappa_neg_hist1 = np.convolve(kappa_neg_hist1, np.ones(5)/5, mode='same')
# kappa_neg_hist2 = np.convolve(kappa_neg_hist2, np.ones(5)/5, mode='same')
# kappa_neg_hist3 = np.convolve(kappa_neg_hist3, np.ones(5)/5, mode='same')
plt.figure(figsize=(8, 6))
plt.plot(bins, kappa_neg_cum1, label=f'Oc={cosmo1["Oc"]}')
plt.plot(bins, kappa_neg_cum2, label=f'Oc={cosmo2["Oc"]}')
plt.plot(bins, kappa_neg_cum3, label=f'Oc={cosmo3["Oc"]}')
plt.xlabel('Convergence κ at -1/2 Defect Locations', fontsize=14)
plt.ylabel('Probability Density', fontsize=14)
plt.title('Convergence κ Distribution at -1/2 Defects', fontsize=16)
plt.legend()
plt.grid()
plt.tight_layout()
plt.savefig("outfiles/kappa_neg_pdf_comparison.png", dpi=300, bbox_inches='tight')

# all kappa at defects
kappa_defects1 = np.concatenate([kappa_pos1, kappa_neg1])
kappa_defects2 = np.concatenate([kappa_pos2, kappa_neg2])
kappa_defects3 = np.concatenate([kappa_pos3, kappa_neg3])
bins = np.linspace(-0.005, 0.005, 50)
kappa_defects_cum1 = np.array([np.sum(kappa_defects1 < b) for b in bins]) / 4 / np.pi
kappa_defects_cum2 = np.array([np.sum(kappa_defects2 < b) for b in bins]) / 4 / np.pi
kappa_defects_cum3 = np.array([np.sum(kappa_defects3 < b) for b in bins]) / 4 / np.pi

plt.figure(figsize=(8, 6))
plt.plot(bins, kappa_defects_cum1, label=f'Oc={cosmo1["Oc"]}')
plt.plot(bins, kappa_defects_cum2, label=f'Oc={cosmo2["Oc"]}')
plt.plot(bins, kappa_defects_cum3, label=f'Oc={cosmo3["Oc"]}')
plt.xlabel('Convergence κ at Defect Locations', fontsize=14)
plt.ylabel('Cumulative Number < κ', fontsize=14)
plt.title('Cumulative Count of Defects with κ < threshold', fontsize=16)
plt.legend()
plt.grid()
plt.tight_layout()
plt.savefig("outfiles/kappa_defects_pdf_comparison.png", dpi=300, bbox_inches='tight')

# kappa pdf
kappa_all1 = df1.kappa_bar
kappa_all2 = df2.kappa_bar
kappa_all3 = df3.kappa_bar
bin_edges = np.array([-0.005, -0.0025, 0.0, 0.0025, 0.005])
bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
kappa_all_hist1, _ = np.histogram(kappa_all1, bins=bin_edges, density=True)
kappa_all_hist2, _ = np.histogram(kappa_all2, bins=bin_edges, density=True)
kappa_all_hist3, _ = np.histogram(kappa_all3, bins=bin_edges, density=True)
# kappa_all_hist1 = np.convolve(kappa_all_hist1, np.ones(5)/5, mode='same')
# kappa_all_hist2 = np.convolve(kappa_all_hist2, np.ones(5)/5, mode='same')
# kappa_all_hist3 = np.convolve(kappa_all_hist3, np.ones(5)/5, mode='same')
plt.figure(figsize=(8, 6))
plt.plot(bin_centers, kappa_all_hist1, label=f'Oc={cosmo1["Oc"]}')
plt.plot(bin_centers, kappa_all_hist2, label=f'Oc={cosmo2["Oc"]}')
plt.plot(bin_centers, kappa_all_hist3, label=f'Oc={cosmo3["Oc"]}')
plt.xlabel('Convergence κ', fontsize=14)
plt.ylabel('Probability Density', fontsize=14)
plt.title('Convergence κ Full Map Distribution', fontsize=16)
plt.legend()
plt.grid()
plt.tight_layout()
plt.savefig("outfiles/kappa_pdf_comparison.png", dpi=300, bbox_inches='tight')


# # put defect pdf and kappa pdf together
# plt.figure(figsize=(8, 6))
# plt.plot(bin_centers, kappa_all_hist1, label=f'Full Map κ (Oc={cosmo1["Oc"]})', color='black', linestyle='-')
# plt.plot(bin_centers, kappa_pos_hist1, label=f'+1/2 Defects κ (Oc={cosmo1["Oc"]})', color='blue', linestyle='--')
# plt.plot(bin_centers, kappa_neg_hist1, label=f'-1/2 Defects κ (Oc={cosmo1["Oc"]})', color='red', linestyle=':')
# plt.xlabel('Convergence κ', fontsize=14)
# plt.ylabel('Probability Density', fontsize=14)
# plt.title('Convergence κ Distribution Comparison (Oc=0.6)', fontsize=16)
# plt.legend()
# plt.grid()
# plt.tight_layout()
# plt.savefig("outfiles/kappa_pdf_defect_comparison_cosmo1.png", dpi=300, bbox_inches='tight')

# plt.figure(figsize=(8, 6))
# plt.plot(bin_centers, kappa_all_hist2, label=f'Full Map κ (Oc={cosmo2["Oc"]})', color='black', linestyle='-')
# plt.plot(bin_centers, kappa_pos_hist2, label=f'+1/2 Defects κ (Oc={cosmo2["Oc"]})', color='blue', linestyle='--')
# plt.plot(bin_centers, kappa_neg_hist2, label=f'-1/2 Defects κ (Oc={cosmo2["Oc"]})', color='red', linestyle=':')
# plt.xlabel('Convergence κ', fontsize=14)
# plt.ylabel('Probability Density', fontsize=14)
# plt.title('Convergence κ Distribution Comparison (Oc=0.8)', fontsize=16)
# plt.legend()
# plt.grid()
# plt.tight_layout()
# plt.savefig("outfiles/kappa_pdf_defect_comparison_cosmo2.png", dpi= 300, bbox_inches='tight')   

# plt.figure(figsize=(8, 6))
# plt.plot(bin_centers, kappa_all_hist3, label=f'Full Map κ (Oc={cosmo3["Oc"]})', color='black', linestyle='-')
# plt.plot(bin_centers, kappa_pos_hist3, label=f'+1/2 Defects κ (Oc={cosmo3["Oc"]})', color='blue', linestyle='--')
# plt.plot(bin_centers, kappa_neg_hist3, label=f'-1/2 Defects κ (Oc={cosmo3["Oc"]})', color='red', linestyle=':')
# plt.xlabel('Convergence κ', fontsize=14)
# plt.ylabel('Probability Density', fontsize=14)
# plt.title('Convergence κ Distribution Comparison (Oc=1.0)', fontsize=16)
# plt.legend()
# plt.grid()
# plt.tight_layout()
# plt.savefig("outfiles/kappa_pdf_defect_comparison_cosmo3.png", dpi=300, bbox_inches='tight')

# # pdf ratio 
# kappa_pos_ratio1 = kappa_pos_hist1 / kappa_all_hist1
# kappa_neg_ratio1 = kappa_neg_hist1 / kappa_all_hist1
# kappa_pos_ratio2 = kappa_pos_hist2 / kappa_all_hist2
# kappa_neg_ratio2 = kappa_neg_hist2 / kappa_all_hist2
# kappa_pos_ratio3 = kappa_pos_hist3 / kappa_all_hist3
# kappa_neg_ratio3 = kappa_neg_hist3 / kappa_all_hist3
# plt.figure(figsize=(8, 6))
# plt.plot(bin_centers, kappa_pos_ratio1, label=f'+1/2 Defects (Oc={cosmo1["Oc"]})', linestyle='-', c='blue')
# plt.plot(bin_centers, kappa_neg_ratio1, label=f'-1/2 Defects (Oc={cosmo1["Oc"]})', linestyle='--', c='blue')
# plt.plot(bin_centers, kappa_pos_ratio2, label=f'+1/2 Defects (Oc={cosmo2["Oc"]})', linestyle='-', c='green')
# plt.plot(bin_centers, kappa_neg_ratio2, label=f'-1/2 Defects (Oc={cosmo2["Oc"]})', linestyle='--', c='green')
# plt.plot(bin_centers, kappa_pos_ratio3, label=f'+1/2 Defects (Oc={cosmo3["Oc"]})', linestyle='-', c='red')
# plt.plot(bin_centers, kappa_neg_ratio3, label=f'-1/2 Defects (Oc={cosmo3["Oc"]})', linestyle='--', c='red')
# plt.xlabel('Convergence κ', fontsize=14)
# plt.ylabel('PDF Ratio', fontsize=14)
# plt.title('Convergence κ PDF Ratio: Defects vs Full Map', fontsize=16)
# plt.legend()
# plt.grid()
# plt.tight_layout()
# plt.savefig("outfiles/kappa_pdf_ratio_comparison.png", dpi=300, bbox_inches='tight')

# defect correlation function
theta1, xi_dd_pos1, varxi_dd_pos1 = df1.correlation_function(charge1_path="outfiles/defects_cosmo1_charge_+0.5.fits", charge2_path="outfiles/defects_cosmo1_charge_+0.5.fits", min_sep=10., max_sep=3000., n_bins=20)
theta2, xi_dd_pos2, varxi_dd_pos2 = df2.correlation_function(charge1_path="outfiles/defects_cosmo2_charge_+0.5.fits", charge2_path="outfiles/defects_cosmo2_charge_+0.5.fits", min_sep=10., max_sep=3000., n_bins=20)
theta3, xi_dd_pos3, varxi_dd_pos3 = df3.correlation_function(charge1_path="outfiles/defects_cosmo3_charge_+0.5.fits", charge2_path="outfiles/defects_cosmo3_charge_+0.5.fits", min_sep=10., max_sep=3000., n_bins=20)
plt.figure(figsize=(8, 6))
plt.errorbar(theta1, xi_dd_pos1, yerr=np.sqrt(varxi_dd_pos1), fmt='o-', label=f'Oc={cosmo1["Oc"]}')
plt.errorbar(theta2, xi_dd_pos2, yerr=np.sqrt(varxi_dd_pos2), fmt='s--', label=f'Oc={cosmo2["Oc"]}')
plt.errorbar(theta3, xi_dd_pos3, yerr=np.sqrt(varxi_dd_pos3), fmt='^-.', label=f'Oc={cosmo3["Oc"]}')
plt.ylim(-2, 2)
plt.xlabel('Angular Separation [arcmin]', fontsize=14)
plt.ylabel('Defect-Defect Correlation ξ_dd', fontsize=14)
plt.title('Defect-Defect Correlation Function for +1/2 Defects', fontsize=16)
plt.xscale('log')
plt.legend()
plt.grid()
plt.tight_layout()
plt.savefig("outfiles/defect_defect_correlation_pos_comparison.png", dpi=300, bbox_inches='tight')

theta1, xi_dd_neg1, varxi_dd_neg1 = df1.correlation_function(charge1_path="outfiles/defects_cosmo1_charge_-0.5.fits", charge2_path="outfiles/defects_cosmo1_charge_-0.5.fits", min_sep=10., max_sep=3000., n_bins=20)
theta2, xi_dd_neg2, varxi_dd_neg2 = df2.correlation_function(charge1_path="outfiles/defects_cosmo2_charge_-0.5.fits", charge2_path="outfiles/defects_cosmo2_charge_-0.5.fits", min_sep=10., max_sep=3000., n_bins=20)
theta3, xi_dd_neg3, varxi_dd_neg3 = df3.correlation_function(charge1_path="outfiles/defects_cosmo3_charge_-0.5.fits", charge2_path="outfiles/defects_cosmo3_charge_-0.5.fits", min_sep=10., max_sep=3000., n_bins=20)
plt.figure(figsize=(8, 6))
plt.errorbar(theta1, xi_dd_neg1, yerr=np.sqrt(varxi_dd_neg1), fmt='o-', label=f'Oc={cosmo1["Oc"]}')
plt.errorbar(theta2, xi_dd_neg2, yerr=np.sqrt(varxi_dd_neg2), fmt='s--', label=f'Oc={cosmo2["Oc"]}')
plt.errorbar(theta3, xi_dd_neg3, yerr=np.sqrt(varxi_dd_neg3), fmt='^-.', label=f'Oc={cosmo3["Oc"]}')
plt.ylim(-2, 2)
plt.xlabel('Angular Separation [arcmin]', fontsize=14)
plt.ylabel('Defect-Defect Correlation ξ_dd', fontsize=14)
plt.title('Defect-Defect Correlation Function for -1/2 Defects', fontsize=16)
plt.xscale('log')
plt.legend()
plt.grid()
plt.tight_layout()
plt.savefig("outfiles/defect_defect_correlation_neg_comparison.png", dpi=300, bbox_inches='tight')

theta1, xi_dd_neg_pos1, varxi_dd_neg_pos1, covmat_dd_neg_pos1 = df1.correlation_function(charge1_path="outfiles/defects_cosmo1_charge_+0.5.fits", charge2_path="outfiles/defects_cosmo1_charge_-0.5.fits", min_sep=10., max_sep=3000., n_bins=20)
theta2, xi_dd_neg_pos2, varxi_dd_neg_pos2, covmat_dd_neg_pos2 = df2.correlation_function(charge1_path="outfiles/defects_cosmo2_charge_+0.5.fits", charge2_path="outfiles/defects_cosmo2_charge_-0.5.fits", min_sep=10., max_sep=3000., n_bins=20)
theta3, xi_dd_neg_pos3, varxi_dd_neg_pos3, covmat_dd_neg_pos3 = df3.correlation_function(charge1_path="outfiles/defects_cosmo3_charge_+0.5.fits", charge2_path="outfiles/defects_cosmo3_charge_-0.5.fits", min_sep=10., max_sep=3000., n_bins=20)
plt.figure(figsize=(8, 6))
plt.errorbar(theta1, xi_dd_neg_pos1, yerr=np.sqrt(varxi_dd_neg_pos1), fmt='o-', label=f'Oc={cosmo1["Oc"]}')
plt.errorbar(theta2, xi_dd_neg_pos2, yerr=np.sqrt(varxi_dd_neg_pos2), fmt='s--', label=f'Oc={cosmo2["Oc"]}')
plt.errorbar(theta3, xi_dd_neg_pos3, yerr=np.sqrt(varxi_dd_neg_pos3), fmt='^-.', label=f'Oc={cosmo3["Oc"]}')
plt.ylim(-2, 2)
plt.xlabel('Angular Separation [arcmin]', fontsize=14)
plt.ylabel('Defect-Defect Correlation ξ_dd', fontsize=14)
plt.title('Defect-Defect Correlation Function for +1/2 and -1/2 Defects', fontsize=16)
plt.xscale('log')
plt.legend()
plt.grid()
plt.tight_layout()
plt.savefig("outfiles/defect_defect_correlation_neg_pos_comparison.png", dpi=300, bbox_inches='tight')

# defect-kappa cross-correlation function
theta1, xi_dk_pos1, varxi_dk_pos1, covmat_dk_pos1 = df1.defect_convergence_correlation(charge=0.5, min_sep=10., max_sep=3000, n_bins=20)
theta2, xi_dk_pos2, varxi_dk_pos2, covmat_dk_pos2 = df2.defect_convergence_correlation(charge=0.5, min_sep=10., max_sep=3000, n_bins=20)
theta3, xi_dk_pos3, varxi_dk_pos3, covmat_dk_pos3 = df3.defect_convergence_correlation(charge=0.5, min_sep=10., max_sep=3000, n_bins=20)
plt.figure(figsize=(8, 6))
plt.errorbar(theta1, xi_dk_pos1, yerr=np.sqrt(varxi_dk_pos1), fmt='o-', label=f'Oc={cosmo1["Oc"]}')
plt.errorbar(theta2, xi_dk_pos2, yerr=np.sqrt(varxi_dk_pos2), fmt='s--', label=f'Oc={cosmo2["Oc"]}')
plt.errorbar(theta3, xi_dk_pos3, yerr=np.sqrt(varxi_dk_pos3), fmt='^-.', label=f'Oc={cosmo3["Oc"]}')
plt.xlabel('Angular Separation [arcmin]', fontsize=14)
plt.ylabel('Defect-Convergence Correlation ξ_dκ', fontsize=14)
plt.title('Defect-Convergence Correlation Function for +1/2 Defects', fontsize=16)
plt.xscale('log')
plt.legend()
plt.grid()
plt.tight_layout()
plt.savefig("outfiles/defect_convergence_correlation_pos_comparison.png", dpi=300, bbox_inches='tight')

theta1, xi_dk_neg1, varxi_dk_neg1, covmat_dk_neg1 = df1.defect_convergence_correlation(charge=-0.5, min_sep=10., max_sep=3000, n_bins=20)
theta2, xi_dk_neg2, varxi_dk_neg2, covmat_dk_neg2 = df2.defect_convergence_correlation(charge=-0.5, min_sep=10., max_sep=3000, n_bins=20)
theta3, xi_dk_neg3, varxi_dk_neg3, covmat_dk_neg3 = df3.defect_convergence_correlation(charge=-0.5, min_sep=10., max_sep=3000, n_bins=20)
plt.figure(figsize=(8, 6))
plt.errorbar(theta1, xi_dk_neg1, yerr=np.sqrt(varxi_dk_neg1), fmt='o-', label=f'Oc={cosmo1["Oc"]}')
plt.errorbar(theta2, xi_dk_neg2, yerr=np.sqrt(varxi_dk_neg2), fmt='s--', label=f'Oc={cosmo2["Oc"]}')
plt.errorbar(theta3, xi_dk_neg3, yerr=np.sqrt(varxi_dk_neg3), fmt='^-.', label=f'Oc={cosmo3["Oc"]}')
plt.xlabel('Angular Separation [arcmin]', fontsize=14)
plt.ylabel('Defect-Convergence Correlation ξ_dκ', fontsize=14)
plt.title('Defect-Convergence Correlation Function for -1/2 Defects', fontsize=16)
plt.xscale('log')
plt.legend()
plt.grid()
plt.tight_layout()
plt.savefig("outfiles/defect_convergence_correlation_neg_comparison.png", dpi=300, bbox_inches='tight')

# convergence KK correlation function
theta1, xi_kk1, varxi_kk1, covmat_kk1 = df1.convergence_correlation(min_sep=10., max_sep=3000., n_bins=20)
theta2, xi_kk2, varxi_kk2, covmat_kk2 = df2.convergence_correlation(min_sep=10., max_sep=3000., n_bins=20)
theta3, xi_kk3, varxi_kk3, covmat_kk3 = df3.convergence_correlation(min_sep=10., max_sep=3000., n_bins=20)
plt.figure(figsize=(8, 6))
plt.errorbar(theta1, xi_kk1, yerr=np.sqrt(varxi_kk1), fmt='o-', label=f'Oc={cosmo1["Oc"]}')
plt.errorbar(theta2, xi_kk2, yerr=np.sqrt(varxi_kk2), fmt='s--', label=f'Oc={cosmo2["Oc"]}')
plt.errorbar(theta3, xi_kk3, yerr=np.sqrt(varxi_kk3), fmt='^-.', label=f'Oc={cosmo3["Oc"]}')
plt.xlabel('Angular Separation [arcmin]', fontsize=14)
plt.ylabel('Convergence Correlation ξ_κκ', fontsize=14)
plt.title('Convergence Correlation Function', fontsize=16)
plt.xscale('log')
plt.legend()
plt.grid()
plt.tight_layout()
plt.savefig("outfiles/convergence_correlation_comparison.png", dpi=300, bbox_inches='tight')