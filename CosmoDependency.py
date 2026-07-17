from GenMapData import GenMap
from DefectFinder import DefectFinder
import matplotlib.pyplot as plt
import numpy as np
import glob
from scipy.special import jvp


def cl_psi_from_kappa(cl_kappa):
    ells = np.arange(len(cl_kappa))
    cl_psi = np.zeros_like(cl_kappa)
    valid = ells >= 2
    denom = (ells[valid] ** 2) * ((ells[valid] + 1) ** 2)
    cl_psi[valid] = 4.0 * cl_kappa[valid] / denom
    return cl_psi


def gaussian_filter_ell(ells, sigma_rad):
    return np.exp(-0.5 * ells * (ells + 1) * sigma_rad**2)


def corr_theta_derivative_from_cl_bessel(cl_psi, theta, order):
    ells = np.arange(len(cl_psi), dtype=np.float64)
    pref = (ells / (2.0 * np.pi)) * cl_psi
    x = np.outer(ells, theta)
    kernel = jvp(0, x, n=int(order))
    return np.sum((pref * (ells ** int(order)))[:, None] * kernel, axis=0)


def compute_c0_moments(cl_psi):
    ells = np.arange(len(cl_psi), dtype=np.float64)
    pref = (ells / (2.0 * np.pi)) * cl_psi
    c4_0 = np.sum(pref * (3.0 / 8.0) * ells**4)
    c6_0 = np.sum(pref * (-5.0 / 16.0) * ells**6)
    return c4_0, c6_0


def compute_h_u_and_derivative_analytic(theta, c1, c2, c3, c4, c5, c6, c4_0):
    r = theta
    l0 = c4_0 / 3.0
    l = c3 / r - 2.0 * c2 / (r**2) + 2.0 * c1 / (r**3)
    m = c4
    n = 3.0 * c2 / (r**2) - 3.0 * c1 / (r**3)
    p = -c5
    q = -c4 / r + 3.0 * c3 / (r**2) - 6.0 * c2 / (r**3) + 6.0 * c1 / (r**4)
    rr = -3.0 * c3 / (r**2) + 9.0 * c2 / (r**3) - 9.0 * c1 / (r**4)

    scomb = m + n - 2.0 * l
    a = l0**2 - (scomb**2) / 16.0
    b = l0**2 - l**2
    detk = a * b

    l1 = c4 / r - 3.0 * c3 / (r**2) + 6.0 * c2 / (r**3) - 6.0 * c1 / (r**4)
    l2 = c5 / r - 4.0 * c4 / (r**2) + 12.0 * c3 / (r**3) - 24.0 * c2 / (r**4) + 24.0 * c1 / (r**5)
    m1 = c5
    m2 = c6
    n1 = 3.0 * c3 / (r**2) - 9.0 * c2 / (r**3) + 9.0 * c1 / (r**4)
    n2 = 3.0 * c4 / (r**2) - 15.0 * c3 / (r**3) + 36.0 * c2 / (r**4) - 36.0 * c1 / (r**5)
    p1 = -c6
    q1 = -c5 / r + 4.0 * c4 / (r**2) - 12.0 * c3 / (r**3) + 24.0 * c2 / (r**4) - 24.0 * c1 / (r**5)
    q2 = -c6 / r + 5.0 * c5 / (r**2) - 20.0 * c4 / (r**3) + 60.0 * c3 / (r**4) - 120.0 * c2 / (r**5) + 120.0 * c1 / (r**6)
    rr1 = -3.0 * c4 / (r**2) + 15.0 * c3 / (r**3) - 36.0 * c2 / (r**4) + 36.0 * c1 / (r**5)
    rr2 = -3.0 * c5 / (r**2) + 21.0 * c4 / (r**3) - 81.0 * c3 / (r**4) + 180.0 * c2 / (r**5) - 180.0 * c1 / (r**6)

    s1c = m1 + n1 - 2.0 * l1
    s2c = m2 + n2 - 2.0 * l2
    a1 = -0.125 * scomb * s1c
    a2 = -0.125 * (s1c**2 + scomb * s2c)
    b1 = -2.0 * l * l1
    b2 = -2.0 * (l1**2 + l * l2)
    detk1 = a1 * b + a * b1
    detk2 = a2 * b + 2.0 * a1 * b1 + a * b2

    s = np.sqrt(detk)
    invs = 1.0 / s
    s1 = detk1 / (2.0 * s)
    s2 = detk2 / (2.0 * s) - (detk1**2) / (4.0 * s**3)
    invs1 = -s1 * invs**2
    invs2 = -s2 * invs**2 + 2.0 * (s1**2) * invs**3

    t = q - rr
    t1 = q1 - rr1
    t2 = q2 - rr2
    g = t**2
    g1 = 2.0 * t * t1
    g2 = 2.0 * (t1**2 + t * t2)

    aa1 = 0.25 * (g * invs + r * g1 * invs + r * g * invs1)
    aa2 = 0.25 * (2.0 * g1 * invs + 2.0 * g * invs1 + r * g2 * invs + 2.0 * r * g1 * invs1 + r * g * invs2)
    mm = q * (p + rr - 2.0 * q)
    mm1 = q1 * (p + rr - 2.0 * q) + q * (p1 + rr1 - 2.0 * q1)
    bb = 0.25 * mm * invs
    bb1 = 0.25 * (mm1 * invs + mm * invs1)

    h_u = aa1 + bb
    dh_u = aa2 + bb1
    return h_u, dh_u


def compute_g_u_theory_from_cl_kappa(cl_kappa, theta_arcmin, smooth_arcmin=30.0):
    ells = np.arange(len(cl_kappa))
    cl_psi = cl_psi_from_kappa(cl_kappa)

    sigma_rad = np.deg2rad(smooth_arcmin / 60.0)
    cl_psi *= gaussian_filter_ell(ells, sigma_rad) ** 2

    theta = np.deg2rad(theta_arcmin / 60.0)
    c1 = corr_theta_derivative_from_cl_bessel(cl_psi, theta, 1)
    c2 = corr_theta_derivative_from_cl_bessel(cl_psi, theta, 2)
    c3 = corr_theta_derivative_from_cl_bessel(cl_psi, theta, 3)
    c4 = corr_theta_derivative_from_cl_bessel(cl_psi, theta, 4)
    c5 = corr_theta_derivative_from_cl_bessel(cl_psi, theta, 5)
    c6 = corr_theta_derivative_from_cl_bessel(cl_psi, theta, 6)
    c4_0, c6_0 = compute_c0_moments(cl_psi)
    _, dh_u = compute_h_u_and_derivative_analytic(theta, c1, c2, c3, c4, c5, c6, c4_0)
    d_u = np.abs(3.0 * c6_0 / (10.0 * np.pi * c4_0))
    g_u = dh_u / (4.0 * np.pi**2 * d_u**2 * theta)
    return g_u

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

# Correlation measurement setup (tuned for sparse defect catalogs).
MIN_SEP_ARCMIN = 10.0
MAX_SEP_ARCMIN = 1500.0
N_BINS = 15
RANDOM_MULTIPLIER = 300
RANDOM_SEED_BASE = 12345
# generate maps for the three cosmologies
kappa1_unsmoothed, kappa1, gamm11, gamm21, phi1 = GenMap(cosmo1, nside=res, lmax=res*3, random_seed=24, random_type='gaussian', smooth_angle=np.radians(90/60), smooth_type='post')
kappa2_unsmoothed, kappa2, gamm12, gamm22, phi2 = GenMap(cosmo2, nside=res, lmax=res*3, random_seed=24, random_type='gaussian', smooth_angle=np.radians(90/60), smooth_type='post')
kappa3_unsmoothed, kappa3, gamm13, gamm23, phi3 = GenMap(cosmo3, nside=res, lmax=res*3, random_seed=24, random_type='gaussian', smooth_angle=np.radians(90/60), smooth_type='post')

np.savez("outfiles/map_data_cosmo1.npz", kappa_bar_unsmoothed=kappa1_unsmoothed, kappa_bar=kappa1, gamm1_bar=gamm11, gamm2_bar=gamm21, phi_bar=phi1)
np.savez("outfiles/map_data_cosmo2.npz", kappa_bar_unsmoothed=kappa2_unsmoothed, kappa_bar=kappa2, gamm1_bar=gamm12, gamm2_bar=gamm22, phi_bar=phi2)
np.savez("outfiles/map_data_cosmo3.npz", kappa_bar_unsmoothed=kappa3_unsmoothed, kappa_bar=kappa3, gamm1_bar=gamm13, gamm2_bar=gamm23, phi_bar=phi3)

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

# df1.find_peaks(type='peaks', savepath="outfiles/cosmo1")
# df2.find_peaks(type='peaks', savepath="outfiles/cosmo2")
# df3.find_peaks(type='peaks', savepath="outfiles/cosmo3")

# df1.find_peaks(type='minima', savepath="outfiles/cosmo1")
# df2.find_peaks(type='minima', savepath="outfiles/cosmo2")
# df3.find_peaks(type='minima', savepath="outfiles/cosmo3")

# # print numbers
# print(df1.number(), df1.number_estimation())
# print(df2.number(), df2.number_estimation())
# print(df3.number(), df3.number_estimation())

# nearest_pos_dist1 = df1.nearest_distance(charge=0.5, k=1)
# nearest_pos_dist2 = df2.nearest_distance(charge=0.5, k=4)
# nearest_pos_dist3 = df3.nearest_distance(charge=0.5, k=8)
# plt.figure(figsize=(8, 6))
# plt.hist(nearest_pos_dist1, bins=30, density=False, cumulative=True, histtype='step', label=f'Oc={cosmo1["Oc"]}')
# plt.hist(nearest_pos_dist2, bins=30, density=False, cumulative=True, histtype='step', label=f'Oc={cosmo2["Oc"]}')
# plt.hist(nearest_pos_dist3, bins=30, density=False, cumulative=True, histtype='step', label=f'Oc={cosmo3["Oc"]}')
# # plt.xscale('log')
# plt.xlabel('Nearest +1/2 Defect Distance [arcmin]', fontsize=14)
# plt.ylabel('Cumulative Number Distribution', fontsize=14)
# plt.title('Nearest +1/2 Defect Distance Distribution', fontsize=16)
# plt.legend()
# plt.grid()
# plt.tight_layout()
# plt.savefig("outfiles/nearest_pos_distance_comparison.png", dpi=300, bbox_inches='tight')


# nearest_neg_dist1 = df1.nearest_distance(charge=-0.5, k=1)
# nearest_neg_dist2 = df2.nearest_distance(charge=-0.5, k=4)
# nearest_neg_dist3 = df3.nearest_distance(charge=-0.5, k=8)

# plt.figure(figsize=(8, 6))
# plt.hist(nearest_neg_dist1, bins=30, density=False, cumulative=True, histtype='step', label=f'Oc={cosmo1["Oc"]}')
# plt.hist(nearest_neg_dist2, bins=30, density=False, cumulative=True, histtype='step', label=f'Oc={cosmo2["Oc"]}')
# plt.hist(nearest_neg_dist3, bins=30, density=False, cumulative=True, histtype='step', label=f'Oc={cosmo3["Oc"]}')
# # plt.xscale('log')
# plt.xlabel('Nearest -1/2 Defect Distance [arcmin]', fontsize=14)
# plt.ylabel('Cumulative Number Distribution', fontsize=14)
# plt.title('Nearest -1/2 Defect Distance Distribution', fontsize=16)
# plt.legend()
# plt.grid()
# plt.tight_layout()
# plt.savefig("outfiles/nearest_neg_distance_comparison.png", dpi=300, bbox_inches='tight')



# nearest_opp_pos_dist1 = df1.nearest_opposite_distance(charge=0.5, k=1)
# nearest_opp_pos_dist2 = df2.nearest_opposite_distance(charge=0.5, k=4)
# nearest_opp_pos_dist3 = df3.nearest_opposite_distance(charge=0.5, k=8)
# plt.figure(figsize=(8, 6))
# plt.hist(nearest_opp_pos_dist1, bins=30, density=False, cumulative=True, histtype='step', label=f'Oc={cosmo1["Oc"]}')
# plt.hist(nearest_opp_pos_dist2, bins=30, density=False, cumulative=True, histtype='step', label=f'Oc={cosmo2["Oc"]}')
# plt.hist(nearest_opp_pos_dist3, bins=30, density=False, cumulative=True, histtype='step', label=f'Oc={cosmo3["Oc"]}')
# # plt.xscale('log')
# plt.xlabel('Nearest Opposite Charge Defect Distance [arcmin]', fontsize=14)
# plt.ylabel('Cumulative Number Distribution', fontsize=14)
# plt.title('Nearest Opposite Charge Defect Distance Distribution', fontsize=16)
# plt.legend()
# plt.grid()
# plt.tight_layout()
# plt.savefig("outfiles/nearest_opp_pos_distance_comparison.png", dpi=300, bbox_inches='tight')

# nearest_opp_neg_dist1 = df1.nearest_opposite_distance(charge=-0.5, k=1)
# nearest_opp_neg_dist2 = df2.nearest_opposite_distance(charge=-0.5, k=4)
# nearest_opp_neg_dist3 = df3.nearest_opposite_distance(charge=-0.5, k=8)
# plt.figure(figsize=(8, 6))
# plt.hist(nearest_opp_neg_dist1, bins=30, density=False, cumulative=True, histtype='step', label=f'Oc={cosmo1["Oc"]}')
# plt.hist(nearest_opp_neg_dist2, bins=30, density=False, cumulative=True, histtype='step', label=f'Oc={cosmo2["Oc"]}')
# plt.hist(nearest_opp_neg_dist3, bins=30, density=False, cumulative=True, histtype='step', label=f'Oc={cosmo3["Oc"]}')
# # plt.xscale('log')
# plt.xlabel('Nearest Opposite Charge Defect Distance [arcmin]', fontsize=14)
# plt.ylabel('Cumulative Number Distribution', fontsize=14)
# plt.title('Nearest Opposite Charge Defect Distance Distribution', fontsize=16)
# plt.legend()
# plt.grid()
# plt.tight_layout()
# plt.savefig("outfiles/nearest_opp_neg_distance_comparison.png", dpi=300, bbox_inches='tight')

# # kappa dist
# kappa_pos1 = df1.convergence_dist(charge=0.5)
# kappa_pos2 = df2.convergence_dist(charge=0.5)
# kappa_pos3 = df3.convergence_dist(charge=0.5)
# bins = np.array([-0.01, -0.005, 0.005, 0.01])
# bins = np.linspace(-0.01, 0.01, 50)

# kappa_pos_cum1 = np.array([np.sum(kappa_pos1 < b) for b in bins]) / 4 / np.pi
# kappa_pos_cum2 = np.array([np.sum(kappa_pos2 < b) for b in bins]) / 4 / np.pi
# kappa_pos_cum3 = np.array([np.sum(kappa_pos3 < b) for b in bins]) / 4 / np.pi

# plt.figure(figsize=(8, 6))
# plt.plot(bins, kappa_pos_cum1, label=f'Oc={cosmo1["Oc"]}')
# plt.plot(bins, kappa_pos_cum2, label=f'Oc={cosmo2["Oc"]}')
# plt.plot(bins, kappa_pos_cum3, label=f'Oc={cosmo3["Oc"]}')
# plt.xlabel('Convergence κ at +1/2 Defect Locations', fontsize=14)
# plt.ylabel('Cumulative Number < κ', fontsize=14)
# plt.title('Cumulative Count of +1/2 Defects with κ < threshold', fontsize=16)
# plt.legend()
# plt.grid()
# plt.tight_layout()
# plt.savefig("outfiles/kappa_pos_pdf_comparison.png", dpi=300, bbox_inches='tight')

# kappa_neg1 = df1.convergence_dist(charge=-0.5)
# kappa_neg2 = df2.convergence_dist(charge=-0.5)
# kappa_neg3 = df3.convergence_dist(charge=-0.5)
# bins = np.array([-0.01, -0.005, 0.005, 0.01])
# bins = np.linspace(-0.01, 0.01, 50)
# kappa_neg_cum1 = np.array([np.sum(kappa_neg1 < b) for b in bins]) / 4 / np.pi
# kappa_neg_cum2 = np.array([np.sum(kappa_neg2 < b) for b in bins]) / 4 / np.pi
# kappa_neg_cum3 = np.array([np.sum(kappa_neg3 < b) for b in bins]) / 4 / np.pi
# # smooth the histogram using a simple moving average
# # kappa_neg_hist1 = np.convolve(kappa_neg_hist1, np.ones(5)/5, mode='same')
# # kappa_neg_hist2 = np.convolve(kappa_neg_hist2, np.ones(5)/5, mode='same')
# # kappa_neg_hist3 = np.convolve(kappa_neg_hist3, np.ones(5)/5, mode='same')
# plt.figure(figsize=(8, 6))
# plt.plot(bins, kappa_neg_cum1, label=f'Oc={cosmo1["Oc"]}')
# plt.plot(bins, kappa_neg_cum2, label=f'Oc={cosmo2["Oc"]}')
# plt.plot(bins, kappa_neg_cum3, label=f'Oc={cosmo3["Oc"]}')
# plt.xlabel('Convergence κ at -1/2 Defect Locations', fontsize=14)
# plt.ylabel('Probability Density', fontsize=14)
# plt.title('Convergence κ Distribution at -1/2 Defects', fontsize=16)
# plt.legend()
# plt.grid()
# plt.tight_layout()
# plt.savefig("outfiles/kappa_neg_pdf_comparison.png", dpi=300, bbox_inches='tight')

# # all kappa at defects
# kappa_defects1 = np.concatenate([kappa_pos1, kappa_neg1])
# kappa_defects2 = np.concatenate([kappa_pos2, kappa_neg2])
# kappa_defects3 = np.concatenate([kappa_pos3, kappa_neg3])
# bins = np.linspace(-0.01, 0.01, 50)
# kappa_defects_cum1 = np.array([np.sum(kappa_defects1 < b) for b in bins]) / 4 / np.pi
# kappa_defects_cum2 = np.array([np.sum(kappa_defects2 < b) for b in bins]) / 4 / np.pi
# kappa_defects_cum3 = np.array([np.sum(kappa_defects3 < b) for b in bins]) / 4 / np.pi

# plt.figure(figsize=(8, 6))
# plt.plot(bins, kappa_defects_cum1, label=f'Oc={cosmo1["Oc"]}')
# plt.plot(bins, kappa_defects_cum2, label=f'Oc={cosmo2["Oc"]}')
# plt.plot(bins, kappa_defects_cum3, label=f'Oc={cosmo3["Oc"]}')
# plt.xlabel('Convergence κ at Defect Locations', fontsize=14)
# plt.ylabel('Cumulative Number < κ', fontsize=14)
# plt.title('Cumulative Count of Defects with κ < threshold', fontsize=16)
# plt.legend()
# plt.grid()
# plt.tight_layout()
# plt.savefig("outfiles/kappa_defects_pdf_comparison.png", dpi=300, bbox_inches='tight')

# # kappa pdf
# kappa_all1 = df1.kappa_bar
# kappa_all2 = df2.kappa_bar
# kappa_all3 = df3.kappa_bar
# bin_edges = np.array([-0.01, -0.005, 0.0, 0.005, 0.01])
# bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
# kappa_all_hist1, _ = np.histogram(kappa_all1, bins=bin_edges, density=True)
# kappa_all_hist2, _ = np.histogram(kappa_all2, bins=bin_edges, density=True)
# kappa_all_hist3, _ = np.histogram(kappa_all3, bins=bin_edges, density=True)
# # kappa_all_hist1 = np.convolve(kappa_all_hist1, np.ones(5)/5, mode='same')
# # kappa_all_hist2 = np.convolve(kappa_all_hist2, np.ones(5)/5, mode='same')
# # kappa_all_hist3 = np.convolve(kappa_all_hist3, np.ones(5)/5, mode='same')
# plt.figure(figsize=(8, 6))
# plt.plot(bin_centers, kappa_all_hist1, label=f'Oc={cosmo1["Oc"]}')
# plt.plot(bin_centers, kappa_all_hist2, label=f'Oc={cosmo2["Oc"]}')
# plt.plot(bin_centers, kappa_all_hist3, label=f'Oc={cosmo3["Oc"]}')
# plt.xlabel('Convergence κ', fontsize=14)
# plt.ylabel('Probability Density', fontsize=14)
# plt.title('Convergence κ Full Map Distribution', fontsize=16)
# plt.legend()
# plt.grid()
# plt.tight_layout()
# plt.savefig("outfiles/kappa_pdf_comparison.png", dpi=300, bbox_inches='tight')

# unsigned defect correlation function + theory comparison
cosmo_cfgs = {
    "cosmo1": (cosmo1, df1),
    "cosmo2": (cosmo2, df2),
    "cosmo3": (cosmo3, df3),
}

plt.figure(figsize=(8, 6))
results = {}
for label, (cosmo_cfg, df_obj) in cosmo_cfgs.items():
    idx = int(label[-1])
    charge_paths = sorted(glob.glob(f"outfiles/defects_{label}_charge_*.fits"))
    if len(charge_paths) == 0:
        raise FileNotFoundError(f"No charge-split FITS found for {label}.")

    theta_arcmin, xi_data, varxi_data, cov_data = df_obj.correlation_function_unsigned(
        charge_paths=charge_paths,
        min_sep=MIN_SEP_ARCMIN,
        max_sep=MAX_SEP_ARCMIN,
        n_bins=N_BINS,
        random_multiplier=RANDOM_MULTIPLIER,
        random_seed=RANDOM_SEED_BASE + idx,
    )
    ell_map, cl_kappa_map = df_obj.convergence_power_spectrum(lmax=3 * res - 1)
    xi_theory = compute_g_u_theory_from_cl_kappa(cl_kappa_map, theta_arcmin, smooth_arcmin=90.0)
    valid = np.isfinite(xi_data) & np.isfinite(varxi_data) & (varxi_data > 0) & np.isfinite(xi_theory)

    results[label] = {
        "theta_arcmin": theta_arcmin,
        "xi_data": xi_data,
        "varxi_data": varxi_data,
        "cov_data": cov_data,
        "xi_theory": xi_theory,
        "ell_map": ell_map,
        "cl_kappa_map": cl_kappa_map,
        "charge_paths": np.array(charge_paths),
        "valid_mask": valid,
    }

    line = plt.errorbar(
        theta_arcmin[valid],
        xi_data[valid],
        yerr=np.sqrt(varxi_data[valid]),
        fmt='o',
        ms=4,
        capsize=2,
        label=f'{label} data (Oc={cosmo_cfg["Oc"]:.2f})'
    )
    plt.plot(
        theta_arcmin[valid],
        xi_theory[valid],
        '--',
        color=line[0].get_color(),
        label=f'{label} theory'
    )

np.savez("outfiles/cosmo_unsigned_cf_vs_theory.npz", **results)

plt.xscale('log')
plt.xlabel('Angular Separation [arcmin]', fontsize=14)
plt.ylabel('Correlation', fontsize=14)
plt.title('Unsigned Defect Correlation: Data vs Theory', fontsize=16)
plt.grid(alpha=0.3)
plt.legend(fontsize=9)
plt.tight_layout()
plt.savefig("outfiles/defect_unsigned_correlation_vs_theory_cosmo123.png", dpi=300, bbox_inches='tight')

# signed density correlation function + theory comparison
plt.figure(figsize=(8, 6))
results_signed = {}
for label, (cosmo_cfg, df_obj) in cosmo_cfgs.items():
    idx = int(label[-1])
    charge_paths = sorted(glob.glob(f"outfiles/defects_{label}_charge_*.fits"))
    if len(charge_paths) == 0:
        raise FileNotFoundError(f"No charge-split FITS found for {label}.")

    theta_arcmin, xi_data, varxi_data, cov_data = df_obj.correlation_function_signed(
        charge_paths=charge_paths,
        min_sep=MIN_SEP_ARCMIN,
        max_sep=MAX_SEP_ARCMIN,
        n_bins=N_BINS,
        random_multiplier=RANDOM_MULTIPLIER,
        random_seed=RANDOM_SEED_BASE + 100 + idx,
    )
    ell_map, cl_kappa_map = df_obj.convergence_power_spectrum(lmax=3 * res - 1)
    xi_theory = compute_g_u_theory_from_cl_kappa(cl_kappa_map, theta_arcmin, smooth_arcmin=90.0)
    valid = np.isfinite(xi_data) & np.isfinite(varxi_data) & (varxi_data > 0) & np.isfinite(xi_theory)

    results_signed[label] = {
        "theta_arcmin": theta_arcmin,
        "xi_data_signed": xi_data,
        "varxi_data_signed": varxi_data,
        "cov_data_signed": cov_data,
        "xi_theory": xi_theory,
        "ell_map": ell_map,
        "cl_kappa_map": cl_kappa_map,
        "charge_paths": np.array(charge_paths),
        "valid_mask": valid,
    }

    line = plt.errorbar(
        theta_arcmin[valid],
        xi_data[valid],
        yerr=np.sqrt(varxi_data[valid]),
        fmt='o',
        ms=4,
        capsize=2,
        label=f'{label} signed density data (Oc={cosmo_cfg["Oc"]:.2f})'
    )
    plt.plot(
        theta_arcmin[valid],
        xi_theory[valid],
        '--',
        color=line[0].get_color(),
        label=f'{label} theory'
    )

np.savez("outfiles/cosmo_signed_cf_vs_theory.npz", **results_signed)

plt.xscale('log')
plt.xlabel('Angular Separation [arcmin]', fontsize=14)
plt.ylabel('Correlation', fontsize=14)
plt.title('Signed Density Correlation: Data vs Theory', fontsize=16)
plt.grid(alpha=0.3)
plt.legend(fontsize=9)
plt.tight_layout()
plt.savefig("outfiles/defect_signed_correlation_vs_theory_cosmo123.png", dpi=300, bbox_inches='tight')

# # defect-kappa cross-correlation function
# theta1, xi_dk_pos1, varxi_dk_pos1, covmat_dk_pos1 = df1.defect_convergence_correlation(charge=0.5, min_sep=10., max_sep=3000, n_bins=20)
# theta2, xi_dk_pos2, varxi_dk_pos2, covmat_dk_pos2 = df2.defect_convergence_correlation(charge=0.5, min_sep=10., max_sep=3000, n_bins=20)
# theta3, xi_dk_pos3, varxi_dk_pos3, covmat_dk_pos3 = df3.defect_convergence_correlation(charge=0.5, min_sep=10., max_sep=3000, n_bins=20)
# plt.figure(figsize=(8, 6))
# plt.errorbar(theta1, xi_dk_pos1, yerr=np.sqrt(varxi_dk_pos1), fmt='o-', label=f'Oc={cosmo1["Oc"]}')
# plt.errorbar(theta2, xi_dk_pos2, yerr=np.sqrt(varxi_dk_pos2), fmt='s--', label=f'Oc={cosmo2["Oc"]}')
# plt.errorbar(theta3, xi_dk_pos3, yerr=np.sqrt(varxi_dk_pos3), fmt='^-.', label=f'Oc={cosmo3["Oc"]}')
# plt.xlabel('Angular Separation [arcmin]', fontsize=14)
# plt.ylabel('Defect-Convergence Correlation ξ_dκ', fontsize=14)
# plt.title('Defect-Convergence Correlation Function for +1/2 Defects', fontsize=16)
# plt.xscale('log')
# plt.legend()
# plt.grid()
# plt.tight_layout()
# plt.savefig("outfiles/defect_convergence_correlation_pos_comparison.png", dpi=300, bbox_inches='tight')

# theta1, xi_dk_neg1, varxi_dk_neg1, covmat_dk_neg1 = df1.defect_convergence_correlation(charge=-0.5, min_sep=10., max_sep=3000, n_bins=20)
# theta2, xi_dk_neg2, varxi_dk_neg2, covmat_dk_neg2 = df2.defect_convergence_correlation(charge=-0.5, min_sep=10., max_sep=3000, n_bins=20)
# theta3, xi_dk_neg3, varxi_dk_neg3, covmat_dk_neg3 = df3.defect_convergence_correlation(charge=-0.5, min_sep=10., max_sep=3000, n_bins=20)
# plt.figure(figsize=(8, 6))
# plt.errorbar(theta1, xi_dk_neg1, yerr=np.sqrt(varxi_dk_neg1), fmt='o-', label=f'Oc={cosmo1["Oc"]}')
# plt.errorbar(theta2, xi_dk_neg2, yerr=np.sqrt(varxi_dk_neg2), fmt='s--', label=f'Oc={cosmo2["Oc"]}')
# plt.errorbar(theta3, xi_dk_neg3, yerr=np.sqrt(varxi_dk_neg3), fmt='^-.', label=f'Oc={cosmo3["Oc"]}')
# plt.xlabel('Angular Separation [arcmin]', fontsize=14)
# plt.ylabel('Defect-Convergence Correlation ξ_dκ', fontsize=14)
# plt.title('Defect-Convergence Correlation Function for -1/2 Defects', fontsize=16)
# plt.xscale('log')
# plt.legend()
# plt.grid()
# plt.tight_layout()
# plt.savefig("outfiles/defect_convergence_correlation_neg_comparison.png", dpi=300, bbox_inches='tight')

# # convergence KK correlation function
# theta1, xi_kk1, varxi_kk1, covmat_kk1 = df1.convergence_correlation(min_sep=10., max_sep=3000., n_bins=20)
# theta2, xi_kk2, varxi_kk2, covmat_kk2 = df2.convergence_correlation(min_sep=10., max_sep=3000., n_bins=20)
# theta3, xi_kk3, varxi_kk3, covmat_kk3 = df3.convergence_correlation(min_sep=10., max_sep=3000., n_bins=20)
# plt.figure(figsize=(8, 6))
# plt.errorbar(theta1, xi_kk1, yerr=np.sqrt(varxi_kk1), fmt='o-', label=f'Oc={cosmo1["Oc"]}')
# plt.errorbar(theta2, xi_kk2, yerr=np.sqrt(varxi_kk2), fmt='s--', label=f'Oc={cosmo2["Oc"]}')
# plt.errorbar(theta3, xi_kk3, yerr=np.sqrt(varxi_kk3), fmt='^-.', label=f'Oc={cosmo3["Oc"]}')
# plt.xlabel('Angular Separation [arcmin]', fontsize=14)
# plt.ylabel('Convergence Correlation ξ_κκ', fontsize=14)
# plt.title('Convergence Correlation Function', fontsize=16)
# plt.xscale('log')
# plt.legend()
# plt.grid()
# plt.tight_layout()
# plt.savefig("outfiles/convergence_correlation_comparison.png", dpi=300, bbox_inches='tight')

# # defect-peak 2PCF, pos-peak, neg-minima
# theta1, xi_pos_peak1, varxi_pos_peak1, covmat_pos_peak1 = df1.defect_peak_correlation(charge_path="outfiles/defects_cosmo1_charge_+0.5.fits", peak_path="outfiles/cosmo1_peaks.fits", min_sep=10., max_sep=3000, n_bins=20)
# theta2, xi_pos_peak2, varxi_pos_peak2, covmat_pos_peak2 = df2.defect_peak_correlation(charge_path="outfiles/defects_cosmo2_charge_+0.5.fits", peak_path="outfiles/cosmo2_peaks.fits", min_sep=10., max_sep=3000, n_bins=20)
# theta3, xi_pos_peak3, varxi_pos_peak3, covmat_pos_peak3 = df3.defect_peak_correlation(charge_path="outfiles/defects_cosmo3_charge_+0.5.fits", peak_path="outfiles/cosmo3_peaks.fits", min_sep=10., max_sep=3000, n_bins=20)

# plt.figure(figsize=(8, 6))
# plt.errorbar(theta1, xi_pos_peak1, yerr=np.sqrt(varxi_pos_peak1), fmt='o-', label=f'Oc={cosmo1["Oc"]}')
# plt.errorbar(theta2, xi_pos_peak2, yerr=np.sqrt(varxi_pos_peak2), fmt='s--', label=f'Oc={cosmo2["Oc"]}')
# plt.errorbar(theta3, xi_pos_peak3, yerr=np.sqrt(varxi_pos_peak3), fmt='^-.', label=f'Oc={cosmo3["Oc"]}')
# plt.xlabel('Angular Separation [arcmin]', fontsize=14)
# plt.ylabel('Cross-Correlation ξ', fontsize=14)
# plt.title('Correlation: +1/2 Defects vs Peaks', fontsize=16)
# plt.xscale('log')
# plt.legend()
# plt.grid()
# plt.tight_layout()
# plt.savefig("outfiles/defect_peak_correlation_pos_comparison.png", dpi=300, bbox_inches='tight')

# theta1, xi_neg_min1, varxi_neg_min1, covmat_neg_min1 = df1.defect_peak_correlation(charge_path="outfiles/defects_cosmo1_charge_-0.5.fits", peak_path="outfiles/cosmo1_minima.fits", min_sep=10., max_sep=3000, n_bins=20)
# theta2, xi_neg_min2, varxi_neg_min2, covmat_neg_min2 = df2.defect_peak_correlation(charge_path="outfiles/defects_cosmo2_charge_-0.5.fits", peak_path="outfiles/cosmo2_minima.fits", min_sep=10., max_sep=3000, n_bins=20)
# theta3, xi_neg_min3, varxi_neg_min3, covmat_neg_min3 = df3.defect_peak_correlation(charge_path="outfiles/defects_cosmo3_charge_-0.5.fits", peak_path="outfiles/cosmo3_minima.fits", min_sep=10., max_sep=3000, n_bins=20)

# plt.figure(figsize=(8, 6))
# plt.errorbar(theta1, xi_neg_min1, yerr=np.sqrt(varxi_neg_min1), fmt='o-', label=f'Oc={cosmo1["Oc"]}')
# plt.errorbar(theta2, xi_neg_min2, yerr=np.sqrt(varxi_neg_min2), fmt='s--', label=f'Oc={cosmo2["Oc"]}')
# plt.errorbar(theta3, xi_neg_min3, yerr=np.sqrt(varxi_neg_min3), fmt='^-.', label=f'Oc={cosmo3["Oc"]}')
# plt.xlabel('Angular Separation [arcmin]', fontsize=14)
# plt.ylabel('Cross-Correlation ξ', fontsize=14)
# plt.title('Correlation: -1/2 Defects vs Minima', fontsize=16)
# plt.xscale('log')
# plt.legend()
# plt.grid()
# plt.tight_layout()
# plt.savefig("outfiles/defect_minima_correlation_neg_comparison.png", dpi=300, bbox_inches='tight')

# # Pos-Minima
# theta1, xi_pos_min1, varxi_pos_min1, covmat_pos_min1 = df1.defect_peak_correlation(charge_path="outfiles/defects_cosmo1_charge_+0.5.fits", peak_path="outfiles/cosmo1_minima.fits", min_sep=10., max_sep=3000, n_bins=20)
# theta2, xi_pos_min2, varxi_pos_min2, covmat_pos_min2 = df2.defect_peak_correlation(charge_path="outfiles/defects_cosmo2_charge_+0.5.fits", peak_path="outfiles/cosmo2_minima.fits", min_sep=10., max_sep=3000, n_bins=20)
# theta3, xi_pos_min3, varxi_pos_min3, covmat_pos_min3 = df3.defect_peak_correlation(charge_path="outfiles/defects_cosmo3_charge_+0.5.fits", peak_path="outfiles/cosmo3_minima.fits", min_sep=10., max_sep=3000, n_bins=20)

# plt.figure(figsize=(8, 6))
# plt.errorbar(theta1, xi_pos_min1, yerr=np.sqrt(varxi_pos_min1), fmt='o-', label=f'Oc={cosmo1["Oc"]}')
# plt.errorbar(theta2, xi_pos_min2, yerr=np.sqrt(varxi_pos_min2), fmt='s--', label=f'Oc={cosmo2["Oc"]}')
# plt.errorbar(theta3, xi_pos_min3, yerr=np.sqrt(varxi_pos_min3), fmt='^-.', label=f'Oc={cosmo3["Oc"]}')
# plt.xlabel('Angular Separation [arcmin]', fontsize=14)
# plt.ylabel('Cross-Correlation ξ', fontsize=14)
# plt.title('Correlation: +1/2 Defects vs Minima', fontsize=16)
# plt.xscale('log')
# plt.legend()
# plt.grid()
# plt.tight_layout()
# plt.savefig("outfiles/defect_minima_correlation_pos_comparison.png", dpi=300, bbox_inches='tight')

# # Neg-Peak
# theta1, xi_neg_peak1, varxi_neg_peak1, covmat_neg_peak1 = df1.defect_peak_correlation(charge_path="outfiles/defects_cosmo1_charge_-0.5.fits", peak_path="outfiles/cosmo1_peaks.fits", min_sep=10., max_sep=3000, n_bins=20)
# theta2, xi_neg_peak2, varxi_neg_peak2, covmat_neg_peak2 = df2.defect_peak_correlation(charge_path="outfiles/defects_cosmo2_charge_-0.5.fits", peak_path="outfiles/cosmo2_peaks.fits", min_sep=10., max_sep=3000, n_bins=20)
# theta3, xi_neg_peak3, varxi_neg_peak3, covmat_neg_peak3 = df3.defect_peak_correlation(charge_path="outfiles/defects_cosmo3_charge_-0.5.fits", peak_path="outfiles/cosmo3_peaks.fits", min_sep=10., max_sep=3000, n_bins=20)

# plt.figure(figsize=(8, 6))
# plt.errorbar(theta1, xi_neg_peak1, yerr=np.sqrt(varxi_neg_peak1), fmt='o-', label=f'Oc={cosmo1["Oc"]}')
# plt.errorbar(theta2, xi_neg_peak2, yerr=np.sqrt(varxi_neg_peak2), fmt='s--', label=f'Oc={cosmo2["Oc"]}')
# plt.errorbar(theta3, xi_neg_peak3, yerr=np.sqrt(varxi_neg_peak3), fmt='^-.', label=f'Oc={cosmo3["Oc"]}')
# plt.xlabel('Angular Separation [arcmin]', fontsize=14)
# plt.ylabel('Cross-Correlation ξ', fontsize=14)
# plt.title('Correlation: -1/2 Defects vs Peaks', fontsize=16)
# plt.xscale('log')
# plt.legend()
# plt.grid()
# plt.tight_layout()
# plt.savefig("outfiles/defect_peak_correlation_neg_comparison.png", dpi=300, bbox_inches='tight')
