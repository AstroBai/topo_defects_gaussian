import numpy as np
from tqdm import tqdm
import os
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde, norm


DATA_DIR = "/home/jbai/cephfs_copy/cephfs/jbai/DefectFisherGaussian"
# read pp,nn,pn for s90, 4 cosmologies, 100 realisations each
# {DATA_DIR}/TrainPosPos/train_{i}_s90_seed_{seed}_pos_pos_corr.npz"
# gQ = (pp + nn - 2*pn)/4

# first load fisher data
fid = {'Om': 0.3,
       'sigma8': 0.8}
Ob = 0.049
h = 0.67
w = -1.0
# surrounding cosmology params, e.g. Om +/- delta_Om
delta_params = {'Om': 0.1,
                'sigma8': 0.1}

gQ_s90 = []
gQ_s90_stds = []
pp_s90 = []
nn_s90 = []
pn_s90 = []
for i in range(4):
    gQ_s90_rands = []
    pp_s90_rands = []
    nn_s90_rands = []
    pn_s90_rands = []
    for seed in tqdm(range(100), desc=f"Loading data for cosmology {i}"):
        
        data = np.load(f"{DATA_DIR}/TrainPosPos/train_{i}_s90_seed_{seed}_pos_pos_corr.npz")
        pp = data['xi_pp']
        data = np.load(f"{DATA_DIR}/TrainNegNeg/train_{i}_s90_seed_{seed}_neg_neg_corr.npz")
        nn = data['xi_nn']
        data = np.load(f"{DATA_DIR}/TrainPosNeg/train_{i}_s90_seed_{seed}_pos_neg_corr.npz")
        pn = data['xi_pn']
        gQ = (pp + nn - 2*pn)/4
        gQ_s90_rands.append(gQ)
        pp_s90_rands.append(pp)
        nn_s90_rands.append(nn)
        pn_s90_rands.append(pn)
    pp_s90.append(np.mean(pp_s90_rands, axis=0))
    nn_s90.append(np.mean(nn_s90_rands, axis=0))
    pn_s90.append(np.mean(pn_s90_rands, axis=0))
    mean_gQ = np.mean(gQ_s90_rands, axis=0)
    std_gQ = np.std(gQ_s90_rands, axis=0)
    gQ_s90.append(mean_gQ)
    gQ_s90_stds.append(std_gQ)
pp_nn_pn_s90 = np.concatenate([pp_s90, nn_s90, pn_s90], axis=1)
print(pp_nn_pn_s90.shape) 

# min_sep=90., max_sep=600, n_bins=10 theta is meanr, bintype=log
bin_edges = np.logspace(np.log10(90), np.log10(600), 11)
theta = 0.5 * (bin_edges[:-1] + bin_edges[1:])
print(theta)
# load data for covmat
COV_DIR = '/home/jbai/cephfs_copy/cephfs/jbai/DefectMockGaussian'
# /PosNeg/realisation_{i}_s{s}_pos_neg_corr.npz
cov_data = []
cov_pp_nn_pn = []
for i in tqdm(range(1000), desc="Loading covariance data"):
    s = 90
    data = np.load(f"{COV_DIR}/PosPos/realisation_{i}_s{s}_pos_pos_corr.npz")
    pp = data['xi_pp']
    data = np.load(f"{COV_DIR}/NegNeg/realisation_{i}_s{s}_neg_neg_corr.npz")
    nn = data['xi_nn']
    data = np.load(f"{COV_DIR}/PosNeg/realisation_{i}_s{s}_pos_neg_corr.npz")
    pn = data['xi_pn']
    gQ = (pp + nn - 2*pn)/4
    dv_pp_nn_pn = np.concatenate([pp, nn, pn])
    cov_pp_nn_pn.append(dv_pp_nn_pn)
    cov_data.append(gQ)
cov_data = np.array(cov_data)
cov_pp_nn_pn = np.array(cov_pp_nn_pn)
# check mean and std of cov_data
print("Mean of cov_data:", np.mean(cov_data, axis=0))
print("Std of cov_data:", np.std(cov_data, axis=0))
# check covariance matrix
cov_matrix = np.cov(cov_data, rowvar=False)
cov_matrix_pp_nn_pn = np.cov(cov_pp_nn_pn, rowvar=False)
# compute fisher matrix
# first compute derivative of gQ w.r.t. Om and sigma8 using finite difference
# we have 4 cosmologies: (Om, sigma8) = (fid), (fid + delta), (fid - delta), (fid + delta_Om, fid - delta_sigma8), (fid - delta_Om, fid + delta_sigma8)
# we can use central difference to compute derivative
# e.g. dgQ/dOm = (gQ(Om + delta_Om) - gQ(Om - delta_Om))/(2*delta_Om)
gQ_s90 = np.array(gQ_s90)
gQ_s90 = gQ_s90.reshape(4, -1) # shape (4, n_theta)
delta_Om = delta_params['Om']
delta_sigma8 = delta_params['sigma8']
dgQ_dOm = (gQ_s90[1] - gQ_s90[0])/(2*delta_Om)
dgQ_dsigma8 = (gQ_s90[3] - gQ_s90[2])/(2*delta_sigma8)

pp_nn_pn_s90 = np.array(pp_nn_pn_s90)
dppnnpn_dOm = (pp_nn_pn_s90[1] - pp_nn_pn_s90[0])/(2*delta_Om)
dppnnpn_dsigma8 = (pp_nn_pn_s90[3] - pp_nn_pn_s90[2])/(2*delta_sigma8)

# compute all rand derivatives for error bars
dgQ_dOm_rands = []
dgQ_dsigma8_rands = []
for seed in tqdm(range(100), desc="Computing derivatives for error bars"):
    gQ_s90_rands = []
    for i in range(4):
        data = np.load(f"{DATA_DIR}/TrainPosPos/train_{i}_s90_seed_{seed}_pos_pos_corr.npz")
        pp = data['xi_pp']
        data = np.load(f"{DATA_DIR}/TrainNegNeg/train_{i}_s90_seed_{seed}_neg_neg_corr.npz")
        nn = data['xi_nn']
        data = np.load(f"{DATA_DIR}/TrainPosNeg/train_{i}_s90_seed_{seed}_pos_neg_corr.npz")
        pn = data['xi_pn']
        
        gQ = (pp + nn - 2*pn)/4
        gQ_s90_rands.append(gQ)
    gQ_s90_rands = np.array(gQ_s90_rands)
    dgQ_dOm_rand = (gQ_s90_rands[1] - gQ_s90_rands[0])/(2*delta_Om)
    dgQ_dsigma8_rand = (gQ_s90_rands[3] - gQ_s90_rands[2])/(2*delta_sigma8)
    dgQ_dOm_rands.append(dgQ_dOm_rand)
    dgQ_dsigma8_rands.append(dgQ_dsigma8_rand)
dgQ_dOm_rands = np.array(dgQ_dOm_rands)
dgQ_dsigma8_rands = np.array(dgQ_dsigma8_rands)
dgQ_dOm_stds = np.std(dgQ_dOm_rands, axis=0)
dgQ_dsigma8_stds = np.std(dgQ_dsigma8_rands, axis=0)

# now compute fisher matrix
inv_cov = np.linalg.inv(cov_matrix)
dv_len = len(theta)
n_realisation = cov_data.shape[0]
hartlap_factor = (n_realisation - dv_len - 2) / (n_realisation - 1)
inv_cov *= hartlap_factor
Fisher = np.zeros((2, 2))
Fisher[0, 0] = np.dot(dgQ_dOm, np.dot(inv_cov, dgQ_dOm))
Fisher[0, 1] = np.dot(dgQ_dOm, np.dot(inv_cov, dgQ_dsigma8))
Fisher[1, 0] = np.dot(dgQ_dsigma8, np.dot(inv_cov, dgQ_dOm))
Fisher[1, 1] = np.dot(dgQ_dsigma8, np.dot(inv_cov, dgQ_dsigma8))
print("Fisher matrix:\n", Fisher)
# compute parameter constraints
cov_params = np.linalg.inv(Fisher)
sigma_Om = np.sqrt(cov_params[0, 0])
sigma_sigma8 = np.sqrt(cov_params[1, 1])
print("Sigma Om:", sigma_Om)
print("Sigma sigma8:", sigma_sigma8)

# compute fisher matrix using pp, nn, pn data
inv_cov_pp_nn_pn = np.linalg.inv(cov_matrix_pp_nn_pn)
inv_cov_pp_nn_pn *= hartlap_factor
Fisher_pp_nn_pn = np.zeros((2, 2))
Fisher_pp_nn_pn[0, 0] = np.dot(dppnnpn_dOm, np.dot(inv_cov_pp_nn_pn, dppnnpn_dOm))
Fisher_pp_nn_pn[0, 1] = np.dot(dppnnpn_dOm, np.dot(inv_cov_pp_nn_pn, dppnnpn_dsigma8))
Fisher_pp_nn_pn[1, 0] = np.dot(dppnnpn_dsigma8, np.dot(inv_cov_pp_nn_pn, dppnnpn_dOm))
Fisher_pp_nn_pn[1, 1] = np.dot(dppnnpn_dsigma8, np.dot(inv_cov_pp_nn_pn, dppnnpn_dsigma8))
print("Fisher matrix using pp, nn, pn:\n", Fisher_pp_nn_pn)
cov_params_pp_nn_pn = np.linalg.inv(Fisher_pp_nn_pn)
sigma_Om_pp_nn_pn = np.sqrt(cov_params_pp_nn_pn[0, 0])
sigma_sigma8_pp_nn_pn = np.sqrt(cov_params_pp_nn_pn[1, 1])
print("Sigma Om using pp, nn, pn:", sigma_Om_pp_nn_pn)
print("Sigma sigma8 using pp, nn, pn:", sigma_sigma8_pp_nn_pn)

# analytical fisher derivative with jax
from DefectCF import DefectCF
cosmo_params = {
    "Omega_c": fid['Om'] - Ob,
    "sigma8": fid['sigma8'],
    "h": h,
    "w": w,
}
ell = np.arange(1, 512*3)
defect_cf = DefectCF(cosmo_params, ell_max=512*3, theta_arcmin=theta, smooth_arcmin=90)
fisher_derivative_Om = defect_cf.get_fisher_derivative("Omega_c")
fisher_derivative_sigma8 = defect_cf.get_fisher_derivative("sigma8")
Fisher_analytic = np.zeros((2, 2))
Fisher_analytic[0, 0] = np.dot(fisher_derivative_Om, np.dot(inv_cov, fisher_derivative_Om))
Fisher_analytic[0, 1] = np.dot(fisher_derivative_Om, np.dot(inv_cov, fisher_derivative_sigma8))
Fisher_analytic[1, 0] = np.dot(fisher_derivative_sigma8, np.dot(inv_cov, fisher_derivative_Om))
Fisher_analytic[1, 1] = np.dot(fisher_derivative_sigma8, np.dot(inv_cov, fisher_derivative_sigma8))
print("Analytical Fisher matrix:\n", Fisher_analytic)
cov_params_analytic = np.linalg.inv(Fisher_analytic)
sigma_Om_analytic = np.sqrt(cov_params_analytic[0, 0])
sigma_sigma8_analytic = np.sqrt(cov_params_analytic[1, 1])
print("Analytical Sigma Om:", sigma_Om_analytic)
print("Analytical Sigma sigma8:", sigma_sigma8_analytic)

from getdist import plots, MCSamples
from getdist.gaussian_mixtures import GaussianND
import matplotlib as mpl
mpl.rcParams['xtick.direction'] = 'in'
mpl.rcParams['ytick.direction'] = 'in'
mpl.rcParams['font.size'] = 6
# use tab10 colors
color_map = plt.get_cmap('tab10')
# plot constraints 
g = plots.get_subplot_plotter(width_inch=3)
g.settings.num_plot_contours = 1
samples = GaussianND(mean=[fid['Om'], fid['sigma8']], cov=cov_params, names=['Om', 'sigma8'], labels=[r'\Omega_{\rm m}', r'\sigma_8'])
samples_analytic = GaussianND(mean=[fid['Om'], fid['sigma8']], cov=cov_params_analytic, names=['Om', 'sigma8'], labels=[r'\Omega_{\rm m}', r'\sigma_8'])
samples_pp_nn_pn = GaussianND(mean=[fid['Om'], fid['sigma8']], cov=cov_params_pp_nn_pn, names=['Om', 'sigma8'], labels=[r'\Omega_{\rm m}', r'\sigma_8'])

# Build two sigma8-derivative variants per bin from |v| ± std (keep original sign):
# closest-to-zero branch gives weaker sigma8 response (outer contour),
# farthest-from-zero branch gives stronger sigma8 response (inner contour).
dg = dgQ_dsigma8
sd = dgQ_dsigma8_stds
abs_dg = np.abs(dg)
sign_dg = np.sign(dg)
sign_dg[sign_dg == 0] = 1.0
abs_closest = np.clip(abs_dg - sd, 0.0, None)
abs_furthest = abs_dg + sd
dg_sigma8_closest = sign_dg * abs_closest
dg_sigma8_furthest = sign_dg * abs_furthest

# Fisher variants using only the sigma8 derivative variants
def fisher_with_sigma8_variant(dg_sigma8_variant):
    F = np.zeros((2, 2))
    F[0, 0] = np.dot(dgQ_dOm, np.dot(inv_cov, dgQ_dOm))
    F[0, 1] = np.dot(dgQ_dOm, np.dot(inv_cov, dg_sigma8_variant))
    F[1, 0] = np.dot(dg_sigma8_variant, np.dot(inv_cov, dgQ_dOm))
    F[1, 1] = np.dot(dg_sigma8_variant, np.dot(inv_cov, dg_sigma8_variant))
    return F

F_closest = fisher_with_sigma8_variant(dg_sigma8_closest)
F_furthest = fisher_with_sigma8_variant(dg_sigma8_furthest)

def robust_inverse_fisher(F, eps_list=(1e-10, 1e-8, 1e-6, 1e-4, 1e-2)):
    try:
        return np.linalg.inv(F)
    except np.linalg.LinAlgError:
        I = np.eye(F.shape[0])
        for eps in eps_list:
            try:
                return np.linalg.inv(F + eps * I)
            except np.linalg.LinAlgError:
                continue
        raise

cov_params_closest = robust_inverse_fisher(F_closest)
cov_params_furthest = robust_inverse_fisher(F_furthest)

samples_closest = GaussianND(mean=[fid['Om'], fid['sigma8']], cov=cov_params_closest, names=['Om', 'sigma8'], labels=[r'\Omega_{\rm m}', r'\sigma_8'])
samples_furthest = GaussianND(mean=[fid['Om'], fid['sigma8']], cov=cov_params_furthest, names=['Om', 'sigma8'], labels=[r'\Omega_{\rm m}', r'\sigma_8'])

# one sigma contour (numerical vs analytical)
g.plot_2d([samples, samples_analytic, samples_closest], ['Om', 'sigma8'], 
          filled=False, colors=['red', 'black', 'red'], ls=['solid', 'dashed', 'dotted'],
          contour_labels=[r'Numerical $g_{\rm Q}(\theta)$', r'Analytical $g_{\rm Q}(\theta)$', r'Conservative Numerical Bound'], contour_colors=['red', 'black', 'red'], contour_ls=['solid', 'dashed', 'dashdot'],
          lims = [0.3-1, 0.3+1, 0.8-2, 0.8+2],
          lws=2)


g.add_legend(legend_labels=[r'Numerical $g_{\rm Q}(\theta)$', r'Analytical $g_{\rm Q}(\theta)$', r'Conservative Numerical Bound'], legend_loc='upper right', fontsize=4)
plt.savefig(f"outfiles/Fisher_validation.png",dpi=300)
del g

# plot derivative of gQ w.r.t. Om and sigma8
from matplotlib.ticker import FixedLocator, FixedFormatter
fig, axs = plt.subplots(1, 2, figsize=(7, 3), sharey=True, sharex=True)
xticks = [100, 200, 300, 400, 600]
xtick_labels = [r'$10^2$', r'$2\times10^2$', r'$3\times10^2$', r'$4\times10^2$', r'$6\times10^2$']
ax = axs[0]
ax.errorbar(theta, dgQ_dOm, yerr=dgQ_dOm_stds/np.sqrt(100), label=r'$\partial g_{\rm Q} / \partial \Omega_{\rm m}$ Numerical', capsize=3, fmt='o')
ax.plot(theta, fisher_derivative_Om, label=r'$\partial g_{\rm Q} / \partial \Omega_{\rm m}$ Analytical', linestyle='dashed')
ax.set_xscale('log')
ax.set_xlim(90, 600)
ax.xaxis.set_major_locator(FixedLocator(xticks))
ax.xaxis.set_major_formatter(FixedFormatter(xtick_labels[:-1] + ['']))
ax.set_xlabel(r'$\theta$ [arcmin]')
ax.set_ylabel(r'$\partial g_{\rm Q} / \partial p$')
ax.legend()    
ax = axs[1]
ax.errorbar(theta, dgQ_dsigma8, yerr=dgQ_dsigma8_stds/np.sqrt(100), label=r'$\partial g_{\rm Q} / \partial \sigma_8$ Numerical', capsize=3, fmt='o')
ax.plot(theta, fisher_derivative_sigma8, label=r'$\partial g_{\rm Q} / \partial \sigma_8$ Analytical', linestyle='dashed')
ax.set_xscale('log')
ax.set_xlabel(r'$\theta$ [arcmin]')
ax.set_xlim(90, 600)
ax.xaxis.set_major_locator(FixedLocator(xticks))
ax.xaxis.set_major_formatter(FixedFormatter([''] + xtick_labels[1:]))
ax.legend()
plt.subplots_adjust(hspace=0, wspace=0)
plt.savefig(f"outfiles/Fisher_derivative_validation.png", dpi=300, bbox_inches='tight')