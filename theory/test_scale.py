import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import jax_cosmo as jc
import matplotlib.pyplot as plt
from tqdm import tqdm

from jax.tree_util import register_pytree_node_class

# ============================================================
# 1. Multipole range
# ============================================================
ell_full = jnp.arange(2, 5000)
ell_cuts = jnp.logspace(1.5, jnp.log10(5000), 100).astype(int)

# ============================================================
# 2. Source redshift distribution
# ============================================================
@register_pytree_node_class
class CustomNZ(jc.redshift.redshift_distribution):
    def pz_fn(self, z):
        A = 1.8048
        a = 0.4170
        b = 4.8685
        c = 0.7841
        return A * (z**a + z**(a*b)) / (z**b + c)

nz = CustomNZ(zmax=3.0)  # 确保 z_max 足够大
wl_probe = jc.probes.WeakLensing([nz])

# ============================================================
# 3. Cosmology constructor
# ============================================================
def make_cosmo(Omega_m, sigma8):
    return jc.Cosmology(
        Omega_c=Omega_m - 0.05,
        Omega_b=0.05,
        h=0.67,
        n_s=0.965,
        sigma8=sigma8,
        w0=-1.0,
        wa=0.0,
        Omega_k=0.0
    )

# ============================================================
# 4. Convergence power spectrum + shape noise
# ============================================================
def Cl_kappa(Omega_m, sigma8, ell=ell_full, with_noise=True):
    cosmo = make_cosmo(Omega_m, sigma8)
    Cl_sig = jc.angular_cl.angular_cl(cosmo, ell, [wl_probe])[0]
    
    # -----------------------
    # Shape noise
    # -----------------------
    sigma_e = 0.26                 
    n_eff_arcmin2 = 30.0           
    arcmin2_to_sr = (180.0 / jnp.pi)**2 * 60.0**2
    n_eff = n_eff_arcmin2 * arcmin2_to_sr
    Nl = sigma_e**2 / (2.0 * n_eff)
    
    if with_noise:
        return Cl_sig + Nl
    else:
        return Cl_sig

# ============================================================
# 5. Full defect formula (带 low/high ell cut)
# ============================================================
def defect_full(params, ell_max, with_noise):
    Omega_m, sigma8 = params
    ell = ell_full
    Cl_k = Cl_kappa(Omega_m, sigma8, ell, with_noise=with_noise)
    
    # Survey mask: 5x5 deg^2 -> ell_min ~ 36
    ell_min = 2
    mask = jnp.where((ell <= ell_max) & (ell >= ell_min), 1.0, 0.0)
    
    # Integrand common factor
    common = (2*ell + 1) / (ell * (ell+1)) * Cl_k * mask
    
    # Numerator & Denominator weights
    num = jnp.sum(common * (ell-1)*(ell+2)*(ell-2)*(ell+3))
    den = jnp.sum(common * (ell-1)*(ell+2))
    
    return (3.0 / (4.0 * jnp.pi)) * (num / den)

# ============================================================
# 6. Gradient w.r.t parameters
# ============================================================
def defect_noisy(params, ell_max):
    return defect_full(params, ell_max, with_noise=True)

def defect_noiseless(params, ell_max):
    return defect_full(params, ell_max, with_noise=False)

# d_defect_dParams_noisy = jax.grad(defect_noisy, argnums=0)
# d_defect_dParams_noisy_jit = jax.jit(d_defect_dParams_noisy)

# d_defect_dParams_noiseless = jax.grad(defect_noiseless, argnums=0)
# d_defect_dParams_noiseless_jit = jax.jit(d_defect_dParams_noiseless)


# # ============================================================
# # 7. Evaluate derivatives for different ell_max
# # ============================================================
# # Om0 = 0.3
# # s8_0 = 0.8
# # params0 = jnp.array([Om0, s8_0])
# # ell_cuts_arr = jnp.array(ell_cuts)

# # d_defect_dOm_noisy_vals = []
# # d_defect_ds8_noisy_vals = []
# # d_defect_dOm_noiseless_vals = []
# # d_defect_ds8_noiseless_vals = []

# # for ell_max in tqdm(ell_cuts_arr):
# #     # Noisy case
# #     grads_n = d_defect_dParams_noisy_jit(params0, ell_max)
# #     d_defect_dOm_noisy_vals.append(grads_n[0])
# #     d_defect_ds8_noisy_vals.append(grads_n[1])
    
# #     # Noiseless case
# #     grads_nl = d_defect_dParams_noiseless_jit(params0, ell_max)
# #     d_defect_dOm_noiseless_vals.append(grads_nl[0])
# #     d_defect_ds8_noiseless_vals.append(grads_nl[1])

# # d_defect_dOm_noisy_vals = jnp.array(d_defect_dOm_noisy_vals)
# # d_defect_ds8_noisy_vals = jnp.array(d_defect_ds8_noisy_vals)
# # d_defect_dOm_noiseless_vals = jnp.array(d_defect_dOm_noiseless_vals)
# # d_defect_ds8_noiseless_vals = jnp.array(d_defect_ds8_noiseless_vals)

# # # ============================================================
# # # 8. Plot
# # # ============================================================
# import matplotlib as mpl
# mpl.rcParams['xtick.direction'] = 'in'
# mpl.rcParams['ytick.direction'] = 'in'
# mpl.rcParams['font.size'] = 8
# color_map = plt.get_cmap('tab10')

# # fig, ax1 = plt.subplots(figsize=(4, 3))

# # # Plot Noisy (Solid)
# # ax1.plot(ell_cuts_arr, d_defect_dOm_noisy_vals, 
# #          label=r'$\partial d / \partial \Omega_{\rm m}$ (Noisy)', 
# #          color=color_map(0), linestyle='-')
# # ax1.plot(ell_cuts_arr, d_defect_ds8_noisy_vals, 
# #          label=r'$\partial d / \partial \sigma_8$ (Noisy)', 
# #          color=color_map(1), linestyle='-')

# # # Plot Noiseless (Dashed)
# # ax1.plot(ell_cuts_arr, d_defect_dOm_noiseless_vals, 
# #          label=r'$\partial d / \partial \Omega_{\rm m}$ (Noise-free)', 
# #          color=color_map(0), linestyle='--')
# # ax1.plot(ell_cuts_arr, d_defect_ds8_noiseless_vals, 
# #          label=r'$\partial d / \partial \sigma_8$ (Noise-free)', 
# #          color=color_map(1), linestyle='--')

# # ax1.set_xlabel(r'$\ell_{\rm max}$')
# # ax1.set_ylabel(r'$\partial d / \partial p$')
# # ax1.axhline(0, color='gray', linestyle='--', linewidth=0.8, alpha=0.5)
# # ax1.axvline(300, color='blue', linestyle='--', linewidth=0.8, alpha=0.8)
# # ax1.axvline(2500, color='red', linestyle='--', linewidth=0.8, alpha=0.8)
# # # ax1.set_xscale('log')
# # ax1.set_xticks([300, 1300, 2500, 4000])
# # ax1.set_xticklabels([r'Full-sky(300)', r'$1300$', r'Euclid-like(2500)', r'$4000$'])
# # ax1.minorticks_off()
# # # Use a slightly smaller font for legend to fit everything
# # ax1.legend(loc='best', fontsize=6)

# # plt.tight_layout()
# # plt.savefig("outfiles/derivative_vs_ellmax_with_noise.png", dpi=300)
# # plt.close()

# # ============================================================
# # 9. Plot Density vs Parameters (Combined Panel)
# # ============================================================
# print("Calculating density vs parameters...")

# l_max_fixed = 2500
# Om_range = jnp.linspace(0.1, 0.5, 50)
# s8_range = jnp.linspace(0.6, 1.0, 50)

# # JIT compile the evaluation functions for speed
# defect_noisy_jit = jax.jit(defect_noisy)
# defect_noiseless_jit = jax.jit(defect_noiseless)

# # --- Define function for Pure Noise case ---
# def defect_pure_noise(ell_max):
#     # Calculate Nl (Shape noise) same as in Cl_kappa
#     sigma_e = 0.26                 
#     n_eff_arcmin2 = 30.0           
#     arcmin2_to_sr = (180.0 / jnp.pi)**2 * 60.0**2
#     n_eff = n_eff_arcmin2 * arcmin2_to_sr
#     Nl = sigma_e**2 / (2.0 * n_eff)
    
#     # Cl is just constant noise
#     ell = ell_full
#     # FIX: Ensure float dtype, otherwise Nl (small float) is cast to int(0) by full_like(ell)
#     Cl_k = jnp.ones_like(ell, dtype=jnp.float64) * Nl
    
#     # Survey mask
#     ell_min = 2
#     mask = jnp.where((ell <= ell_max) & (ell >= ell_min), 1.0, 0.0)
    
#     common = (2*ell + 1) / (ell * (ell+1)) * Cl_k * mask
#     num = jnp.sum(common * (ell-1)*(ell+2)*(ell-2)*(ell+3))
#     den = jnp.sum(common * (ell-1)*(ell+2))
    
#     return (3.0 / (4.0 * jnp.pi)) * (num / den)

# defect_pure_noise_jit = jax.jit(defect_pure_noise)
# density_noise_val = defect_pure_noise_jit(l_max_fixed)
# print(f"  Pure noise density: {density_noise_val:.5f}")

# # 1. Varying Om (Fixed s8=0.8)
# densities_Om_noisy = []
# densities_Om_clean = []
# s8_fixed = 0.8
# print("  Varying Omega_m...")
# for om in tqdm(Om_range, leave=False):
#     p = jnp.array([om, s8_fixed])
#     densities_Om_noisy.append(defect_noisy_jit(p, l_max_fixed))
#     densities_Om_clean.append(defect_noiseless_jit(p, l_max_fixed))

# # 2. Varying s8 (Fixed Om=0.3)
# densities_s8_noisy = []
# densities_s8_clean = []
# Om_fixed = 0.3
# print("  Varying sigma8...")
# for s8 in tqdm(s8_range, leave=False):
#     p = jnp.array([Om_fixed, s8])
#     densities_s8_noisy.append(defect_noisy_jit(p, l_max_fixed))
#     densities_s8_clean.append(defect_noiseless_jit(p, l_max_fixed))

# # Convert to arrays
# densities_Om_noisy = jnp.array(densities_Om_noisy)
# densities_Om_clean = jnp.array(densities_Om_clean)
# densities_s8_noisy = jnp.array(densities_s8_noisy)
# densities_s8_clean = jnp.array(densities_s8_clean)

# # --- Plotting Combined ---
# fig, ax1 = plt.subplots(figsize=(4, 3))
# ax2 = ax1.twiny() # Create a second x-axis sharing the same y-axis

# # Plot Lines associated with Omega_m (Blue tones, ax1)
# l1, = ax1.plot(Om_range, densities_Om_noisy, color='C0', linestyle='-', label=r'Noisy ($\Omega_{\rm m}$)')
# l2, = ax1.plot(Om_range, densities_Om_clean, color='C0', linestyle='--', label=r'Noise-free ($\Omega_{\rm m}$)')

# # Plot Lines associated with Sigma_8 (Orange tones, ax2)
# l3, = ax2.plot(s8_range, densities_s8_noisy, color='C1', linestyle='-', label=r'Noisy ($\sigma_8$)')
# l4, = ax2.plot(s8_range, densities_s8_clean, color='C1', linestyle='--', label=r'Noise-free ($\sigma_8$)')
# # Plot Pure Noise (Horizontal line)
# l5 = ax1.axhline(density_noise_val, color='k', linestyle='-.', label='Noise only')

# # Setup Axes Labels
# ax1.set_xticks([0.1, 0.2, 0.3, 0.4, 0.5])
# ax2.set_xticks([0.6, 0.7, 0.8, 0.9, 1.0])
# ax1.set_xlabel(r'$\Omega_{\rm m}$')
# ax2.set_xlabel(r'$\sigma_8$')
# ax1.set_ylabel(r'Defect Density')
# ax1.text(0.7, 0.1, r'$\ell_{\rm max} = 2500$', transform=ax1.transAxes, fontsize=9, verticalalignment='top')

# # Grid and Ticks
# # ax1.grid(True, linestyle=':', alpha=0.6)
# ax1.minorticks_off() # Change from on to off
# ax2.minorticks_off() # Change from on to off

# # Combined Legend
# lines = [l1, l2, l3, l4, l5]
# labels = [l.get_label() for l in lines]
# # Place legend effectively
# ax1.legend(lines, labels, loc=(0.65, 0.6), fontsize=6, framealpha=0.9)
# plt.yscale('log')
# plt.tight_layout()
# plt.savefig("outfiles/density_vs_params_lmax2500.png", dpi=300)
# print("Saved outfiles/density_vs_params_lmax2500.png")


# compute 3rd order derivative for a fiducial, w/o noise, at lmax=300 only for Om, use finite difference
params_fid = jnp.array([0.3, 0.8])
ell_max_test = 300
def finite_diff_3rd_order(func, params, ell_max, h=1e-5):
    # 3rd order central finite difference for the first parameter (Omega_m)
    p_plus_2h = params.at[0].add(2*h)
    p_plus_h = params.at[0].add(h)
    p_minus_h = params.at[0].add(-h)
    p_minus_2h = params.at[0].add(-2*h)
    
    f_plus_2h = func(p_plus_2h, ell_max)
    f_plus_h = func(p_plus_h, ell_max)
    f_minus_h = func(p_minus_h, ell_max)
    f_minus_2h = func(p_minus_2h, ell_max)
    
    return (f_minus_2h - 2*f_minus_h + 2*f_plus_h - f_plus_2h) / (2 * h**3)

third_derivative_om = finite_diff_3rd_order(defect_noiseless, params_fid, ell_max_test)
print(f"3rd order derivative w.r.t Omega_m at params={params_fid}, ell_max={ell_max_test}: {third_derivative_om:.5e}")
