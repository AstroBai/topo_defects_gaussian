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

d_defect_dParams_noisy = jax.grad(defect_noisy, argnums=0)
d_defect_dParams_noisy_jit = jax.jit(d_defect_dParams_noisy)

d_defect_dParams_noiseless = jax.grad(defect_noiseless, argnums=0)
d_defect_dParams_noiseless_jit = jax.jit(d_defect_dParams_noiseless)

# ============================================================
# 7. Evaluate derivatives for different ell_max
# ============================================================
Om0 = 0.3
s8_0 = 0.8
params0 = jnp.array([Om0, s8_0])
ell_cuts_arr = jnp.array(ell_cuts)

d_defect_dOm_noisy_vals = []
d_defect_ds8_noisy_vals = []
d_defect_dOm_noiseless_vals = []
d_defect_ds8_noiseless_vals = []

for ell_max in tqdm(ell_cuts_arr):
    # Noisy case
    grads_n = d_defect_dParams_noisy_jit(params0, ell_max)
    d_defect_dOm_noisy_vals.append(grads_n[0])
    d_defect_ds8_noisy_vals.append(grads_n[1])
    
    # Noiseless case
    grads_nl = d_defect_dParams_noiseless_jit(params0, ell_max)
    d_defect_dOm_noiseless_vals.append(grads_nl[0])
    d_defect_ds8_noiseless_vals.append(grads_nl[1])

d_defect_dOm_noisy_vals = jnp.array(d_defect_dOm_noisy_vals)
d_defect_ds8_noisy_vals = jnp.array(d_defect_ds8_noisy_vals)
d_defect_dOm_noiseless_vals = jnp.array(d_defect_dOm_noiseless_vals)
d_defect_ds8_noiseless_vals = jnp.array(d_defect_ds8_noiseless_vals)

# ============================================================
# 8. Plot
# ============================================================
import matplotlib as mpl
mpl.rcParams['xtick.direction'] = 'in'
mpl.rcParams['ytick.direction'] = 'in'
mpl.rcParams['font.size'] = 8
color_map = plt.get_cmap('tab10')

fig, ax1 = plt.subplots(figsize=(4, 3))

# Plot Noisy (Solid)
ax1.plot(ell_cuts_arr, d_defect_dOm_noisy_vals, 
         label=r'$\partial d / \partial \Omega_{\rm m}$ (Noisy)', 
         color=color_map(0), linestyle='-')
ax1.plot(ell_cuts_arr, d_defect_ds8_noisy_vals, 
         label=r'$\partial d / \partial \sigma_8$ (Noisy)', 
         color=color_map(1), linestyle='-')

# Plot Noiseless (Dashed)
ax1.plot(ell_cuts_arr, d_defect_dOm_noiseless_vals, 
         label=r'$\partial d / \partial \Omega_{\rm m}$ (Noise-free)', 
         color=color_map(0), linestyle='--')
ax1.plot(ell_cuts_arr, d_defect_ds8_noiseless_vals, 
         label=r'$\partial d / \partial \sigma_8$ (Noise-free)', 
         color=color_map(1), linestyle='--')

ax1.set_xlabel(r'$\ell_{\rm max}$')
ax1.set_ylabel(r'$\partial d / \partial p$')
ax1.axhline(0, color='gray', linestyle='--', linewidth=0.8, alpha=0.5)
ax1.axvline(300, color='blue', linestyle='--', linewidth=0.8, alpha=0.8)
ax1.axvline(2500, color='red', linestyle='--', linewidth=0.8, alpha=0.8)
# ax1.set_xscale('log')
ax1.set_xticks([300, 1300, 2500, 4000])
ax1.set_xticklabels([r'Full-sky(300)', r'$1300$', r'Euclid-like(2500)', r'$4000$'])
ax1.minorticks_off()
# Use a slightly smaller font for legend to fit everything
ax1.legend(loc='best', fontsize=6)

plt.tight_layout()
plt.savefig("outfiles/derivative_vs_ellmax_with_noise.png", dpi=300)
plt.close()