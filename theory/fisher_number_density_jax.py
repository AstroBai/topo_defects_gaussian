import os

# Force CPU to avoid CUDA plugin issues on shared nodes.
os.environ.setdefault("JAX_PLATFORMS", "cuda")

import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import jax_cosmo as jc
import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
from getdist.gaussian_mixtures import GaussianND
from getdist import plots

from jax.tree_util import register_pytree_node_class


@register_pytree_node_class
class CustomNZ(jc.redshift.redshift_distribution):
    """Source redshift distribution used in test_scale.py."""

    def pz_fn(self, z):
        A = 1.8048
        a = 0.4170
        b = 4.8685
        c = 0.7841
        return A * (z**a + z ** (a * b)) / (z**b + c)


def make_cosmo(omega_m, sigma8):
    return jc.Cosmology(
        Omega_c=omega_m - 0.05,
        Omega_b=0.05,
        h=0.67,
        n_s=0.965,
        sigma8=sigma8,
        w0=-1.0,
        wa=0.0,
        Omega_k=0.0,
    )


def noise_cl_shape(ell, sigma_e=0.26, n_eff_arcmin2=30.0):
    """Shape-noise power spectrum (white noise)."""
    arcmin2_to_sr = (180.0 / jnp.pi)**2 * 60.0**2
    n_eff_sr = n_eff_arcmin2 * arcmin2_to_sr
    nl = sigma_e**2 / (2.0 * n_eff_sr)
    return jnp.ones_like(ell, dtype=jnp.float64) * nl


def cl_kappa(omega_m, sigma8, ell, wl_probe, include_noise=True, sigma_e=0.26, n_eff_arcmin2=30.0):
    cosmo = make_cosmo(omega_m, sigma8)
    # one WL bin -> first (and only) spectrum
    cl_sig = jc.angular_cl.angular_cl(cosmo, ell, [wl_probe])[0]
    if include_noise:
        return cl_sig + noise_cl_shape(ell, sigma_e=sigma_e, n_eff_arcmin2=n_eff_arcmin2)
    return cl_sig
    
def gaussian_beam_sq(ell, sigma_rad):
    """Power-spectrum smoothing factor for Gaussian beam."""
    return jnp.exp(-ell * (ell + 1.0) * sigma_rad**2)


def defect_density_from_cl(cl, ell, ell_min, ell_max):
    """Same estimator structure as in test_scale.py, with ell cuts."""
    mask = jnp.where((ell >= ell_min) & (ell <= ell_max), 1.0, 0.0)

    pref = (2.0 * ell + 1.0) / (ell * (ell + 1.0))
    w_num = pref * (ell - 1.0) * (ell + 2.0) * (ell - 2.0) * (ell + 3.0) * mask
    w_den = pref * (ell - 1.0) * (ell + 2.0) * mask

    num = jnp.sum(w_num * cl)
    den = jnp.sum(w_den * cl)

    density = (3.0 / (4.0 * jnp.pi)) * (num / den)
    return density, num, den, w_num, w_den


def defect_counts_datavector(params, ell, ell_min, ell_max, wl_probe, sigma_list_rad, area_sr,
                             include_noise=True, sigma_e=0.26, n_eff_arcmin2=30.0):
    """Datavector: defect counts at multiple Gaussian smoothing scales."""
    omega_m, sigma8 = params
    cl = cl_kappa(
        omega_m, sigma8, ell, wl_probe,
        include_noise=include_noise,
        sigma_e=sigma_e,
        n_eff_arcmin2=n_eff_arcmin2,
    )
    densities = []
    for sigma_rad in sigma_list_rad:
        cl_s = cl * gaussian_beam_sq(ell, sigma_rad)
        density, _, _, _, _ = defect_density_from_cl(cl_s, ell, ell_min, ell_max)
        densities.append(density)
    densities = jnp.array(densities)
    # Convert density [sr^-1] to counts in survey area.
    return densities * area_sr


def datavector_cov_cosmic_variance(cl, ell, ell_min, ell_max, f_sky, sigma_list_rad, area_sr):
    """Propagate Gaussian cosmic variance of Cl to datavector covariance."""
    A = 3.0 / (4.0 * jnp.pi)
    data_vec = []
    jac_rows = []

    for sigma_rad in sigma_list_rad:
        b2 = gaussian_beam_sq(ell, sigma_rad)
        cl_s = cl * b2
        density, num, den, w_num, w_den = defect_density_from_cl(cl_s, ell, ell_min, ell_max)

        # dd/dCl_s at this smoothing
        dd_dcls = A * (w_num * den - num * w_den) / (den**2)
        # Chain rule: dCl_s/dCl = B_l^2
        dd_dcl = dd_dcls * b2

        data_vec.append(density)
        jac_rows.append(dd_dcl)

    data_vec = jnp.array(data_vec)
    jac = jnp.stack(jac_rows, axis=0)  # [n_data, n_ell]

    # Var(C_ell) = 2 C_ell^2 / ((2ell+1) f_sky)
    var_cl = 2.0 * cl**2 / ((2.0 * ell + 1.0) * f_sky)

    weighted_jac = jac * var_cl[None, :]
    cov_dens = weighted_jac @ jac.T  # [n_data, n_data]

    # Convert density covariance to counts covariance.
    counts = data_vec * area_sr
    cov_counts = cov_dens * area_sr**2
    return counts, cov_counts


def fisher_number_density(params_fid, ell, ell_min, ell_max, f_sky, wl_probe, sigma_list_rad, area_sr,
                          include_noise=True, sigma_e=0.26, n_eff_arcmin2=30.0):
    cl_fid = cl_kappa(
        params_fid[0], params_fid[1], ell, wl_probe,
        include_noise=include_noise,
        sigma_e=sigma_e,
        n_eff_arcmin2=n_eff_arcmin2,
    )
    data_fid, cov_data = datavector_cov_cosmic_variance(
        cl_fid, ell, ell_min, ell_max, f_sky, sigma_list_rad, area_sr
    )

    jac_fn = jax.jacrev(defect_counts_datavector, argnums=0)
    grad = jac_fn(
        params_fid, ell, ell_min, ell_max, wl_probe, sigma_list_rad, area_sr,
        include_noise, sigma_e, n_eff_arcmin2
    )
    # grad shape: [n_data, 2]

    cov_inv = jnp.linalg.pinv(cov_data)
    fisher = grad.T @ cov_inv @ grad
    return data_fid, cov_data, grad, fisher


def main():
    # -----------------------------
    # Fixed configuration
    # -----------------------------
    ell_grid_max = 5000
    ell_min = 2
    ell_max = 5000
    area_deg2 = 15000
    params_fid = jnp.array([0.3, 0.8], dtype=jnp.float64)
    include_noise = True
    sigma_e = 0.26
    n_eff_arcmin2 = 30.0
    smoothing_arcmin = jnp.array([1.0, 2.0, 3.0, 4.0, 5.0, 8.0, 10.0, 15.0, 20.0, 30.0, 40.0, 60.0, 80.0], dtype=jnp.float64)
    sigma_list_rad = jnp.deg2rad(smoothing_arcmin / 60.0)

    ell = jnp.arange(2, ell_grid_max + 1)

    # Source distribution / probe
    nz = CustomNZ(zmax=3.0)
    wl_probe = jc.probes.WeakLensing([nz])

    # f_sky from survey area
    full_sky_deg2 = 4.0 * jnp.pi * (180.0 / jnp.pi) ** 2
    f_sky = area_deg2 / full_sky_deg2
    area_sr = 4.0 * jnp.pi * f_sky

    data_fid, cov_data, grad, fisher = fisher_number_density(
        params_fid=params_fid,
        ell=ell,
        ell_min=ell_min,
        ell_max=ell_max,
        f_sky=f_sky,
        wl_probe=wl_probe,
        sigma_list_rad=sigma_list_rad,
        area_sr=area_sr,
        include_noise=include_noise,
        sigma_e=sigma_e,
        n_eff_arcmin2=n_eff_arcmin2,
    )

    sigma_data = jnp.sqrt(jnp.diag(cov_data))

    # Fixed-other-parameter projections from full Fisher.
    sigma_om_fix = 1.0 / jnp.sqrt(fisher[0, 0])
    sigma_s8_fix = 1.0 / jnp.sqrt(fisher[1, 1])

    print("=== Fisher Forecast (JAX): Number Density + Cosmic Variance ===")
    print(f"ell grid : [2, {ell_grid_max}] (computed)")
    print(f"ell cut  : [{ell_min}, {ell_max}] (fixed)")
    print(f"smoothing (arcmin): {np.asarray(smoothing_arcmin)}")
    print(f"include noise: {include_noise}, sigma_e={sigma_e}, n_eff={n_eff_arcmin2} arcmin^-2")
    print(f"area     : {area_deg2:.3f} deg^2")
    print(f"f_sky    : {float(f_sky):.6e}")
    print(f"fiducial : Omega_m={float(params_fid[0]):.5f}, sigma8={float(params_fid[1]):.5f}")
    print("data vector N_defects:", np.asarray(data_fid))
    print("sigma(data):", np.asarray(sigma_data))
    print("grad shape [n_data, 2]:", np.asarray(grad).shape)
    print(f"sigma(Omega_m) [fixed sigma8]={float(sigma_om_fix):.6e}")
    print(f"sigma(sigma8)  [fixed Omega_m]={float(sigma_s8_fix):.6e}")

    # -----------------------------
    # Fisher contour plot (getdist)
    # -----------------------------
    mpl.rcParams['xtick.direction'] = 'in'
    mpl.rcParams['ytick.direction'] = 'in'
    mpl.rcParams['font.size'] = 8

    fisher_np = np.asarray(fisher, dtype=float)
    cov = np.linalg.pinv(fisher_np)

    sample = GaussianND(
        mean=np.asarray(params_fid, dtype=float),
        cov=cov,
        names=['Om', 'sigma8'],
        labels=['\\Omega_{\\rm m}', '\\sigma_8']
    )

    g = plots.get_subplot_plotter(width_inch=4)
    g.settings.num_plot_contours = 2
    g.settings.linewidth_contour = 1.5
    g.settings.alpha_factor_contour_lines = 1.0
    g.triangle_plot(
        [sample],
        params=['Om', 'sigma8'],
        filled=True,
        contour_colors=['#1f77b4'],
    )

    g.add_text('ell_max=2500\nGaussian smoothing datavector + CV', x=0.06, y=0.95, color='black')
    plt.savefig('outfiles/fisher_number_density_jax_contour_getdist.png', dpi=300)
    plt.close()

    # Data vector diagnostic plot
    fig, ax = plt.subplots(figsize=(4.2, 3.0))
    ax.errorbar(np.asarray(smoothing_arcmin), np.asarray(data_fid), yerr=np.asarray(sigma_data), fmt='o-', color='#1f77b4', capsize=2)
    ax.set_xlabel('Gaussian smoothing [arcmin]')
    ax.set_ylabel('Defect counts in survey area')
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_title('Datavector used in Fisher')
    plt.tight_layout()
    plt.savefig('outfiles/fisher_number_density_jax_datavector.png', dpi=300)
    plt.close()


if __name__ == "__main__":
    main()
